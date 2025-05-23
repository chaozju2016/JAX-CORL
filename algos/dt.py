# source https://github.com/nikhilbarhate99/min-decision-transformer
# https://arxiv.org/abs/2106.01345
import collections
import os
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import minari
import brax
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
# os.environ["WANDB_MODE"] = "offline"


class DTConfig(BaseModel):
    # GENERAL
    algo: str = "DT"
    project: str = "align-decision-transformer"
    seed: int = 0
    env_name: str = "HalfCheetah-v5"
    dataset_name: str = "mujoco/halfcheetah/expert-v0"
    experiment_name: str = "CLIP"
    batch_size: int = 64
    num_eval_episodes: int = 5
    max_eval_ep_len: int = 1000
    max_steps: int = 20000
    eval_interval: int = 2000
    # NETWORK
    context_len: int = 20
    n_blocks: int = 3
    embed_dim: int = 128
    n_heads: int = 1
    dropout_p: float = 0.1
    lr: float = 0.0008
    wt_decay: float = 1e-4
    beta: Sequence = (0.9, 0.999)
    clip_grads: float = 0.25
    warmup_steps: int = 10000
    # DT SPECIFIC
    rtg_scale: int = 1000
    rtg_target: int = None
    # 添加多模态对齐参数
    alignment_weight: float = 0.1  # 对齐损失权重
    alignment_temp: float = 0.1  # 对比学习温度参数


conf_dict = OmegaConf.from_cli()
config: DTConfig = DTConfig(**conf_dict)

# RTG target is specific to each environment
if "HalfCheetah" in config.env_name:
    rtg_target = 12000
elif "Hopper" in config.env_name:
    rtg_target = 3600
elif "Walker" in config.env_name:
    rtg_target = 5000
elif "Humanoid" in config.env_name:
    rtg_target = 12000
elif "InvertedDoublePendulum" in config.env_name:
    rtg_target = 9000
elif "Pusher" in config.env_name:
    rtg_target = 0  # Pusher是一个负回报环境，通常目标是最小化惩罚
elif "Reacher" in config.env_name:
    rtg_target = -4  # Reacher也是一个负回报环境
elif "Swimmer" in config.env_name:
    rtg_target = 300
elif "AntMaze" in config.env_name:
    # AntMaze环境使用稀疏奖励，目标通常是1.0
    rtg_target = 1000
else:
    raise ValueError("Environment not supported.")
config.rtg_target = rtg_target


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MaskedCausalAttention(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, cross: jnp.ndarray = None, training=True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, T, C = x.shape
        input_tensor = x if cross is None else cross

        N, D = self.n_heads, C // self.n_heads
        assert (
            cross is None or cross.shape == x.shape
        ), f"Our cross shape {cross.shape if cross is not None else None} must match x shape {x.shape}"
        # rearrange q, k, v as (B, N, T, D)
        q = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        k = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(input_tensor)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        v = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(input_tensor)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        # causal mask
        ones = jnp.ones((self.max_T, self.max_T))
        mask = jnp.tril(ones).reshape(1, 1, self.max_T, self.max_T)
        # weights (B, N, T, T) jax
        weights = jnp.einsum("bntd,bnfd->bntf", q, k) / jnp.sqrt(D)
        # causal mask applied to weights
        weights = jnp.where(mask[..., :T, :T] == 0, -jnp.inf, weights[..., :T, :T])
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)
        # attention (B, N, T, D)
        attention = nn.Dropout(self.drop_p, deterministic=not training)(
            jnp.einsum("bntf,bnfd->bntd", normalized_weights, v)
        )
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * D)
        out = nn.Dropout(self.drop_p, deterministic=not training)(
            nn.Dense(self.h_dim)(attention)
        )
        return out, normalized_weights


class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training=True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        attn, normalized_weights = MaskedCausalAttention(
            self.h_dim, self.max_T, self.n_heads, self.drop_p
        )(x, training=training)
        x = x + attn  # residual
        x = nn.LayerNorm()(x)
        # MLP
        out = nn.Dense(4 * self.h_dim, kernel_init=self.kernel_init)(x)
        out = nn.gelu(out)
        out = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(out)
        out = nn.Dropout(self.drop_p, deterministic=not training)(out)
        # residual
        x = x + out
        x = nn.LayerNorm()(x)
        return x, normalized_weights


class DecisionTransformer(nn.Module):
    state_dim: int
    act_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    max_timestep: int = 4096
    kernel_init: Callable = default_init()

    def setup(self) -> None:
        self.blocks = [
            Block(self.h_dim, 3 * self.context_len, self.n_heads, self.drop_p)
            for _ in range(self.n_blocks)
        ]
        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm()
        self.embed_timestep = nn.Embed(self.max_timestep, self.h_dim)
        self.embed_rtg = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.embed_state = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        # continuous actions
        self.embed_action = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.use_action_tanh = True
        # prediction heads
        self.predict_rtg = nn.Dense(1, kernel_init=self.kernel_init)
        self.predict_state = nn.Dense(self.state_dim, kernel_init=self.kernel_init)
        self.predict_action = nn.Dense(self.act_dim, kernel_init=self.kernel_init)

        # alignment projector
        self.alignment_projector = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        # self.action_projector = nn.Dense(self.h_dim, kernel_init=self.kernel_init)

        # alignment attention
        self.alignment_attention = MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.context_len,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
        )

    def __call__(
        self,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
        training=True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[int, jnp.ndarray]]:
        B, T, _ = states.shape
        attns = {}

        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = (
            jnp.stack((returns_embeddings, state_embeddings, action_embeddings), axis=1)
            .transpose(0, 2, 1, 3)
            .reshape(B, 3 * T, self.h_dim)
        )
        h = self.embed_ln(h)
        # transformer and prediction
        for i, block in enumerate(self.blocks):
            h, attn = block(h, training=training)
            attns[i] = attn.copy()
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)
        # get predictions
        return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict next action given r, s
        if self.use_action_tanh:
            action_preds = jnp.tanh(action_preds)

        # return state_preds, action_preds, return_preds, attns

        # 在返回前，提取重塑后的状态和动作表示
        # reshape后h的形状是 (B, 3, T, h_dim)
        return_repr = h[:, 0]  # 提取回报表示
        state_repr = h[:, 1]  # 提取状态表示
        action_repr = h[:, 2]  # 提取动作表示

        use_r_condition_attn = True
        if use_r_condition_attn:
            # 用state_repr,action_repr作为Q值，return_repr作为KV值，计算增强后的state_repr
            state_repr_conditioned, sr_weight = self.alignment_attention(
                x=state_repr, cross=return_repr, training=training
            )
            action_repr_conditioned, ar_weight = self.alignment_attention(
                x=action_repr, cross=return_repr, training=training
            )
            state_proj = self.alignment_projector(state_repr_conditioned)
            action_proj = self.alignment_projector(action_repr_conditioned)

        else:
            state_proj = self.alignment_projector(state_repr)
            action_proj = self.alignment_projector(action_repr)

        # 归一化表示（提高稳定性）
        state_proj = state_proj / (
            jnp.linalg.norm(state_proj, axis=-1, keepdims=True) + 1e-8
        )
        action_proj = action_proj / (
            jnp.linalg.norm(action_proj, axis=-1, keepdims=True) + 1e-8
        )

        return state_preds, action_preds, return_preds, attns, (state_proj, action_proj)


def discount_cumsum(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_traj_minari(dataset_name):
    """
    Load and process a Minari dataset to extract trajectories.

    Args:
        dataset_name: Name of the Minari dataset (e.g., "mujoco/halfcheetah/expert-v0")

    Returns:
        paths: List of episode dictionaries containing trajectory data
        obs_mean: Mean of observations across the entire dataset
        obs_std: Standard deviation of observations across the entire dataset
    """
    print("processing: ", dataset_name)

    # Load the Minari dataset
    dataset = minari.load_dataset(dataset_name)
    episodes_generator = dataset.iterate_episodes()

    paths = []
    all_observations = []

    # Process each trajectory
    for traj in episodes_generator:
        obs = (
            jnp.concat(
                [
                    traj.observations["achieved_goal"],
                    traj.observations["desired_goal"],
                    traj.observations["observation"],
                ],
                axis=-1,
            )
            if "antmaze" in dataset_name
            else traj.observations
        )
        episode_data = {
            "observations": obs[:-1],  # Exclude the last observation
            "next_observations": obs[1:],  # Start from the second observation
            "actions": traj.actions,
            "rewards": traj.rewards,
            "terminals": traj.terminations,
        }
        paths.append(episode_data)
        all_observations.append(obs)

    # Calculate statistics
    all_observations = np.vstack(all_observations)
    obs_mean = all_observations.mean(axis=0)
    obs_std = all_observations.std(axis=0)

    # Calculate returns and print statistics
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )

    return paths, obs_mean, obs_std

class Trajectory(NamedTuple):
    timesteps: np.ndarray  # num_ep x max_len
    states: np.ndarray  # num_ep x max_len x state_dim
    actions: np.ndarray  # num_ep x max_len x act_dim
    returns_to_go: np.ndarray  # num_ep x max_len x 1
    masks: np.ndarray  # num_ep x max_len


def padd_by_zero(arr: jnp.ndarray, pad_to: int) -> jnp.ndarray:
    return np.pad(arr, ((0, pad_to - arr.shape[0]), (0, 0)), mode="constant")


def make_padded_trajectories(
    config: DTConfig,
) -> Tuple[Trajectory, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # trajectories, mean, std = get_traj(config.env_name)
    trajectories, mean, std = get_traj_minari(config.dataset_name)
    # Calculate returns to go for all trajectories
    # Normalize states
    max_len = 0
    traj_lengths = []
    for traj in trajectories:
        traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0) / config.rtg_scale
        traj["observations"] = (traj["observations"] - mean) / std
        max_len = max(max_len, traj["observations"].shape[0])
        traj_lengths.append(traj["observations"].shape[0])
    # Pad trajectories
    padded_trajectories = {key: [] for key in Trajectory._fields}
    for traj in trajectories:
        timesteps = np.arange(0, len(traj["observations"]))
        padded_trajectories["timesteps"].append(
            padd_by_zero(timesteps.reshape(-1, 1), max_len).reshape(-1)
        )
        padded_trajectories["states"].append(
            padd_by_zero(traj["observations"], max_len)
        )
        padded_trajectories["actions"].append(padd_by_zero(traj["actions"], max_len))
        padded_trajectories["returns_to_go"].append(
            padd_by_zero(traj["returns_to_go"].reshape(-1, 1), max_len)
        )
        padded_trajectories["masks"].append(
            padd_by_zero(
                np.ones((len(traj["observations"]), 1)).reshape(-1, 1), max_len
            ).reshape(-1)
        )
    return (
        Trajectory(
            timesteps=np.stack(padded_trajectories["timesteps"]),
            states=np.stack(padded_trajectories["states"]),
            actions=np.stack(padded_trajectories["actions"]),
            returns_to_go=np.stack(padded_trajectories["returns_to_go"]),
            masks=np.stack(padded_trajectories["masks"]),
        ),
        len(trajectories),
        jnp.array(traj_lengths),
        mean,
        std,
    )


def sample_start_idx(
    rng: jax.random.PRNGKey,
    traj_idx: int,
    padded_traj_length: jnp.ndarray,
    context_len: int,
) -> jnp.ndarray:
    """
    Determine the start_idx for given trajectory, the trajectories are padded to max_len.
    Therefore, naively sample from 0, max_len will produce bunch of all zero data.
    To avoid that, we refer padded_traj_length, the list of actual trajectry length + context_len
    """
    traj_len = padded_traj_length[traj_idx]
    start_idx = jax.random.randint(rng, (1,), 0, traj_len - context_len - 1)
    return start_idx


def extract_traj(
    traj_idx: jnp.ndarray, start_idx: jnp.ndarray, traj: Trajectory, context_len: int
) -> Trajectory:
    """
    Extract the trajectory with context_len for given traj_idx and start_idx
    """
    return jax.tree_util.tree_map(
        lambda x: jax.lax.dynamic_slice_in_dim(x[traj_idx], start_idx, context_len),
        traj,
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_traj_batch(
    rng,
    traj: Trajectory,
    batch_size: int,
    context_len: int,
    episode_num: int,
    padded_traj_lengths: jnp.ndarray,
) -> Trajectory:
    traj_idx = jax.random.randint(rng, (batch_size,), 0, episode_num)  # B
    start_idx = jax.vmap(sample_start_idx, in_axes=(0, 0, None, None))(
        jax.random.split(rng, batch_size), traj_idx, padded_traj_lengths, context_len
    ).reshape(
        -1
    )  # B
    return jax.vmap(extract_traj, in_axes=(0, 0, None, None))(
        traj_idx, start_idx, traj, context_len
    )


class DTTrainState(NamedTuple):
    transformer: TrainState


class DT(object):

    @classmethod
    def update(
        self, train_state: DTTrainState, batch: Trajectory, rng: jax.random.PRNGKey
    ) -> Tuple[Any, jnp.ndarray]:
        timesteps, states, actions, returns_to_go, traj_mask = (
            batch.timesteps,
            batch.states,
            batch.actions,
            batch.returns_to_go,
            batch.masks,
        )

        def loss_fn(params):
            (
                state_preds,
                action_preds,
                return_preds,
                attns,
                (state_proj, action_proj),
            ) = train_state.transformer.apply_fn(
                params,
                timesteps,
                states,
                actions,
                returns_to_go,
                rngs={"dropout": rng},
            )  # B x T x state_dim, B x T x act_dim, B x T x 1
            # mask actions
            actions_masked = actions * traj_mask[:, :, None]
            action_preds_masked = action_preds * traj_mask[:, :, None]
            # Calculate mean squared error loss
            action_loss = jnp.mean(jnp.square(action_preds_masked - actions_masked))
            # return action_loss

            # 计算对齐损失 (InfoNCE对比损失)
            # 计算相似度矩阵
            similarity = jnp.einsum("btd,bsd->bts", state_proj, action_proj)
            similarity = similarity / config.alignment_temp

            # 创建掩码，只考虑有效轨迹部分
            valid_mask = traj_mask[:, :, None] * traj_mask[:, None, :]

            # 正样本是同一时间步的状态和动作
            diag_mask = jnp.eye(state_proj.shape[1])[None, :, :] * valid_mask

            # 防止数值不稳定
            max_sim = jnp.max(similarity, axis=-1, keepdims=True)
            exp_sim = jnp.exp(similarity - max_sim)

            # 计算正样本相似度
            pos_sim = jnp.sum(exp_sim * diag_mask, axis=-1) + 1e-8
            # 计算所有可能的相似度
            all_sim = jnp.sum(exp_sim * valid_mask, axis=-1) + 1e-8

            # 对比损失: -log(pos_sim / all_sim)
            alignment_loss = -jnp.log(pos_sim / all_sim)
            alignment_loss = jnp.mean(alignment_loss * traj_mask)  # 只考虑有效部分

            # 总损失
            total_loss = action_loss + config.alignment_weight * alignment_loss

            return total_loss, (action_loss, alignment_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        # loss, grad = grad_fn(train_state.transformer.params)
        (total_loss, (action_loss, alignment_loss)), grad = grad_fn(
            train_state.transformer.params
        )

        # Apply gradient clipping
        transformer = train_state.transformer.apply_gradients(grads=grad)
        # return train_state._replace(transformer=transformer), loss
        return train_state._replace(transformer=transformer), {
            "loss": total_loss,
            "action_loss": action_loss,
            "alignment_loss": alignment_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: DTTrainState,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray]]:
        (
            state_preds,
            action_preds,
            return_preds,
            attns,
            (state_proj, action_proj),
        ) = train_state.transformer.apply_fn(
            train_state.transformer.params,
            timesteps,
            states,
            actions,
            returns_to_go,
            training=False,
        )
        return action_preds, attns


def create_dt_train_state(
    rng: jax.random.PRNGKey, state_dim: int, act_dim: int, config: DTConfig
) -> DTTrainState:
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=config.n_blocks,
        h_dim=config.embed_dim,
        context_len=config.context_len,
        n_heads=config.n_heads,
        drop_p=config.dropout_p,
    )
    rng, init_rng = jax.random.split(rng)
    # initialize params
    params = model.init(
        init_rng,
        timesteps=jnp.zeros((1, config.context_len), jnp.int32),
        states=jnp.zeros((1, config.context_len, state_dim), jnp.float32),
        actions=jnp.zeros((1, config.context_len, act_dim), jnp.float32),
        returns_to_go=jnp.zeros((1, config.context_len, 1), jnp.float32),
        training=False,
    )
    # optimizer
    scheduler = optax.cosine_decay_schedule(
        init_value=config.lr, decay_steps=config.warmup_steps
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_grads),
        optax.scale_by_schedule(scheduler),
        optax.adamw(
            learning_rate=config.lr,
            weight_decay=config.wt_decay,
            b1=config.beta[0],
            b2=config.beta[1],
        ),
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return DTTrainState(train_state)


def evaluate(
    policy_fn: Callable,
    train_state: DTTrainState,
    env: gym.Env,
    config: DTConfig,
    state_mean=0,
    state_std=1,
) -> float:
    eval_batch_size = 1  # required for forward pass
    results = {}
    total_reward = 0
    total_timesteps = 0
    if "mujoco" in config.dataset_name:
        state_dim = env.observation_space.shape[0]
    elif "antmaze" in config.dataset_name:
        state_dim = sum([part.shape[0] for part in env.observation_space.values()])
    act_dim = env.action_space.shape[0]
    # same as timesteps used for training the transformer
    timesteps = jnp.arange(0, config.max_eval_ep_len, 1, jnp.int32)
    # repeat
    timesteps = jnp.repeat(timesteps[None, :], eval_batch_size, axis=0)
    for _ in range(config.num_eval_episodes):
        # zeros place holders
        actions = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, act_dim), dtype=jnp.float32
        )
        states = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, state_dim), dtype=jnp.float32
        )
        rewards_to_go = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, 1), dtype=jnp.float32
        )
        # init episode
        running_state, info = env.reset()
        running_reward = 0
        running_rtg = config.rtg_target / config.rtg_scale
        for t in tqdm(range(config.max_eval_ep_len)):
            total_timesteps += 1
            # add state in placeholder and normalize
            if "antmaze" in config.dataset_name:
                running_state = jnp.concat(
                    [
                        running_state["achieved_goal"],
                        running_state["desired_goal"],
                        running_state["observation"],
                    ],
                    axis=-1,
                )
            states = states.at[0, t].set((running_state - state_mean) / state_std)
            # calcualate running rtg and add in placeholder
            running_rtg = running_rtg - (running_reward / config.rtg_scale)
            rewards_to_go = rewards_to_go.at[0, t].set(running_rtg)
            if t < config.context_len:
                act_preds, attns = policy_fn(
                    train_state,
                    timesteps[:, : t + 1],
                    states[:, : t + 1],
                    actions[:, : t + 1],
                    rewards_to_go[:, : t + 1],
                )
                act = act_preds[0, -1]
            else:
                act_preds, attns = policy_fn(
                    train_state,
                    timesteps[:, t - config.context_len + 1 : t + 1],
                    states[:, t - config.context_len + 1 : t + 1],
                    actions[:, t - config.context_len + 1 : t + 1],
                    rewards_to_go[:, t - config.context_len + 1 : t + 1],
                )
                act = act_preds[0, -1]
            running_state, running_reward, done, truncated, next_info = env.step(act)
            # add action in placeholder
            actions = actions.at[0, t].set(act)
            total_reward += running_reward
            if done or truncated:
                break
    normalized_score = (
        normalize_score(
            score=total_reward / config.num_eval_episodes, env_id=config.env_name
        )
        * 100
    )
    # env.get_normalized_score(total_reward / config.num_eval_episodes) * 100
    return normalized_score


def normalize_score(score, env_id="HalfCheetah-v5"):
    # MuJoCo环境的参考分数
    reference_scores = {
        "Ant-v5": {"min": 0.0, "max": 6000.0},
        "HalfCheetah-v5": {"min": -280.0, "max": 12000.0},
        "Hopper-v5": {"min": -20.0, "max": 3800.0},
        "Humanoid-v5": {"min": 0.0, "max": 12000.0},
        "Walker2d-v5": {"min": 0.0, "max": 5000.0},
        "InvertedDoublePendulum-v5": {"min": 0.0, "max": 9100.0},
        "Pusher-v5": {"min": -100.0, "max": 0.0},  # Pusher通常是负回报环境
        "Reacher-v5": {"min": -40.0, "max": -1.0},  # Reacher也是负回报环境
        "Swimmer-v5": {"min": 0.0, "max": 360.0},
        "AntMaze_UMaze-v5": {"min": 0.0, "max": 1000.0},
        "AntMaze_Medium-v5": {"min": 0.0, "max": 1000.0},
        "AntMaze_Large-v5": {"min": 0.0, "max": 1000.0},
        # 添加其他MuJoCo环境
    }

    if env_id in reference_scores:
        min_score = reference_scores[env_id]["min"]
        max_score = reference_scores[env_id]["max"]
        normalized = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))  # 裁剪到[0,1]区间
    else:
        return score  # 如果环境未知，返回原始分数


if __name__ == "__main__":
    wandb.init(
        project=config.project,
        name=f"{config.experiment_name}-{config.dataset_name}-seed-{config.seed}",
        config=config,
    )
    dataset = minari.load_dataset(config.dataset_name)
    # env = dataset.recover_environment()
    env = gym.make(config.env_name)
    rng = jax.random.PRNGKey(config.seed)
    if "mujoco" in config.dataset_name:
        state_dim = env.observation_space.shape[0]
    elif "antmaze" in config.dataset_name:
        # state_dim = env.observation_space["observation"].shape[0]
        state_dim = sum([part.shape[0] for part in env.observation_space.values()])
    act_dim = env.action_space.shape[0]
    trajectories, episode_num, traj_lengths, state_mean, state_std = (
        make_padded_trajectories(config)
    )
    # create trainer
    rng, subkey = jax.random.split(rng)
    # from fdt import create_flamingo_dt_train_state
    # train_state = create_flamingo_dt_train_state(subkey, state_dim, act_dim, config)
    train_state = create_dt_train_state(subkey, state_dim, act_dim, config)

    algo = DT()
    update_fn = jax.jit(algo.update)

    # Create checkpoint directory
    ckpt_dir = os.path.join(
        "checkpoints", config.project, config.env_name, f"seed_{config.seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize best score tracker
    best_score = float("-inf")

    for i in tqdm(range(1, config.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, data_rng, update_rng = jax.random.split(rng, 3)
        traj_batch = sample_traj_batch(
            data_rng,
            trajectories,
            config.batch_size,
            config.context_len,
            episode_num,
            traj_lengths,
        )  # B x T x D
        # train_state, action_loss = update_fn(train_state, traj_batch, update_rng)  # update parameters
        train_state, losses = update_fn(
            train_state, traj_batch, update_rng
        )  # update parameters

        if i % config.eval_interval == 0:
            # evaluate on env
            normalized_score = evaluate(
                algo.get_action, train_state, env, config, state_mean, state_std
            )
            print(i, normalized_score)
            wandb.log(
                {
                    # "action_loss": action_loss,
                    "action_loss": losses["action_loss"],
                    "alignment_loss": losses["alignment_loss"],
                    "total_loss": losses["loss"],
                    f"{config.env_name}/normalized_score": normalized_score,
                    "step": i,
                }
            )

            # Save checkpoint if score is better than or equal to the best score
            if normalized_score >= best_score:
                best_score = normalized_score
                ckpt_path = os.path.join(ckpt_dir, f"step_{i}_score_{best_score:.2f}")
                with open(ckpt_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(train_state))
                print(f"New best model saved with score: {best_score:.2f} at step {i}")

    # final evaluation
    normalized_score = evaluate(
        algo.get_action, train_state, env, config, state_mean, state_std
    )
    wandb.log(
        {
            # "action_loss": action_loss,
            "action_loss": losses["action_loss"],
            "alignment_loss": losses["alignment_loss"],
            "total_loss": losses["loss"],
            f"{config.env_name}/normalized_score": normalized_score,
            "step": i + 1,
        }
    )
    wandb.log({f"{config.env_name}/final_normalized_score": normalized_score})
    wandb.log({f"{config.env_name}/best_score": best_score})

    # Save final model
    final_ckpt_path = os.path.join(ckpt_dir, "final_model")
    with open(final_ckpt_path, "wb") as f:
        f.write(flax.serialization.to_bytes(train_state))
    print(f"Final model saved at: {final_ckpt_path}")

    wandb.finish()
