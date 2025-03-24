import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Tuple, Sequence
from functools import partial


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class GatedCrossAttention(nn.Module):
    """Cross-attention module for feature enhancement between modalities."""
    h_dim: int
    n_heads: int
    drop_p: float = 0.1
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x, y, training=True) -> jnp.ndarray:
        """
        Args:
            x: queries tensor (B, T, C)
            y: keys/values tensor (B, T, C)
        """
        B, T, C = x.shape
        head_dim = C // self.n_heads
        
        x_norm = nn.LayerNorm()(x)
        
        # Project queries, keys, and values
        q = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x_norm)
        k = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(y)
        v = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(y)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, H, T, D)
        v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, H, T, D)
        
        # Compute attention
        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention
        attention = jnp.matmul(weights, v)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, self.h_dim)
        
        # Apply dropout with explicit deterministic parameter
        attention_dropout = nn.Dropout(rate=self.drop_p)(
            attention, deterministic=not training
        )
        
        return nn.Dense(self.h_dim, kernel_init=self.kernel_init)(attention_dropout)


class GatedCrossAttentionBlock(nn.Module):
    """Block combining cross-attention and feedforward with gating mechanisms."""
    h_dim: int
    n_heads: int
    drop_p: float = 0.1
    kernel_init: Callable = default_init()

    def setup(self):
        self.attn = GatedCrossAttention(
            h_dim=self.h_dim, 
            n_heads=self.n_heads, 
            drop_p=self.drop_p,
            kernel_init=self.kernel_init
        )
        
        # Feedforward network components
        self.ff_norm = nn.LayerNorm()
        self.ff_dense1 = nn.Dense(4 * self.h_dim, kernel_init=self.kernel_init)
        self.ff_dense2 = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.ff_dropout = nn.Dropout(self.drop_p)
        
        # Gating parameters
        self.attn_gate = self.param('attn_gate', jax.nn.initializers.zeros, (1,))
        self.ff_gate = self.param('ff_gate', jax.nn.initializers.zeros, (1,))

    def __call__(self, x, y, training=True) -> jnp.ndarray:
        # Cross-attention with gating
        attn_out = self.attn(x, y, training=training)
        x = x + attn_out * jnp.tanh(self.attn_gate)
        
        # Feedforward with gating
        ff_norm = self.ff_norm(x)
        ff_hidden = self.ff_dense1(ff_norm)
        ff_hidden = nn.gelu(ff_hidden)
        ff_out = self.ff_dense2(ff_hidden)
        ff_out = self.ff_dropout(ff_out, deterministic=not training)
        x = x + ff_out * jnp.tanh(self.ff_gate)
        
        return x


class MaskedCausalAttention(nn.Module):
    """Original DT causal self-attention module."""
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        
        # rearrange q, k, v as (B, N, T, D)
        q = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        k = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        v = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
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
        
        return nn.Dropout(self.drop_p, deterministic=not training)(
            nn.Dense(self.h_dim)(attention)
        )


class Block(nn.Module):
    """Original DT transformer block."""
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + MaskedCausalAttention(
            self.h_dim, self.max_T, self.n_heads, self.drop_p
        )(
            x, training=training
        )  # residual
        x = nn.LayerNorm()(x)
        
        # MLP
        out = nn.Dense(4 * self.h_dim, kernel_init=self.kernel_init)(x)
        out = nn.gelu(out)
        out = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(out)
        out = nn.Dropout(self.drop_p, deterministic=not training)(out)
        
        # residual
        x = x + out
        x = nn.LayerNorm()(x)
        
        return x


class FlamingoDecisionTransformer(nn.Module):
    """Decision Transformer enhanced with Flamingo-style gated cross-attention."""
    state_dim: int
    act_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    max_timestep: int = 4096
    kernel_init: Callable = default_init()

    def setup(self):
        # Embedding layers
        self.embed_ln = nn.LayerNorm()
        self.embed_timestep = nn.Embed(self.max_timestep, self.h_dim)
        self.embed_rtg = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.embed_state = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.embed_action = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        
        # Gated cross-attention blocks for feature enhancement
        # RTG attends to state - returns-to-go modality enhanced by state information
        self.rtg_state_attn = GatedCrossAttentionBlock(
            h_dim=self.h_dim,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            kernel_init=self.kernel_init
        )
        
        # State attends to action - state modality enhanced by action information
        self.state_action_attn = GatedCrossAttentionBlock(
            h_dim=self.h_dim,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            kernel_init=self.kernel_init
        )
        
        # Cross-agent attention (degenerates to self-attention in single-agent case)
        self.cross_agent_attn = GatedCrossAttentionBlock(
            h_dim=self.h_dim,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            kernel_init=self.kernel_init
        )
        
        # Original transformer blocks
        self.blocks = [
            Block(self.h_dim, 3 * self.context_len, self.n_heads, self.drop_p)
            for _ in range(self.n_blocks)
        ]
        
        # Prediction heads
        self.predict_rtg = nn.Dense(1, kernel_init=self.kernel_init)
        self.predict_state = nn.Dense(self.state_dim, kernel_init=self.kernel_init)
        self.predict_action = nn.Dense(self.act_dim, kernel_init=self.kernel_init)
        
        self.use_action_tanh = True

    def __call__(
        self,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
        training=True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B, T, _ = states.shape

        # Embed timesteps
        time_embeddings = self.embed_timestep(timesteps)
        
        # Embed states, actions, and returns with timestep positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        
        # Apply cross-modal attention for feature enhancement:
        # 1. RTG attends to state
        rtg_enhanced = self.rtg_state_attn(rtg_embeddings, state_embeddings, training=training)
        
        # 2. State attends to action
        state_enhanced = self.state_action_attn(state_embeddings, action_embeddings, training=training)
        
        # 3. Cross-agent attention (degenerates to self-attention in single-agent)
        state_enhanced = self.cross_agent_attn(state_enhanced, state_enhanced, training=training)
        
        # Stack rtg, states and actions and reshape sequence as in original DT
        # (r1, s1, a1, r2, s2, a2 ...)
        h = (
            jnp.stack((rtg_enhanced, state_enhanced, action_embeddings), axis=1)
            .transpose(0, 2, 1, 3)
            .reshape(B, 3 * T, self.h_dim)
        )
        h = self.embed_ln(h)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, training=training)
        
        # Reshape to get predictions
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)
        
        # Generate predictions
        return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict action given r, s
        
        if self.use_action_tanh:
            action_preds = jnp.tanh(action_preds)
            
        return state_preds, action_preds, return_preds


# Modified train state creation function to use FlamingoDecisionTransformer
def create_flamingo_dt_train_state(
    rng: jax.random.PRNGKey, state_dim: int, act_dim: int, config, DTTrainState=None
):
    """
    Create a train state for the Flamingo Decision Transformer.
    
    Args:
        rng: JAX random key
        state_dim: State dimension 
        act_dim: Action dimension
        config: Configuration object
        DTTrainState: The DTTrainState class/namedtuple from the original DT implementation
                     (Pass this to avoid circular imports)
    
    Returns:
        A DTTrainState instance with the Flamingo DT model
    """
    from flax.training.train_state import TrainState
    import optax
    
    # If DTTrainState is not provided, try to import it
    if DTTrainState is None:
        # This might cause circular import issues, so it's better to pass DTTrainState
        from collections import namedtuple
        DTTrainState = namedtuple('DTTrainState', ['transformer'])
    
    model = FlamingoDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=config.n_blocks,
        h_dim=config.embed_dim,
        context_len=config.context_len,
        n_heads=config.n_heads,
        drop_p=config.dropout_p,
    )
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init(
        {'params': init_rng, 'dropout': jax.random.PRNGKey(0)},  # Fix for Flax dropout
        timesteps=jnp.zeros((1, config.context_len), jnp.int32),
        states=jnp.zeros((1, config.context_len, state_dim), jnp.float32),
        actions=jnp.zeros((1, config.context_len, act_dim), jnp.float32),
        returns_to_go=jnp.zeros((1, config.context_len, 1), jnp.float32),
        training=False,
    )
    
    # Optimizer setup (same as original DT)
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
    
    # Create and return train state
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    return DTTrainState(train_state)