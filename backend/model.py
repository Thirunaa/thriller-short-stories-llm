"""A compact GPT (decoder-only transformer) built with Flax NNX.

Small enough to train on CPU, but architecturally a real GPT: learned token +
positional embeddings, pre-norm transformer blocks with causal self-attention and
a GELU MLP, optional weight tying between the embedding and the output head.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nnx.Linear(cfg.n_embd, 3 * cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.c_proj = nnx.Linear(cfg.n_embd, cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.attn_dropout = nnx.Dropout(cfg.dropout, rngs=rngs)
        self.resid_dropout = nnx.Dropout(cfg.dropout, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(mask, att, jnp.finfo(att.dtype).min)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(cfg.n_embd, 4 * cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.proj = nnx.Linear(4 * cfg.n_embd, cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.dropout = nnx.Dropout(cfg.dropout, rngs=rngs)

    def __call__(self, x):
        return self.dropout(self.proj(jax.nn.gelu(self.fc(x))))


class Block(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, rngs=rngs)
        self.ln2 = nnx.LayerNorm(cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.wte = nnx.Embed(cfg.vocab_size, cfg.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(cfg.block_size, cfg.n_embd, rngs=rngs)
        self.drop = nnx.Dropout(cfg.dropout, rngs=rngs)
        self.blocks = nnx.data([Block(cfg, rngs=rngs) for _ in range(cfg.n_layer)])
        self.ln_f = nnx.LayerNorm(cfg.n_embd, use_bias=cfg.bias, rngs=rngs)
        # Untied head only allocated when weight tying is off.
        self.head = None if cfg.tie_weights else nnx.Linear(
            cfg.n_embd, cfg.vocab_size, use_bias=False, rngs=rngs
        )

    def hidden(self, idx):
        """Transformer trunk up to the final norm -> (B, T, n_embd)."""
        B, T = idx.shape
        pos = jnp.arange(T)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def _project(self, h):
        if self.head is None:                       # tied head: reuse embedding matrix
            return h @ self.wte.embedding.T
        return self.head(h)

    def __call__(self, idx):
        return self._project(self.hidden(idx))

    def logits_last(self, idx):
        """Logits for only the final position -> (B, vocab). ~vocab/T cheaper than
        the full forward, which is the dominant cost during autoregressive decoding."""
        return self._project(self.hidden(idx)[:, -1, :])

    def logits_at(self, idx, pos):
        """Logits at an arbitrary position `pos` (dynamic) -> (B, vocab). Lets us
        right-pad a fixed (1, block_size) buffer and read the last *real* token, so
        the prompt keeps correct positional indices and padding is ignored by the
        causal mask (positions > pos are never attended)."""
        h = self.hidden(idx)
        return self._project(jnp.take(h, pos, axis=1))


def build_model(cfg: ModelConfig, seed: int = 0) -> MiniGPT:
    return MiniGPT(cfg, rngs=nnx.Rngs(seed))


def param_count(model: MiniGPT) -> int:
    state = nnx.state(model, nnx.Param)
    return int(sum(x.size for x in jax.tree.leaves(state)))
