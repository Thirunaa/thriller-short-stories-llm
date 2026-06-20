"""Autoregressive sampling from a MiniGPT, with temperature + top-k.

Speed: the per-token forward is JIT-compiled and projects logits for ONLY the last
position (the 50k-vocab head, otherwise computed over every position, dominates the
cost). It runs on a fixed (1, block_size) window (left-padded with the EOT document
separator) so it compiles once and reuses the kernel every step. Sampling happens
host-side in numpy, which is cheap on a single (vocab,) vector and lets us suppress
an over-eager EOT so short prompts still produce a story.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import jax
from flax import nnx

import tokenizer as tok
from model import MiniGPT

# Always generate at least this many tokens before honoring an EOT stop, so an
# undertrained model doesn't immediately end and return an empty story.
_MIN_TOKENS = 12


@nnx.jit
def _logits_last(model: MiniGPT, idx):
    return model.logits_last(idx)            # (1, vocab)


def _sample(logits: np.ndarray, temperature: float, top_k: Optional[int],
            rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        keep = np.argpartition(logits, -k)[-k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(probs.shape[-1], p=probs))


def generate_ids(
    model: MiniGPT,
    prompt_ids: List[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    seed: int = 0,
    stop_at_eot: bool = True,
) -> List[int]:
    """Return only the newly generated token ids (prompt excluded)."""
    model.eval()
    block_size = model.cfg.block_size
    rng = np.random.default_rng(seed)

    window = ([tok.EOT] * block_size + list(prompt_ids))[-block_size:]
    idx = jax.numpy.asarray(window, dtype=jax.numpy.int32)[None, :]

    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = np.array(_logits_last(model, idx)[0], dtype=np.float32)  # copy -> writable
        logits[tok.REAL_VOCAB:] = -np.inf                     # never emit padded ids
        if not (stop_at_eot and len(generated) >= _MIN_TOKENS):
            logits[tok.EOT] = -np.inf                          # suppress early EOT
        tid = _sample(logits, temperature, top_k, rng)
        if stop_at_eot and tid == tok.EOT:
            break
        generated.append(tid)
        next_id = jax.numpy.asarray([[tid]], dtype=jax.numpy.int32)
        idx = jax.numpy.concatenate([idx[:, 1:], next_id], axis=1)
    return generated


def generate_text(
    model: MiniGPT,
    user_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    seed: int = 0,
) -> str:
    prompt_ids = tok.encode_prompt(user_text)
    new_ids = generate_ids(model, prompt_ids, max_new_tokens, temperature, top_k, seed)
    return tok.decode(new_ids).strip()
