"""Autoregressive sampling from a MiniGPT, with temperature + top-k + rep-penalty.

Speed/correctness: the per-token forward is JIT-compiled on a fixed (1, block_size)
buffer with the prompt RIGHT-padded (tokens at positions 0..L-1, padding to the
right). `model.logits_at(idx, pos)` reads logits at the last *real* position, so the
prompt keeps correct positional indices and the causal mask ignores the padding.
This compiles once and computes only the last position's 50k-wide head (the dominant
cost). Sampling happens host-side in numpy, which is cheap on one (vocab,) vector and
lets us apply a repetition penalty and suppress an over-eager EOT.
"""
from __future__ import annotations

import re
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
def _logits_at(model: MiniGPT, idx, pos):
    return model.logits_at(idx, pos)         # (1, vocab) at position `pos`


def _sample(logits: np.ndarray, temperature: float, top_k: Optional[int],
            rng: np.random.Generator, recent_ids=None, rep_penalty: float = 1.3) -> int:
    # Repetition penalty: damp tokens generated recently. This is what stops a tiny,
    # undertrained model from collapsing into "the the the" / newline-spam loops.
    if rep_penalty and rep_penalty != 1.0 and recent_ids:
        for tid in set(recent_ids):
            v = logits[tid]
            logits[tid] = v / rep_penalty if v > 0 else v * rep_penalty
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

    # Right-padded fixed buffer: prompt at positions 0..L-1 (correct positional
    # indices); the causal mask ignores the padding to the right of `pos`.
    ids = list(prompt_ids)[-block_size:] or [tok.EOT]
    buf = np.zeros(block_size, dtype=np.int32)
    buf[: len(ids)] = ids
    pos = len(ids) - 1                       # index of the last real token

    recent = list(ids[-48:])
    generated: List[int] = []
    for _ in range(max_new_tokens):
        idx = jax.numpy.asarray(buf)[None, :]
        logits = np.array(
            _logits_at(model, idx, jax.numpy.asarray(pos, dtype=jax.numpy.int32))[0],
            dtype=np.float32,
        )
        logits[tok.REAL_VOCAB:] = -np.inf                     # never emit padded ids
        if not (stop_at_eot and len(generated) >= _MIN_TOKENS):
            logits[tok.EOT] = -np.inf                          # suppress early EOT
        tid = _sample(logits, temperature, top_k, rng, recent_ids=recent[-48:])
        if stop_at_eot and tid == tok.EOT:
            break
        generated.append(tid)
        recent.append(tid)
        if pos < block_size - 1:
            pos += 1
            buf[pos] = tid
        else:
            buf = np.roll(buf, -1)            # buffer full -> slide window left
            buf[-1] = tid
    return generated


def generate_text(
    model: MiniGPT,
    user_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    seed: int = 0,
) -> str:
    """Story-mode generation: the user's text is the opening of the story, which
    the model continues. Returns the full story (seed + continuation)."""
    seed_ids = tok.encode_ordinary(user_text.strip())
    new_ids = generate_ids(model, seed_ids, max_new_tokens, temperature, top_k, seed)
    text = tok.decode(seed_ids + new_ids)
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse newline spam
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
