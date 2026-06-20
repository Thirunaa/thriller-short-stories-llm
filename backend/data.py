"""Binary token-shard loading and batching (nanoGPT-style memmap).

Tokens are stored as flat uint16 arrays on disk (train.bin / val.bin); batches are
random contiguous windows of length block_size.
"""
from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import jax.numpy as jnp

import tokenizer as tok
from config import DATA_DIR


def split_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"{split}.bin")


def has_data() -> bool:
    return os.path.exists(split_path("train"))


def load_split(split: str) -> np.ndarray:
    return np.memmap(split_path(split), dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64) for i in ix])
    return jnp.asarray(x, dtype=jnp.int32), jnp.asarray(y, dtype=jnp.int32)


def tokens_from_conversations(conversations: List[List[Dict[str, str]]]) -> np.ndarray:
    """Flatten a list of conversations into one uint16 token array (for fine-tuning)."""
    ids: List[int] = []
    for messages in conversations:
        ids.extend(tok.encode_conversation(messages))
    return np.asarray(ids, dtype=np.uint16)
