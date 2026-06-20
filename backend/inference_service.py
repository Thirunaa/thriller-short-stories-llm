"""Holds the live model in memory and supports atomic hot-swap reloads.

A single global ModelService instance is shared by the API and the continuous
fine-tune worker. Generation and reloads are serialized with a lock (CPU
autoregressive decoding is single-threaded anyway).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config import ModelConfig, PRETRAIN_CKPT_DIR, FINETUNE_CKPT_DIR
from checkpointing import Checkpointer
from model import MiniGPT, build_model, param_count
from generate import generate_text


class ModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self.model: Optional[MiniGPT] = None
        self.cfg: Optional[ModelConfig] = None
        self.version: int = 0
        self.source: str = "untrained"
        self.meta: dict = {}
        self.loaded_at: float = 0.0
        self.pretrain_ck = Checkpointer(PRETRAIN_CKPT_DIR)
        self.finetune_ck = Checkpointer(FINETUNE_CKPT_DIR)

    # -- which checkpoint should we serve? --------------------------------
    def _best_checkpointer(self):
        """Prefer the newest fine-tuned model, else the pretrained one."""
        if self.finetune_ck.has_checkpoint():
            return self.finetune_ck, "finetune"
        if self.pretrain_ck.has_checkpoint():
            return self.pretrain_ck, "pretrain"
        return None, "untrained"

    def load_initial(self) -> None:
        ck, source = self._best_checkpointer()
        with self._lock:
            if ck is None:
                # No checkpoint yet -- serve a random model so the API still boots.
                self.cfg = ModelConfig()
                self.model = build_model(self.cfg, seed=0)
                self.version = 0
                self.source = "untrained"
                self.meta = {}
            else:
                model, cfg, meta = ck.restore()
                self.model, self.cfg, self.meta = model, cfg, meta
                self.version = ck.latest_step() or 0
                self.source = source
            self.loaded_at = time.time()

    def reload_latest(self) -> bool:
        """Reload the newest checkpoint if it is newer than what we serve."""
        ck, source = self._best_checkpointer()
        if ck is None:
            return False
        latest = ck.latest_step() or 0
        if source == self.source and latest <= self.version:
            return False
        model, cfg, meta = ck.restore()
        with self._lock:
            self.model, self.cfg, self.meta = model, cfg, meta
            self.version, self.source = latest, source
            self.loaded_at = time.time()
        return True

    def swap_in(self, model: MiniGPT, cfg: ModelConfig, version: int, meta: dict, source: str = "finetune") -> None:
        """Directly install an already-trained model (used by the fine-tune worker)."""
        with self._lock:
            self.model, self.cfg, self.meta = model, cfg, meta
            self.version, self.source = version, source
            self.loaded_at = time.time()

    # -- inference --------------------------------------------------------
    def generate(self, user_text: str, max_new_tokens: int, temperature: float,
                 top_k: Optional[int], seed: int) -> str:
        with self._lock:
            if self.model is None:
                raise RuntimeError("model not loaded")
            return generate_text(self.model, user_text, max_new_tokens, temperature, top_k, seed)

    def status(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "trained": self.source != "untrained",
            "params": param_count(self.model) if self.model is not None else 0,
            "config": self.cfg.__dict__ if self.cfg else {},
            "meta": self.meta,
            "loaded_at": self.loaded_at,
        }


# Process-wide singleton.
service = ModelService()
