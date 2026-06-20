"""Continuous-improvement worker.

A background daemon thread watches the feedback queue. Once enough fresh
thumbs-up samples accumulate (or an admin triggers it), it:

  1. restores a fresh copy of the currently-served weights (the base),
  2. fine-tunes them on the (prompt -> preferred story) feedback pairs,
  3. saves a new versioned checkpoint, and
  4. hot-swaps the live model in the ModelService.

This closes the loop: human preference -> data -> new weights -> better serving.
"""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np

import data
import feedback
from config import TrainConfig, ContinuousConfig, FINETUNE_CKPT_DIR
from checkpointing import Checkpointer
from train import make_optimizer, train_step
from inference_service import service


class ContinuousTrainer:
    def __init__(self, cfg: ContinuousConfig | None = None):
        self.cfg = cfg or ContinuousConfig()
        self.finetune_ck = Checkpointer(FINETUNE_CKPT_DIR)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._trigger = threading.Event()
        self._busy = threading.Lock()
        self.rounds_completed = 0
        self.in_progress = False
        self.last_result: dict = {"status": "idle"}

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="continuous-trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._trigger.set()

    def trigger_now(self) -> dict:
        """Ask the worker to run a fine-tune round on its next tick."""
        self._trigger.set()
        return {"status": "scheduled", "pending_positive": feedback.count_pending_positive()}

    def status(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "running": bool(self._thread and self._thread.is_alive()),
            "in_progress": self.in_progress,
            "rounds_completed": self.rounds_completed,
            "min_new_samples": self.cfg.min_new_samples,
            "poll_seconds": self.cfg.poll_seconds,
            "finetune_iters": self.cfg.finetune_iters,
            "pending_positive": feedback.count_pending_positive(),
            "last_result": self.last_result,
        }

    # -- worker loop ------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            triggered = self._trigger.wait(timeout=self.cfg.poll_seconds)
            self._trigger.clear()
            if self._stop.is_set():
                break
            try:
                pending = feedback.count_pending_positive()
                if triggered or (self.cfg.enabled and pending >= self.cfg.min_new_samples):
                    self.run_round()
            except Exception:
                self.last_result = {"status": "error", "error": traceback.format_exc()}

    # -- one fine-tune round ---------------------------------------------
    def run_round(self) -> dict:
        with self._busy:
            self.in_progress = True
            try:
                result = self._run_round_impl()
            finally:
                self.in_progress = False
            self.last_result = result
            return result

    def _run_round_impl(self) -> dict:
        samples = feedback.fetch_pending_training_samples()
        if not samples:
            return {"status": "no_samples", "at": time.time()}

        base_ck, source = service._best_checkpointer()
        if base_ck is None or source == "untrained":
            return {"status": "no_base_model",
                    "detail": "pretrain the model first (python train.py)", "at": time.time()}

        # Fresh copy of the served weights so we never mutate the live model mid-request.
        model, cfg, _ = base_ck.restore()

        # Story-mode: fine-tune on the raw preferred-story prose (assistant turn).
        texts = []
        for s in samples:
            for m in s["messages"]:
                if m.get("role") == "assistant" and (m.get("content") or "").strip():
                    texts.append(m["content"])
        tokens = data.tokens_from_texts(texts)
        block_size = min(cfg.block_size, max(16, len(tokens) // 4))
        if len(tokens) < block_size + 2:
            return {"status": "insufficient_tokens", "tokens": int(len(tokens)), "at": time.time()}

        tcfg = TrainConfig(
            batch_size=min(8, max(1, len(tokens) // block_size)),
            block_size=block_size,
        ).finetune_variant(self.cfg.finetune_iters)

        rng = np.random.default_rng()
        optimizer = make_optimizer(model, tcfg)
        model.train()
        last_loss = None
        for _ in range(tcfg.max_iters):
            xb, yb = data.get_batch(tokens, tcfg.batch_size, tcfg.block_size, rng)
            last_loss = float(train_step(model, optimizer, xb, yb))

        new_version = max(service.version, self.finetune_ck.latest_step() or 0) + 1
        meta = {
            "kind": "finetune",
            "source": "feedback",
            "samples": len(samples),
            "tokens": int(len(tokens)),
            "final_loss": round(last_loss, 4) if last_loss is not None else None,
            "base_version": service.version,
        }
        self.finetune_ck.save(new_version, model, cfg, meta)
        feedback.mark_samples_used([s["fb_id"] for s in samples])
        service.swap_in(model, cfg, new_version, meta, source="finetune")
        self.rounds_completed += 1

        return {"status": "improved", "new_version": new_version, **meta, "at": time.time()}


trainer = ContinuousTrainer()
