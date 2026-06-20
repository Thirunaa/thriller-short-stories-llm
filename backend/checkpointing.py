"""Orbax-based checkpointing for the NNX model.

Each checkpoint root (e.g. checkpoints/pretrain) is an Orbax CheckpointManager
directory holding versioned `step_N` states, plus two sidecar JSON files:
  - model_config.json : architecture needed to rebuild the abstract model
  - meta.json         : training metadata (val loss, lineage, sample counts ...)
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import orbax.checkpoint as ocp
from flax import nnx

from config import ModelConfig
from model import MiniGPT, build_model


def _abs(path: str) -> str:
    return os.path.abspath(path)


class Checkpointer:
    def __init__(self, root: str, max_to_keep: int = 3):
        self.root = _abs(root)
        os.makedirs(self.root, exist_ok=True)
        self._cfg_path = os.path.join(self.root, "model_config.json")
        self._meta_path = os.path.join(self.root, "meta.json")
        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        self.mngr = ocp.CheckpointManager(os.path.join(self.root, "states"), options=options)

    # -- saving -----------------------------------------------------------
    def save(self, step: int, model: MiniGPT, cfg: ModelConfig, meta: dict) -> None:
        _, state = nnx.split(model)
        self.mngr.save(step, args=ocp.args.StandardSave(state))
        self.mngr.wait_until_finished()
        cfg.to_json(self._cfg_path)
        meta = {**meta, "step": step}
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # -- loading ----------------------------------------------------------
    def latest_step(self) -> Optional[int]:
        return self.mngr.latest_step()

    def has_checkpoint(self) -> bool:
        return os.path.exists(self._cfg_path) and self.latest_step() is not None

    def load_config(self) -> ModelConfig:
        return ModelConfig.from_json(self._cfg_path)

    def load_meta(self) -> dict:
        if not os.path.exists(self._meta_path):
            return {}
        with open(self._meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def restore(self, step: Optional[int] = None) -> Tuple[MiniGPT, ModelConfig, dict]:
        cfg = self.load_config()
        if step is None:
            step = self.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint found under {self.root}")

        # Build an abstract (shapes-only) model so Orbax knows the tree structure.
        abstract = nnx.eval_shape(lambda: build_model(cfg))
        graphdef, abstract_state = nnx.split(abstract)
        state = self.mngr.restore(step, args=ocp.args.StandardRestore(abstract_state))
        model = nnx.merge(graphdef, state)
        return model, cfg, self.load_meta()
