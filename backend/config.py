"""Central configuration: model hyper-parameters, training schedule, and paths.

Everything is config-driven so the same codebase runs as a tiny CPU model here
(the defaults) or scales up to a GPU box by overriding a few fields.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, replace

# ---------------------------------------------------------------------------
# Paths (all relative to the backend/ directory this file lives in)
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data_cache")
CKPT_ROOT = os.path.join(BACKEND_DIR, "checkpoints")
PRETRAIN_CKPT_DIR = os.path.join(CKPT_ROOT, "pretrain")
FINETUNE_CKPT_DIR = os.path.join(CKPT_ROOT, "finetune")
DB_PATH = os.path.join(BACKEND_DIR, "feedback.db")

for _d in (DATA_DIR, CKPT_ROOT):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # 50257 real gpt2 tokens, padded up to a multiple of 64 for matmul efficiency.
    vocab_size: int = 50304
    block_size: int = 128          # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False             # bias in Linear / LayerNorm
    tie_weights: bool = True       # share token embedding with the output head

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return ModelConfig(**json.load(f))


# ---------------------------------------------------------------------------
# Training schedule (used by both pretraining and continuous fine-tuning)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 128
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_iters: int = 2000
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 40
    log_interval: int = 20
    seed: int = 1337

    def finetune_variant(self, max_iters: int) -> "TrainConfig":
        """A gentler schedule used when continuously fine-tuning on feedback."""
        return replace(
            self,
            learning_rate=5e-5,
            min_lr=1e-5,
            max_iters=max_iters,
            warmup_iters=max(2, max_iters // 10),
            lr_decay_iters=max_iters,
            eval_interval=max(1, max_iters),  # only eval at the end
        )


# Tunable knobs for the continuous-improvement worker.
@dataclass
class ContinuousConfig:
    min_new_samples: int = 8       # fine-tune once this many fresh good samples queue up
    poll_seconds: int = 30         # how often the worker checks the feedback queue
    finetune_iters: int = 60       # gradient steps per fine-tune round
    enabled: bool = True
