"""Training primitives + the pretraining CLI.

The reusable pieces (make_optimizer, train_step, estimate_loss, run_training) are
shared by both pretraining (this file's `main`) and the continuous fine-tune worker.

    python train.py --max-iters 2000 --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import optax
from flax import nnx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import data
import tokenizer as tok
from config import ModelConfig, TrainConfig, PRETRAIN_CKPT_DIR
from model import MiniGPT, build_model, param_count
from checkpointing import Checkpointer
from generate import generate_text


def make_optimizer(model: MiniGPT, tcfg: TrainConfig) -> nnx.Optimizer:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tcfg.learning_rate,
        warmup_steps=tcfg.warmup_iters,
        decay_steps=max(tcfg.lr_decay_iters, tcfg.warmup_iters + 1),
        end_value=tcfg.min_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.adamw(schedule, b1=tcfg.beta1, b2=tcfg.beta2, weight_decay=tcfg.weight_decay),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def _loss_fn(model: MiniGPT, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, x, y):
    loss, grads = nnx.value_and_grad(_loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MiniGPT, x, y):
    return _loss_fn(model, x, y)


def estimate_loss(model: MiniGPT, dataset, tcfg: TrainConfig, rng) -> float:
    model.eval()
    losses = []
    for _ in range(tcfg.eval_iters):
        xb, yb = data.get_batch(dataset, tcfg.batch_size, tcfg.block_size, rng)
        losses.append(float(eval_step(model, xb, yb)))
    model.train()
    return float(np.mean(losses))


def run_training(
    model: MiniGPT,
    cfg: ModelConfig,
    tcfg: TrainConfig,
    train_data: np.ndarray,
    ckptr: Checkpointer,
    val_data=None,
    meta_extra: dict | None = None,
    start_step: int = 0,
    verbose: bool = True,
) -> float:
    """Train `model` in place; checkpoint the best val loss. Returns best val loss."""
    rng = np.random.default_rng(tcfg.seed)
    optimizer = make_optimizer(model, tcfg)
    model.train()
    best_val = float("inf")
    t0 = time.time()

    for it in range(1, tcfg.max_iters + 1):
        xb, yb = data.get_batch(train_data, tcfg.batch_size, tcfg.block_size, rng)
        loss = train_step(model, optimizer, xb, yb)

        if verbose and (it % tcfg.log_interval == 0 or it == 1):
            dt = time.time() - t0
            print(f"  iter {it}/{tcfg.max_iters}  loss {float(loss):.4f}  "
                  f"({it / max(dt, 1e-9):.1f} it/s)")

        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            eval_src = val_data if (val_data is not None and len(val_data) > tcfg.block_size + 1) else train_data
            val = estimate_loss(model, eval_src, tcfg, rng)
            if verbose:
                print(f"  [eval] iter {it}  val_loss {val:.4f}")
            if val < best_val:
                best_val = val
                meta = {"val_loss": round(val, 4), "kind": "train"}
                if meta_extra:
                    meta.update(meta_extra)
                ckptr.save(start_step + it, model, cfg, meta)
                if verbose:
                    print(f"  [ckpt] saved step {start_step + it} (val {val:.4f})")

    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=ModelConfig.n_layer)
    ap.add_argument("--n-head", type=int, default=ModelConfig.n_head)
    ap.add_argument("--n-embd", type=int, default=ModelConfig.n_embd)
    ap.add_argument("--block-size", type=int, default=ModelConfig.block_size)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--max-iters", type=int, default=TrainConfig.max_iters)
    ap.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    ap.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    ap.add_argument("--resume", action="store_true", help="continue from latest pretrain checkpoint")
    args = ap.parse_args()

    if not data.has_data():
        raise SystemExit("No token shards found. Run:  python prepare_data.py")

    cfg = ModelConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
    )
    tcfg = TrainConfig(
        batch_size=args.batch_size, block_size=args.block_size, learning_rate=args.lr,
        max_iters=args.max_iters, lr_decay_iters=args.max_iters, eval_interval=args.eval_interval,
    )

    ckptr = Checkpointer(PRETRAIN_CKPT_DIR)
    start_step = 0
    if args.resume and ckptr.has_checkpoint():
        model, cfg, _ = ckptr.restore()
        start_step = ckptr.latest_step() or 0
        print(f"Resumed from step {start_step}")
    else:
        model = build_model(cfg, seed=tcfg.seed)

    print(f"Model: {param_count(model):,} params  "
          f"(L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd} ctx={cfg.block_size})")

    train_data = data.load_split("train")
    val_data = data.load_split("val") if data.has_data() else None
    print(f"Train tokens: {len(train_data):,}")

    best = run_training(model, cfg, tcfg, train_data, ckptr, val_data,
                        meta_extra={"source": "pretrain"}, start_step=start_step)
    print(f"\nBest val loss: {best:.4f}")
    print("\nSample generation:")
    print(generate_text(model, "Write a short thriller about a missed train.",
                        max_new_tokens=120, temperature=0.8, top_k=40, seed=0))


if __name__ == "__main__":
    main()
