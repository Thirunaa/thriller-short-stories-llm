"""Download the HF writing dataset, render conversations, tokenize, and write
train.bin / val.bin token shards.

    python prepare_data.py --max-rows 3000

Streams the JSONL so it never loads the whole file into memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# UTF-8 stdout so progress printing never crashes on the Windows cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fsspec

import tokenizer as tok
from config import DATA_DIR
from data import split_path

DATASET_URL = "hf://datasets/Nopm/Opus_WritingStruct/claude_dataset.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DATASET_URL)
    ap.add_argument("--max-rows", type=int, default=3000,
                    help="cap the number of conversations (keeps the CPU demo fast)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-chars", type=int, default=40,
                    help="skip trivially short conversations")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Streaming {args.url} (max {args.max_rows} rows)...")

    all_ids: list[int] = []
    rows = 0
    with fsspec.open(args.url, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages")
            if not messages:
                continue
            text = tok.render_conversation(messages)
            if len(text) < args.min_chars:
                continue
            all_ids.extend(tok.encode_conversation(messages))
            rows += 1
            if rows % 250 == 0:
                print(f"  {rows} rows -> {len(all_ids):,} tokens")
            if rows >= args.max_rows:
                break

    if not all_ids:
        raise SystemExit("No data parsed -- check the dataset URL / schema.")

    ids = np.asarray(all_ids, dtype=np.uint16)
    n_val = int(len(ids) * args.val_frac)
    train_ids, val_ids = ids[:-n_val] if n_val else ids, ids[-n_val:] if n_val else ids[:0]

    train_ids.tofile(split_path("train"))
    val_ids.tofile(split_path("val"))

    meta = {
        "rows": rows,
        "total_tokens": int(len(ids)),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "vocab": tok.REAL_VOCAB,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {rows} rows, {len(ids):,} tokens "
          f"(train={len(train_ids):,}, val={len(val_ids):,})")
    print(f"Wrote {split_path('train')} and {split_path('val')}")


if __name__ == "__main__":
    main()
