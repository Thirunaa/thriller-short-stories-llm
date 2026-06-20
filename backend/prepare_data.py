"""Tokenize training conversations into train.bin / val.bin token shards.

Sources (combined, then document-shuffled so domains interleave):
  - the local thriller/horror corpus built by datagen (data_cache/thriller_corpus.jsonl)
  - optionally the HF Opus_WritingStruct dataset, for general writing fluency

    # thriller corpus + a little general data (default):
    python prepare_data.py

    # only the scraped thriller/horror corpus:
    python prepare_data.py --no-hf

    # original behaviour (HF only):
    python prepare_data.py --no-local --max-rows 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fsspec

import tokenizer as tok
from config import DATA_DIR
from data import split_path
from textclean import clean_story_text

DATASET_URL = "hf://datasets/Nopm/Opus_WritingStruct/claude_dataset.jsonl"
DEFAULT_LOCAL = os.path.join(DATA_DIR, "thriller_corpus.jsonl")


def docs_from_local(path: str, min_chars: int) -> list[list[int]]:
    """Story-mode: train on cleaned plot/episode PROSE only (the assistant turn),
    with NO instruction scaffolding. A tiny model otherwise overfits to the few
    repeated prompt templates instead of learning to write stories."""
    docs = []
    seen = set()
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages") or []
            story = ""
            for m in messages:
                if m.get("role") == "assistant":
                    story = m.get("content") or ""
            story = clean_story_text(story)
            if len(story) < min_chars:
                dropped += 1
                continue
            key = story[:200]
            if key in seen:                 # drop near-duplicates
                dropped += 1
                continue
            seen.add(key)
            docs.append(tok.encode_ordinary(story) + [tok.EOT])
    if dropped:
        print(f"  (cleaned/deduped: dropped {dropped} short/duplicate docs)")
    return docs


def docs_from_hf(url: str, max_rows: int, min_chars: int) -> list[list[int]]:
    docs = []
    with fsspec.open(url, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages")
            if not messages or len(tok.render_conversation(messages)) < min_chars:
                continue
            docs.append(tok.encode_conversation(messages))
            if len(docs) >= max_rows:
                break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DATASET_URL)
    ap.add_argument("--local", default=DEFAULT_LOCAL, help="local corpus JSONL ({messages:[...]})")
    ap.add_argument("--no-local", action="store_true", help="ignore the local thriller corpus")
    ap.add_argument("--no-hf", action="store_true", help="ignore the HF dataset")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap on HF conversations to mix in (default 0: pure story prose)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    docs: list[list[int]] = []
    counts = {}

    if not args.no_local:
        import glob
        corpus_files = ([args.local] if args.local != DEFAULT_LOCAL
                        else sorted(glob.glob(os.path.join(DATA_DIR, "*_corpus.jsonl"))))
        for path in corpus_files:
            if not os.path.exists(path):
                continue
            print(f"Loading local corpus {os.path.basename(path)} ...")
            local_docs = docs_from_local(path, args.min_chars)
            docs += local_docs
            counts[os.path.basename(path)] = len(local_docs)
            print(f"  {len(local_docs)} documents")

    if not args.no_hf and args.max_rows > 0:
        print(f"Streaming HF {args.url} (max {args.max_rows}) ...")
        hf_docs = docs_from_hf(args.url, args.max_rows, args.min_chars)
        docs += hf_docs
        counts["hf"] = len(hf_docs)
        print(f"  {len(hf_docs)} HF documents")

    if not docs:
        raise SystemExit("No documents collected. Run datagen.build_corpus first, or pass --max-rows.")

    # Shuffle at the document level so thriller plots and general data interleave.
    rng = np.random.default_rng(args.seed)
    rng.shuffle(docs)

    ids = np.fromiter((t for d in docs for t in d), dtype=np.uint16)
    n_val = int(len(ids) * args.val_frac)
    train_ids = ids[:-n_val] if n_val else ids
    val_ids = ids[-n_val:] if n_val else ids[:0]

    train_ids.tofile(split_path("train"))
    val_ids.tofile(split_path("val"))

    meta = {
        "documents": len(docs),
        "sources": counts,
        "total_tokens": int(len(ids)),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "vocab": tok.REAL_VOCAB,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {len(docs)} docs ({counts}), {len(ids):,} tokens "
          f"(train={len(train_ids):,}, val={len(val_ids):,})")


if __name__ == "__main__":
    main()
