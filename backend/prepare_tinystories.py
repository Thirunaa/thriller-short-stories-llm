"""Tokenize TinyStories (+ a light thriller mix) into train/val shards.

TinyStories is short, clean, GPT-4-written stories designed so SMALL models learn
genuinely coherent, grammatical narrative. That coherence is what a 30-50M model
can actually achieve; plot summaries / archaic prose can't teach it.

    python prepare_tinystories.py --ts-tokens 100000000

Streams the raw text file so it never downloads the whole multi-GB corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import tokenizer as tok
from config import DATA_DIR
from data import split_path
from prepare_data import docs_from_local

# GPT-4-generated split — cleaner than the original.
TS_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
SEP = "<|endoftext|>"


def stream_tinystories(url: str, max_tokens: int) -> list[list[int]]:
    """Stream the text file, split on <|endoftext|>, tokenize each story."""
    docs: list[list[int]] = []
    total = 0
    buf = ""
    req = urllib.request.Request(url, headers={"User-Agent": "ThrillerForge/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        while total < max_tokens:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while SEP in buf:
                story, buf = buf.split(SEP, 1)
                story = story.strip()
                if len(story) < 80:
                    continue
                ids = tok.encode_ordinary(story) + [tok.EOT]
                docs.append(ids)
                total += len(ids)
                if len(docs) % 20000 == 0:
                    print(f"  {len(docs):,} stories -> {total/1e6:.1f}M tokens", flush=True)
            if total >= max_tokens:
                break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-tokens", type=int, default=100_000_000,
                    help="approx TinyStories tokens to ingest")
    ap.add_argument("--no-thriller", action="store_true",
                    help="skip the light thriller-corpus mix")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    print(f"Streaming TinyStories (target {args.ts_tokens/1e6:.0f}M tokens)...", flush=True)
    docs = stream_tinystories(TS_URL, args.ts_tokens)
    counts = {"tinystories": len(docs)}
    print(f"  TinyStories: {len(docs):,} stories", flush=True)

    if not args.no_thriller:
        import glob
        for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_corpus.jsonl"))):
            extra = docs_from_local(path, min_chars=120)
            docs += extra
            counts[os.path.basename(path)] = len(extra)
            print(f"  + {os.path.basename(path)}: {len(extra):,} docs", flush=True)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(docs)

    ids = np.fromiter((t for d in docs for t in d), dtype=np.uint16)
    n_val = int(len(ids) * args.val_frac)
    train_ids = ids[:-n_val] if n_val else ids
    val_ids = ids[-n_val:] if n_val else ids[:0]
    train_ids.tofile(split_path("train"))
    val_ids.tofile(split_path("val"))

    meta = {"documents": len(docs), "sources": counts,
            "total_tokens": int(len(ids)), "train_tokens": int(len(train_ids)),
            "val_tokens": int(len(val_ids)), "vocab": tok.REAL_VOCAB}
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. {len(docs):,} docs ({counts}), {len(ids):,} tokens "
          f"(train={len(train_ids):,}, val={len(val_ids):,})")


if __name__ == "__main__":
    main()
