"""Public-domain thriller / horror / mystery prose from Project Gutenberg.

Plot summaries are telegraphic and teach poor grammar; full narrative prose
(dialogue, paragraphs, sentence flow) is the real grammar/style signal. This
downloads a curated set of public-domain classics, strips the Gutenberg
boilerplate, un-wraps the hard line breaks into paragraphs, and emits
~chunk-sized documents into prose_corpus.jsonl (same schema as the plot corpus).

    python -m datagen.gutenberg
"""
from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import urllib.request

from datagen.common import DATAGEN_DIR, USER_AGENT, write_jsonl

BACKEND_DATA = os.path.dirname(DATAGEN_DIR)
PROSE_PATH = os.path.join(BACKEND_DATA, "prose_corpus.jsonl")

# Curated public-domain thriller / horror / mystery works (Gutenberg ebook IDs).
BOOKS = [
    (345, "Dracula"),
    (84, "Frankenstein"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde"),
    (1661, "The Adventures of Sherlock Holmes"),
    (2852, "The Hound of the Baskervilles"),
    (2097, "The Sign of the Four"),
    (244, "A Study in Scarlet"),
    (108, "The Return of Sherlock Holmes"),
    (155, "The Moonstone"),
    (583, "The Woman in White"),
    (863, "The Mysterious Affair at Styles"),
    (174, "The Picture of Dorian Gray"),
    (209, "The Turn of the Screw"),
    (10007, "Carmilla"),
    (175, "The Phantom of the Opera"),
    (389, "The Great God Pan"),
    (8492, "The King in Yellow"),
    (5230, "The Invisible Man"),
    (36, "The War of the Worlds"),
    (2148, "The Works of Edgar Allan Poe, Volume 2"),
    (2149, "The Works of Edgar Allan Poe, Volume 3"),
    (730, "Oliver Twist"),
    (1142, "The Murders in the Rue Morgue"),
    (768, "Wuthering Heights"),
]

CHUNK_CHARS = 1800        # ~450 tokens per training document
MIN_CHARS = 300

_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*", re.I | re.S)
_PRODUCED = re.compile(r"^.*?(produced by|transcriber'?s note).*$", re.I | re.M)


def fetch_book(book_id: int) -> str | None:
    for url in (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            raw = urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            continue
        if len(raw) < 5000:
            continue
        # Gutenberg .txt are utf-8 OR cp1252 — strict-decode in order so smart
        # quotes/apostrophes don't turn into replacement characters.
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    return None


def strip_boilerplate(text: str) -> str:
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[: m.start()]
    text = _PRODUCED.sub("", text)
    return text


def to_paragraphs(text: str) -> list[str]:
    """Un-wrap Gutenberg's hard line breaks: blank line = paragraph break,
    single newline = soft wrap (join with space)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n[ \t]*\n", text)
    out = []
    for p in paras:
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        p = re.sub(r"[ \t]+", " ", p)
        # skip chapter-number-only / all-caps headers and tiny fragments
        if len(p) < 40:
            continue
        if p.isupper() and len(p) < 80:
            continue
        # skip front matter: tables of contents, copyright/title pages
        if p.count("CHAPTER") >= 3 or p.count("VOLUME") >= 3 or "CONTENTS" in p[:60].upper():
            continue
        if len(p) < 400 and re.search(
            r"copyright|all rights reserved|project gutenberg|e-?book|"
            r"transcriber|illustrat|frontispiece|printed in", p, re.I):
            continue
        out.append(p)
    return out


def chunk_paragraphs(paras: list[str]) -> list[str]:
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 > CHUNK_CHARS and len(buf) >= MIN_CHARS:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if len(buf) >= MIN_CHARS:
        chunks.append(buf.strip())
    return chunks


def main():
    rows = []
    total_chars = 0
    for book_id, title in BOOKS:
        raw = fetch_book(book_id)
        if not raw:
            print(f"  MISS {title} ({book_id})", flush=True)
            continue
        paras = to_paragraphs(strip_boilerplate(raw))
        chunks = chunk_paragraphs(paras)
        for ch in chunks:
            rows.append({
                "messages": [{"role": "assistant", "content": ch}],
                "meta": {"kind": "prose", "title": title, "gutenberg_id": book_id},
            })
        total_chars += sum(len(c) for c in chunks)
        print(f"  OK  {title}: {len(chunks)} chunks", flush=True)

    write_jsonl(PROSE_PATH, rows)
    print(f"\nWrote {PROSE_PATH}: {len(rows)} prose docs (~{total_chars/1e6:.1f}M chars)")


if __name__ == "__main__":
    main()
