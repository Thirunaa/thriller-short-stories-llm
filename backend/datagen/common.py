"""Shared helpers for the data-gathering pipeline: paths, a polite HTTP session,
title normalization, and tiny JSON(L) cache utilities.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from typing import Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAGEN_DIR = os.path.join(BACKEND_DIR, "data_cache", "datagen")
IMDB_DIR = os.path.join(DATAGEN_DIR, "imdb")
RAW_DIR = os.path.join(DATAGEN_DIR, "raw")
for _d in (DATAGEN_DIR, IMDB_DIR, RAW_DIR):
    os.makedirs(_d, exist_ok=True)

# Wikipedia asks for a descriptive User-Agent with contact info.
USER_AGENT = (
    "ThrillerForge-DataGen/1.0 (educational MiniGPT training corpus; "
    "https://github.com/Thirunaa/thriller-short-stories-llm)"
)

_last_request_ts = {"t": 0.0}


def http_get(url: str, *, params: Optional[dict] = None, min_interval: float = 0.2,
             timeout: int = 30, retries: int = 3) -> Optional[str]:
    """GET text with a shared User-Agent, polite spacing, and simple retries.

    Returns the body as str, or None on persistent failure.
    """
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        # global rate limit across all callers
        gap = min_interval - (time.time() - _last_request_ts["t"])
        if gap > 0:
            time.sleep(gap)
        _last_request_ts["t"] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 + attempt)
    return None


def download_file(url: str, dest: str, *, force: bool = False) -> str:
    """Download a (large) binary file to dest with caching."""
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


_PAREN = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9]+")
_ARTICLES = re.compile(r"^(the|a|an)\s+")


def norm_title(title: str) -> str:
    """Aggressive normalization for matching titles across datasets."""
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = t.lower().strip()
    t = _PAREN.sub(" ", t)
    t = _ARTICLES.sub("", t)
    t = _NONWORD.sub(" ", t)
    return " ".join(t.split())


def match_key(title: str, year) -> str:
    try:
        y = int(year)
    except (ValueError, TypeError):
        y = 0
    return f"{norm_title(title)}|{y}"


# -- tiny JSONL cache --------------------------------------------------------
def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
