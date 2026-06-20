"""SQLite store for generations and human feedback.

Feedback is the fuel for continuous improvement: a thumbs-up (optionally with an
edited/improved story) becomes a (prompt -> preferred story) training pair that the
fine-tune worker later learns from.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import List, Dict, Optional

from config import DB_PATH


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id            TEXT PRIMARY KEY,
                prompt        TEXT NOT NULL,
                output        TEXT NOT NULL,
                params        TEXT,
                model_version INTEGER,
                created_at    REAL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id               TEXT PRIMARY KEY,
                generation_id    TEXT NOT NULL,
                rating           INTEGER NOT NULL,   -- 1 = up, -1 = down
                edited_text      TEXT,
                created_at       REAL,
                used_in_training INTEGER DEFAULT 0,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            );
            """
        )


def add_generation(prompt: str, output: str, params: dict, model_version: int) -> str:
    gen_id = uuid.uuid4().hex
    with _conn() as c:
        c.execute(
            "INSERT INTO generations VALUES (?,?,?,?,?,?)",
            (gen_id, prompt, output, json.dumps(params), model_version, time.time()),
        )
    return gen_id


def add_feedback(generation_id: str, rating: int, edited_text: Optional[str] = None) -> str:
    fb_id = uuid.uuid4().hex
    with _conn() as c:
        c.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,0)",
            (fb_id, generation_id, 1 if rating >= 0 else -1, edited_text, time.time()),
        )
    return fb_id


def count_pending_positive() -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE rating = 1 AND used_in_training = 0"
        ).fetchone()
    return int(row["n"])


def fetch_pending_training_samples() -> List[Dict]:
    """Return positive, unused feedback joined to its generation, as training pairs."""
    with _conn() as c:
        rows = c.execute(
            """
            SELECT f.id AS fb_id, g.prompt AS prompt,
                   COALESCE(NULLIF(f.edited_text, ''), g.output) AS answer
            FROM feedback f JOIN generations g ON f.generation_id = g.id
            WHERE f.rating = 1 AND f.used_in_training = 0
            """
        ).fetchall()
    samples = []
    for r in rows:
        samples.append({
            "fb_id": r["fb_id"],
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["answer"]},
            ],
        })
    return samples


def mark_samples_used(fb_ids: List[str]) -> None:
    if not fb_ids:
        return
    with _conn() as c:
        c.executemany(
            "UPDATE feedback SET used_in_training = 1 WHERE id = ?",
            [(i,) for i in fb_ids],
        )


def stats() -> dict:
    with _conn() as c:
        g = c.execute("SELECT COUNT(*) n FROM generations").fetchone()["n"]
        up = c.execute("SELECT COUNT(*) n FROM feedback WHERE rating=1").fetchone()["n"]
        down = c.execute("SELECT COUNT(*) n FROM feedback WHERE rating=-1").fetchone()["n"]
        used = c.execute("SELECT COUNT(*) n FROM feedback WHERE used_in_training=1").fetchone()["n"]
        pending = c.execute(
            "SELECT COUNT(*) n FROM feedback WHERE rating=1 AND used_in_training=0"
        ).fetchone()["n"]
    return {
        "generations": int(g),
        "thumbs_up": int(up),
        "thumbs_down": int(down),
        "used_in_training": int(used),
        "pending_positive": int(pending),
    }
