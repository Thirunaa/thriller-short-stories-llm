"""Orchestrate the data gathering and emit a chat-format training corpus.

  movies:  IMDb-selected best thriller/horror  ->  CMU plot (fast) or Wikipedia
  series:  IMDb-selected best thriller/horror  ->  Wikipedia episode summaries

Everything is cached to JSONL so the run is resumable (re-running only fetches
what is still missing). Final output: data_cache/thriller_corpus.jsonl, with rows
{"messages": [{user}, {assistant}]} compatible with prepare_data.py.

    python -m datagen.build_corpus --movies-per-genre 1000 --series-per-genre 100
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datagen.common import DATAGEN_DIR, match_key, load_jsonl, write_jsonl
from datagen import plots_cmu, wiki

BACKEND_DATA = os.path.dirname(DATAGEN_DIR)  # data_cache/
CORPUS_PATH = os.path.join(BACKEND_DATA, "thriller_corpus.jsonl")
MOVIE_CACHE = os.path.join(DATAGEN_DIR, "movie_plots.jsonl")
SERIES_CACHE = os.path.join(DATAGEN_DIR, "series_episodes.jsonl")

MAX_PLOT_CHARS = 4000
MIN_PLOT_CHARS = 200

MOVIE_PROMPTS = [
    "Write the plot of a gripping {g} film.",
    "Give me a chilling {g} story.",
    "Write a tense {g} movie synopsis.",
    "Tell me an original {g} story.",
]
SERIES_PROMPTS = [
    "Write a {g} TV episode synopsis.",
    "Write a suspenseful {g} episode.",
    "Give me a {g} series episode plot.",
]


def load_selection(kind: str, genre: str) -> list:
    path = os.path.join(DATAGEN_DIR, f"select_{kind}_{genre}.csv")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_plot(text: str) -> str:
    text = " ".join(text.split("\n\n"))  # flatten paragraphs into one block
    text = text.strip()
    if len(text) > MAX_PLOT_CHARS:
        cut = text[:MAX_PLOT_CHARS]
        text = cut[: cut.rfind(". ") + 1] or cut  # end on a sentence if possible
    return text.strip()


# --------------------------------------------------------------------------- #
# Movies
# --------------------------------------------------------------------------- #
def gather_movies(per_genre: int, max_wiki: int) -> list:
    cmu = plots_cmu.build_index()
    print(f"CMU index ready: {len(cmu):,} plots", flush=True)

    cache = {r["tconst"]: r for r in load_jsonl(MOVIE_CACHE)}
    wiki_calls = 0

    for genre in ("thriller", "horror"):
        have = sum(1 for r in cache.values() if r.get("genre") == genre and r.get("plot"))
        print(f"\n[movies/{genre}] starting (already have {have})", flush=True)
        for row in load_selection("movie", genre):
            if have >= per_genre:
                break
            tconst = row["tconst"]
            if tconst in cache:
                if cache[tconst].get("plot") and cache[tconst].get("genre") == genre:
                    have += 1
                continue

            title, year = row["title"], row["year"]
            plot = cmu.get(match_key(title, year)) or cmu.get(match_key(row["original"], year)) \
                or cmu.get(match_key(title, ""))
            source = "cmu"
            if not plot and wiki_calls < max_wiki:
                plot = wiki.find_plot(title, year)
                wiki_calls += 1
                source = "wikipedia"

            plot = clean_plot(plot) if plot else None
            if plot and len(plot) < MIN_PLOT_CHARS:
                plot = None
            cache[tconst] = {"tconst": tconst, "title": title, "year": year,
                             "genre": genre, "source": source if plot else None, "plot": plot}
            if plot:
                have += 1
                if have % 50 == 0:
                    print(f"  {genre}: {have}/{per_genre} (wiki calls={wiki_calls})", flush=True)
                    write_jsonl(MOVIE_CACHE, list(cache.values()))
        write_jsonl(MOVIE_CACHE, list(cache.values()))
        print(f"[movies/{genre}] collected {have} plots", flush=True)

    return [r for r in cache.values() if r.get("plot")]


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
def gather_series(per_genre: int) -> list:
    cache = {r["tconst"]: r for r in load_jsonl(SERIES_CACHE)}

    for genre in ("thriller", "horror"):
        have = sum(1 for r in cache.values() if r.get("genre") == genre and r.get("episodes"))
        print(f"\n[series/{genre}] starting (already have {have})", flush=True)
        for row in load_selection("series", genre):
            if have >= per_genre:
                break
            tconst = row["tconst"]
            if tconst in cache:
                if cache[tconst].get("episodes") and cache[tconst].get("genre") == genre:
                    have += 1
                continue

            title, year = row["title"], row["year"]
            eps = wiki.episode_summaries(title, year)
            eps = [{"title": e["title"], "summary": clean_plot(e["summary"])}
                   for e in eps if e.get("summary") and len(e["summary"]) >= 120]
            cache[tconst] = {"tconst": tconst, "series": title, "year": year,
                             "genre": genre, "episodes": eps}
            if eps:
                have += 1
                print(f"  {genre}: {have}/{per_genre}  {title} ({len(eps)} eps)", flush=True)
                write_jsonl(SERIES_CACHE, list(cache.values()))
        write_jsonl(SERIES_CACHE, list(cache.values()))
        print(f"[series/{genre}] collected {have} series with episodes", flush=True)

    return [r for r in cache.values() if r.get("episodes")]


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #
def build_corpus(movies: list, series: list) -> list:
    rows = []
    seen = set()
    for i, m in enumerate(movies):
        if m["tconst"] in seen:
            continue
        seen.add(m["tconst"])
        g = m["genre"]
        prompt = MOVIE_PROMPTS[i % len(MOVIE_PROMPTS)].format(g=g)
        answer = f"{m['title']}\n\n{m['plot']}" if i % 2 == 0 else m["plot"]
        rows.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ], "meta": {"kind": "movie", "genre": g, "tconst": m["tconst"]}})

    j = 0
    for s in series:
        g = s["genre"]
        for ep in s["episodes"]:
            prompt = SERIES_PROMPTS[j % len(SERIES_PROMPTS)].format(g=g)
            title = ep.get("title") or s["series"]
            answer = f"{title}\n\n{ep['summary']}" if j % 2 == 0 else ep["summary"]
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ], "meta": {"kind": "episode", "genre": g, "series": s["series"]}})
            j += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies-per-genre", type=int, default=1000)
    ap.add_argument("--series-per-genre", type=int, default=100)
    ap.add_argument("--max-wiki", type=int, default=4000,
                    help="cap on live Wikipedia movie-plot fetches")
    args = ap.parse_args()

    movies = gather_movies(args.movies_per_genre, args.max_wiki)
    series = gather_series(args.series_per_genre)
    corpus = build_corpus(movies, series)
    write_jsonl(CORPUS_PATH, corpus)

    n_movie = sum(1 for r in corpus if r["meta"]["kind"] == "movie")
    n_ep = sum(1 for r in corpus if r["meta"]["kind"] == "episode")
    chars = sum(len(r["messages"][1]["content"]) for r in corpus)
    print(f"\n=== corpus written: {CORPUS_PATH} ===")
    print(f"  movie samples : {n_movie}")
    print(f"  episode samples: {n_ep}")
    print(f"  total samples : {len(corpus)}  (~{chars/1e6:.1f}M chars)")


if __name__ == "__main__":
    main()
