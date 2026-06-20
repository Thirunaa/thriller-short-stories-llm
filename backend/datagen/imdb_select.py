"""Select the 'best' thriller / horror movies and series from IMDb's official
datasets (https://datasets.imdbws.com/), ranked by an IMDb-style weighted rating.

Outputs CSVs under data_cache/datagen/ with: tconst, title, year, rating, votes.

    python -m datagen.imdb_select
"""
from __future__ import annotations

import csv
import gzip
import math
import os
from collections import defaultdict

from datagen.common import IMDB_DIR, DATAGEN_DIR, download_file

BASE = "https://datasets.imdbws.com/"
FILES = {
    "basics": "title.basics.tsv.gz",
    "ratings": "title.ratings.tsv.gz",
    "episode": "title.episode.tsv.gz",
}

MOVIE_TYPES = {"movie"}
SERIES_TYPES = {"tvSeries", "tvMiniSeries"}

# candidates kept per (kind, genre) — headroom for plot/episode dropout downstream
MOVIE_CANDIDATES = 2000
SERIES_CANDIDATES = 160

# weighted-rating prior: minimum votes (m) and how it shapes "best"
MOVIE_MIN_VOTES = 25000
SERIES_MIN_VOTES = 5000


def ensure_files() -> dict:
    paths = {}
    for key, fname in FILES.items():
        dest = os.path.join(IMDB_DIR, fname)
        print(f"  fetching {fname} ...", flush=True)
        download_file(BASE + fname, dest)
        paths[key] = dest
    return paths


def load_ratings(path: str) -> dict:
    ratings = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            tconst, avg, votes = line.rstrip("\n").split("\t")
            try:
                ratings[tconst] = (float(avg), int(votes))
            except ValueError:
                continue
    return ratings


def scan_basics(path: str, ratings: dict):
    """Yield candidate rows (with ratings) for movies and series in our genres."""
    wanted_types = MOVIE_TYPES | SERIES_TYPES
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            # tconst,titleType,primaryTitle,originalTitle,isAdult,startYear,endYear,runtime,genres
            tconst, ttype, primary, original, is_adult, start_year = row[0], row[1], row[2], row[3], row[4], row[5]
            if ttype not in wanted_types or is_adult == "1":
                continue
            genres = row[8]
            if "Thriller" not in genres and "Horror" not in genres:
                continue
            rv = ratings.get(tconst)
            if not rv:
                continue
            yield {
                "tconst": tconst,
                "ttype": ttype,
                "title": primary,
                "original": original if original != "\\N" else primary,
                "year": start_year if start_year != "\\N" else "",
                "rating": rv[0],
                "votes": rv[1],
                "genres": genres,
            }


def weighted_rating(rating: float, votes: int, m: int, c: float) -> float:
    return (votes / (votes + m)) * rating + (m / (votes + m)) * c


def select():
    paths = ensure_files()
    print("Loading ratings ...", flush=True)
    ratings = load_ratings(paths["ratings"])
    print(f"  {len(ratings):,} rated titles", flush=True)

    print("Scanning title.basics (this reads ~1GB, takes a minute) ...", flush=True)
    buckets = defaultdict(list)  # (kind, genre) -> rows
    n = 0
    for r in scan_basics(paths["basics"], ratings):
        kind = "movie" if r["ttype"] in MOVIE_TYPES else "series"
        for genre in ("Thriller", "Horror"):
            if genre in r["genres"]:
                buckets[(kind, genre.lower())].append(r)
        n += 1
    print(f"  {n:,} genre+rated candidates", flush=True)

    selections = {}
    for (kind, genre), rows in buckets.items():
        c = sum(x["rating"] for x in rows) / max(len(rows), 1)
        m = MOVIE_MIN_VOTES if kind == "movie" else SERIES_MIN_VOTES
        for x in rows:
            x["wr"] = weighted_rating(x["rating"], x["votes"], m, c)
        rows.sort(key=lambda x: x["wr"], reverse=True)
        keep = MOVIE_CANDIDATES if kind == "movie" else SERIES_CANDIDATES
        top = rows[:keep]
        out = os.path.join(DATAGEN_DIR, f"select_{kind}_{genre}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tconst", "title", "original", "year", "rating", "votes", "wr", "genres"])
            for x in top:
                w.writerow([x["tconst"], x["title"], x["original"], x["year"],
                            x["rating"], x["votes"], round(x["wr"], 3), x["genres"]])
        selections[(kind, genre)] = top
        sample = ", ".join(f"{x['title']}({x['year']})" for x in top[:5])
        print(f"  {kind}/{genre}: kept {len(top)}  e.g. {sample}", flush=True)

    return selections


if __name__ == "__main__":
    select()
