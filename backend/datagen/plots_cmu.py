"""CMU Movie Summary Corpus -> a {normalized title|year: plot} lookup.

~42k Wikipedia-sourced plot summaries with metadata (countries/languages), freely
available. Used as the fast, no-network bulk source for movie plots before falling
back to live Wikipedia fetches for whatever is missing.
"""
from __future__ import annotations

import os
import tarfile

from datagen.common import RAW_DIR, download_file, match_key

CMU_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
CMU_TGZ = os.path.join(RAW_DIR, "MovieSummaries.tar.gz")
CMU_DIR = os.path.join(RAW_DIR, "MovieSummaries")


def ensure_cmu() -> str:
    download_file(CMU_URL, CMU_TGZ)
    if not os.path.isdir(CMU_DIR):
        with tarfile.open(CMU_TGZ, "r:gz") as tar:
            tar.extractall(RAW_DIR)
    return CMU_DIR


def build_index() -> dict:
    """Return {match_key(title, year): plot}."""
    ensure_cmu()
    # plot_summaries.txt: <wiki_movie_id>\t<plot>
    plots = {}
    with open(os.path.join(CMU_DIR, "plot_summaries.txt"), "r", encoding="utf-8") as f:
        for line in f:
            mid, _, summary = line.partition("\t")
            if summary.strip():
                plots[mid.strip()] = summary.strip()

    # movie.metadata.tsv: id, freebase, name, release_date, box, runtime, langs, countries, genres
    index = {}
    with open(os.path.join(CMU_DIR, "movie.metadata.tsv"), "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            mid, name, release = cols[0], cols[2], cols[3]
            plot = plots.get(mid)
            if not plot:
                continue
            year = release[:4] if release else ""
            index[match_key(name, year)] = plot
            # also index without year to catch off-by-one release-year mismatches
            index.setdefault(match_key(name, ""), plot)
    return index


if __name__ == "__main__":
    idx = build_index()
    print(f"CMU index: {len(idx):,} keyed plots")
