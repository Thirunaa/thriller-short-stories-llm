"""Keyless Wikipedia extraction:
  - find_plot(title, year): the 'Plot'/'Synopsis' section of a film article
  - episode_summaries(series_title): episode-by-episode summaries from the
    "List of <series> episodes" page's episode tables

Uses the MediaWiki action API (action=parse) and stdlib HTML parsing only.
Coverage is best-effort: not every title resolves and not every series/episode
has a published summary.
"""
from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from typing import List, Optional

from datagen.common import http_get

API = "https://en.wikipedia.org/w/api.php"

PLOT_HEADINGS = {"plot", "plot summary", "synopsis", "premise", "story", "plot synopsis"}


def _api(params: dict) -> Optional[dict]:
    params = {**params, "format": "json", "formatversion": "2"}
    body = http_get(API, params=params, min_interval=0.15)
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# HTML -> text helpers
# --------------------------------------------------------------------------- #
class _PlainText(HTMLParser):
    SKIP = {"sup", "style", "script", "table", "ref"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "li") and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        t = html.unescape("".join(self.parts))
        t = re.sub(r"\[[0-9a-z]+\]", "", t)          # leftover [1] [a] citation marks
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n\n", t)
        return t.strip()


def _strip_html(fragment: str) -> str:
    p = _PlainText()
    p.feed(fragment)
    return p.text()


# --------------------------------------------------------------------------- #
# Movie plot
# --------------------------------------------------------------------------- #
def _resolve_page(title: str, year) -> Optional[str]:
    candidates = [f"{title} ({year} film)", f"{title} (film)", title] if year else [f"{title} (film)", title]
    for cand in candidates:
        data = _api({"action": "query", "titles": cand, "redirects": "1"})
        if not data:
            continue
        pages = data.get("query", {}).get("pages", [])
        if pages and not pages[0].get("missing"):
            return pages[0]["title"]
    return None


def find_plot(title: str, year) -> Optional[str]:
    page = _resolve_page(title, year)
    if not page:
        return None
    sec = _api({"action": "parse", "page": page, "prop": "sections"})
    if not sec:
        return None
    idx = None
    for s in sec.get("parse", {}).get("sections", []):
        if s.get("line", "").strip().lower() in PLOT_HEADINGS:
            idx = s.get("index")
            break
    if idx is None:
        return None
    body = _api({"action": "parse", "page": page, "section": idx, "prop": "text"})
    if not body:
        return None
    text = _strip_html(body.get("parse", {}).get("text", ""))
    # drop a leading repeated heading line like "Plot"
    text = re.sub(r"^(plot[\w \-]*|synopsis|premise|story)\s*\n", "", text, flags=re.I)
    return text.strip() or None


# --------------------------------------------------------------------------- #
# Series episode summaries
# --------------------------------------------------------------------------- #
class _EpisodeTable(HTMLParser):
    """Collect (episode_title, summary) pairs from Wikipedia episode tables."""

    def __init__(self):
        super().__init__()
        self.episodes: List[dict] = []
        self._mode = None          # 'title' | 'desc' | None
        self._buf: List[str] = []
        self._skip_depth = 0
        self._cur_title: Optional[str] = None

    def _cls(self, attrs) -> str:
        for k, v in attrs:
            if k == "class":
                return v or ""
        return ""

    def handle_starttag(self, tag, attrs):
        cls = self._cls(attrs)
        if tag == "td" and "summary" in cls and self._mode is None:
            self._mode, self._buf = "title", []
        elif tag == "td" and "description" in cls and self._mode is None:
            self._mode, self._buf = "desc", []
        elif self._mode and tag in ("sup", "style", "script"):
            self._skip_depth += 1
        elif self._mode == "desc" and tag in ("p", "br"):
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if self._mode and tag in ("sup", "style", "script") and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "td" and self._mode:
            text = html.unescape("".join(self._buf)).strip()
            text = re.sub(r"\[[0-9a-z]+\]", "", text)
            text = re.sub(r"[ \t]+", " ", text).strip()
            if self._mode == "title":
                self._cur_title = text.strip('"')
            elif self._mode == "desc" and len(text) > 40:
                self.episodes.append({"title": self._cur_title, "summary": text})
                self._cur_title = None
            self._mode = None

    def handle_data(self, data):
        if self._mode and self._skip_depth == 0:
            self._buf.append(data)


def _page_html(page: str) -> Optional[str]:
    data = _api({"action": "parse", "page": page, "prop": "text"})
    if not data or "parse" not in data:
        return None
    return data["parse"]["text"]


def episode_summaries(series_title: str, year=None) -> List[dict]:
    """Best-effort episode summaries. Tries the dedicated episode-list page first,
    then the main article."""
    candidates = [
        f"List of {series_title} episodes",
        f"{series_title} ({year} TV series)" if year else None,
        f"{series_title} (TV series)",
        series_title,
    ]
    for cand in [c for c in candidates if c]:
        resolved = _resolve_exists(cand)
        if not resolved:
            continue
        body = _page_html(resolved)
        if not body:
            continue
        parser = _EpisodeTable()
        parser.feed(body)
        if parser.episodes:
            return parser.episodes

    # Fallback: many large series keep episode summaries in per-season sub-articles
    # ("<series> (season N)") rather than on the list page.
    all_eps: List[dict] = []
    misses = 0
    for n in range(1, 13):
        page = _resolve_exists(f"{series_title} (season {n})")
        if not page:
            misses += 1
            if all_eps or misses >= 2:
                break
            continue
        body = _page_html(page)
        if not body:
            continue
        parser = _EpisodeTable()
        parser.feed(body)
        all_eps.extend(parser.episodes)
    return all_eps


def _resolve_exists(title: str) -> Optional[str]:
    data = _api({"action": "query", "titles": title, "redirects": "1"})
    if not data:
        return None
    pages = data.get("query", {}).get("pages", [])
    if pages and not pages[0].get("missing"):
        return pages[0]["title"]
    return None
