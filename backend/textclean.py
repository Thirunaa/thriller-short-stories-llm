"""Text cleaning for the story corpus.

For a small model, data quality dominates. Plot/episode text scraped from
Wikipedia/CMU carries residual markup and non-narrative noise that wastes the
model's limited capacity. This normalizes everything to clean narrative prose.
"""
from __future__ import annotations

import html
import re
import unicodedata

# Order matters: refs before generic tags; templates before link unwrapping.
_REF_BLOCK = re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I)
_REF_SELF = re.compile(r"<ref[^>]*/>", re.I)
_HTML_TAG = re.compile(r"<[^>]+>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")                 # {{cite ...}} (one nesting level)
_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")  # [[a|b]] -> b
_EXTLINK = re.compile(r"\[https?://\S+?(?:\s+([^\]]+))?\]")  # [http... text] -> text
_URL = re.compile(r"https?://\S+")
_CITE = re.compile(r"\[(?:\d{1,3}|[a-z]|citation needed|edit|note \d+)\]", re.I)
_BOLD_ITAL = re.compile(r"'{2,5}")
_HEADING = re.compile(r"^=+\s*.*?\s*=+\s*$", re.M)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?])")

# Lines that are clearly non-narrative (credits / meta).
_NOISE_LINE = re.compile(
    r"^\s*(cast|see also|references|external links|notes|production|reception|"
    r"release|soundtrack|in other media)\s*:?\s*$",
    re.I,
)


def clean_story_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    text = _REF_BLOCK.sub(" ", text)
    text = _REF_SELF.sub(" ", text)
    # Run template/link passes twice to catch shallow nesting.
    for _ in range(2):
        text = _TEMPLATE.sub(" ", text)
        text = _WIKILINK.sub(r"\1", text)
        text = _EXTLINK.sub(lambda m: m.group(1) or " ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CITE.sub("", text)
    text = _BOLD_ITAL.sub("", text)
    text = _HEADING.sub("", text)

    lines = [ln for ln in text.split("\n") if not _NOISE_LINE.match(ln)]
    text = "\n".join(lines)

    text = _SPACE_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    # strip leftover bracket/brace fragments from unbalanced markup
    text = text.replace("{{", "").replace("}}", "").replace("[[", "").replace("]]", "")
    return text.strip()
