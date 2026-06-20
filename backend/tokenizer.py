"""tiktoken (GPT-2 BPE) wrapper plus the chat formatting used for training/inference.

The MiniGPT is trained on conversations rendered as a flat text stream:

    ### User:
    {user text}
    ### Assistant:
    {assistant text}<|endoftext|>

so at inference we prompt with the same template up to "### Assistant:\n" and let
the model continue.
"""
from __future__ import annotations

from typing import List, Dict

import tiktoken

_enc = tiktoken.get_encoding("gpt2")

EOT = _enc.eot_token          # 50256, the <|endoftext|> document separator
REAL_VOCAB = _enc.n_vocab     # 50257

USER_TAG = "### User:\n"
ASSISTANT_TAG = "\n### Assistant:\n"


def encode_ordinary(text: str) -> List[int]:
    """Encode user/assistant text, treating any literal special tokens as plain text."""
    return _enc.encode_ordinary(text)


def encode_with_special(text: str) -> List[int]:
    return _enc.encode(text, allowed_special="all")


def decode(ids: List[int]) -> str:
    # Guard against padded-vocab ids that have no real token.
    ids = [int(i) for i in ids if 0 <= int(i) < REAL_VOCAB]
    return _enc.decode(ids)


def render_conversation(messages: List[Dict[str, str]]) -> str:
    """Render one dataset row (list of {role, content}) into a training string."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            parts.append(ASSISTANT_TAG + content)
        else:  # user / system both become a user turn
            parts.append(USER_TAG + content)
    return "".join(parts)


def encode_conversation(messages: List[Dict[str, str]]) -> List[int]:
    """Token ids for one conversation, terminated by the EOT separator."""
    return encode_ordinary(render_conversation(messages)) + [EOT]


def build_prompt(user_text: str) -> str:
    """Format a single user message into the inference prompt the model expects."""
    return USER_TAG + user_text.strip() + ASSISTANT_TAG


def encode_prompt(user_text: str) -> List[int]:
    return encode_ordinary(build_prompt(user_text))
