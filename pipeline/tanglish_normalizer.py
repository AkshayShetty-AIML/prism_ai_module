"""
tanglish_normalizer.py — Phase 1b: Text Normalization
Engineer B | PRISM AI Processing Pipeline

Normalizes Tanglish/Hinglish slang into standard English equivalents
so downstream LLM (Eng C) receives cleaner, more consistent text.

Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.

Output fields added to record:
    normalised_text : str — slang-normalized version of content
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load slang dictionary from data/slang_map.json
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_SLANG_MAP_PATH = _DATA_DIR / "slang_map.json"


def _load_slang_dict() -> dict[str, str]:
    """
    Load and merge Tanglish + Hinglish entries into a single flat dict.
    Keys are lowercased for case-insensitive matching.
    Sorts by length (longest first) to handle multi-word phrases before single words.
    """
    flat: dict[str, str] = {}
    try:
        with open(_SLANG_MAP_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for section in ("tanglish", "hinglish"):
            for slang, replacement in raw.get(section, {}).items():
                flat[slang.lower().strip()] = replacement
    except FileNotFoundError:
        # Fallback inline dictionary if JSON not found
        flat = _INLINE_FALLBACK
    return dict(sorted(flat.items(), key=lambda kv: len(kv[0]), reverse=True))


# Minimal inline fallback — used if slang_map.json is missing during tests
_INLINE_FALLBACK: dict[str, str] = {
    "semma": "excellent",
    "vera level": "outstanding",
    "mass": "impressive",
    "nalla": "good",
    "mokkai": "boring",
    "padam": "film",
    "ayyo": "dismay",
    "bakwaas": "nonsense",
    "zabardast": "outstanding",
    "bekar": "useless",
    "mast": "great",
    "yaar": "friend",
    "bhai": "brother",
    "ekdum": "absolutely",
    "timepass": "time wasting",
    "paisa vasool": "worth the money",
    "bindaas": "carefree/excellent",
    "thala": "fan term for actor Vijay",
    "thalapathy": "fan term for actor Vijay",
    "fdfs": "opening day premiere",
    "bgm": "background music",
    "machane": "friend",
    "macha": "friend",
    "dei": "hey",
    "kuthu": "energetic dance number",
}

# Load once at module import time
SLANG_DICT: dict[str, str] = _load_slang_dict()

# ---------------------------------------------------------------------------
# Filler / noise tokens to strip (non-semantic particles)
# ---------------------------------------------------------------------------

_FILLER_TOKENS: frozenset[str] = frozenset({
    "da", "di", "ra", "la", "na", "pa", "pa da",
})

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _collapse_repeated_chars(text: str) -> str:
    """
    'Semmaaaaaa' → 'Semma', 'nallaaaa' → 'nallaa'
    Collapses 3+ consecutive identical characters to 2 (preserves intentional doubles).
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _remove_emoji_clusters(text: str) -> str:
    """Remove emoji characters to reduce noise for downstream NLP."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002700-\U000027BF"   # dingbats
        "\U0001F900-\U0001F9FF"   # supplemental symbols
        "\U00002600-\U000026FF"   # misc symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def _strip_urls(text: str) -> str:
    """Remove http/https URLs from text."""
    return re.sub(r"https?://\S+", "", text)


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _replace_slang(text: str, slang_dict: dict[str, str]) -> str:
    """
    Replace slang phrases/words with their normalised equivalents.
    Uses longest-match-first to handle multi-word phrases correctly.
    Case-insensitive matching; preserves surrounding whitespace rhythm.
    """
    result = text
    for slang, replacement in slang_dict.items():
        if not slang:
            continue
        # Word-boundary aware replacement (handles multi-word phrases)
        # Escape special regex chars in slang key
        pattern = r"(?<!\w)" + re.escape(slang) + r"(?!\w)"
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _strip_fillers(text: str) -> str:
    """Strip standalone filler particles (da, di, etc.) at word boundaries."""
    for filler in _FILLER_TOKENS:
        pattern = r"(?<!\w)" + re.escape(filler) + r"(?!\w)"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """
    Apply full normalization pipeline to a single text string.

    Steps:
        1. Strip URLs
        2. Remove emoji clusters
        3. Collapse repeated characters (semmmmaaa → semma)
        4. Replace slang (longest-match-first)
        5. Strip filler tokens
        6. Normalize whitespace

    Args:
        text: Raw comment/post text string.

    Returns:
        Normalised text string with slang replaced by English equivalents.
    """
    if not text or not text.strip():
        return text

    text = _strip_urls(text)
    text = _remove_emoji_clusters(text)
    text = _collapse_repeated_chars(text)
    text = _replace_slang(text, SLANG_DICT)
    text = _strip_fillers(text)
    text = _normalise_whitespace(text)
    return text


def process(record: dict) -> dict:
    """
    Pipeline entry-point for Phase 1b.
    Normalises the record's 'content' and stores result as 'normalised_text'.

    Args:
        record: Item dict (must contain 'content' key).
                Should already have 'language_detected' from Phase 1a.

    Returns:
        Same dict enriched with 'normalised_text' field.
    """
    content = record.get("content", "")
    record["normalised_text"] = normalise(content)
    return record


# ---------------------------------------------------------------------------
# Eng A interface alias — matches pipeline_runner.py contract exactly
# Usage: from pipeline.tanglish_normalizer import normalise_tanglish
# ---------------------------------------------------------------------------
def normalise_tanglish(record: dict) -> dict:
    """
    Eng A contract alias (Phase 1b).
    Wraps process() so pipeline_runner.py can call normalise_tanglish(record).
    """
    return process(record)
