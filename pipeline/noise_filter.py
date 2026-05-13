"""
noise_filter.py — Phase 1c: Relevance Scoring + Deduplication
Engineer B | PRISM AI Processing Pipeline

Filters out irrelevant, duplicate, and too-short records so the expensive
LLM step (Phase 3) only runs on genuinely useful data.

Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.
NOTE: The seen_hashes dedup store is module-level and shared across calls
      within a single batch. Call reset_dedup_store() between batches.

Output fields added to record:
    is_relevant     : bool   — False = stop pipeline here, skip LLM
    relevance_score : float  — 0.0–1.0
    filter_reason   : str|None — None | "duplicate" | "too_short" | "off_topic"
    is_duplicate    : bool
    text_hash       : str    — MD5 of normalised_text
"""

from __future__ import annotations

import hashlib
import re

# ---------------------------------------------------------------------------
# Domain context words — film/celebrity ecosystem (Indian focus)
# ---------------------------------------------------------------------------

FILM_WORDS: frozenset[str] = frozenset({
    # English film terms
    "movie", "film", "scene", "actor", "actress", "review", "trailer",
    "song", "director", "bgm", "screenplay", "cinema", "release",
    "padam", "flop", "hit", "blockbuster", "sequel", "premiere",
    "ott", "streaming", "screen", "box office", "collection",
    "casting", "character", "plot", "climax", "interval",
    # Tamil transliterations
    "padam", "paatha", "kadha", "fdfs", "kuthu", "thala", "thalapathy",
    # Hindi transliterations
    "filmy", "dhamaal",
    # Platforms
    "youtube", "twitter", "reddit", "instagram",
    # Celebrity-related
    "celebrity", "star", "fans", "fandom", "trending",
})

# Weight constants (must sum to 1.0 across all signals)
_W_KEYWORD = 0.40
_W_DOMAIN = 0.40
_W_LENGTH = 0.20

# Relevance threshold — records below this are filtered out
RELEVANCE_THRESHOLD = 0.40

# Dedup store: MD5 hash → True (module-level, reset per batch)
_seen_hashes: set[str] = set()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def reset_dedup_store() -> None:
    """
    Clear the in-memory deduplication hash store.
    Call this at the start of every new batch to avoid cross-batch false positives.
    """
    global _seen_hashes
    _seen_hashes = set()


def compute_text_hash(normalised_text: str) -> str:
    """Return MD5 hex-digest of the normalised text (lowercased + stripped)."""
    cleaned = normalised_text.lower().strip()
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def _word_count(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split()) if text.strip() else 0


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Case-insensitive check if keyword (or any keyword token) is in text."""
    if not keyword:
        return False
    text_lower = text.lower()
    # Full keyword phrase
    if keyword.lower() in text_lower:
        return True
    # Any significant keyword token (length > 3 to skip "a", "of", etc.)
    for token in keyword.lower().split():
        if len(token) > 3 and token in text_lower:
            return True
    return False


def _domain_hits(text: str) -> int:
    """Count how many FILM_WORDS appear in the text (lowercased)."""
    text_lower = text.lower()
    hits = 0
    for word in FILM_WORDS:
        # Use word-boundary check for single-word entries
        if " " in word:
            if word in text_lower:
                hits += 1
        else:
            if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text_lower):
                hits += 1
    return hits


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_relevance(text: str, keyword: str) -> dict:
    """
    Compute a relevance score for a piece of text against a keyword.

    Algorithm (from spec §7.4):
        Signal 1: Keyword presence      → +0.40
        Signal 2: Domain context words  → up to +0.40 (0.10 per hit, cap 4 hits)
        Signal 3: Minimum length        → +0.20 (≥5 words), +0.10 (3-4 words)

    Args:
        text:    The normalised text of the record.
        keyword: The keyword/entity being tracked (e.g., "Leo movie").

    Returns:
        dict:
            relevance_score (float)
            is_relevant     (bool)
            filter_reason   (str | None)
    """
    score = 0.0

    # Signal 1 — Keyword presence (0.40)
    if _keyword_in_text(keyword, text):
        score += _W_KEYWORD

    # Signal 2 — Domain context (up to 0.40)
    hits = _domain_hits(text)
    score += min(hits * 0.10, _W_DOMAIN)

    # Signal 3 — Length (0.20 or 0.10)
    wc = _word_count(text)
    if wc >= 5:
        score += _W_LENGTH
    elif wc >= 3:
        score += _W_LENGTH / 2

    score = round(min(score, 1.0), 3)
    is_relevant = score >= RELEVANCE_THRESHOLD
    filter_reason = None if is_relevant else "off_topic"

    return {
        "relevance_score": score,
        "is_relevant": is_relevant,
        "filter_reason": filter_reason,
    }


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def filter_record(record: dict) -> dict:
    """
    Apply noise filtering to a single record.

    Checks (in order):
        1. Too short          → filter_reason = "too_short"
        2. Duplicate (MD5)    → filter_reason = "duplicate"
        3. Relevance score    → filter_reason = "off_topic" if below threshold

    Args:
        record: Item dict. Expected keys:
                    normalised_text (str)  — from Phase 1b
                    content         (str)  — raw text (fallback)
                    keyword         (str)  — batch keyword

    Returns:
        Same dict enriched with: is_relevant, relevance_score,
        filter_reason, is_duplicate, text_hash.
    """
    global _seen_hashes

    text = record.get("normalised_text") or record.get("content", "")
    keyword = record.get("keyword", "")

    # --- Compute hash for dedup ---
    text_hash = compute_text_hash(text)
    record["text_hash"] = text_hash

    # --- Guard: too short ---
    wc = _word_count(text)
    if wc < 3:
        record.update({
            "is_relevant": False,
            "relevance_score": 0.0,
            "filter_reason": "too_short",
            "is_duplicate": False,
        })
        return record

    # --- Guard: duplicate ---
    if text_hash in _seen_hashes:
        record.update({
            "is_relevant": False,
            "relevance_score": 0.0,
            "filter_reason": "duplicate",
            "is_duplicate": True,
        })
        return record

    # Mark as seen
    _seen_hashes.add(text_hash)

    # --- Relevance scoring ---
    result = score_relevance(text, keyword)
    record.update(result)
    record["is_duplicate"] = False

    return record


# Alias for pipeline_runner.py convention
def process(record: dict) -> dict:
    """Alias for filter_record() to match pipeline interface convention."""
    return filter_record(record)


# ---------------------------------------------------------------------------
# Eng A interface alias — matches pipeline_runner.py contract exactly
# Usage: from pipeline.noise_filter import filter_noise
# ---------------------------------------------------------------------------
def filter_noise(record: dict) -> dict:
    """
    Eng A contract alias (Phase 1c).
    Wraps filter_record() so pipeline_runner.py can call filter_noise(record).
    """
    return filter_record(record)
