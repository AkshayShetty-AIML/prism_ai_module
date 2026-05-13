"""
promo_classifier.py — Phase 2b: Promotional vs Organic Classification
Engineer B | PRISM AI Processing Pipeline

Rule-based classifier that flags promotional / sponsored content so it
can be separated from genuine audience sentiment in analytics.

Implements the algorithm in spec §7.5 exactly.

Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.

Output fields added to record:
    is_promotional  : bool
    content_type    : str   — "promotional" | "organic"
    promo_signals   : list[str] — signal names that fired
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Signal sets (spec §7.5 + expanded coverage for Indian market)
# ---------------------------------------------------------------------------

PROMO_HASHTAGS: frozenset[str] = frozenset({
    "#ad", "#sponsored", "#collab", "#gifted", "#partnership",
    "#paidpromotion", "#promotedpost", "#paidpartnership",
    "#brandedcontent", "#advertisement", "#promo", "#promotion",
    "#paid", "#spon", "#sp",
})

CTA_PHRASES: tuple[str, ...] = (
    # English CTAs
    "watch now", "book tickets", "streaming on", "available on",
    "in cinemas", "link in bio", "use code", "swipe up",
    "click here", "limited offer", "book now", "grab your tickets",
    "out now", "releasing on", "now in theatres", "now showing",
    "buy now", "shop now", "subscribe to", "follow us",
    "dm for", "comment below", "tag a friend", "share this",
    # Indian-market specifics
    "bookmyshow", "paytm movies", "pvr cinemas", "inox",
    "amazon prime video", "netflix india", "hotstar", "sony liv",
    "zee5", "aha video",
)

# Brand-indicator terms (used for verified-brand signal)
_BRAND_INDICATORS: tuple[str, ...] = (
    "official", "presents", "production", "studio", "entertainment",
    "films", "pictures", "pvt ltd", "corporation", "corp",
    "media", "records", "music", "label",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_hashtags(text: str) -> list[str]:
    """Return lowercased hashtags found in text."""
    return [tag.lower() for tag in re.findall(r"#\w+", text)]


def _has_promo_hashtag(text: str) -> bool:
    tags = _extract_hashtags(text)
    return any(tag in PROMO_HASHTAGS for tag in tags)


def _has_cta_language(text: str) -> bool:
    text_lower = text.lower()
    return any(cta in text_lower for cta in CTA_PHRASES)


def _is_verified_brand(author: dict | None) -> bool:
    """
    True if the author is verified AND bio/username contains brand keywords.
    This catches official studio/label accounts promoting content.
    """
    if author is None:
        return False
    if not author.get("verified", False):
        return False

    # Check username for brand-indicator terms (bio_text is not provided by Team 1)
    username = (author.get("username") or "").lower()

    return any(indicator in username for indicator in _BRAND_INDICATORS)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify(text: str, author: dict | None = None) -> dict:
    """
    Classify whether a piece of text is promotional or organic.

    Args:
        text:   The raw or normalised content text.
        author: Author metadata dict (optional).

    Returns:
        dict with keys:
            is_promotional (bool)
            content_type   (str)
            promo_signals  (list[str])
    """
    signals: list[str] = []

    # Signal 1: Promotional hashtag
    if _has_promo_hashtag(text):
        signals.append("promo_hashtag")

    # Signal 2: CTA language
    if _has_cta_language(text):
        signals.append("cta_language")

    # Signal 3: Verified brand account
    if _is_verified_brand(author):
        signals.append("verified_brand")

    is_promotional = len(signals) > 0
    return {
        "is_promotional": is_promotional,
        "content_type": "promotional" if is_promotional else "organic",
        "promo_signals": signals,
    }


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def classify_promo(record: dict) -> dict:
    """
    Pipeline entry-point for Phase 2b.
    Uses normalised_text if available, falls back to raw content.

    Args:
        record: Item dict with 'content' and optionally 'normalised_text', 'author'.

    Returns:
        Same dict enriched with is_promotional, content_type, promo_signals.
    """
    text = record.get("normalised_text") or record.get("content", "")
    author = record.get("author")
    result = classify(text, author)
    record.update(result)
    return record


# Alias
process = classify_promo
