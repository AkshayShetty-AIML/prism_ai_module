"""
bot_detector.py — Phase 2a: Bot / Human Classification
Engineer B | PRISM AI Processing Pipeline

Heuristic bot detection based solely on author metadata signals.
Implements the algorithm defined in spec §7.1 exactly.

Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.

Output fields added to record:
    bot_flag        : str   — "bot" | "human"
    bot_confidence  : float — 0.0–1.0  (composite weighted score)
    bot_flags       : list[str] — signal names that fired
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Thresholds & weights (spec §7.1)
# ---------------------------------------------------------------------------

_BOT_THRESHOLD = 0.65        # score >= threshold → "bot"

# Signal weights
_W_ACCOUNT_AGE = 0.30
_W_POST_FREQ   = 0.25
_W_FOLLOW_RATIO = 0.15
_W_PROFILE_PIC = 0.08
_W_BIO         = 0.07
_W_PROTECTED   = -0.10      # negative: bots rarely protect accounts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Any) -> datetime | None:
    """Parse ISO8601 string or datetime object into timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            # Handle Z suffix
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def _account_age_days(account_created_at: Any) -> int | None:
    """Return account age in days, or None if date unavailable."""
    dt = _parse_date(account_created_at)
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt
    return max(delta.days, 0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def detect_bot(author: dict | None) -> dict:
    """
    Run the 5-signal heuristic bot detection algorithm (spec §7.1).

    Args:
        author: Author metadata dict from the incoming batch item.
                May be None (e.g. web-crawl sources without author data).

    Returns:
        dict with keys:
            bot_flag        (str)
            bot_confidence  (float)
            bot_flags       (list[str])
    """
    # Null author → treat as human, flag the absence
    if author is None:
        return {
            "bot_flag": "human",
            "bot_confidence": 0.0,
            "bot_flags": ["no_author_data"],
        }

    score: float = 0.0
    flags: list[str] = []

    # ── Signal 5 (checked first): Verified Override ──────────────────────
    # Verified accounts are immediately human — no further scoring needed.
    if author.get("verified", False):
        return {
            "bot_flag": "human",
            "bot_confidence": 0.0,
            "bot_flags": [],
        }

    # ── Signal 1: Account Age (weight 0.30) ──────────────────────────────
    age_days = _account_age_days(author.get("account_created_at"))
    if age_days is None:
        age_days = 365  # unknown age → assume moderate; don't penalise

    if age_days < 7:
        score += _W_ACCOUNT_AGE       # 0.30
        flags.append("very_new_account")
    elif age_days < 30:
        score += 0.20
        flags.append("new_account")
    elif age_days < 90:
        score += 0.05

    # ── Signal 2: Posting Frequency (weight 0.25) ─────────────────────────
    post_count = author.get("post_count", 0) or 0
    posts_per_day = post_count / max(age_days, 1)

    if posts_per_day > 100:
        score += _W_POST_FREQ         # 0.25
        flags.append("extreme_frequency")
    elif posts_per_day > 50:
        score += 0.15
        flags.append("high_frequency")

    # ── Signal 3: Follower Ratio (weight 0.15) ────────────────────────────
    follower_count  = author.get("follower_count", 0) or 0
    following_count = author.get("following_count", 0) or 0

    if following_count > 300:
        ratio = follower_count / following_count  # already safe (> 300 denominator)
        if ratio < 0.05:
            score += _W_FOLLOW_RATIO  # 0.15
            flags.append("suspicious_follower_ratio")
        elif ratio < 0.15:
            score += 0.08

    # ── Signal 4: Profile Completeness (weight 0.15 total) ───────────────
    if not author.get("profile_picture_present", True):
        score += _W_PROFILE_PIC       # 0.08
        flags.append("no_profile_pic")

    if not author.get("bio_present", True):
        score += _W_BIO               # 0.07
        flags.append("no_bio")

    # ── Signal 6: Protected Account (negative weight) ────────────────────
    if author.get("account_protected", False):
        score += _W_PROTECTED         # -0.10

    # ── Composite Rule: new + high-frequency = guaranteed bot ─────────────
    if age_days < 30 and posts_per_day > 100:
        score = max(score, 0.70)
        if "new_account_high_frequency" not in flags:
            flags.append("new_account_high_frequency")

    score = _clamp(score, 0.0, 1.0)
    bot_flag = "bot" if score >= _BOT_THRESHOLD else "human"

    return {
        "bot_flag": bot_flag,
        "bot_confidence": round(score, 3),
        "bot_flags": flags,
    }


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def classify_bot(record: dict) -> dict:
    """
    Pipeline entry-point for Phase 2a.
    Extracts author from record and runs bot detection.

    Args:
        record: Item dict (must contain 'author' key).

    Returns:
        Same dict enriched with bot_flag, bot_confidence, bot_flags.
    """
    author = record.get("author")
    result = detect_bot(author)
    record.update(result)
    return record


# Alias
process = classify_bot
