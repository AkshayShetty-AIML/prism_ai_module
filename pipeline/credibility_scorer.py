"""
credibility_scorer.py — Phase 2c: Genuine User Credibility Tiers
Engineer B | PRISM AI Processing Pipeline

Assigns a credibility tier to each record based on author quality signals.
Runs AFTER bot detection (Phase 2a) — requires 'bot_flag' in record.

Implements the algorithm in spec §7.2 exactly.

Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.

Output fields added to record:
    credibility_tier : str — "high" | "medium" | "low" | "bot"
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Tier thresholds (spec §7.2)
# ---------------------------------------------------------------------------

_TIER_HIGH   = 8
_TIER_MEDIUM = 4
# < 4 → "low"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _account_age_days(account_created_at: Any) -> int:
    dt = _parse_date(account_created_at)
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max((now - dt).days, 0)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def score_credibility(author: dict | None, bot_flag: str = "human") -> str:
    """
    Calculate credibility tier for a user.

    Args:
        author:   Author metadata dict from the batch item.
        bot_flag: Result from Phase 2a bot detection ("bot" | "human").

    Returns:
        str: "high" | "medium" | "low" | "bot"
    """
    # Bots always get "bot" tier, regardless of other signals
    if bot_flag == "bot":
        return "bot"

    # No author data → can't assess → default "low"
    if author is None:
        return "low"

    points = 0

    # ── Account age: older = more credible ───────────────────────────────
    age_days = _account_age_days(author.get("account_created_at"))
    if age_days > 365:
        points += 3
    elif age_days > 180:
        points += 2
    elif age_days > 30:
        points += 1

    # ── Follower count ────────────────────────────────────────────────────
    follower_count = author.get("follower_count", 0) or 0
    if follower_count > 10_000:
        points += 3
    elif follower_count > 1_000:
        points += 2
    elif follower_count > 100:
        points += 1

    # ── Profile completeness ──────────────────────────────────────────────
    if author.get("bio_present", False):
        points += 1
    if author.get("profile_picture_present", False):
        points += 1
    if author.get("verified", False):
        points += 3          # verified = highest credibility boost

    # ── Engagement consistency ────────────────────────────────────────────
    # engagement_rate = avg_likes_per_post / max(follower_count, 1)
    # NOTE: The batch payload does not include avg_likes_per_post directly;
    #       we approximate using engagement.likes for the single post provided.
    engagement = author.get("_engagement_snapshot")  # injected by pipeline_runner if available
    post_count = author.get("post_count", 0) or 0

    if engagement and post_count > 0:
        likes = engagement.get("likes", 0) or 0
        # Rough per-post estimate from this single sample
        avg_likes = likes  # single observation — treated as representative sample
        engagement_rate = avg_likes / max(follower_count, 1)
        if engagement_rate > 0.02:
            points += 2
        elif engagement_rate > 0.005:
            points += 1

    # ── Tier assignment ───────────────────────────────────────────────────
    if points >= _TIER_HIGH:
        return "high"
    elif points >= _TIER_MEDIUM:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def assess_credibility(record: dict) -> dict:
    """
    Pipeline entry-point for Phase 2c.
    Reads bot_flag (from Phase 2a) and author from the record.

    NOTE: Injects engagement snapshot into author temporarily so the
          credibility scorer can use engagement data from the record level.

    Args:
        record: Item dict (must have 'bot_flag' and 'author' keys).

    Returns:
        Same dict enriched with 'credibility_tier'.
    """
    author = record.get("author")
    bot_flag = record.get("bot_flag", "human")

    # Temporarily attach engagement snapshot so scorer can use it
    engagement = record.get("engagement")
    if author is not None and engagement is not None:
        author["_engagement_snapshot"] = engagement

    tier = score_credibility(author, bot_flag)
    record["credibility_tier"] = tier

    # Clean up temp key
    if author is not None:
        author.pop("_engagement_snapshot", None)

    return record


# Alias
process = assess_credibility
