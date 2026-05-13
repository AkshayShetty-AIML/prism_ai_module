"""
PRISM — Impact Scorer (Eng C · Phase 4)

Computes a 0.0–100.0 impact score for each record based on engagement
metrics and author reach. Assigns an impact tier and viral flag.

Pure mathematical function — no LLM calls, no DB calls.

Interface contract (from Eng A):
    def score_impact(record: dict) -> dict

Formula (from spec §7.3):
    engagement = likes + (replies × 1.5) + (shares × 2.0) + (views × 0.01)
    reach_factor = log10(follower_count + 1) / log10(1_000_001)
    raw_score = engagement × (1 + reach_factor)
    impact_score = min((raw_score / 10000) × 100, 100.0)
"""

import logging
import math

logger = logging.getLogger("prism.pipeline.impact")

# Tier thresholds (from spec §7.3)
_TIER_THRESHOLDS = [
    (80, "Viral"),
    (60, "High Impact"),
    (40, "Notable"),
]
_DEFAULT_TIER = "Low Impact"

# Calibration cap for normalising raw score to 0–100 range (POC value)
_CALIBRATION_CAP = 10_000


def score_impact(record: dict) -> dict:
    """
    Phase 4: Compute impact score from engagement metrics and author reach.

    Parameters
    ----------
    record : dict
        Must contain (all optional — missing values default to 0):
        - engagement.likes (int)
        - engagement.replies (int)
        - engagement.shares (int)
        - engagement.views (int)
        - author.follower_count (int)

    Returns
    -------
    dict
        The same record enriched with:
        - impact_score (float): 0.0–100.0
        - impact_tier (str): Viral | High Impact | Notable | Low Impact
        - viral_flag (bool): True if impact_score > 80
    """
    # ── Extract metrics (default to 0 if missing/None) ──
    engagement_data = record.get("engagement") or {}
    likes = _safe_int(engagement_data.get("likes"))
    replies = _safe_int(engagement_data.get("replies"))
    shares = _safe_int(engagement_data.get("shares"))
    views = _safe_int(engagement_data.get("views"))

    author_data = record.get("author") or {}
    follower_count = _safe_int(author_data.get("follower_count"))

    # ── 1. Engagement component (weighted) ──
    engagement = likes + (replies * 1.5) + (shares * 2.0) + (views * 0.01)

    # ── 2. Reach factor: log-normalised follower count ──
    # 0 followers → 0.0,  1K → ~0.5,  1M → 1.0
    reach_factor = math.log10(follower_count + 1) / math.log10(1_000_001)

    # ── 3. Raw score ──
    raw_score = engagement * (1 + reach_factor)

    # ── 4. Normalise to 0–100 ──
    impact_score = min((raw_score / _CALIBRATION_CAP) * 100, 100.0)
    impact_score = round(impact_score, 1)

    # ── 5. Tier assignment ──
    tier = _DEFAULT_TIER
    for threshold, tier_name in _TIER_THRESHOLDS:
        if impact_score > threshold:
            tier = tier_name
            break

    # ── 6. Enrich record ──
    record["impact_score"] = impact_score
    record["impact_tier"] = tier
    record["viral_flag"] = impact_score > 80

    logger.debug(
        "Item '%s': impact=%.1f tier=%s viral=%s",
        record.get("item_id", "unknown"),
        impact_score, tier, record["viral_flag"],
    )

    return record


def _safe_int(value) -> int:
    """Safely convert a value to int, defaulting to 0 on None or error."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
