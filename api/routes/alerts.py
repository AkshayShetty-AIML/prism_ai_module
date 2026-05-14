"""
api/routes/alerts.py — Alert metrics endpoint (Flow 5).

GET /alerts/metrics — Returns aggregated metrics for a keyword within a time window.
Synchronous — returns immediately, no job polling needed.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from db.mongo_client import (
    get_sentiment_distribution,
    get_bot_activity,
    get_crisis_severity_counts,
    get_keyword_volume,
)

router = APIRouter()


def parse_duration(duration: str) -> timedelta:
    """
    Parse duration string like '6h', '24h', '7d' into a timedelta.
    """
    unit = duration[-1]
    value = int(duration[:-1])
    if unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "m":
        return timedelta(minutes=value)
    return timedelta(hours=value)


@router.get("/alerts/metrics")
async def alert_metrics(
    keyword: str = Query(..., description="Tracked keyword"),
    duration: str = Query("6h", description="Window size, e.g. '6h', '24h', '7d'"),
    reference_window: str = Query("previous_6h", description="Reference window for delta, e.g. 'previous_6h'"),
    metrics: str = Query(
        "sentiment_distribution,sentiment_delta,crisis_severity_count,bot_activity_count,keyword_volume",
        description="Comma-separated metrics to return",
    ),
):
    """
    Returns aggregated alert metrics for a keyword within a time window.

    Synchronous endpoint — Backend polls this periodically to check
    if any thresholds are crossed (sentiment spikes, crisis events, bot surges).

    Available metrics:
    - sentiment_distribution: positive/neutral/negative percentages
    - sentiment_delta: shift vs reference window
    - crisis_severity_count: count by severity level
    - bot_activity_count: bot items and percentage
    - keyword_volume: total items + delta vs reference
    """
    now = datetime.now(timezone.utc)
    window_delta = parse_duration(duration)

    date_to = now.isoformat()
    date_from = (now - window_delta).isoformat()

    # Parse reference window for delta calculations
    ref_duration_str = reference_window.replace("previous_", "")
    ref_delta = parse_duration(ref_duration_str)
    ref_to = date_from  # reference ends where current starts
    ref_from = (now - window_delta - ref_delta).isoformat()

    requested_metrics = [m.strip() for m in metrics.split(",")]
    result_metrics = {}

    if "sentiment_distribution" in requested_metrics:
        result_metrics["sentiment_distribution"] = await get_sentiment_distribution(
            keyword=keyword, date_from=date_from, date_to=date_to,
        )

    if "sentiment_delta" in requested_metrics:
        current = await get_sentiment_distribution(
            keyword=keyword, date_from=date_from, date_to=date_to,
        )
        reference = await get_sentiment_distribution(
            keyword=keyword, date_from=ref_from, date_to=ref_to,
        )
        result_metrics["sentiment_delta"] = {
            "negative_shift_pct": round(
                current.get("negative_pct", 0) - reference.get("negative_pct", 0), 1
            ),
            "positive_shift_pct": round(
                current.get("positive_pct", 0) - reference.get("positive_pct", 0), 1
            ),
        }

    if "crisis_severity_count" in requested_metrics:
        result_metrics["crisis_severity_count"] = await get_crisis_severity_counts(
            keyword=keyword, date_from=date_from, date_to=date_to,
        )

    if "bot_activity_count" in requested_metrics:
        result_metrics["bot_activity_count"] = await get_bot_activity(
            keyword=keyword, date_from=date_from, date_to=date_to,
        )

    if "keyword_volume" in requested_metrics:
        result_metrics["keyword_volume"] = await get_keyword_volume(
            keyword=keyword,
            date_from=date_from, date_to=date_to,
            ref_from=ref_from, ref_to=ref_to,
        )

    return {
        "keyword": keyword,
        "window": {"from": date_from, "to": date_to},
        "metrics": result_metrics,
    }