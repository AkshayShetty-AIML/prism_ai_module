"""
PRISM — Report Generator (Eng C · Flow 4 On-Demand)

Aggregates metrics from DB2 and generates an AI insight report.

Signature (from Eng A):
    async def generate_report(keyword, date_from, date_to, segments) -> dict
"""

import logging
from typing import Optional

from db.mongo_client import (
    get_db,
    get_sentiment_distribution,
    get_bot_activity,
    get_crisis_severity_counts,
)
from prompts.report_prompt import build_report_prompt
from llm.llm_client import call_llm

logger = logging.getLogger("prism.pipeline.report")


async def _aggregate_metrics(
    keyword: str,
    date_from: Optional[str],
    date_to: Optional[str],
) -> dict:
    """Pull and combine all metrics needed for the report prompt."""
    db = get_db()

    match_q: dict = {"keyword": keyword, "is_relevant": True}
    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to
        match_q["processed_at"] = date_filter

    sentiment = await get_sentiment_distribution(keyword, date_from, date_to)
    bot = await get_bot_activity(keyword, date_from, date_to)
    crisis = await get_crisis_severity_counts(keyword, date_from, date_to)
    total = await db["analyzed_records"].count_documents(match_q)

    platform_cursor = db["analyzed_records"].aggregate([
        {"$match": match_q},
        {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ])
    platform_raw = await platform_cursor.to_list(length=20)
    platforms = {r["_id"]: r["count"] for r in platform_raw if r["_id"]}

    top_cursor = (
        db["analyzed_records"]
        .find(match_q, {"_id": 0, "content": 1, "platform": 1, "sentiment": 1, "impact_score": 1})
        .sort("impact_score", -1)
        .limit(5)
    )
    top_posts = await top_cursor.to_list(length=5)

    total_bot = bot.get("total_bot_items", 0)
    bot_pct = round(total_bot / total * 100, 1) if total > 0 else 0

    return {
        "total_records": total,
        "positive_pct": sentiment.get("positive_pct", 0),
        "neutral_pct": sentiment.get("neutral_pct", 0),
        "negative_pct": sentiment.get("negative_pct", 0),
        "bot_pct": bot_pct,
        "platforms": platforms,
        "crisis_severity": crisis,
        "top_posts": top_posts,
    }


async def generate_report(
    keyword: str,
    date_from: str,
    date_to: str,
    segments: list[str],
) -> dict:
    """
    Flow 4: Aggregate DB2 metrics and generate an AI insight report.

    Parameters
    ----------
    keyword : str
        The tracked keyword/entity.
    date_from, date_to : str
        ISO8601 date range for the report window.
    segments : list[str]
        Requested report segments (e.g. ["sentiment", "bot_activity"]).

    Returns
    -------
    dict
        Report payload with narrative_summary, key_insights, recommendations, etc.
    """
    try:
        metrics = await _aggregate_metrics(keyword, date_from, date_to)
    except Exception as e:
        logger.error("Failed to aggregate metrics for report: %s", e)
        raise RuntimeError(f"Data aggregation failed: {e}") from e

    system, prompt = build_report_prompt(
        keyword=keyword,
        date_from=date_from,
        date_to=date_to,
        metrics=metrics,
        segments=segments,
    )

    result = call_llm(prompt, system=system, use_cache=False)

    if not result:
        logger.error("LLM failed to generate report for keyword '%s'", keyword)
        raise RuntimeError("AI report generation failed after retries.")

    logger.info(
        "Report generated for '%s' (%d records)",
        keyword, metrics["total_records"],
    )

    return {
        "narrative_summary": result.get("narrative_summary", ""),
        "key_insights": result.get("key_insights", []),
        "sentiment_breakdown": result.get("sentiment_breakdown", {}),
        "bot_activity_note": result.get("bot_activity_note", ""),
        "platform_notes": result.get("platform_notes", ""),
        "recommendations": result.get("recommendations", []),
        "data_summary": {
            "total_records": metrics["total_records"],
            "date_from": date_from,
            "date_to": date_to,
        },
    }
