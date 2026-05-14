"""
api/routes/sentiment.py — Sentiment analysis endpoints.

GET /sentiment/aggregate — Sentiment distribution for a keyword
GET /sentiment/trend     — Hourly sentiment over time
GET /sentiment/viral     — Viral posts (impact_score > 80)
GET /keywords/compare    — Compare sentiment across multiple keywords
"""

from typing import Optional

from fastapi import APIRouter, Query

from db.mongo_client import (
    get_sentiment_distribution,
    get_sentiment_trend,
    get_viral_records,
    get_keywords_compare,
)

router = APIRouter()


@router.get("/sentiment/aggregate")
async def sentiment_aggregate(
    keyword: str = Query(..., description="Tracked keyword to aggregate sentiment for"),
    date_from: Optional[str] = Query(None, description="ISO8601 start date"),
    date_to: Optional[str] = Query(None, description="ISO8601 end date"),
):
    """
    Returns sentiment percentages (positive/neutral/negative) for a keyword.

    Optional date range narrows the window. Without dates, aggregates all time.
    """
    result = await get_sentiment_distribution(
        keyword=keyword,
        date_from=date_from,
        date_to=date_to,
    )
    return result


@router.get("/sentiment/trend")
async def sentiment_trend(
    keyword: str = Query(..., description="Tracked keyword"),
    date_from: Optional[str] = Query(None, description="ISO8601 start date"),
    date_to: Optional[str] = Query(None, description="ISO8601 end date"),
):
    """
    Returns hourly sentiment counts over time.

    Each entry: {"hour": "2026-05-10T08:00:00Z", "positive": 12, "neutral": 5, "negative": 3}
    Sorted chronologically. Powers the trend graph on the dashboard.
    """
    trend = await get_sentiment_trend(
        keyword=keyword,
        date_from=date_from,
        date_to=date_to,
    )
    return {"keyword": keyword, "trend": trend}


@router.get("/sentiment/viral")
async def sentiment_viral(
    keyword: str = Query(..., description="Tracked keyword"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Returns viral posts (impact_score > 80) sorted by impact score.

    These are high-visibility posts driving the conversation.
    """
    result = await get_viral_records(
        keyword=keyword,
        page=page,
        page_size=page_size,
    )
    return result


@router.get("/keywords/compare")
async def keywords_compare(
    keywords: str = Query(..., description="Comma-separated keywords to compare"),
):
    """
    Compare sentiment distribution across multiple keywords.

    Pass keywords as comma-separated: ?keywords=Leo movie,Jailer movie
    Returns sentiment percentages for each keyword side by side.
    """
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if not keyword_list:
        return {"comparisons": []}

    comparisons = await get_keywords_compare(keyword_list)
    return {"comparisons": comparisons}