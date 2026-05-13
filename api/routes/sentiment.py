"""
api/routes/sentiment.py — Sentiment analysis endpoints.

GET /sentiment/aggregate — Sentiment distribution for a keyword
GET /sentiment/trend     — Hourly sentiment over time (Day 4)
GET /sentiment/viral     — Viral posts (Day 4)
"""

from typing import Optional

from fastapi import APIRouter, Query

from db.mongo_client import get_sentiment_distribution

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


# GET /sentiment/trend — Day 4
# GET /sentiment/viral — Day 4