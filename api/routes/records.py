"""
api/routes/records.py — Fetch analyzed records (Flow 1 output).

GET /records       — Filtered + paginated list
GET /records/{id}  — Single record by item_id
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from db.mongo_client import get_records, get_record

router = APIRouter()


@router.get("/records")
async def list_records(
    keyword: Optional[str] = Query(None, description="Filter by tracked keyword"),
    platform: Optional[str] = Query(None, description="youtube|twitter|reddit|external"),
    sentiment: Optional[str] = Query(None, description="positive|neutral|negative"),
    is_bot: Optional[bool] = Query(None, description="True = bots only, False = humans only"),
    is_promotional: Optional[bool] = Query(None, description="True = promo only, False = organic only"),
    date_from: Optional[str] = Query(None, description="ISO8601 start date"),
    date_to: Optional[str] = Query(None, description="ISO8601 end date"),
    batch_id: Optional[str] = Query(None, description="Filter by batch ID"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
):
    """
    Fetch analyzed records with optional filters and pagination.

    Team 1 calls this after a batch is completed to retrieve processed results.
    All filters are optional — no filters returns all records.
    """
    result = await get_records(
        keyword=keyword,
        platform=platform,
        sentiment=sentiment,
        is_bot=is_bot,
        is_promotional=is_promotional,
        date_from=date_from,
        date_to=date_to,
        batch_id=batch_id,
        page=page,
        page_size=page_size,
    )
    return result


@router.get("/records/{item_id}")
async def get_single_record(item_id: str):
    """Fetch a single analyzed record by its item_id."""
    record = await get_record(item_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {item_id} not found")
    return record