"""
db/mongo_client.py — MongoDB connection and CRUD helpers for PRISM AI Module (DB2).

Collections:
    - analyzed_records: stores fully processed records from the 5-phase pipeline
    - batch_jobs: tracks async job status for batch, engagement, crisis, report flows

Usage:
    from db.mongo_client import connect, save_record, create_job, get_job, update_job

    # At FastAPI startup:
    db = await connect()

    # In pipeline:
    await save_record(record)

    # In routes:
    job_id = await create_job({...})
    job = await get_job(job_id)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

# ── Module-level state ──────────────────────────────────────────────
_client: AsyncIOMotorClient | None = None
_db = None


# ── Connection ──────────────────────────────────────────────────────

async def connect():
    """
    Initialise the Motor async client and return the database object.
    Call once at FastAPI startup.  Safe to call multiple times (idempotent).
    Also creates indexes (MongoDB ignores if they already exist).
    """
    global _client, _db

    if _db is not None:
        return _db

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "prism_db2")

    _client = AsyncIOMotorClient(mongo_uri)
    _db = _client[db_name]

    # ── Create indexes (idempotent) ─────────────────────────────────
    records = _db["analyzed_records"]
    await records.create_index([("keyword", 1), ("processed_at", -1)])
    await records.create_index([("platform", 1)])
    await records.create_index([("sentiment", 1)])
    await records.create_index([("viral_flag", 1), ("impact_score", -1)])
    await records.create_index([("bot_flag", 1)])
    await records.create_index([("crisis_severity", 1)])
    await records.create_index([("batch_id", 1)])
    await records.create_index([("text_hash", 1)])

    jobs = _db["batch_jobs"]
    await jobs.create_index([("job_id", 1)], unique=True)
    await jobs.create_index([("batch_id", 1)])

    return _db


def get_db():
    """
    Return the current database object.
    Raises RuntimeError if connect() has not been called yet.
    """
    if _db is None:
        raise RuntimeError("Database not initialised — call connect() first")
    return _db


async def close():
    """Close the Motor client.  Call at FastAPI shutdown."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


# ── analyzed_records helpers ────────────────────────────────────────

async def save_record(record: dict) -> str:
    """
    Upsert a single processed record into analyzed_records.
    Uses item_id as the unique key — if the record already exists it is
    replaced, preventing duplicates on re-processing or retries.

    Returns the item_id.
    """
    db = get_db()
    item_id = record.get("item_id")
    if not item_id:
        raise ValueError("record must contain 'item_id'")

    await db["analyzed_records"].update_one(
        {"item_id": item_id},
        {"$set": record},
        upsert=True,
    )
    return item_id


async def save_records_bulk(records: list[dict]) -> int:
    """
    Bulk upsert a list of records.  Returns count of records written.
    Useful at the end of a batch when you want fewer round-trips.
    """
    if not records:
        return 0

    db = get_db()
    from pymongo import UpdateOne

    ops = [
        UpdateOne(
            {"item_id": r["item_id"]},
            {"$set": r},
            upsert=True,
        )
        for r in records
        if r.get("item_id")
    ]
    if ops:
        result = await db["analyzed_records"].bulk_write(ops)
        return result.upserted_count + result.modified_count
    return 0


async def get_record(item_id: str) -> dict | None:
    """Fetch a single analyzed record by item_id."""
    db = get_db()
    doc = await db["analyzed_records"].find_one(
        {"item_id": item_id}, {"_id": 0}
    )
    return doc


async def get_records(
    keyword: str | None = None,
    platform: str | None = None,
    sentiment: str | None = None,
    is_bot: bool | None = None,
    is_promotional: bool | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    batch_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """
    Query analyzed_records with optional filters and pagination.

    Returns:
        {
            "items": [record, ...],
            "total": int,
            "page": int,
            "page_size": int
        }
    """
    db = get_db()
    query: dict = {}

    if keyword:
        query["keyword"] = keyword
    if platform:
        query["platform"] = platform
    if sentiment:
        query["sentiment"] = sentiment
    if is_bot is not None:
        query["bot_flag"] = "bot" if is_bot else "human"
    if is_promotional is not None:
        query["is_promotional"] = is_promotional
    if batch_id:
        query["batch_id"] = batch_id

    # Date range filter on processed_at
    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from  # ISO8601 string comparison
        if date_to:
            date_filter["$lte"] = date_to
        query["processed_at"] = date_filter

    collection = db["analyzed_records"]

    total = await collection.count_documents(query)

    skip = (page - 1) * page_size
    cursor = (
        collection.find(query, {"_id": 0})
        .sort("processed_at", -1)
        .skip(skip)
        .limit(page_size)
    )
    items = await cursor.to_list(length=page_size)

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ── batch_jobs helpers ──────────────────────────────────────────────

async def create_job(
    batch_id: str | None = None,
    job_type: str = "batch",
    total_items: int = 0,
) -> dict:
    """
    Create a new job entry in batch_jobs.

    Args:
        batch_id:    The batch ID (for batch jobs) or request_id (for on-demand).
        job_type:    One of "batch", "engagement", "crisis", "report".
        total_items: Number of items to process (batch jobs).

    Returns:
        The full job document (includes generated job_id).
    """
    db = get_db()

    job = {
        "job_id": str(uuid4()),
        "batch_id": batch_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "total_items": total_items,
        "processed": 0,
        "failed": 0,
        "filtered": 0,
        "error": None,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
    }

    await db["batch_jobs"].insert_one(job)
    return job


async def get_job(job_id: str) -> dict | None:
    """
    Fetch a job by job_id.  Returns None if not found.
    """
    db = get_db()
    doc = await db["batch_jobs"].find_one(
        {"job_id": job_id}, {"_id": 0}
    )
    return doc


async def update_job(job_id: str, updates: dict) -> bool:
    """
    Partially update a job document.

    Common update patterns:
        # Mark as processing
        await update_job(job_id, {"status": "processing"})

        # Increment progress after processing one record
        await update_job(job_id, {
            "processed": new_count,
            "progress": int((new_count / total) * 100),
        })

        # Mark as completed
        await update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        # Store on-demand result (crisis/engagement/report)
        await update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": { ... },
        })

        # Mark as failed
        await update_job(job_id, {
            "status": "failed",
            "error": "LLM timeout after 3 retries",
        })

    Returns True if a document was matched, False otherwise.
    """
    db = get_db()
    result = await db["batch_jobs"].update_one(
        {"job_id": job_id},
        {"$set": updates},
    )
    return result.matched_count > 0


# ── Aggregation helpers (for alerts, reports, sentiment endpoints) ──

async def get_sentiment_distribution(
    keyword: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """
    Returns sentiment percentages for a keyword within an optional date window.

    Returns:
        {
            "positive_pct": float,
            "neutral_pct": float,
            "negative_pct": float,
            "total_items_analyzed": int
        }
    """
    db = get_db()
    match_stage: dict = {"keyword": keyword, "is_relevant": True}

    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to
        match_stage["processed_at"] = date_filter

    pipeline = [
        {"$match": match_stage},
        {
            "$group": {
                "_id": "$sentiment",
                "count": {"$sum": 1},
            }
        },
    ]

    cursor = db["analyzed_records"].aggregate(pipeline)
    results = await cursor.to_list(length=10)

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        if r["_id"] in counts:
            counts[r["_id"]] = r["count"]

    total = sum(counts.values())
    if total == 0:
        return {
            "positive_pct": 0,
            "neutral_pct": 0,
            "negative_pct": 0,
            "total_items_analyzed": 0,
        }

    return {
        "positive_pct": round(counts["positive"] / total * 100, 1),
        "neutral_pct": round(counts["neutral"] / total * 100, 1),
        "negative_pct": round(counts["negative"] / total * 100, 1),
        "total_items_analyzed": total,
    }


async def get_bot_activity(
    keyword: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """
    Returns bot activity stats for a keyword.

    Returns:
        {"total_bot_items": int, "pct_of_total": float}
    """
    db = get_db()
    match_stage: dict = {"keyword": keyword, "is_relevant": True}

    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to
        match_stage["processed_at"] = date_filter

    total = await db["analyzed_records"].count_documents(match_stage)
    bot_count = await db["analyzed_records"].count_documents(
        {**match_stage, "bot_flag": "bot"}
    )

    return {
        "total_bot_items": bot_count,
        "pct_of_total": round(bot_count / total * 100, 1) if total > 0 else 0,
    }


async def get_records_by_theme_group(
    theme_group_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Fetch records belonging to a crisis theme group.
    Used by Flow 2 (engagement generator) to get sample comments.
    """
    db = get_db()
    cursor = (
        db["analyzed_records"]
        .find(
            {"crisis_theme_group": theme_group_id},
            {"_id": 0, "content": 1, "platform": 1, "sentiment": 1},
        )
        .limit(limit)
    )
    return await cursor.to_list(length=limit)