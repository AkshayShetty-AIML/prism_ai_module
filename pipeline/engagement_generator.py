"""
PRISM — Engagement Generator (Eng C · Flow 2 On-Demand)

Generates multi-tone reply drafts for a cluster of negative comments.
Fetches comments from DB2 internally, calls LLM, returns structured reply payload.

Signature (from Eng A):
    async def generate_engagement(theme_group_id: str, keyword: str) -> dict
"""

import datetime
import logging
import os
import uuid

from prompts.engagement_prompt import build_engagement_prompt
from llm.llm_client import call_llm

logger = logging.getLogger("prism.pipeline.engagement")

VALID_TONES = frozenset([
    "empathetic", "professional", "apologetic", "informative", "clarifying",
])


def _get_db():
    """
    Get the MongoDB database connection.
    Uses motor (async) or pymongo (sync) depending on availability.
    Falls back to None if DB is not configured yet (Eng A dependency).
    """
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "prism_db2")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client[db_name]
    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", e)
        return None


async def generate_engagement(theme_group_id: str, keyword: str) -> dict:
    """
    Flow 2: Generate multi-tone reply drafts for a theme group.

    Fetches comments from DB2 internally, builds the engagement prompt,
    calls the LLM, validates the response, and returns structured reply drafts.

    Parameters
    ----------
    theme_group_id : str
        The cluster/theme group UUID.
    keyword : str
        The tracked keyword (e.g. "Leo movie").

    Returns
    -------
    dict
        Structured payload with suggested_replies, or error dict.
    """
    # Get DB connection internally (per Eng A's signature — no db param)
    db = _get_db()
    if db is None:
        logger.error("Cannot generate engagement — DB not available")
        return {"error": "Database connection unavailable"}

    try:
        system, prompt = build_engagement_prompt(theme_group_id, keyword, db)
    except Exception as e:
        logger.error("DB fetch failed for theme '%s': %s", theme_group_id, e)
        return {"error": f"Database error: {e}"}

    if not prompt:
        logger.warning("No comments for theme_group_id: %s", theme_group_id)
        return {"error": f"No comments found for theme_group_id: {theme_group_id}"}

    result = call_llm(prompt, system=system, use_cache=False)

    if not result:
        logger.error("LLM failed for engagement theme '%s'", theme_group_id)
        return {"error": "LLM failed to generate engagement replies after retries"}

    # Validate reply structure
    replies = result.get("suggested_replies", [])
    if not isinstance(replies, list):
        replies = []

    validated = []
    for i, r in enumerate(replies):
        if not isinstance(r, dict):
            continue
        tone = r.get("tone", "professional")
        if tone not in VALID_TONES:
            tone = "professional"
        validated.append({
            "reply_id": r.get("reply_id", f"r{i+1}"),
            "tone": tone,
            "text": str(r.get("text", "")),
            "suitable_for": r.get("suitable_for", []),
            "target_intent": str(r.get("target_intent", "")),
        })

    if len(validated) < 3:
        logger.warning("Only %d replies (expected 3-5) for '%s'",
                       len(validated), theme_group_id)

    output = {
        "request_id": str(uuid.uuid4()),
        "theme_group_id": theme_group_id,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "theme_summary": str(result.get("theme_summary", "Unknown theme")),
        "suggested_replies": validated,
        "confidence_note": str(result.get("confidence_note", "")),
    }

    logger.info("Generated %d replies for theme '%s'",
                len(validated), theme_group_id)
    return output
