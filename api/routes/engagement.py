"""
api/routes/engagement.py — Comment engagement endpoint (Flow 2).

POST /engagement/generate — Generate multi-tone reply drafts for a negative theme group.
Async polling pattern: returns job_id immediately, poll GET /jobs/{job_id} for result.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import EngagementRequest, EngagementJobResponse
from db.mongo_client import create_job, update_job
from pipeline.engagement_generator import generate_engagement

router = APIRouter()


async def process_engagement(job_id: str, theme_group_id: str, keyword: str):
    """
    Background task: calls Eng C's engagement generator,
    stores the result in the job document.
    """
    await update_job(job_id, {"status": "processing"})

    try:
        result = await generate_engagement(
            theme_group_id=theme_group_id,
            keyword=keyword,
        )

        await update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        })

    except Exception as e:
        await update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


@router.post("/engagement/generate", response_model=EngagementJobResponse)
async def engagement_generate(request: EngagementRequest, background_tasks: BackgroundTasks):
    """
    Generate multi-tone reply drafts for a negative theme group.

    The AI fetches comments for the theme_group_id from DB2, understands
    the common complaint, and generates 3-5 draft replies in different tones
    (empathetic, professional, apologetic, informative, clarifying).
    """
    job = await create_job(
        batch_id=request.request_id,
        job_type="engagement",
        total_items=1,
    )

    background_tasks.add_task(
        process_engagement,
        job_id=job["job_id"],
        theme_group_id=request.theme_group_id,
        keyword=request.keyword,
    )

    return EngagementJobResponse(
        request_id=request.request_id,
        job_id=job["job_id"],
        status="queued",
        submitted_at=job["submitted_at"],
    )