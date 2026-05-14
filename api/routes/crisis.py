"""
api/routes/crisis.py — Crisis management endpoint (Flow 3).

POST /crisis/generate — Generate AI crisis advisory from free-form description.
Async polling pattern: returns job_id immediately, poll GET /jobs/{job_id} for result.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks

from api.schemas import CrisisRequest, CrisisJobResponse
from db.mongo_client import create_job, update_job
from pipeline.crisis_generator import generate_crisis_advisory

router = APIRouter()


async def process_crisis(job_id: str, crisis_description: str, keyword: str | None):
    """
    Background task: calls Eng C's crisis generator,
    stores the advisory text in the job document.
    """
    await update_job(job_id, {"status": "processing"})

    try:
        # Returns plain text string (not JSON) — POC scope
        advisory = await generate_crisis_advisory(
            crisis_description=crisis_description,
            keyword=keyword,
        )

        await update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": {
                "crisis_response": advisory,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    except Exception as e:
        await update_job(job_id, {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


@router.post("/crisis/generate", response_model=CrisisJobResponse)
async def crisis_generate(request: CrisisRequest, background_tasks: BackgroundTasks):
    """
    Generate a crisis management advisory from a free-form description.

    The user types a crisis situation in the dashboard, and the AI returns
    a 3-5 paragraph advisory covering immediate actions, 24-hour plan,
    common mistakes to avoid, and suggested statement tone.
    """
    job = await create_job(
        batch_id=request.request_id,
        job_type="crisis",
        total_items=1,
    )

    background_tasks.add_task(
        process_crisis,
        job_id=job["job_id"],
        crisis_description=request.crisis_description,
        keyword=request.keyword,
    )

    return CrisisJobResponse(
        request_id=request.request_id,
        job_id=job["job_id"],
        status="queued",
        submitted_at=job["submitted_at"],
    )