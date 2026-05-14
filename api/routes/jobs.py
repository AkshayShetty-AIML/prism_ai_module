"""
api/routes/jobs.py — Universal job polling endpoint.

GET /jobs/{job_id} — Poll status + result for any async job
                     (batch, engagement, crisis, report).
"""

from fastapi import APIRouter, HTTPException

from db.mongo_client import get_job

router = APIRouter()


@router.get("/jobs/{job_id}")
async def poll_job(job_id: str):
    """
    Poll the status and result of any async job.

    Works for all job types: batch, engagement, crisis, report.

    When status is "completed":
    - Batch jobs: result is null (records are in analyzed_records, fetch via GET /records)
    - Engagement jobs: result contains theme_summary + suggested_replies
    - Crisis jobs: result contains crisis_response text
    - Report jobs: result contains summary_text + segments + key_insights

    When status is "failed": error field contains the reason.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response = {
        "job_id": job["job_id"],
        "type": job.get("type"),
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "completed_at": job.get("completed_at"),
    }

    # Include result only when completed (for on-demand jobs)
    if job["status"] == "completed" and job.get("result"):
        response["result"] = job["result"]

    # Include batch counters for batch jobs
    if job.get("type") == "batch":
        response["total_items"] = job.get("total_items", 0)
        response["processed"] = job.get("processed", 0)
        response["failed"] = job.get("failed", 0)
        response["filtered"] = job.get("filtered", 0)

    return response