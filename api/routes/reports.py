"""
api/routes/reports.py — Report generation endpoint (Flow 4).

POST /reports/generate — Generate AI insight report for a keyword + date range.
Async polling pattern: returns job_id immediately, poll GET /jobs/{job_id} for result.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks

from api.schemas import ReportRequest, ReportJobResponse
from db.mongo_client import create_job, update_job
from pipeline.report_generator import generate_report

router = APIRouter()


async def process_report(
    job_id: str,
    keyword: str,
    date_from: str,
    date_to: str,
    segments: list[str],
    include_summary: bool,
):
    """
    Background task: aggregates data from DB2, calls Eng C's report generator
    for the narrative, and stores the full report in the job document.
    """
    await update_job(job_id, {"status": "processing"})

    try:
        result = await generate_report(
            keyword=keyword,
            date_from=date_from,
            date_to=date_to,
            segments=segments,
        )

        # Add metadata
        result["request_id"] = job_id
        result["keyword"] = keyword
        result["generated_at"] = datetime.now(timezone.utc).isoformat()

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


@router.post("/reports/generate", response_model=ReportJobResponse)
async def reports_generate(request: ReportRequest, background_tasks: BackgroundTasks):
    """
    Generate an AI insight report for a keyword within a date range.

    The AI aggregates sentiment, platform, bot/human, promo/organic data from DB2,
    identifies high-impact posts, and generates a narrative summary with key insights.
    Frontend handles PDF export from the returned JSON.
    """
    job = await create_job(
        batch_id=request.request_id,
        job_type="report",
        total_items=1,
    )

    background_tasks.add_task(
        process_report,
        job_id=job["job_id"],
        keyword=request.keyword,
        date_from=request.date_range.from_date,
        date_to=request.date_range.to_date,
        segments=request.segments,
        include_summary=request.include_summary,
    )

    return ReportJobResponse(
        request_id=request.request_id,
        job_id=job["job_id"],
        status="queued",
        submitted_at=job["submitted_at"],
    )