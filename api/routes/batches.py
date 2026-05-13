"""
api/routes/batches.py — Batch processing endpoints (Flow 1).

POST /batches/submit  — Submit raw data for processing (async, returns job_id)
GET  /batches/{job_id}/status — Poll job status
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import BatchSubmitRequest, BatchSubmitResponse, JobStatusResponse
from db.mongo_client import create_job, get_job, save_record, update_job
from pipeline.pipeline_runner import process_record

router = APIRouter()


# ── Background task: processes the entire batch ─────────────────────

async def process_batch(job_id: str, batch_id: str, keyword: str, items: list[dict]):
    """
    Runs in the background after the API returns the job_id to Team 1.

    For each item:
      1. Attach batch-level metadata (batch_id, keyword)
      2. Run through process_record() — the 5-phase pipeline
      3. Save result to analyzed_records
      4. Update job progress

    On Day 3, process_record() will be wired to Eng B + Eng C modules.
    For now it's a passthrough stub.
    """
    total = len(items)

    # Mark job as processing
    await update_job(job_id, {"status": "processing"})

    processed = 0
    failed = 0
    filtered = 0

    for item in items:
        try:
            # Attach batch-level info to each record
            record = item
            record["batch_id"] = batch_id
            record["keyword"] = keyword

            # ── Run through 5-phase pipeline ──
            record = await process_record(record)

            # Track filtered records
            if not record.get("is_relevant", True):
                filtered += 1

            # Save processed record to DB
            await save_record(record)
            processed += 1

        except Exception as e:
            # Never crash — log error, mark as failed, continue to next item
            failed += 1
            try:
                record["pipeline_error"] = str(e)
                record["pipeline_stage_stopped"] = "error"
                record["processed_at"] = datetime.now(timezone.utc).isoformat()
                await save_record(record)
            except Exception:
                pass  # If even error-saving fails, just move on

        # Update job progress after each record
        progress = int(((processed + failed) / total) * 100)
        await update_job(job_id, {
            "processed": processed,
            "failed": failed,
            "filtered": filtered,
            "progress": progress,
        })

    # Mark job as completed (or failed if everything failed)
    final_status = "failed" if failed == total else "completed"
    await update_job(job_id, {
        "status": final_status,
        "progress": 100,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


# ── Routes ──────────────────────────────────────────────────────────

@router.post("/batches/submit", response_model=BatchSubmitResponse)
async def submit_batch(request: BatchSubmitRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch of raw posts/comments for processing.

    Team 1 sends the batch → we create a job → kick off background processing
    → immediately return the job_id so they can poll for status.
    """
    # Create job entry in batch_jobs collection
    job = await create_job(
        batch_id=request.batch_id,
        job_type="batch",
        total_items=len(request.items),
    )

    # Convert Pydantic models to plain dicts for pipeline processing
    items_as_dicts = [item.model_dump() for item in request.items]

    # Kick off background processing — this returns immediately
    background_tasks.add_task(
        process_batch,
        job_id=job["job_id"],
        batch_id=request.batch_id,
        keyword=request.keyword,
        items=items_as_dicts,
    )

    return BatchSubmitResponse(
        batch_id=request.batch_id,
        job_id=job["job_id"],
        status="queued",
        submitted_at=request.submitted_at,
    )


@router.get("/batches/{job_id}/status", response_model=JobStatusResponse)
async def get_batch_status(job_id: str):
    """
    Poll the status of a batch processing job.

    Team 1 calls this repeatedly until status is "completed" or "failed".
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        error=job.get("error"),
        completed_at=job.get("completed_at"),
    )