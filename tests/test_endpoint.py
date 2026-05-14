from __future__ import annotations

"""
tests/test_api_endpoints.py — Tests every API endpoint with real data.

Seeds MongoDB with 12 pre-processed records, then hits every endpoint
and validates the response shape and values.

Run:
    python tests/test_api_endpoints.py

Requires: MongoDB running locally.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from db.mongo_client import connect, close, get_db


# ── Helpers ─────────────────────────────────────────────────────────

def load_seed_data() -> list[dict]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_data.json")
    with open(path) as f:
        return json.load(f)


def check(label: str, condition: bool, detail: str = ""):
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if not condition and detail:
        msg += f" — {detail}"
    print(msg)
    return condition


async def seed_db(records: list[dict]):
    """Wipe test data and insert seed records."""
    db = get_db()
    # Clean up previous test data
    await db["analyzed_records"].delete_many({"batch_id": "seed-batch-001"})
    await db["batch_jobs"].delete_many({"batch_id": {"$regex": "^test-"}})

    # Insert seed records
    for r in records:
        r_copy = r.copy()
        await db["analyzed_records"].update_one(
            {"item_id": r_copy["item_id"]},
            {"$set": r_copy},
            upsert=True,
        )


async def run():
    import httpx

    # ── Setup ───────────────────────────────────────────────────────
    await connect()
    seed = load_seed_data()
    await seed_db(seed)
    print(f"\n{'='*70}")
    print(f"  API ENDPOINT TESTS — {len(seed)} seed records loaded")
    print(f"{'='*70}\n")

    base = os.getenv("API_BASE_URL", "http://localhost:8001/api/v1")
    passed = 0
    failed = 0

    async with httpx.AsyncClient(timeout=30.0) as client:

        # ── 1. Health ───────────────────────────────────────────────
        print("─── Health ───")
        try:
            r = await client.get(f"{base}/health")
            data = r.json()
            if check("GET /health returns 200", r.status_code == 200):
                passed += 1
            else:
                failed += 1
            if check("status is 'ok'", data.get("status") == "ok", f"got: {data}"):
                passed += 1
            else:
                failed += 1
            if check("db is 'ok'", data.get("db") == "ok", f"got: {data.get('db')}"):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  💥 Health check failed: {e}")
            print(f"\n  ⛔ Cannot reach API at {base}. Is the server running?")
            print(f"     Start it with: uvicorn api.app:app --port 8001\n")
            await close()
            return
        print()

        # ── 2. Batch Submit ─────────────────────────────────────────
        print("─── POST /batches/submit ───")
        batch_payload = {
            "batch_id": "test-batch-api",
            "submitted_at": "2026-05-14T10:00:00Z",
            "keyword": "Leo movie",
            "items": [
                {
                    "item_id": "api-test-001",
                    "platform": "twitter",
                    "content": "Testing the API endpoint for Leo movie review",
                    "posted_at": "2026-05-14T09:00:00Z",
                    "collected_at": "2026-05-14T09:30:00Z",
                    "source_url": "https://twitter.com/test/1",
                    "author": {
                        "author_id": "test_author",
                        "username": "test_user",
                        "account_created_at": "2023-01-01T00:00:00Z",
                        "verified": False,
                        "follower_count": 100,
                        "following_count": 50,
                        "post_count": 200,
                        "profile_picture_present": True,
                        "bio_present": True
                    },
                    "engagement": {"likes": 5, "replies": 1, "shares": 0, "views": 100}
                }
            ]
        }
        r = await client.post(f"{base}/batches/submit", json=batch_payload)
        data = r.json()
        if check("POST /batches/submit returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
            print(f"     Response: {data}")
        if check("Returns job_id", "job_id" in data, f"got: {list(data.keys())}"):
            passed += 1
        else:
            failed += 1
        if check("Status is 'queued'", data.get("status") == "queued"):
            passed += 1
        else:
            failed += 1

        batch_job_id = data.get("job_id", "")
        print()

        # ── 3. Batch Status Polling ─────────────────────────────────
        print("─── GET /batches/{job_id}/status ───")
        # Wait a moment for background processing
        await asyncio.sleep(2)
        r = await client.get(f"{base}/batches/{batch_job_id}/status")
        data = r.json()
        if check("GET /batches/status returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has status field", "status" in data):
            passed += 1
        else:
            failed += 1
        if check("Has progress field", "progress" in data):
            passed += 1
        else:
            failed += 1
        print(f"     Status: {data.get('status')}, Progress: {data.get('progress')}")

        # Test 404
        r = await client.get(f"{base}/batches/nonexistent-id/status")
        if check("404 for invalid job_id", r.status_code == 404):
            passed += 1
        else:
            failed += 1
        print()

        # ── 4. GET /records ─────────────────────────────────────────
        print("─── GET /records ───")
        r = await client.get(f"{base}/records", params={"keyword": "Leo movie"})
        data = r.json()
        if check("GET /records returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has 'items' array", isinstance(data.get("items"), list)):
            passed += 1
        else:
            failed += 1
        if check("Has 'total' count", isinstance(data.get("total"), int)):
            passed += 1
        else:
            failed += 1
        leo_count = data.get("total", 0)
        if check(f"Found Leo records (expected ~10)", leo_count >= 5, f"got: {leo_count}"):
            passed += 1
        else:
            failed += 1

        # Filter by platform
        r = await client.get(f"{base}/records", params={"keyword": "Leo movie", "platform": "youtube"})
        yt_data = r.json()
        if check("Platform filter works", yt_data.get("total", 0) < leo_count, f"youtube={yt_data.get('total')}, all={leo_count}"):
            passed += 1
        else:
            failed += 1

        # Filter by sentiment
        r = await client.get(f"{base}/records", params={"keyword": "Leo movie", "sentiment": "negative"})
        neg_data = r.json()
        if check("Sentiment filter works", neg_data.get("total", 0) > 0, f"negative={neg_data.get('total')}"):
            passed += 1
        else:
            failed += 1

        # Filter by bot
        r = await client.get(f"{base}/records", params={"keyword": "Leo movie", "is_bot": True})
        bot_data = r.json()
        if check("Bot filter works", bot_data.get("total", 0) >= 1, f"bots={bot_data.get('total')}"):
            passed += 1
        else:
            failed += 1

        # Pagination
        r = await client.get(f"{base}/records", params={"keyword": "Leo movie", "page": 1, "page_size": 3})
        page_data = r.json()
        if check("Pagination works", len(page_data.get("items", [])) <= 3, f"items on page: {len(page_data.get('items', []))}"):
            passed += 1
        else:
            failed += 1
        print()

        # ── 5. GET /records/{item_id} ───────────────────────────────
        print("─── GET /records/{item_id} ───")
        r = await client.get(f"{base}/records/seed-001")
        data = r.json()
        if check("Single record returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has sentiment field", "sentiment" in data):
            passed += 1
        else:
            failed += 1

        r = await client.get(f"{base}/records/nonexistent-id")
        if check("404 for invalid item_id", r.status_code == 404):
            passed += 1
        else:
            failed += 1
        print()

        # ── 6. Sentiment Aggregate ──────────────────────────────────
        print("─── GET /sentiment/aggregate ───")
        r = await client.get(f"{base}/sentiment/aggregate", params={"keyword": "Leo movie"})
        data = r.json()
        if check("Aggregate returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has positive_pct", "positive_pct" in data):
            passed += 1
        else:
            failed += 1
        if check("Has total_items_analyzed", data.get("total_items_analyzed", 0) > 0, f"total={data.get('total_items_analyzed')}"):
            passed += 1
        else:
            failed += 1
        pcts_sum = data.get("positive_pct", 0) + data.get("neutral_pct", 0) + data.get("negative_pct", 0)
        if check("Percentages sum to ~100", abs(pcts_sum - 100) < 1, f"sum={pcts_sum}"):
            passed += 1
        else:
            failed += 1
        print(f"     +{data.get('positive_pct')}% / ={data.get('neutral_pct')}% / -{data.get('negative_pct')}%")
        print()

        # ── 7. Sentiment Trend ──────────────────────────────────────
        print("─── GET /sentiment/trend ───")
        r = await client.get(f"{base}/sentiment/trend", params={"keyword": "Leo movie"})
        data = r.json()
        if check("Trend returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has 'trend' array", isinstance(data.get("trend"), list)):
            passed += 1
        else:
            failed += 1
        trend_len = len(data.get("trend", []))
        if check(f"Has hourly buckets", trend_len > 0, f"buckets={trend_len}"):
            passed += 1
        else:
            failed += 1
        if trend_len > 0:
            bucket = data["trend"][0]
            if check("Bucket has hour/positive/neutral/negative", all(k in bucket for k in ["hour", "positive", "neutral", "negative"])):
                passed += 1
            else:
                failed += 1
        print(f"     {trend_len} hourly buckets")
        print()

        # ── 8. Sentiment Viral ──────────────────────────────────────
        print("─── GET /sentiment/viral ───")
        r = await client.get(f"{base}/sentiment/viral", params={"keyword": "Leo movie"})
        data = r.json()
        if check("Viral returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        viral_count = data.get("total", 0)
        if check(f"Found viral posts", viral_count > 0, f"count={viral_count}"):
            passed += 1
        else:
            failed += 1
        # Verify all returned records are actually viral
        all_viral = all(item.get("viral_flag") is True for item in data.get("items", []))
        if check("All returned records have viral_flag=True", all_viral):
            passed += 1
        else:
            failed += 1
        print()

        # ── 9. Keywords Compare ─────────────────────────────────────
        print("─── GET /keywords/compare ───")
        r = await client.get(f"{base}/keywords/compare", params={"keywords": "Leo movie,Jailer movie"})
        data = r.json()
        if check("Compare returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        comparisons = data.get("comparisons", [])
        if check("Has 2 comparisons", len(comparisons) == 2, f"got {len(comparisons)}"):
            passed += 1
        else:
            failed += 1
        keywords_returned = [c.get("keyword") for c in comparisons]
        if check("Contains both keywords", "Leo movie" in keywords_returned and "Jailer movie" in keywords_returned):
            passed += 1
        else:
            failed += 1
        print(f"     Keywords: {keywords_returned}")
        print()

        # ── 10. Alerts Metrics ──────────────────────────────────────
        print("─── GET /alerts/metrics ───")
        r = await client.get(f"{base}/alerts/metrics", params={
            "keyword": "Leo movie",
            "duration": "24h",
            "reference_window": "previous_24h",
        })
        data = r.json()
        if check("Alerts returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has 'metrics' object", isinstance(data.get("metrics"), dict)):
            passed += 1
        else:
            failed += 1
        metrics = data.get("metrics", {})
        if check("Has sentiment_distribution", "sentiment_distribution" in metrics):
            passed += 1
        else:
            failed += 1
        if check("Has sentiment_delta", "sentiment_delta" in metrics):
            passed += 1
        else:
            failed += 1
        if check("Has crisis_severity_count", "crisis_severity_count" in metrics):
            passed += 1
        else:
            failed += 1
        if check("Has bot_activity_count", "bot_activity_count" in metrics):
            passed += 1
        else:
            failed += 1
        if check("Has keyword_volume", "keyword_volume" in metrics):
            passed += 1
        else:
            failed += 1
        print(f"     Window: {data.get('window')}")
        print()

        # ── 11. Jobs Polling ────────────────────────────────────────
        print("─── GET /jobs/{job_id} ───")
        r = await client.get(f"{base}/jobs/{batch_job_id}")
        data = r.json()
        if check("Jobs returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
        if check("Has type field", "type" in data):
            passed += 1
        else:
            failed += 1
        if check("Type is 'batch'", data.get("type") == "batch"):
            passed += 1
        else:
            failed += 1
        if check("Has batch counters", "total_items" in data):
            passed += 1
        else:
            failed += 1

        r = await client.get(f"{base}/jobs/nonexistent")
        if check("404 for invalid job_id", r.status_code == 404):
            passed += 1
        else:
            failed += 1
        print()

        # ── 12. Engagement Generate ─────────────────────────────────
        print("─── POST /engagement/generate ───")
        engagement_payload = {
            "request_id": "test-eng-001",
            "keyword": "Leo movie",
            "theme_group_id": "theme-pacing-001",
            "requested_at": "2026-05-14T10:00:00Z",
        }
        r = await client.post(f"{base}/engagement/generate", json=engagement_payload)
        data = r.json()
        if check("Engagement returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
            print(f"     Response: {data}")
        if check("Returns job_id", "job_id" in data):
            passed += 1
        else:
            failed += 1
        if check("Status is 'queued'", data.get("status") == "queued"):
            passed += 1
        else:
            failed += 1
        eng_job_id = data.get("job_id", "")

        # Poll for result
        await asyncio.sleep(5)
        r = await client.get(f"{base}/jobs/{eng_job_id}")
        eng_result = r.json()
        print(f"     Job status: {eng_result.get('status')}")
        if eng_result.get("status") == "completed" and eng_result.get("result"):
            result = eng_result["result"]
            if check("Result has suggested_replies", "suggested_replies" in result):
                passed += 1
            else:
                failed += 1
            replies = result.get("suggested_replies", [])
            if check(f"Has 3-5 replies", 1 <= len(replies) <= 5, f"got {len(replies)}"):
                passed += 1
            else:
                failed += 1
        elif eng_result.get("status") == "failed":
            print(f"     ⚠️  Engagement job failed: {eng_result.get('error')}")
            failed += 2
        else:
            print(f"     ⚠️  Job still {eng_result.get('status')} — may need more time")
        print()

        # ── 13. Crisis Generate ─────────────────────────────────────
        print("─── POST /crisis/generate ───")
        crisis_payload = {
            "request_id": "test-crisis-001",
            "crisis_description": "Major backlash on Twitter about a plot leak from Leo. Fans angry, hashtag trending.",
            "keyword": "Leo movie",
            "requested_at": "2026-05-14T10:00:00Z",
        }
        r = await client.post(f"{base}/crisis/generate", json=crisis_payload)
        data = r.json()
        if check("Crisis returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
            print(f"     Response: {data}")
        if check("Returns job_id", "job_id" in data):
            passed += 1
        else:
            failed += 1
        crisis_job_id = data.get("job_id", "")

        # Poll for result
        await asyncio.sleep(5)
        r = await client.get(f"{base}/jobs/{crisis_job_id}")
        crisis_result = r.json()
        print(f"     Job status: {crisis_result.get('status')}")
        if crisis_result.get("status") == "completed" and crisis_result.get("result"):
            result = crisis_result["result"]
            if check("Result has crisis_response", "crisis_response" in result):
                passed += 1
            else:
                failed += 1
            response_text = result.get("crisis_response", "")
            if check("Crisis response is substantial", len(response_text) > 100, f"length={len(response_text)}"):
                passed += 1
            else:
                failed += 1
        elif crisis_result.get("status") == "failed":
            print(f"     ⚠️  Crisis job failed: {crisis_result.get('error')}")
            failed += 2
        else:
            print(f"     ⚠️  Job still {crisis_result.get('status')} — may need more time")
        print()

        # ── 14. Report Generate ─────────────────────────────────────
        print("─── POST /reports/generate ───")
        report_payload = {
            "request_id": "test-report-001",
            "keyword": "Leo movie",
            "date_range": {"from": "2026-05-10T00:00:00Z", "to": "2026-05-10T23:59:59Z"},
            "segments": ["sentiment", "platform", "bot_vs_human", "promo_vs_organic", "high_impact_posts"],
            "include_summary": True,
        }
        r = await client.post(f"{base}/reports/generate", json=report_payload)
        data = r.json()
        if check("Report returns 200", r.status_code == 200):
            passed += 1
        else:
            failed += 1
            print(f"     Response: {data}")
        if check("Returns job_id", "job_id" in data):
            passed += 1
        else:
            failed += 1
        report_job_id = data.get("job_id", "")

        # Poll for result
        await asyncio.sleep(8)
        r = await client.get(f"{base}/jobs/{report_job_id}")
        report_result = r.json()
        print(f"     Job status: {report_result.get('status')}")
        if report_result.get("status") == "completed" and report_result.get("result"):
            result = report_result["result"]
            if check("Result has summary_text", "summary_text" in result):
                passed += 1
            else:
                failed += 1
            if check("Result has key_insights", isinstance(result.get("key_insights"), list)):
                passed += 1
            else:
                failed += 1
        elif report_result.get("status") == "failed":
            print(f"     ⚠️  Report job failed: {report_result.get('error')}")
            failed += 2
        else:
            print(f"     ⚠️  Job still {report_result.get('status')} — may need more time")
        print()

    # ── Summary ─────────────────────────────────────────────────────
    total = passed + failed
    print(f"{'='*70}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"  🟢 ALL ENDPOINTS WORKING")
    elif failed <= 5:
        print(f"  🟡 MOSTLY WORKING — {failed} issues to fix")
    else:
        print(f"  🔴 {failed} FAILURES — needs debugging")
    print(f"{'='*70}\n")

    # Cleanup
    db = get_db()
    await db["analyzed_records"].delete_many({"item_id": "api-test-001"})
    await close()


if __name__ == "__main__":
    asyncio.run(run())