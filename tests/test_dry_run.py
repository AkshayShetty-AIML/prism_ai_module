from __future__ import annotations

"""
tests/test_pipeline_dry_run.py — Day 3 dry run: mock records through full pipeline.

Processes all records ONCE, then runs 20 focused tests that validate
real constraints — field existence, value validity, score math, 
bot/promo detection accuracy, and DB persistence.

Run:
    pytest tests/test_pipeline_dry_run.py -v
    python tests/test_pipeline_dry_run.py        # standalone detailed output

Requires: GROQ_API_KEY set in environment or .env file.
"""

import asyncio
import json
import os
import sys
import traceback

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

os.environ["LLM_PROVIDER"] = "groq"

if not os.getenv("GROQ_API_KEY"):
    pytest.skip("GROQ_API_KEY not set — skipping dry run tests", allow_module_level=True)

from pipeline.pipeline_runner import process_record
from db.mongo_client import connect, save_record, get_record, close


# ── Expected fields per phase ───────────────────────────────────────

PHASE1_FIELDS = [
    "language_detected", "language_confidence", "has_code_mixing",
    "normalised_text",
    "is_relevant", "relevance_score", "filter_reason",
    "is_duplicate", "text_hash",
]

PHASE2_FIELDS = [
    "bot_flag", "bot_confidence", "bot_flags",
    "is_promotional", "content_type", "promo_signals",
    "credibility_tier",
]

PHASE3_FIELDS = [
    "sentiment", "positive_score", "neutral_score", "negative_score",
    "confidence_score", "dominant_emotion", "emotion_tags",
    "is_sarcastic", "sentiment_reasoning", "sentiment_failed",
    "low_confidence", "crisis_severity",
    "model_used", "tokens_used", "prompt_version",
]

PHASE4_FIELDS = [
    "impact_score", "impact_tier", "viral_flag",
]

VALID_SENTIMENTS = {"positive", "neutral", "negative", None}
VALID_BOT_FLAGS = {"bot", "human"}
VALID_CREDIBILITY_TIERS = {"high", "medium", "low", "bot"}
VALID_CRISIS = {"none", "low", "medium", "severe"}
VALID_CONTENT_TYPES = {"promotional", "organic"}
VALID_IMPACT_TIERS = {"Viral", "High Impact", "Notable", "Low Impact"}
VALID_EMOTIONS = {
    "excited", "happy", "disappointed", "angry", "neutral",
    "surprised", "sad", "sarcastic", "confused", "praise",
}
VALID_LANGUAGES = {"english", "tanglish", "hinglish", "hindi", "tamil", "mixed"}


# ── Helpers ─────────────────────────────────────────────────────────

def load_mock_data() -> list[dict]:
    mock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_data.json")
    with open(mock_path, "r") as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) else data


def prepare_record(raw: dict) -> dict:
    record = raw.copy()
    record["batch_id"] = "test-batch-001"
    record["keyword"] = "Leo movie"
    return record


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def mock_records():
    return load_mock_data()


@pytest.fixture(scope="module")
async def processed_records(mock_records):
    """Process all records once, reuse across every test."""
    await connect()
    results = []
    for raw in mock_records:
        record = prepare_record(raw)
        result = await process_record(record)
        results.append(result)
    yield results
    await close()


# Helper filters
def completed(records):
    return [r for r in records if r.get("pipeline_stage_stopped") == "complete"]

def relevant(records):
    return [r for r in records if r.get("is_relevant", True)]

def filtered_out(records):
    return [r for r in records if not r.get("is_relevant", True)]


# ═══════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_no_crashes(processed_records):
    """Every record must return a dict with pipeline_stage_stopped."""
    crashed = [
        r.get("item_id", "?") for r in processed_records
        if r is None or "pipeline_stage_stopped" not in r
    ]
    assert not crashed, f"Records crashed: {crashed}"


@pytest.mark.asyncio
async def test_no_pipeline_errors(processed_records):
    """No record should end with stage='error'."""
    errors = [
        f"{r.get('item_id')}: {r.get('pipeline_error')}"
        for r in processed_records
        if r.get("pipeline_stage_stopped") == "error"
    ]
    assert not errors, f"Pipeline errors:\n" + "\n".join(f"  {e}" for e in errors)


@pytest.mark.asyncio
async def test_phase1_fields(processed_records):
    """All records must have Phase 1 preprocessing fields."""
    bad = []
    for r in processed_records:
        missing = [f for f in PHASE1_FIELDS if f not in r]
        if missing:
            bad.append(f"{r.get('item_id')}: {missing}")
    assert not bad, f"Phase 1 missing:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_phase2_fields(processed_records):
    """Relevant records must have Phase 2 classifier fields."""
    bad = []
    for r in relevant(processed_records):
        missing = [f for f in PHASE2_FIELDS if f not in r]
        if missing:
            bad.append(f"{r.get('item_id')}: {missing}")
    assert not bad, f"Phase 2 missing:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_phase3_fields(processed_records):
    """Completed records must have Phase 3 sentiment fields."""
    bad = []
    for r in completed(processed_records):
        missing = [f for f in PHASE3_FIELDS if f not in r]
        if missing:
            bad.append(f"{r.get('item_id')}: {missing}")
    assert not bad, f"Phase 3 missing:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_phase4_fields(processed_records):
    """Completed records must have Phase 4 impact fields."""
    bad = []
    for r in completed(processed_records):
        missing = [f for f in PHASE4_FIELDS if f not in r]
        if missing:
            bad.append(f"{r.get('item_id')}: {missing}")
    assert not bad, f"Phase 4 missing:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_filtered_stop_at_noise_filter(processed_records):
    """Filtered records must have stage='noise_filter' and no sentiment."""
    bad = []
    for r in filtered_out(processed_records):
        if r.get("pipeline_stage_stopped") != "noise_filter":
            bad.append(f"{r.get('item_id')}: stage='{r.get('pipeline_stage_stopped')}'")
        if r.get("sentiment") is not None:
            bad.append(f"{r.get('item_id')}: has sentiment but was filtered")
    assert not bad, f"Filtered record issues:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_language_values(processed_records):
    """language_detected must be from the allowed set."""
    bad = [
        f"{r.get('item_id')}: '{r.get('language_detected')}'"
        for r in processed_records
        if r.get("language_detected") and r["language_detected"] not in VALID_LANGUAGES
    ]
    assert not bad, f"Invalid languages:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_sentiment_values(processed_records):
    """Sentiment must be positive/neutral/negative/null."""
    bad = [
        f"{r.get('item_id')}: '{r.get('sentiment')}'"
        for r in completed(processed_records)
        if r.get("sentiment") not in VALID_SENTIMENTS
    ]
    assert not bad, f"Invalid sentiments:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_scores_sum_to_one(processed_records):
    """Sentiment scores must sum to ~1.0 (tolerance 0.1)."""
    bad = []
    for r in completed(processed_records):
        if r.get("sentiment_failed", False):
            continue
        total = r.get("positive_score", 0) + r.get("neutral_score", 0) + r.get("negative_score", 0)
        if abs(total - 1.0) > 0.1:
            bad.append(f"{r.get('item_id')}: sum={total:.3f}")
    assert not bad, f"Score sums wrong:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_confidence_range(processed_records):
    """confidence_score must be 0.0–1.0."""
    bad = [
        f"{r.get('item_id')}: {r.get('confidence_score')}"
        for r in completed(processed_records)
        if r.get("confidence_score") is not None and not (0.0 <= r["confidence_score"] <= 1.0)
    ]
    assert not bad, f"Confidence out of range:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_emotion_values(processed_records):
    """dominant_emotion must be from the 10-emotion list."""
    bad = [
        f"{r.get('item_id')}: '{r.get('dominant_emotion')}'"
        for r in completed(processed_records)
        if r.get("dominant_emotion") and r["dominant_emotion"] not in VALID_EMOTIONS
    ]
    assert not bad, f"Invalid emotions:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_bot_flag_values(processed_records):
    """bot_flag must be 'bot' or 'human'."""
    bad = [
        f"{r.get('item_id')}: '{r.get('bot_flag')}'"
        for r in relevant(processed_records)
        if r.get("bot_flag") not in VALID_BOT_FLAGS
    ]
    assert not bad, f"Invalid bot_flag:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_credibility_tier_values(processed_records):
    """credibility_tier must be high/medium/low/bot."""
    bad = [
        f"{r.get('item_id')}: '{r.get('credibility_tier')}'"
        for r in relevant(processed_records)
        if r.get("credibility_tier") not in VALID_CREDIBILITY_TIERS
    ]
    assert not bad, f"Invalid credibility:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_content_type_values(processed_records):
    """content_type must be 'promotional' or 'organic'."""
    bad = [
        f"{r.get('item_id')}: '{r.get('content_type')}'"
        for r in relevant(processed_records)
        if r.get("content_type") not in VALID_CONTENT_TYPES
    ]
    assert not bad, f"Invalid content_type:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_crisis_severity_values(processed_records):
    """crisis_severity must be none/low/medium/severe."""
    bad = [
        f"{r.get('item_id')}: '{r.get('crisis_severity')}'"
        for r in completed(processed_records)
        if r.get("crisis_severity") not in VALID_CRISIS
    ]
    assert not bad, f"Invalid crisis_severity:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_impact_score_range(processed_records):
    """impact_score must be 0.0–100.0."""
    bad = [
        f"{r.get('item_id')}: {r.get('impact_score')}"
        for r in completed(processed_records)
        if r.get("impact_score") is not None and not (0.0 <= r["impact_score"] <= 100.0)
    ]
    assert not bad, f"Impact out of range:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_impact_tier_values(processed_records):
    """impact_tier must be Viral/High Impact/Notable/Low Impact."""
    bad = [
        f"{r.get('item_id')}: '{r.get('impact_tier')}'"
        for r in completed(processed_records)
        if r.get("impact_tier") not in VALID_IMPACT_TIERS
    ]
    assert not bad, f"Invalid impact_tier:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_viral_flag_matches_score(processed_records):
    """viral_flag=True iff impact_score > 80."""
    bad = []
    for r in completed(processed_records):
        score = r.get("impact_score")
        flag = r.get("viral_flag")
        if score is not None and flag is not None:
            expected = score > 80
            if flag != expected:
                bad.append(f"{r.get('item_id')}: flag={flag} but score={score}")
    assert not bad, f"viral_flag mismatch:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_obvious_bots_detected(processed_records, mock_records):
    """New accounts with extreme posting frequency must be flagged as bots."""
    from datetime import datetime, timezone
    bad = []
    for raw, proc in zip(mock_records, processed_records):
        if not proc.get("is_relevant", True):
            continue
        author = raw.get("author")
        if not author or not author.get("account_created_at"):
            continue
        try:
            created = datetime.fromisoformat(author["account_created_at"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = max((now - created).days, 1)
            posts_per_day = author.get("post_count", 0) / age_days
            if age_days < 30 and posts_per_day > 100 and proc.get("bot_flag") != "bot":
                bad.append(
                    f"{raw['item_id']}: {age_days}d old, {posts_per_day:.0f} posts/day → '{proc.get('bot_flag')}'"
                )
        except (ValueError, KeyError):
            pass
    assert not bad, f"Obvious bots missed:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_obvious_promos_detected(processed_records, mock_records):
    """Posts with #Ad, #Sponsored, etc. must be flagged as promotional."""
    promo_tags = {"#ad", "#sponsored", "#collab", "#paidpromotion", "#partnership"}
    bad = []
    for raw, proc in zip(mock_records, processed_records):
        if not proc.get("is_relevant", True):
            continue
        content_lower = raw.get("content", "").lower()
        if any(tag in content_lower for tag in promo_tags):
            if not proc.get("is_promotional"):
                bad.append(f"{raw['item_id']}: has promo hashtags but is_promotional=False")
    assert not bad, f"Promos missed:\n" + "\n".join(f"  {b}" for b in bad)


@pytest.mark.asyncio
async def test_db_save_retrieve(processed_records):
    """Every record must save to and retrieve from MongoDB correctly."""
    bad = []
    for r in processed_records:
        try:
            await save_record(r)
            fetched = await get_record(r["item_id"])
            if not fetched:
                bad.append(f"{r['item_id']}: not found after save")
            elif fetched.get("pipeline_stage_stopped") != r.get("pipeline_stage_stopped"):
                bad.append(f"{r['item_id']}: stage mismatch after retrieve")
        except Exception as e:
            bad.append(f"{r['item_id']}: {e}")
    assert not bad, f"DB failures:\n" + "\n".join(f"  {b}" for b in bad)


# ═══════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════

async def run_standalone():
    await connect()
    records = load_mock_data()

    print(f"\n{'='*70}")
    print(f"  DRY RUN — {len(records)} records (LLM: groq)")
    print(f"{'='*70}\n")

    stats = {"complete": 0, "noise_filter": 0, "error": 0}
    issues_total = 0

    for i, raw in enumerate(records):
        record = prepare_record(raw)
        item_id = record.get("item_id", f"rec-{i}")

        try:
            result = await process_record(record)
        except Exception:
            result = record
            result["pipeline_stage_stopped"] = "error"
            result["pipeline_error"] = traceback.format_exc().split("\n")[-2]

        stage = result.get("pipeline_stage_stopped", "???")
        stats[stage] = stats.get(stage, 0) + 1

        # Check fields
        if stage == "complete":
            expected = PHASE1_FIELDS + PHASE2_FIELDS + PHASE3_FIELDS + PHASE4_FIELDS
        elif stage == "noise_filter":
            expected = PHASE1_FIELDS
        else:
            expected = []

        missing = [f for f in expected if f not in result]
        icon = "✅" if stage == "complete" and not missing else \
               "⏭️" if stage == "noise_filter" and not missing else \
               "💥" if stage == "error" else "⚠️"

        print(f"  {icon} {item_id:15s} stage={stage:14s} "
              f"sent={str(result.get('sentiment', '—')):10s} "
              f"bot={str(result.get('bot_flag', '—')):6s} "
              f"impact={result.get('impact_score', '—')}")

        if missing:
            print(f"     Missing: {missing}")
            issues_total += 1
        if stage == "error":
            print(f"     Error: {result.get('pipeline_error')}")
            issues_total += 1

        await save_record(result)

    print(f"\n{'='*70}")
    print(f"  {stats} | Issues: {issues_total}")
    verdict = "🟢 ALL CLEAR" if issues_total == 0 and stats.get("error", 0) == 0 else \
              "🟡 ISSUES" if stats.get("error", 0) == 0 else "🔴 ERRORS"
    print(f"  {verdict}")
    print(f"{'='*70}\n")

    await close()


if __name__ == "__main__":
    asyncio.run(run_standalone())