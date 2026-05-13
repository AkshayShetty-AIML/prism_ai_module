"""
tests/test_pipeline_dry_run.py — Day 3 dry run: 20 mock records through full pipeline.

Loads mock_data.json, runs each record through process_record(),
saves to MongoDB, and verifies every expected field is present.

Run:
    pytest tests/test_pipeline_dry_run.py -v

Requires: GROQ_API_KEY set in environment or .env file.
"""

import asyncio
import json
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# Force Groq as the LLM provider for all tests in this file
os.environ["LLM_PROVIDER"] = "groq"

if not os.getenv("GROQ_API_KEY"):
    pytest.skip("GROQ_API_KEY not set — skipping dry run tests", allow_module_level=True)

from pipeline.pipeline_runner import process_record
from db.mongo_client import connect, save_record, get_record, close


# ── Expected fields after each pipeline phase ───────────────────────

# Fields Eng B adds (Phase 1 + 2)
PHASE1_FIELDS = [
    "language_detected",
    "language_confidence",
    "has_code_mixing",
    "normalised_text",
    "is_relevant",
    "relevance_score",
    "filter_reason",
    "is_duplicate",
    "text_hash",
]

PHASE2_FIELDS = [
    "bot_flag",
    "bot_confidence",
    "bot_flags",
    "is_promotional",
    "content_type",
    "promo_signals",
    "credibility_tier",
]

# Fields Eng C adds (Phase 3 + 4)
PHASE3_FIELDS = [
    "sentiment",
    "positive_score",
    "neutral_score",
    "negative_score",
    "confidence_score",
    "dominant_emotion",
    "emotion_tags",
    "is_sarcastic",
    "sentiment_reasoning",
    "sentiment_failed",
    "low_confidence",
    "crisis_severity",
    "model_used",
    "tokens_used",
    "prompt_version",
]

PHASE4_FIELDS = [
    "impact_score",
    "impact_tier",
    "viral_flag",
]

# Pipeline metadata (Eng A)
METADATA_FIELDS = [
    "pipeline_stage_stopped",
    "processed_at",
]


# ── Load mock data ──────────────────────────────────────────────────

def load_mock_data() -> list[dict]:
    mock_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mock_data.json"
    )
    with open(mock_path, "r") as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) else data


# ── Tests ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def mock_records():
    return load_mock_data()


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    """Connect to MongoDB before tests, close after."""
    await connect()
    yield
    await close()


@pytest.mark.asyncio
async def test_all_records_process_without_crashing(mock_records):
    """Every record should complete without raising an exception."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)
        assert result is not None, f"Record {record['item_id']} returned None"
        assert "pipeline_stage_stopped" in result, (
            f"Record {record['item_id']} missing pipeline_stage_stopped"
        )


@pytest.mark.asyncio
async def test_phase1_fields_present(mock_records):
    """Every record should have Phase 1 (preprocessing) fields."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        for field in PHASE1_FIELDS:
            assert field in result, (
                f"Record {record['item_id']} missing Phase 1 field: {field}"
            )


@pytest.mark.asyncio
async def test_filtered_records_skip_later_phases(mock_records):
    """Records marked is_relevant=False should stop at noise_filter."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if not result.get("is_relevant", True):
            assert result["pipeline_stage_stopped"] == "noise_filter", (
                f"Filtered record {record['item_id']} didn't stop at noise_filter"
            )
            # Should NOT have Phase 3/4 fields
            assert "sentiment" not in result or result.get("sentiment") is None, (
                f"Filtered record {record['item_id']} should not have sentiment"
            )


@pytest.mark.asyncio
async def test_relevant_records_have_all_fields(mock_records):
    """Records that pass noise filter should have fields from all phases."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("is_relevant", True) and result.get("pipeline_stage_stopped") == "complete":
            all_fields = PHASE1_FIELDS + PHASE2_FIELDS + PHASE3_FIELDS + PHASE4_FIELDS + METADATA_FIELDS

            missing = [f for f in all_fields if f not in result]
            assert not missing, (
                f"Record {record['item_id']} missing fields: {missing}"
            )


@pytest.mark.asyncio
async def test_sentiment_values_valid(mock_records):
    """Sentiment should be one of the allowed values."""
    valid_sentiments = {"positive", "neutral", "negative", None}

    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("pipeline_stage_stopped") == "complete":
            assert result.get("sentiment") in valid_sentiments, (
                f"Record {record['item_id']} has invalid sentiment: {result.get('sentiment')}"
            )


@pytest.mark.asyncio
async def test_scores_sum_to_one(mock_records):
    """Positive + neutral + negative scores should sum to ~1.0."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("pipeline_stage_stopped") == "complete" and not result.get("sentiment_failed"):
            total = (
                result.get("positive_score", 0)
                + result.get("neutral_score", 0)
                + result.get("negative_score", 0)
            )
            assert abs(total - 1.0) < 0.05, (
                f"Record {record['item_id']} scores sum to {total}, expected ~1.0"
            )


@pytest.mark.asyncio
async def test_bot_flag_valid(mock_records):
    """Bot flag should be 'bot' or 'human'."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("is_relevant", True):
            assert result.get("bot_flag") in {"bot", "human"}, (
                f"Record {record['item_id']} has invalid bot_flag: {result.get('bot_flag')}"
            )


@pytest.mark.asyncio
async def test_impact_score_range(mock_records):
    """Impact score should be 0.0–100.0."""
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("pipeline_stage_stopped") == "complete":
            score = result.get("impact_score", 0)
            assert 0.0 <= score <= 100.0, (
                f"Record {record['item_id']} impact_score {score} out of range"
            )


@pytest.mark.asyncio
async def test_save_and_retrieve_from_db(mock_records):
    """Records should save to MongoDB and be retrievable."""
    # Process and save first record
    record = mock_records[0].copy()
    record["batch_id"] = "test-batch-001"
    record["keyword"] = "Leo movie"
    result = await process_record(record)
    await save_record(result)

    # Retrieve and verify
    fetched = await get_record(record["item_id"])
    assert fetched is not None, "Saved record not found in MongoDB"
    assert fetched["item_id"] == record["item_id"]
    assert "pipeline_stage_stopped" in fetched


@pytest.mark.asyncio
async def test_no_pipeline_errors(mock_records):
    """No record should have pipeline_stage_stopped = 'error'."""
    errors = []
    for record in mock_records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)

        if result.get("pipeline_stage_stopped") == "error":
            errors.append({
                "item_id": record["item_id"],
                "error": result.get("pipeline_error"),
            })

    assert not errors, f"Pipeline errors found: {errors}"


# ── Summary runner (optional: run standalone) ───────────────────────

async def run_dry_run():
    """Run outside pytest for quick manual verification."""
    await connect()
    records = load_mock_data()

    print(f"\n{'='*60}")
    print(f"PIPELINE DRY RUN — {len(records)} records")
    print(f"{'='*60}\n")

    stats = {"complete": 0, "filtered": 0, "error": 0}

    for record in records:
        record["batch_id"] = "test-batch-001"
        record["keyword"] = "Leo movie"
        result = await process_record(record)
        await save_record(result)

        stage = result.get("pipeline_stage_stopped", "unknown")
        stats[stage] = stats.get(stage, 0) + 1

        # Print summary per record
        status_icon = "✅" if stage == "complete" else "⏭️" if stage == "noise_filter" else "❌"
        sentiment = result.get("sentiment", "—")
        bot = result.get("bot_flag", "—")
        impact = result.get("impact_score", "—")

        print(f"{status_icon} {record['item_id']:12s} | stage={stage:14s} | "
              f"sentiment={sentiment:10s} | bot={bot:6s} | impact={impact}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {stats}")
    print(f"{'='*60}\n")

    await close()


if __name__ == "__main__":
    asyncio.run(run_dry_run())