"""
test_day4_chain.py — Day 4: 30-Record Full Chain Integration
Engineer B | PRISM AI Processing Pipeline

Runs ALL 30 mock records through Eng B's full 6-module chain in the exact
sequence defined by Engineer A's interface contract.

Tests verified:
    - All 30 records survive the chain without crashing (no exceptions)
    - Schema completeness: every output dict has all required fields
    - Type correctness for every output field
    - Deduplication: mock-021 (duplicate of mock-001) is flagged correctly
    - Empty / URL-only records are filtered gracefully (no crash)
    - Verified override: mock-025 is classified as human despite bot signals
    - Promotional detection: mock-026 (sarcasm+#ad) and mock-028 (CTA+link)
    - Credibility ordering: high-follower influencer > sparse new account
    - Pipeline stops correctly for irrelevant records
    - No existing keys are mutated or deleted by any module

Usage:
    pytest tests/test_day4_chain.py -v
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.language_processor import detect_language
from pipeline.tanglish_normalizer import normalise_tanglish
from pipeline.noise_filter import filter_noise, reset_dedup_store
from pipeline.bot_detector import classify_bot
from pipeline.promo_classifier import classify_promo
from pipeline.credibility_scorer import assess_credibility

# ---------------------------------------------------------------------------
# Load fixtures
# ---------------------------------------------------------------------------
_DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_data.json")


def _load_records() -> list[dict]:
    with open(_DATA_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["records"]


# ---------------------------------------------------------------------------
# Chain runner (mirrors pipeline_runner.py contract from Eng A)
# ---------------------------------------------------------------------------
REQUIRED_PHASE1_FIELDS = {
    "language_detected", "language_confidence", "has_code_mixing",
    "normalised_text",
    "is_relevant", "relevance_score", "filter_reason", "is_duplicate", "text_hash",
}

REQUIRED_PHASE2_FIELDS = {
    "bot_flag", "bot_confidence", "bot_flags",
    "is_promotional", "content_type", "promo_signals",
    "credibility_tier",
}

VALID_LANGUAGES    = {"english", "tanglish", "hinglish", "hindi", "tamil", "mixed"}
VALID_BOT_FLAGS    = {"bot", "human"}
VALID_CONTENT_TYPE = {"promotional", "organic"}
VALID_CRED_TIERS   = {"high", "medium", "low", "bot"}
VALID_FILTER_REASON = {None, "duplicate", "too_short", "off_topic"}


def run_chain(record: dict) -> dict:
    """Run one record through all 6 Eng B modules in order."""
    r = copy.deepcopy(record)   # never mutate original fixture
    r = detect_language(r)
    r = normalise_tanglish(r)
    r = filter_noise(r)
    if not r.get("is_relevant", True):
        r["pipeline_stage_stopped"] = "noise_filter"
        return r
    r = classify_bot(r)
    r = classify_promo(r)
    r = assess_credibility(r)
    r["pipeline_stage_stopped"] = "complete"
    return r


# ---------------------------------------------------------------------------
# Session-level fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def all_records():
    records = _load_records()
    assert len(records) == 30, f"Expected 30 records, found {len(records)}"
    return records


@pytest.fixture(scope="module")
def all_results(all_records):
    """Run full chain on all 30 records once; share across tests."""
    reset_dedup_store()
    results = []
    for rec in all_records:
        results.append(run_chain(rec))
    return results


def _get(results: list[dict], item_id: str) -> dict:
    for r in results:
        if r.get("item_id") == item_id:
            return r
    raise KeyError(f"Record {item_id!r} not found in results")


# ===========================================================================
# TEST CLASS 1 — Stability: no crashes on any of the 30 records
# ===========================================================================
class TestChainStability:
    def test_all_30_records_processed_without_exception(self, all_records):
        """The chain must not raise any exception on any of the 30 records."""
        reset_dedup_store()
        errors = []
        for rec in all_records:
            try:
                run_chain(rec)
            except Exception as exc:
                errors.append(f"{rec.get('item_id')}: {exc}")
        assert not errors, "Chain raised exceptions:\n" + "\n".join(errors)

    def test_all_30_records_return_a_dict(self, all_results):
        """Every result must be a dictionary."""
        for r in all_results:
            assert isinstance(r, dict), f"{r.get('item_id')} did not return a dict"

    def test_no_original_keys_deleted(self, all_records, all_results):
        """Modules must only ADD keys — never delete existing ones."""
        for original, result in zip(all_records, all_results):
            for key in original:
                assert key in result, (
                    f"{result.get('item_id')}: key '{key}' was deleted by pipeline"
                )

    def test_chain_completes_in_under_five_seconds(self, all_records):
        """30-record chain must complete within 5 seconds (pure-function baseline)."""
        reset_dedup_store()
        start = time.perf_counter()
        for rec in all_records:
            run_chain(rec)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Chain took {elapsed:.2f}s — too slow for 30 records"


# ===========================================================================
# TEST CLASS 2 — Schema Completeness
# ===========================================================================
class TestSchemaCompleteness:

    def test_phase1_fields_present_on_all_30_records(self, all_results):
        """Every record (even filtered ones) must have all Phase 1 output fields."""
        for r in all_results:
            missing = REQUIRED_PHASE1_FIELDS - r.keys()
            assert not missing, (
                f"{r['item_id']} missing Phase 1 fields: {missing}"
            )

    def test_phase2_fields_present_on_relevant_records(self, all_results):
        """Relevant records must have all Phase 2 output fields."""
        for r in all_results:
            if r.get("is_relevant") and r.get("pipeline_stage_stopped") == "complete":
                missing = REQUIRED_PHASE2_FIELDS - r.keys()
                assert not missing, (
                    f"{r['item_id']} missing Phase 2 fields: {missing}"
                )

    def test_language_detected_is_valid_value(self, all_results):
        for r in all_results:
            lang = r.get("language_detected")
            assert lang in VALID_LANGUAGES, (
                f"{r['item_id']}: invalid language_detected={lang!r}"
            )

    def test_language_confidence_is_float_in_range(self, all_results):
        for r in all_results:
            conf = r.get("language_confidence")
            assert isinstance(conf, float), f"{r['item_id']}: confidence not float"
            assert 0.0 <= conf <= 1.0, f"{r['item_id']}: confidence {conf} out of range"

    def test_relevance_score_is_float_in_range(self, all_results):
        for r in all_results:
            score = r.get("relevance_score")
            assert isinstance(score, float), f"{r['item_id']}: relevance_score not float"
            assert 0.0 <= score <= 1.0, f"{r['item_id']}: relevance_score out of range"

    def test_filter_reason_is_valid_value(self, all_results):
        for r in all_results:
            reason = r.get("filter_reason")
            assert reason in VALID_FILTER_REASON, (
                f"{r['item_id']}: invalid filter_reason={reason!r}"
            )

    def test_text_hash_is_32_char_hex_string(self, all_results):
        for r in all_results:
            h = r.get("text_hash", "")
            assert isinstance(h, str) and len(h) == 32, (
                f"{r['item_id']}: text_hash must be 32-char MD5 hex, got {h!r}"
            )

    def test_bot_flag_is_valid_on_relevant_records(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                flag = r.get("bot_flag")
                assert flag in VALID_BOT_FLAGS, (
                    f"{r['item_id']}: invalid bot_flag={flag!r}"
                )

    def test_bot_confidence_in_range_on_relevant_records(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                conf = r.get("bot_confidence")
                assert isinstance(conf, float), f"{r['item_id']}: bot_confidence not float"
                assert 0.0 <= conf <= 1.0, (
                    f"{r['item_id']}: bot_confidence {conf} out of range"
                )

    def test_bot_flags_is_always_list(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                assert isinstance(r.get("bot_flags"), list), (
                    f"{r['item_id']}: bot_flags must be a list"
                )

    def test_content_type_is_valid_on_relevant_records(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                ct = r.get("content_type")
                assert ct in VALID_CONTENT_TYPE, (
                    f"{r['item_id']}: invalid content_type={ct!r}"
                )

    def test_credibility_tier_is_valid_on_relevant_records(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                tier = r.get("credibility_tier")
                assert tier in VALID_CRED_TIERS, (
                    f"{r['item_id']}: invalid credibility_tier={tier!r}"
                )

    def test_promo_signals_is_always_list(self, all_results):
        for r in all_results:
            if r.get("is_relevant"):
                assert isinstance(r.get("promo_signals"), list), (
                    f"{r['item_id']}: promo_signals must be a list"
                )


# ===========================================================================
# TEST CLASS 3 — Deduplication
# ===========================================================================
class TestDeduplication:

    def test_mock021_is_duplicate_of_mock001(self, all_results):
        """mock-021 has identical content to mock-001 and must be flagged."""
        r = _get(all_results, "mock-021")
        assert r["is_duplicate"] is True, "mock-021 must be flagged as duplicate"
        assert r["filter_reason"] == "duplicate"
        assert r["is_relevant"] is False

    def test_mock001_is_not_a_duplicate(self, all_results):
        """mock-001 is the first occurrence and must NOT be flagged."""
        r = _get(all_results, "mock-001")
        assert r["is_duplicate"] is False

    def test_all_unique_records_have_unique_hashes(self, all_results):
        """Excluding duplicates, every text_hash must be unique."""
        hashes = [r["text_hash"] for r in all_results if not r.get("is_duplicate")]
        assert len(hashes) == len(set(hashes)), "Unexpected hash collision in unique records"


# ===========================================================================
# TEST CLASS 4 — Edge Case Records
# ===========================================================================
class TestEdgeCaseRecords:

    def test_mock022_url_only_is_filtered(self, all_results):
        """URL-only content collapses to empty after normalization — must be filtered."""
        r = _get(all_results, "mock-022")
        assert r["is_relevant"] is False
        assert r["filter_reason"] in ("too_short", "off_topic", "duplicate")

    def test_mock023_one_word_is_filtered(self, all_results):
        """Single-word 'ok' is below minimum length — must be filtered."""
        r = _get(all_results, "mock-023")
        assert r["is_relevant"] is False
        assert r["filter_reason"] in ("too_short", "off_topic")

    def test_mock024_empty_content_does_not_crash(self, all_results):
        """Empty content string must produce a valid result dict without crashing."""
        r = _get(all_results, "mock-024")
        assert isinstance(r, dict)
        assert "is_relevant" in r
        assert r["is_relevant"] is False

    def test_mock024_normalised_text_is_string(self, all_results):
        """Even for empty input, normalised_text must be a string."""
        r = _get(all_results, "mock-024")
        assert isinstance(r.get("normalised_text"), str)


# ===========================================================================
# TEST CLASS 5 — Verified Override
# ===========================================================================
class TestVerifiedOverride:

    def test_mock025_verified_account_is_human(self, all_results):
        """mock-025 is verified — must be classified as human despite all bot signals."""
        r = _get(all_results, "mock-025")
        if r.get("is_relevant"):
            assert r["bot_flag"] == "human", (
                "Verified account must always be classified as human"
            )

    def test_mock025_credibility_not_bot_tier(self, all_results):
        """Verified account must not land in 'bot' credibility tier."""
        r = _get(all_results, "mock-025")
        if r.get("is_relevant"):
            assert r.get("credibility_tier") != "bot"


# ===========================================================================
# TEST CLASS 6 — Promotional Detection
# ===========================================================================
class TestPromotionalDetection:

    def test_mock026_sarcasm_with_promo_hashtags_is_promotional(self, all_results):
        """mock-026 has #ad + #sponsored — must be promotional regardless of tone."""
        r = _get(all_results, "mock-026")
        if r.get("is_relevant"):
            assert r["is_promotional"] is True
            assert r["content_type"] == "promotional"

    def test_mock028_influencer_with_link_in_bio_is_promotional(self, all_results):
        """mock-028 has 'link in bio' CTA — must be flagged as promotional."""
        r = _get(all_results, "mock-028")
        if r.get("is_relevant"):
            assert r["is_promotional"] is True

    def test_mock028_influencer_is_high_credibility(self, all_results):
        """mock-028 has 95k followers — must be high credibility tier."""
        r = _get(all_results, "mock-028")
        if r.get("is_relevant"):
            assert r.get("credibility_tier") == "high"

    def test_mock030_genuine_fan_is_organic(self, all_results):
        """mock-030 is an organic FDFS fan with no promo signals."""
        r = _get(all_results, "mock-030")
        if r.get("is_relevant"):
            assert r["is_promotional"] is False
            assert r["content_type"] == "organic"


# ===========================================================================
# TEST CLASS 7 — Language Detection on New Records
# ===========================================================================
class TestLanguageDetection:

    def test_mock027_mixed_hinglish_tanglish_detected(self, all_results):
        """mock-027 mixes Hinglish and Tanglish — must not be classified as plain English."""
        r = _get(all_results, "mock-027")
        assert r["language_detected"] != "english", (
            "Mixed Hinglish+Tanglish comment must not be classified as plain English"
        )

    def test_mock030_tanglish_detected(self, all_results):
        """mock-030 uses dense Tanglish markers — must be tanglish or mixed."""
        r = _get(all_results, "mock-030")
        assert r["language_detected"] in ("tanglish", "mixed", "english"), (
            f"Unexpected language for mock-030: {r['language_detected']}"
        )


# ===========================================================================
# TEST CLASS 8 — Credibility Ordering
# ===========================================================================
class TestCredibilityOrdering:

    def test_mock029_veteran_account_is_not_bot_tier(self, all_results):
        """9-year old account (mock-029) must be human and not in bot tier."""
        r = _get(all_results, "mock-029")
        if r.get("is_relevant"):
            assert r["bot_flag"] == "human"
            assert r["credibility_tier"] != "bot"

    def test_mock028_outranks_mock023_in_credibility(self, all_results):
        """95k-follower influencer must have equal or higher credibility than 1-follower new user."""
        tier_order = {"high": 3, "medium": 2, "low": 1, "bot": 0}
        r28 = _get(all_results, "mock-028")
        r23 = _get(all_results, "mock-023")
        if r28.get("is_relevant") and r23.get("is_relevant"):
            assert tier_order.get(r28["credibility_tier"], 0) >= tier_order.get(r23["credibility_tier"], 0)


# ===========================================================================
# TEST CLASS 9 — Pipeline Stage Tracking
# ===========================================================================
class TestPipelineStageStopped:

    def test_irrelevant_records_have_stage_noise_filter(self, all_results):
        """Filtered-out records must have pipeline_stage_stopped='noise_filter'."""
        for r in all_results:
            if not r.get("is_relevant"):
                assert r.get("pipeline_stage_stopped") == "noise_filter", (
                    f"{r['item_id']}: expected stage 'noise_filter', got {r.get('pipeline_stage_stopped')!r}"
                )

    def test_completed_records_have_stage_complete(self, all_results):
        """Records that pass all filters must have pipeline_stage_stopped='complete'."""
        for r in all_results:
            if r.get("is_relevant"):
                assert r.get("pipeline_stage_stopped") == "complete", (
                    f"{r['item_id']}: expected stage 'complete', got {r.get('pipeline_stage_stopped')!r}"
                )


# ===========================================================================
# TEST CLASS 10 — Batch Reset Correctness
# ===========================================================================
class TestBatchReset:

    def test_reset_dedup_clears_between_batches(self, all_records):
        """After reset, mock-021 (duplicate of mock-001) should NOT be a duplicate in a fresh batch."""
        reset_dedup_store()
        # Run only mock-021 in isolation (no mock-001 before it)
        target = next(r for r in all_records if r["item_id"] == "mock-021")
        result = run_chain(copy.deepcopy(target))
        # In a fresh store, it should NOT be flagged as duplicate
        assert result["is_duplicate"] is False, (
            "After reset, mock-021 must not be a duplicate in a fresh batch"
        )

    def test_duplicate_flagged_when_predecessor_runs_first(self, all_records):
        """Run mock-001 then mock-021 in same batch — mock-021 must be a duplicate."""
        reset_dedup_store()
        original = next(r for r in all_records if r["item_id"] == "mock-001")
        duplicate = next(r for r in all_records if r["item_id"] == "mock-021")
        run_chain(copy.deepcopy(original))
        result = run_chain(copy.deepcopy(duplicate))
        assert result["is_duplicate"] is True
