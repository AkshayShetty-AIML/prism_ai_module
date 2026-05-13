"""
test_noise_filter.py — Unit tests for Phase 1c: Relevance Scoring + Dedup
Engineer B | PRISM AI Processing Pipeline

Tests for: off-topic filtering, duplicate detection, too-short filtering,
keyword presence, domain word scoring, dedup reset, process() wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.noise_filter import (
    filter_record, score_relevance, reset_dedup_store,
    compute_text_hash, process
)


# Reset dedup store before each test to ensure isolation
@pytest.fixture(autouse=True)
def reset_dedup():
    reset_dedup_store()
    yield
    reset_dedup_store()


# ---------------------------------------------------------------------------
# Test 1: Clearly off-topic text (no keyword, no film words) → filtered
# ---------------------------------------------------------------------------
def test_off_topic_filtered():
    record = {
        "content": "What is the best recipe for chicken biryani with extra spices?",
        "normalised_text": "What is the best recipe for chicken biryani with extra spices?",
        "keyword": "Leo movie",
    }
    result = filter_record(record)
    assert result["is_relevant"] is False
    assert result["filter_reason"] == "off_topic"
    assert result["relevance_score"] < 0.40


# ---------------------------------------------------------------------------
# Test 2: Too-short text (< 3 words) → filtered as too_short
# ---------------------------------------------------------------------------
def test_too_short_filtered():
    record = {
        "content": "Hi",
        "normalised_text": "Hi",
        "keyword": "Leo movie",
    }
    result = filter_record(record)
    assert result["is_relevant"] is False
    assert result["filter_reason"] == "too_short"


# ---------------------------------------------------------------------------
# Test 3: Duplicate text → second occurrence filtered as duplicate
# ---------------------------------------------------------------------------
def test_duplicate_filtered():
    text = "Leo movie is semma padam with great action scenes and BGM."
    record1 = {"content": text, "normalised_text": text, "keyword": "Leo movie"}
    record2 = {"content": text, "normalised_text": text, "keyword": "Leo movie"}

    result1 = filter_record(record1)
    result2 = filter_record(record2)

    assert result1["is_relevant"] is True        # first pass through
    assert result2["is_relevant"] is False       # second is duplicate
    assert result2["filter_reason"] == "duplicate"
    assert result2["is_duplicate"] is True


# ---------------------------------------------------------------------------
# Test 4: Relevant film text with keyword → passes filter
# ---------------------------------------------------------------------------
def test_relevant_with_keyword():
    record = {
        "content": "Leo movie has amazing action scenes. The BGM by Anirudh is brilliant.",
        "normalised_text": "Leo movie has amazing action scenes. The BGM by Anirudh is brilliant.",
        "keyword": "Leo movie",
    }
    result = filter_record(record)
    assert result["is_relevant"] is True
    assert result["relevance_score"] >= 0.40
    assert result["filter_reason"] is None
    assert result["is_duplicate"] is False


# ---------------------------------------------------------------------------
# Test 5: Relevant film text WITHOUT keyword but strong domain words → passes
# ---------------------------------------------------------------------------
def test_relevant_domain_words_no_keyword():
    record = {
        "content": "The movie screenplay was brilliant. Film had great BGM and actor performances.",
        "normalised_text": "The movie screenplay was brilliant. Film had great BGM and actor performances.",
        "keyword": "Leo movie",
    }
    result = filter_record(record)
    # Domain words: movie, screenplay, film, bgm, actor = 5 hits → 0.40 + 0.20 (length) = 0.60
    assert result["is_relevant"] is True
    assert result["relevance_score"] >= 0.40


# ---------------------------------------------------------------------------
# Test 6: score_relevance returns correct field types
# ---------------------------------------------------------------------------
def test_score_relevance_types():
    result = score_relevance("Leo movie is a great film with excellent BGM", "Leo movie")
    assert isinstance(result["relevance_score"], float)
    assert isinstance(result["is_relevant"], bool)
    assert result["filter_reason"] is None or isinstance(result["filter_reason"], str)
    assert 0.0 <= result["relevance_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test 7: Dedup store resets correctly between batches
# ---------------------------------------------------------------------------
def test_dedup_reset_between_batches():
    text = "Leo movie is a fantastic film with great BGM and screenplay."
    record_a = {"content": text, "normalised_text": text, "keyword": "Leo movie"}
    record_b = {"content": text, "normalised_text": text, "keyword": "Leo movie"}

    filter_record(record_a)  # adds to seen_hashes

    reset_dedup_store()      # simulate new batch

    result_b = filter_record(record_b)
    assert result_b["is_relevant"] is True  # NOT filtered after reset
    assert result_b["is_duplicate"] is False


# ---------------------------------------------------------------------------
# Test 8: text_hash is always present and is a 32-char MD5 hex string
# ---------------------------------------------------------------------------
def test_text_hash_present_and_valid():
    record = {
        "content": "Leo movie was excellent and the film had great scenes.",
        "normalised_text": "Leo movie was excellent and the film had great scenes.",
        "keyword": "Leo movie",
    }
    result = filter_record(record)
    assert "text_hash" in result
    assert len(result["text_hash"]) == 32
    assert all(c in "0123456789abcdef" for c in result["text_hash"])


# ---------------------------------------------------------------------------
# Test 9: process() is alias for filter_record()
# ---------------------------------------------------------------------------
def test_process_alias():
    record = {
        "content": "Leo movie has a fantastic screenplay and brilliant BGM.",
        "normalised_text": "Leo movie has a fantastic screenplay and brilliant BGM.",
        "keyword": "Leo movie",
    }
    result = process(record)
    assert "is_relevant" in result
    assert "relevance_score" in result
    assert "filter_reason" in result
    assert "is_duplicate" in result
    assert "text_hash" in result
