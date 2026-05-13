"""
Tests for pipeline/sentiment_tagger.py

Covers: valid LLM response, sarcasm confidence cap, LLM failure,
invalid data types, invalid sentiment/emotion values, score renormalization.
"""

import pytest
from unittest.mock import patch

from pipeline.sentiment_tagger import tag_sentiment


@pytest.fixture
def base_record():
    """Minimal record dict that tag_sentiment expects."""
    return {
        "item_id": "test-123",
        "keyword": "Leo movie",
        "platform": "twitter",
        "language_detected": "english",
        "content": "Sample comment text",
    }


@patch("pipeline.sentiment_tagger.call_llm")
def test_valid_response(mock_llm, base_record):
    """Valid LLM output should enrich record with all sentiment fields."""
    mock_llm.return_value = {
        "sentiment": "positive",
        "positive_score": 0.8,
        "neutral_score": 0.15,
        "negative_score": 0.05,
        "confidence": 0.85,
        "dominant_emotion": "happy",
        "emotion_tags": ["happy", "excited"],
        "is_sarcastic": False,
        "crisis_severity": "none",
        "reasoning": "Clear positive sentiment.",
    }

    result = tag_sentiment(base_record)

    assert result["sentiment_failed"] is False
    assert result["sentiment"] == "positive"
    assert result["confidence_score"] == 0.85
    assert result["dominant_emotion"] == "happy"
    assert result["emotion_tags"] == ["happy", "excited"]
    assert result["low_confidence"] is False
    assert result["is_sarcastic"] is False
    assert result["crisis_severity"] == "none"


@patch("pipeline.sentiment_tagger.call_llm")
def test_sarcasm_caps_confidence(mock_llm, base_record):
    """Sarcasm must force confidence ≤ 0.55 per spec."""
    mock_llm.return_value = {
        "sentiment": "negative",
        "positive_score": 0.1,
        "neutral_score": 0.1,
        "negative_score": 0.8,
        "confidence": 0.90,
        "dominant_emotion": "sarcastic",
        "emotion_tags": ["sarcastic"],
        "is_sarcastic": True,
        "crisis_severity": "low",
        "reasoning": "Sarcastic praise.",
    }

    result = tag_sentiment(base_record)

    assert result["is_sarcastic"] is True
    assert result["confidence_score"] == 0.55
    assert result["low_confidence"] is True


@patch("pipeline.sentiment_tagger.call_llm")
def test_llm_failure_returns_safe_defaults(mock_llm, base_record):
    """Complete LLM failure should set sentiment_failed=True with defaults."""
    mock_llm.return_value = None

    result = tag_sentiment(base_record)

    assert result["sentiment_failed"] is True
    assert result["sentiment"] is None
    assert result["confidence_score"] == 0.0
    assert result["emotion_tags"] == []
    assert result["low_confidence"] is True
    assert "LLM failed" in result["pipeline_error"]


@patch("pipeline.sentiment_tagger.call_llm")
def test_invalid_datatypes_handled(mock_llm, base_record):
    """Non-numeric scores should trigger validation error, not a crash."""
    mock_llm.return_value = {
        "sentiment": "positive",
        "positive_score": "not_a_float",
        "confidence": 0.9,
    }

    result = tag_sentiment(base_record)

    assert result["sentiment_failed"] is True
    assert result["sentiment"] is None
    assert "validation error" in result["pipeline_error"]


@patch("pipeline.sentiment_tagger.call_llm")
def test_invalid_sentiment_value_defaults_to_none(mock_llm, base_record):
    """Invalid sentiment value should be replaced with None."""
    mock_llm.return_value = {
        "sentiment": "very_positive",  # not in VALID_SENTIMENTS
        "positive_score": 0.8,
        "neutral_score": 0.1,
        "negative_score": 0.1,
        "confidence": 0.85,
        "dominant_emotion": "happy",
        "emotion_tags": ["happy"],
        "is_sarcastic": False,
        "crisis_severity": "none",
        "reasoning": "test",
    }

    result = tag_sentiment(base_record)

    assert result["sentiment_failed"] is False
    assert result["sentiment"] is None


@patch("pipeline.sentiment_tagger.call_llm")
def test_invalid_emotions_filtered(mock_llm, base_record):
    """Invalid emotion tags should be filtered out, valid ones kept."""
    mock_llm.return_value = {
        "sentiment": "positive",
        "positive_score": 0.8,
        "neutral_score": 0.1,
        "negative_score": 0.1,
        "confidence": 0.85,
        "dominant_emotion": "happy",
        "emotion_tags": ["happy", "INVALID_EMOTION", "excited"],
        "is_sarcastic": False,
        "crisis_severity": "none",
        "reasoning": "test",
    }

    result = tag_sentiment(base_record)

    assert "happy" in result["emotion_tags"]
    assert "excited" in result["emotion_tags"]
    assert "INVALID_EMOTION" not in result["emotion_tags"]


@patch("pipeline.sentiment_tagger.call_llm")
def test_scores_renormalized(mock_llm, base_record):
    """Scores that don't sum to 1.0 should be renormalized."""
    mock_llm.return_value = {
        "sentiment": "positive",
        "positive_score": 0.6,
        "neutral_score": 0.3,
        "negative_score": 0.3,  # sum = 1.2, not 1.0
        "confidence": 0.8,
        "dominant_emotion": "happy",
        "emotion_tags": ["happy"],
        "is_sarcastic": False,
        "crisis_severity": "none",
        "reasoning": "test",
    }

    result = tag_sentiment(base_record)

    total = result["positive_score"] + result["neutral_score"] + result["negative_score"]
    assert abs(total - 1.0) < 0.01  # Should be renormalized to ~1.0
