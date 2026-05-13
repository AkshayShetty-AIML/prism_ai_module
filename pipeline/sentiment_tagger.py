"""
PRISM — Sentiment Tagger (Eng C · Phase 3)

Core pipeline module that classifies sentiment, emotions, sarcasm, and
crisis severity for each social media record using the LLM.

Interface contract (from Eng A):
    def tag_sentiment(record: dict) -> dict

Rules:
    - Never raises exceptions — sets sentiment_failed=True on failure
    - Validates all LLM JSON fields against allowed values
    - Enforces business rules (sarcasm confidence cap ≤ 0.55)
    - Enriches record with 12+ new fields from the LLM response
"""

import logging
import os

from prompts.sentiment_prompt import (
    build_sentiment_prompt,
    VALID_SENTIMENTS,
    VALID_EMOTIONS,
    VALID_CRISIS_SEVERITIES,
)
from llm.llm_client import call_llm

logger = logging.getLogger("prism.pipeline.sentiment")

# Model name mapping: provider → actual model name (for model_used field)
_MODEL_NAMES = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "claude": "claude-sonnet",
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
}

# Fields set on LLM failure — ensures downstream code never gets KeyErrors
_FAILURE_DEFAULTS = {
    "sentiment_failed": True,
    "sentiment": None,
    "positive_score": 0.0,
    "neutral_score": 0.0,
    "negative_score": 0.0,
    "confidence_score": 0.0,
    "dominant_emotion": None,
    "emotion_tags": [],
    "is_sarcastic": False,
    "crisis_severity": "none",
    "crisis_theme_group": None,
    "sentiment_reasoning": "",
    "low_confidence": True,
    "model_used": None,
    "tokens_used": 0,
    "prompt_version": os.getenv("PROMPT_VERSION", "1.0"),
}


def _set_failure(record: dict, error_msg: str) -> dict:
    """Apply failure defaults and log the error."""
    record.update(_FAILURE_DEFAULTS)
    record["pipeline_error"] = error_msg
    logger.warning("Sentiment tagging failed for item '%s': %s",
                   record.get("item_id", "unknown"), error_msg)
    return record


def _validate_and_enrich(record: dict, result: dict) -> dict:
    """
    Validate the LLM response and enrich the record with sentiment fields.
    Raises ValueError on invalid data types so the caller can handle it.
    """
    # ── Extract and validate scores ──
    pos = float(result.get("positive_score", 0.0))
    neu = float(result.get("neutral_score", 0.0))
    neg = float(result.get("negative_score", 0.0))

    score_sum = pos + neu + neg

    # Renormalize if within tolerance, warn if way off
    if abs(score_sum - 1.0) > 0.05:
        logger.warning(
            "Item '%s': LLM scores sum to %.3f (expected ~1.0)",
            record.get("item_id", "unknown"), score_sum,
        )
    if score_sum > 0 and abs(score_sum - 1.0) > 0.01:
        # Normalize to exactly 1.0
        pos, neu, neg = pos / score_sum, neu / score_sum, neg / score_sum

    record["positive_score"] = round(pos, 4)
    record["neutral_score"] = round(neu, 4)
    record["negative_score"] = round(neg, 4)

    # ── Validate sentiment ──
    sentiment = result.get("sentiment", "").lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        logger.warning(
            "Item '%s': Invalid sentiment '%s', defaulting to None",
            record.get("item_id", "unknown"), sentiment,
        )
        sentiment = None
    record["sentiment"] = sentiment

    # ── Confidence ──
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    record["confidence_score"] = round(confidence, 4)

    # ── Dominant emotion ──
    dominant = result.get("dominant_emotion", "").lower().strip()
    if dominant not in VALID_EMOTIONS:
        logger.warning(
            "Item '%s': Invalid dominant_emotion '%s', defaulting to 'neutral'",
            record.get("item_id", "unknown"), dominant,
        )
        dominant = "neutral"
    record["dominant_emotion"] = dominant

    # ── Emotion tags (filter invalid values) ──
    raw_tags = result.get("emotion_tags", [])
    if isinstance(raw_tags, list):
        record["emotion_tags"] = [
            t.lower().strip() for t in raw_tags
            if isinstance(t, str) and t.lower().strip() in VALID_EMOTIONS
        ]
    else:
        record["emotion_tags"] = []

    # ── Sarcasm ──
    record["is_sarcastic"] = bool(result.get("is_sarcastic", False))

    # ── Crisis severity ──
    severity = result.get("crisis_severity", "none").lower().strip()
    if severity not in VALID_CRISIS_SEVERITIES:
        severity = "none"
    record["crisis_severity"] = severity
    record["crisis_theme_group"] = result.get("crisis_theme_group", None)

    # ── Reasoning ──
    record["sentiment_reasoning"] = str(result.get("reasoning", ""))

    # ── Business logic: Sarcasm confidence cap ──
    if record["is_sarcastic"]:
        record["confidence_score"] = min(record["confidence_score"], 0.55)

    # ── Derived fields ──
    record["low_confidence"] = record["confidence_score"] < 0.6
    record["sentiment_failed"] = False

    # ── Model metadata (Eng A requires actual model name, not provider) ──
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    record["model_used"] = _MODEL_NAMES.get(provider, provider)
    record["tokens_used"] = 0  # TODO: populate from LLM response when SDK supports it
    record["prompt_version"] = os.getenv("PROMPT_VERSION", "1.0")

    return record


def tag_sentiment(record: dict) -> dict:
    """
    Phase 3: Classify sentiment for a single record using the LLM.

    Takes a pre-processed record dict (from Eng B's Phase 1–2 modules),
    calls the LLM with the sentiment prompt, validates the JSON response,
    and enriches the record with 12+ sentiment fields.

    Parameters
    ----------
    record : dict
        A record dict that has passed through Phase 1 (language detection,
        normalisation, noise filtering) and Phase 2 (bot detection, promo
        classification). Must contain at minimum: keyword, platform,
        content or normalised_text.

    Returns
    -------
    dict
        The same record dict enriched with sentiment fields:
        sentiment, positive_score, neutral_score, negative_score,
        confidence_score, dominant_emotion, emotion_tags, is_sarcastic,
        crisis_severity, sentiment_reasoning, low_confidence,
        sentiment_failed, model_used, prompt_version.

        On failure: sentiment_failed=True with all other fields set to
        safe defaults.
    """
    system, prompt = build_sentiment_prompt(record)

    result = call_llm(prompt, system=system, use_cache=True)

    if not result:
        return _set_failure(record, "LLM failed to return valid JSON after retries")

    try:
        return _validate_and_enrich(record, result)
    except (TypeError, ValueError) as e:
        return _set_failure(record, f"LLM output validation error: {e}")
