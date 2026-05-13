"""
pipeline/pipeline_runner.py — Core 5-phase pipeline orchestration.

Eng A owns this file. It calls Eng B (Phase 1-2) and Eng C (Phase 3-4)
modules in sequence. Each module takes a dict and returns an enriched dict.

Flow:
    Phase 1a: detect_language        (Eng B)
    Phase 1b: normalise_tanglish     (Eng B)
    Phase 1c: filter_noise           (Eng B)
    ── if not relevant, stop here ──
    Phase 2a: classify_bot           (Eng B)
    Phase 2b: classify_promo         (Eng B)
    Phase 2c: assess_credibility     (Eng B)
    Phase 3:  tag_sentiment          (Eng C)
    Phase 4:  score_impact           (Eng C)

Usage:
    from pipeline.pipeline_runner import process_record

    record = await process_record(record)
    await save_record(record)
"""

from datetime import datetime, timezone

# ── Eng B imports (Phase 1 + 2) ─────────────────────────────────────
from pipeline.language_processor import detect_language
from pipeline.tanglish_normalizer import normalise_tanglish
from pipeline.noise_filter import filter_noise
from pipeline.bot_detector import classify_bot
from pipeline.promo_classifier import classify_promo
from pipeline.credibility_scorer import assess_credibility

# ── Eng C imports (Phase 3 + 4) ─────────────────────────────────────
from pipeline.sentiment_tagger import tag_sentiment
from pipeline.impact_scorer import score_impact


async def process_record(record: dict) -> dict:
    """
    Run a single record through the full 5-phase pipeline.

    Never raises exceptions — if any phase fails, the error is captured
    in the record and processing stops gracefully.

    Returns the enriched record dict ready to be saved to analyzed_records.
    """
    try:
        # ── Phase 1a: Language Detection (Eng B) ────────────────────
        record = detect_language(record)

        # ── Phase 1b: Tanglish/Hinglish Normalization (Eng B) ───────
        record = normalise_tanglish(record)

        # ── Phase 1c: Noise Filtering + Dedup (Eng B) ──────────────
        record = filter_noise(record)

        # If not relevant, stop here — no point calling LLM
        if not record.get("is_relevant", True):
            record["pipeline_stage_stopped"] = "noise_filter"
            record["processed_at"] = datetime.now(timezone.utc).isoformat()
            return record

        # ── Phase 2a: Bot Detection (Eng B) ─────────────────────────
        record = classify_bot(record)

        # ── Phase 2b: Promotional Classification (Eng B) ────────────
        record = classify_promo(record)

        # ── Phase 2c: Credibility Scoring (Eng B) ───────────────────
        record = assess_credibility(record)

        # ── Phase 3: LLM Sentiment Analysis (Eng C) ────────────────
        record = tag_sentiment(record)

        # ── Phase 4: Impact Scoring (Eng C) ─────────────────────────
        record = score_impact(record)

        # ── Phase 5: Metadata ──────────────────────────────────────
        record["pipeline_stage_stopped"] = "complete"
        record["processed_at"] = datetime.now(timezone.utc).isoformat()

        return record

    except Exception as e:
        # Never crash — capture error and return the record as-is
        record["pipeline_error"] = str(e)
        record["pipeline_stage_stopped"] = "error"
        record["processed_at"] = datetime.now(timezone.utc).isoformat()
        return record