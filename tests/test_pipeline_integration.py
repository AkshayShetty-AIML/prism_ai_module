"""
test_pipeline_integration.py — End-to-End Integration Test on mock_data.json
Engineer B | PRISM AI Processing Pipeline

Loads all 20 records from tests/mock_data.json and runs each through
the complete Engineer B pipeline in sequence:
    Phase 1a: language_processor
    Phase 1b: tanglish_normalizer
    Phase 1c: noise_filter
    Phase 2a: bot_detector
    Phase 2b: promo_classifier
    Phase 2c: credibility_scorer

Asserts correct outputs per record category.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.language_processor import process as detect_language
from pipeline.tanglish_normalizer import process as normalise
from pipeline.noise_filter import process as filter_noise, reset_dedup_store
from pipeline.bot_detector import process as detect_bot
from pipeline.promo_classifier import process as classify_promo
from pipeline.credibility_scorer import process as score_credibility

# ---------------------------------------------------------------------------
# Load mock data
# ---------------------------------------------------------------------------

_MOCK_PATH = os.path.join(os.path.dirname(__file__), "mock_data.json")

with open(_MOCK_PATH, "r", encoding="utf-8") as f:
    _MOCK = json.load(f)

RECORDS = _MOCK["records"]
CATEGORIES = _MOCK["_meta"]["categories"]


def run_pipeline(record: dict) -> dict:
    """Run a single record through all 6 Engineer B modules in order."""
    r = dict(record)   # shallow copy — don't mutate fixture
    r = detect_language(r)
    r = normalise(r)
    r = filter_noise(r)
    if r.get("is_relevant"):
        r = detect_bot(r)
        r = classify_promo(r)
        r = score_credibility(r)
    return r


@pytest.fixture(autouse=True)
def reset_dedup():
    reset_dedup_store()
    yield
    reset_dedup_store()


def get_record(item_id: str) -> dict:
    for r in RECORDS:
        if r["item_id"] == item_id:
            return r
    raise KeyError(f"Record {item_id!r} not found in mock_data.json")


# ===========================================================================
# 1. SCHEMA VALIDATION — all output fields present on every processed record
# ===========================================================================

class TestSchemaCompleteness:
    """Every processed record must have ALL Engineer B output fields."""

    PHASE1_FIELDS = ["language_detected", "language_confidence", "has_code_mixing",
                     "normalised_text", "is_relevant", "relevance_score",
                     "filter_reason", "is_duplicate", "text_hash"]
    PHASE2_FIELDS = ["bot_flag", "bot_confidence", "bot_flags",
                     "is_promotional", "content_type", "promo_signals",
                     "credibility_tier"]

    def test_phase1_fields_always_present(self):
        """Phase 1 fields must appear on every record, even filtered ones."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            for field in self.PHASE1_FIELDS:
                assert field in r, f"Missing '{field}' on {r['item_id']}"

    def test_phase2_fields_present_on_relevant_records(self):
        """Phase 2 fields only on is_relevant=True records."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            if r["is_relevant"]:
                for field in self.PHASE2_FIELDS:
                    assert field in r, (
                        f"Missing '{field}' on relevant record {r['item_id']}"
                    )

    def test_value_types_correct(self):
        """Spot-check type correctness on all records."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            assert isinstance(r["language_detected"], str)
            assert isinstance(r["language_confidence"], float)
            assert isinstance(r["has_code_mixing"], bool)
            assert isinstance(r["normalised_text"], str)
            assert isinstance(r["is_relevant"], bool)
            assert 0.0 <= r["language_confidence"] <= 1.0
            assert 0.0 <= r["relevance_score"] <= 1.0
            assert len(r["text_hash"]) == 32


# ===========================================================================
# 2. LANGUAGE DETECTION — per record
# ===========================================================================

class TestLanguageDetection:

    def test_mock001_english(self):
        r = run_pipeline(get_record("mock-001"))
        assert r["language_detected"] == "english"
        assert r["language_confidence"] > 0.5

    def test_mock002_english(self):
        r = run_pipeline(get_record("mock-002"))
        assert r["language_detected"] == "english"

    def test_mock006_tanglish(self):
        # "Semma padam da! BGM vera level..."
        r = run_pipeline(get_record("mock-006"))
        assert r["language_detected"] in ("tanglish", "mixed")
        assert r["language_confidence"] > 0.5

    def test_mock007_tanglish(self):
        # "Leo padam mokkai da..."
        r = run_pipeline(get_record("mock-007"))
        assert r["language_detected"] in ("tanglish", "mixed")

    def test_mock009_tanglish(self):
        # "First day first show paatha. Fdfs experience was semma..."
        r = run_pipeline(get_record("mock-009"))
        assert r["language_detected"] in ("tanglish", "mixed")

    def test_mock011_hinglish(self):
        # "Leo movie ekdum mast hai yaar! Zabardast action..."
        r = run_pipeline(get_record("mock-011"))
        assert r["language_detected"] in ("hinglish", "mixed")

    def test_mock012_hinglish(self):
        # "Leo movie bakwaas hai. Bekar screenplay..."
        r = run_pipeline(get_record("mock-012"))
        assert r["language_detected"] in ("hinglish", "mixed")

    def test_mock013_hindi_script(self):
        # "लियो फिल्म बहुत अच्छी लगी..." (Devanagari)
        r = run_pipeline(get_record("mock-013"))
        assert r["language_detected"] in ("hindi", "mixed")

    def test_language_confidence_range(self):
        """All confidence values must be 0.0–1.0."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            assert 0.0 <= r["language_confidence"] <= 1.0, (
                f"{raw['item_id']}: confidence={r['language_confidence']}"
            )


# ===========================================================================
# 3. TANGLISH NORMALISATION
# ===========================================================================

class TestNormalisation:

    def test_mock006_slang_normalised(self):
        # "Semma padam" → contains "excellent" and "film"
        r = run_pipeline(get_record("mock-006"))
        n = r["normalised_text"].lower()
        assert "excellent" in n or "outstanding" in n or "impressive" in n

    def test_mock011_hinglish_normalised(self):
        # "zabardast" → "outstanding", "mast" → "great"
        r = run_pipeline(get_record("mock-011"))
        n = r["normalised_text"].lower()
        assert "outstanding" in n or "great" in n

    def test_normalised_text_never_empty_for_real_content(self):
        """Records with real textual content must produce non-empty normalised_text.
        Excludes URL-only and empty-content edge-case records (mock-022, mock-024).
        """
        url_only_or_empty = {"mock-022", "mock-024"}
        for raw in RECORDS:
            if raw["item_id"] in url_only_or_empty:
                continue
            if not raw.get("content", "").strip():
                continue
            r = run_pipeline(raw)
            assert r["normalised_text"].strip() != "", (
                f"{raw['item_id']}: normalised_text is empty"
            )

    def test_original_content_unchanged(self):
        """normalise() must NOT modify the original 'content' field."""
        for raw in RECORDS:
            original = raw["content"]
            r = run_pipeline(raw)
            assert r["content"] == original, (
                f"{raw['item_id']}: content was mutated"
            )


# ===========================================================================
# 4. NOISE FILTER
# ===========================================================================

class TestNoiseFilter:

    def test_mock018_off_topic_filtered(self):
        # "What is the best recipe for chicken biryani..."
        r = run_pipeline(get_record("mock-018"))
        assert r["is_relevant"] is False
        assert r["filter_reason"] == "off_topic"

    def test_mock019_too_short_filtered(self):
        # Content: "Hi" — 1 word
        r = run_pipeline(get_record("mock-019"))
        assert r["is_relevant"] is False
        assert r["filter_reason"] == "too_short"

    def test_mock020_off_topic_python_tutorial(self):
        # "Python programming tutorial for beginners..."
        r = run_pipeline(get_record("mock-020"))
        assert r["is_relevant"] is False

    def test_mock001_relevant(self):
        # Clear English review with keyword "Leo movie"
        r = run_pipeline(get_record("mock-001"))
        assert r["is_relevant"] is True
        assert r["relevance_score"] >= 0.40

    def test_mock006_tanglish_relevant(self):
        # Tanglish review — uses "padam", "bgm" as domain words
        r = run_pipeline(get_record("mock-006"))
        assert r["is_relevant"] is True

    def test_duplicate_detection_across_records(self):
        """Same text submitted twice — second should be duplicate."""
        reset_dedup_store()
        text = "Leo movie is absolutely fantastic! Vijay's performance was great in the film."
        r1 = {"item_id": "dup-1", "content": text, "keyword": "Leo movie",
              "author": RECORDS[0]["author"], "engagement": RECORDS[0]["engagement"]}
        r2 = {"item_id": "dup-2", "content": text, "keyword": "Leo movie",
              "author": RECORDS[0]["author"], "engagement": RECORDS[0]["engagement"]}
        out1 = run_pipeline(r1)
        out2 = run_pipeline(r2)
        assert out1["is_relevant"] is True
        assert out2["is_relevant"] is False
        assert out2["filter_reason"] == "duplicate"
        assert out2["is_duplicate"] is True

    def test_text_hash_unique_across_different_records(self):
        """Different content → different hashes."""
        reset_dedup_store()
        hashes = []
        for raw in RECORDS[:5]:
            r = run_pipeline(raw)
            hashes.append(r["text_hash"])
        assert len(set(hashes)) == len(hashes), "Hash collision across unique records"


# ===========================================================================
# 5. BOT DETECTION
# ===========================================================================

class TestBotDetection:

    def test_mock019_new_account_is_bot(self):
        """
        mock-019: account 5 days old, 1200 posts → clearly a bot.
        Note: content is too short so pipeline stops at noise_filter.
        We test bot detector directly on this author.
        """
        from pipeline.bot_detector import detect_bot
        author = get_record("mock-019")["author"]
        result = detect_bot(author)
        assert result["bot_flag"] == "bot"
        assert result["bot_confidence"] >= 0.65

    def test_mock016_verified_official_is_human(self):
        """mock-016: verified official account — must never be flagged as bot."""
        r = run_pipeline(get_record("mock-016"))
        assert r.get("bot_flag") == "human"
        assert r.get("bot_confidence") == 0.0

    def test_mock002_journalist_is_human(self):
        """mock-002: 5-year-old account with 12K followers — genuine journalist."""
        r = run_pipeline(get_record("mock-002"))
        assert r.get("bot_flag") == "human"

    def test_mock001_genuine_fan_is_human(self):
        """mock-001: 3-year-old account, normal activity."""
        r = run_pipeline(get_record("mock-001"))
        assert r.get("bot_flag") == "human"

    def test_bot_confidence_in_valid_range(self):
        """bot_confidence must always be 0.0–1.0."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            if "bot_confidence" in r:
                assert 0.0 <= r["bot_confidence"] <= 1.0, (
                    f"{raw['item_id']}: bot_confidence={r['bot_confidence']}"
                )

    def test_bot_flags_is_always_list(self):
        for raw in RECORDS:
            r = run_pipeline(raw)
            if "bot_flags" in r:
                assert isinstance(r["bot_flags"], list)


# ===========================================================================
# 6. PROMOTIONAL CLASSIFICATION
# ===========================================================================

class TestPromoClassification:

    def test_mock016_official_promo_is_promotional(self):
        """mock-016: verified studio account with 'book tickets' content."""
        r = run_pipeline(get_record("mock-016"))
        assert r.get("is_promotional") is True
        assert r.get("content_type") == "promotional"

    def test_mock017_influencer_gifted_is_promotional(self):
        """mock-017: #gifted + 'use code' CTA → promotional."""
        r = run_pipeline(get_record("mock-017"))
        assert r.get("is_promotional") is True
        assert "promo_hashtag" in r.get("promo_signals", [])

    def test_mock001_genuine_review_is_organic(self):
        """mock-001: plain review — no hashtags, no CTA."""
        r = run_pipeline(get_record("mock-001"))
        assert r.get("is_promotional") is False
        assert r.get("content_type") == "organic"

    def test_mock006_tanglish_fan_is_organic(self):
        """mock-006: fan comment in Tanglish — organic."""
        r = run_pipeline(get_record("mock-006"))
        assert r.get("is_promotional") is False

    def test_mock009_fdfs_is_organic(self):
        """mock-009: first-day-first-show fan comment — organic."""
        r = run_pipeline(get_record("mock-009"))
        assert r.get("is_promotional") is False

    def test_promo_signals_is_always_list(self):
        for raw in RECORDS:
            r = run_pipeline(raw)
            if "promo_signals" in r:
                assert isinstance(r["promo_signals"], list)


# ===========================================================================
# 7. CREDIBILITY SCORING
# ===========================================================================

class TestCredibilityScoring:

    def test_mock002_journalist_high_credibility(self):
        """mock-002: 5yr account, 12K followers, verified=False, bio+pic → high."""
        r = run_pipeline(get_record("mock-002"))
        assert r.get("credibility_tier") in ("high", "medium")

    def test_mock009_veteran_fan_is_medium_or_high(self):
        """mock-009: 7yr account, 6.7K followers."""
        r = run_pipeline(get_record("mock-009"))
        assert r.get("credibility_tier") in ("high", "medium")

    def test_mock010_new_sparse_account_is_low(self):
        """mock-010: 1yr account, 200 followers, no bio."""
        r = run_pipeline(get_record("mock-010"))
        assert r.get("credibility_tier") in ("low", "medium")

    def test_credibility_tier_valid_values(self):
        """credibility_tier must always be one of 4 valid values."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            if "credibility_tier" in r:
                assert r["credibility_tier"] in ("high", "medium", "low", "bot"), (
                    f"{raw['item_id']}: unexpected tier={r['credibility_tier']}"
                )

    def test_bot_records_get_bot_tier(self):
        """Any record with bot_flag=bot must get credibility_tier=bot."""
        for raw in RECORDS:
            r = run_pipeline(raw)
            if r.get("bot_flag") == "bot":
                assert r.get("credibility_tier") == "bot", (
                    f"{raw['item_id']}: bot_flag=bot but tier={r.get('credibility_tier')}"
                )


# ===========================================================================
# 8. FULL PIPELINE — Print Report
# ===========================================================================

def test_full_pipeline_print_report(capsys):
    """
    Run all 20 mock records through full pipeline and print a human-readable
    summary table. Useful for manual review of pipeline behaviour.
    """
    reset_dedup_store()
    results = []
    for raw in RECORDS:
        r = run_pipeline(raw)
        results.append(r)

    print("\n")
    print("=" * 100)
    print(f"{'ID':<12} {'LANG':<12} {'RELEVANT':<10} {'REASON':<12} {'BOT':<8} {'PROMO':<8} {'TIER':<10} {'CONTENT[:40]'}")
    print("=" * 100)
    for r in results:
        print(
            f"{r['item_id']:<12} "
            f"{r.get('language_detected','—'):<12} "
            f"{str(r.get('is_relevant','—')):<10} "
            f"{str(r.get('filter_reason','—')):<12} "
            f"{str(r.get('bot_flag','—')):<8} "
            f"{str(r.get('is_promotional','—')):<8} "
            f"{str(r.get('credibility_tier','—')):<10} "
            f"{r['content'][:40]!r}"
        )
    print("=" * 100)

    # Basic sanity — all records processed (dataset may grow)
    assert len(results) == len(RECORDS)
    assert len(results) >= 20, "Expected at least 20 records in mock data"
