"""
test_language_processor.py — Unit tests for Phase 1a: Language Detection
Engineer B | PRISM AI Processing Pipeline

10 tests covering: English, Tanglish, Hinglish, Hindi script, Tamil script,
mixed, empty input, code-mixing flag, the process() pipeline wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.language_processor import detect_language, process


# ---------------------------------------------------------------------------
# Test 1: Clear English text → "english"
# ---------------------------------------------------------------------------
def test_english_plain():
    result = detect_language("Leo movie was absolutely fantastic! The screenplay and BGM were outstanding.")
    assert result["language_detected"] == "english"
    assert result["language_confidence"] > 0.5
    assert isinstance(result["has_code_mixing"], bool)


# ---------------------------------------------------------------------------
# Test 2: Pure Tanglish (romanized Tamil) → "tanglish"
# ---------------------------------------------------------------------------
def test_tanglish_detection():
    result = detect_language("Semma padam da! BGM vera level iruku. Thalapathy performance mass.")
    assert result["language_detected"] in ("tanglish", "mixed")
    assert result["language_confidence"] > 0.5


# ---------------------------------------------------------------------------
# Test 3: Tanglish with multiple strong markers → high confidence
# ---------------------------------------------------------------------------
def test_tanglish_high_confidence():
    result = detect_language("Nalla padam vera level. Mokkai illa, mass action scenes, ayyo what a film!")
    assert result["language_detected"] in ("tanglish", "mixed")
    assert result["language_confidence"] >= 0.55


# ---------------------------------------------------------------------------
# Test 4: Pure Hinglish (romanized Hindi) → "hinglish"
# ---------------------------------------------------------------------------
def test_hinglish_detection():
    result = detect_language("Yaar ekdum mast film hai! Zabardast action, bindaas screenplay. Paisa vasool!")
    assert result["language_detected"] in ("hinglish", "mixed")
    assert result["language_confidence"] > 0.5


# ---------------------------------------------------------------------------
# Test 5: Native Devanagari (Hindi) script → "hindi"
# ---------------------------------------------------------------------------
def test_hindi_script_detection():
    result = detect_language("लियो फिल्म बहुत अच्छी लगी। विजय जी का अभिनय शानदार था।")
    assert result["language_detected"] in ("hindi", "mixed")


# ---------------------------------------------------------------------------
# Test 6: Native Tamil Unicode script → "tamil"
# ---------------------------------------------------------------------------
def test_tamil_script_detection():
    result = detect_language("லியோ படம் மிகவும் சிறப்பாக இருந்தது")
    assert result["language_detected"] in ("tamil", "mixed")


# ---------------------------------------------------------------------------
# Test 7: Code-mixing (Tanglish + English) → has_code_mixing may be True
# ---------------------------------------------------------------------------
def test_code_mixing_english_tanglish():
    result = detect_language("The movie was semma and the BGM vera level. Amazing direction!")
    # has_code_mixing should be True OR it's classified as tanglish
    assert result["language_detected"] in ("english", "tanglish", "mixed")
    # Either way, result dict must have the required keys
    assert "has_code_mixing" in result
    assert "language_confidence" in result


# ---------------------------------------------------------------------------
# Test 8: Empty string → default to "english" with confidence 0.0
# ---------------------------------------------------------------------------
def test_empty_text():
    result = detect_language("")
    assert result["language_detected"] == "english"
    assert result["language_confidence"] == 0.0
    assert result["has_code_mixing"] is False


# ---------------------------------------------------------------------------
# Test 9: Whitespace-only string → same as empty
# ---------------------------------------------------------------------------
def test_whitespace_only():
    result = detect_language("   \n\t  ")
    assert result["language_detected"] == "english"
    assert result["language_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 10: process() wrapper enriches record dict correctly
# ---------------------------------------------------------------------------
def test_process_wrapper_enriches_record():
    record = {
        "item_id": "test-001",
        "content": "Semma padam vera level mokkai illa!",
        "keyword": "Leo movie",
    }
    result = process(record)
    # process() must return the same dict object (in-place enrichment)
    assert result is record
    # All three output fields must be present
    assert "language_detected" in result
    assert "language_confidence" in result
    assert "has_code_mixing" in result
    # Values must have correct types
    assert isinstance(result["language_detected"], str)
    assert isinstance(result["language_confidence"], float)
    assert isinstance(result["has_code_mixing"], bool)
    # Language must be one of the valid values
    assert result["language_detected"] in (
        "english", "tanglish", "hinglish", "hindi", "tamil", "mixed"
    )
