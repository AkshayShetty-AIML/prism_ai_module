"""
test_tanglish_normalizer.py — Unit tests for Phase 1b: Text Normalization
Engineer B | PRISM AI Processing Pipeline

8 tests covering: slang replacement, repeated chars, emoji removal,
URL stripping, multi-word phrases, empty input, process() wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.tanglish_normalizer import normalise, process


# ---------------------------------------------------------------------------
# Test 1: Single Tanglish word replacement
# ---------------------------------------------------------------------------
def test_single_slang_replaced():
    result = normalise("semma performance by Vijay!")
    assert "excellent" in result.lower()
    assert "semma" not in result.lower()


# ---------------------------------------------------------------------------
# Test 2: Multiple slang words in one sentence
# ---------------------------------------------------------------------------
def test_multiple_slang_replaced():
    result = normalise("Nalla padam, BGM was mass iruku!")
    # "nalla" → "good", "padam" → "film", "mass" → "impressive"
    assert "good" in result.lower()
    assert "film" in result.lower()
    assert "impressive" in result.lower()


# ---------------------------------------------------------------------------
# Test 3: Multi-word phrase replacement (e.g., "vera level")
# ---------------------------------------------------------------------------
def test_multiword_phrase_replaced():
    result = normalise("This scene was vera level da!")
    assert "outstanding" in result.lower()
    # The individual word "vera" should not appear if phrase matched
    assert "vera level" not in result.lower()


# ---------------------------------------------------------------------------
# Test 4: Repeated characters collapsed
# ---------------------------------------------------------------------------
def test_repeated_chars_collapsed():
    # "Semmmaaaaaa" should collapse to at most 2 consecutive chars
    result = normalise("Semmmaaaaaa padam!")
    # No run of 3+ identical characters should remain
    import re
    assert not re.search(r"(.)\1{2,}", result), f"Repeated chars remain in: {result!r}"


# ---------------------------------------------------------------------------
# Test 5: Emojis are removed
# ---------------------------------------------------------------------------
def test_emojis_removed():
    result = normalise("Semma padam 🔥🔥🔥 vera level 💥")
    assert "🔥" not in result
    assert "💥" not in result
    # Core content should still be there (as normalised form)
    assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Test 6: URLs are stripped
# ---------------------------------------------------------------------------
def test_url_stripped():
    result = normalise("Watch Leo here https://youtube.com/watch?v=abc123 semma!")
    assert "https://" not in result
    assert "youtube.com" not in result


# ---------------------------------------------------------------------------
# Test 7: Empty string returns unchanged
# ---------------------------------------------------------------------------
def test_empty_string():
    result = normalise("")
    assert result == ""


# ---------------------------------------------------------------------------
# Test 8: process() wrapper adds normalised_text field to record
# ---------------------------------------------------------------------------
def test_process_wrapper():
    record = {
        "item_id": "test-001",
        "content": "Semma padam vera level! BGM mokkai illa. Mass performance!",
        "keyword": "Leo movie",
        "language_detected": "tanglish",
    }
    result = process(record)
    assert result is record  # in-place enrichment
    assert "normalised_text" in result
    normalised = result["normalised_text"]
    assert isinstance(normalised, str)
    assert len(normalised) > 0
    # Slang should be replaced in normalised_text
    assert "excellent" in normalised.lower() or "outstanding" in normalised.lower() or "impressive" in normalised.lower()
    # Original content should remain unchanged
    assert result["content"] == "Semma padam vera level! BGM mokkai illa. Mass performance!"
