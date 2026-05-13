"""
language_processor.py — Phase 1a: Language Detection
Engineer B | PRISM AI Processing Pipeline

Detects language of incoming comment/post text.
Pure function: dict in → enriched dict out. No DB, no API, no LLM calls.

Output fields added to record:
    language_detected   : str  — english|tanglish|hinglish|hindi|tamil|mixed
    language_confidence : float — 0.0–1.0
    has_code_mixing     : bool  — True if multiple languages blended
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

# ---------------------------------------------------------------------------
# Tanglish / code-mixing signal markers
# ---------------------------------------------------------------------------

# Tamil-origin words commonly written in Roman script (transliterated)
TANGLISH_MARKERS: frozenset[str] = frozenset({
    "semma", "vera level", "mass", "nalla", "mokkai", "mokka", "padam",
    "ayyo", "enna", "paartha", "theriyum", "therila", "vandhutten",
    "pochu", "machane", "macha", "dei", "ponga", "da", "thala",
    "thalapathy", "verithanam", "masss", "kuthu", "villan",
    "paatha", "kadha", "fdfs", "engayo", "chinna",
})

# Hindi-origin words commonly written in Roman script
HINGLISH_MARKERS: frozenset[str] = frozenset({
    "bakwaas", "zabardast", "zabrdast", "bekar", "mast", "bindaas",
    "ekdum", "bahut", "achha", "bura", "yaar", "bhai", "wah",
    "khatra", "paisa vasool", "timepass", "kadak", "faltu",
    "jhakaas", "dhamaal", "dhamakedar", "filmy", "lag gaya",
    "sahi hai", "scene nahi",
})

# Unicode ranges for Indic scripts
_TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")       # Tamil script block
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")  # Devanagari (Hindi)
_GENERIC_INDIC = re.compile(r"[\u0900-\u0DFF]")     # Any Indic script


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _count_marker_hits(tokens: list[str], markers: frozenset[str]) -> int:
    """Count how many tokens (or bigrams) match a marker set."""
    hits = 0
    for tok in tokens:
        if tok in markers:
            hits += 1
    # Also check bigrams for multi-word markers like "vera level"
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    for bg in bigrams:
        if bg in markers:
            hits += 1
    return hits


def _has_native_script(text: str) -> Tuple[bool, bool]:
    """
    Returns (has_tamil_script, has_devanagari_script).
    Native script = text typed in the actual Unicode block, not romanized.
    """
    has_tamil = bool(_TAMIL_RANGE.search(text))
    has_deva = bool(_DEVANAGARI_RANGE.search(text))
    return has_tamil, has_deva


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_language_detection(text: str) -> dict:
    """
    Detect the language of a social-media post.

    Args:
        text: Raw comment/post text (any length).

    Returns:
        dict with keys:
            language_detected   (str)
            language_confidence (float, 0.0–1.0)
            has_code_mixing     (bool)
    """
    if not text or not text.strip():
        return {
            "language_detected": "english",
            "language_confidence": 0.0,
            "has_code_mixing": False,
        }

    tokens = _tokenize(text)
    has_tamil_script, has_deva_script = _has_native_script(text)

    tanglish_hits = _count_marker_hits(tokens, TANGLISH_MARKERS)
    hinglish_hits = _count_marker_hits(tokens, HINGLISH_MARKERS)

    total_tokens = max(len(tokens), 1)
    tanglish_ratio = tanglish_hits / total_tokens
    hinglish_ratio = hinglish_hits / total_tokens

    # --- Attempt langdetect for base language ---
    base_lang = "en"
    base_conf = 0.8
    try:
        from langdetect import detect_langs  # type: ignore
        results = detect_langs(text)
        if results:
            best = results[0]
            base_lang = best.lang          # e.g. "en", "hi", "ta"
            base_conf = round(float(best.prob), 3)
    except Exception:
        # langdetect unavailable or text too short — fall back gracefully
        pass

    # --- Decision logic ---
    language_detected: str
    language_confidence: float
    has_code_mixing: bool

    # Native Tamil script → "tamil"
    if has_tamil_script and not has_deva_script:
        if tanglish_hits > 0:
            language_detected = "mixed"
            language_confidence = 0.75
            has_code_mixing = True
        else:
            language_detected = "tamil"
            language_confidence = 0.92
            has_code_mixing = False

    # Native Devanagari → "hindi"
    elif has_deva_script and not has_tamil_script:
        if hinglish_hits > 0:
            language_detected = "mixed"
            language_confidence = 0.75
            has_code_mixing = True
        else:
            language_detected = "hindi"
            language_confidence = 0.92
            has_code_mixing = False

    # Both scripts → "mixed"
    elif has_tamil_script and has_deva_script:
        language_detected = "mixed"
        language_confidence = 0.85
        has_code_mixing = True

    # Romanized Tanglish (no native script)
    elif tanglish_ratio >= 0.15:
        code_mix = hinglish_hits > 0
        language_detected = "mixed" if code_mix else "tanglish"
        language_confidence = min(0.60 + tanglish_ratio, 0.95)
        has_code_mixing = code_mix

    # Romanized Hinglish (no native script)
    elif hinglish_ratio >= 0.15:
        code_mix = tanglish_hits > 0
        language_detected = "mixed" if code_mix else "hinglish"
        language_confidence = min(0.60 + hinglish_ratio, 0.95)
        has_code_mixing = code_mix

    # langdetect says Tamil → tag as "tamil"
    elif base_lang == "ta":
        language_detected = "tamil"
        language_confidence = base_conf
        has_code_mixing = tanglish_hits > 0

    # langdetect says Hindi → tag as "hindi"
    elif base_lang == "hi":
        language_detected = "hindi"
        language_confidence = base_conf
        has_code_mixing = hinglish_hits > 0

    # Weak Tanglish signals (1 hit in short text)
    elif tanglish_hits >= 1 and total_tokens <= 10:
        language_detected = "tanglish"
        language_confidence = 0.55
        has_code_mixing = hinglish_hits > 0

    # Default: English
    else:
        language_detected = "english"
        language_confidence = round(base_conf, 3) if base_lang == "en" else 0.70
        has_code_mixing = (tanglish_hits > 0 or hinglish_hits > 0)

    return {
        "language_detected": language_detected,
        "language_confidence": round(language_confidence, 3),
        "has_code_mixing": has_code_mixing,
    }


def process(record: dict) -> dict:
    """
    Pipeline entry-point for Phase 1a.
    Mutates record in-place with language detection fields, then returns it.

    Args:
        record: Incoming item dict (must contain 'content' key).

    Returns:
        Same dict enriched with language_detected, language_confidence,
        has_code_mixing fields.
    """
    content = record.get("content", "")
    lang_result = _run_language_detection(content)
    record.update(lang_result)
    return record


# ---------------------------------------------------------------------------
# Eng A interface contract — Phase 1a public entry-point
# Accepts EITHER a plain str (unit tests) OR a record dict (pipeline).
# pipeline_runner.py: record = detect_language(record)
# unit tests:         result = detect_language("some text")
# ---------------------------------------------------------------------------
def detect_language(text_or_record):  # type: ignore[misc]
    """
    Public entry-point (Eng A contract, Phase 1a).

    Accepts:
        str  → runs language detection, returns result dict (unit-test mode)
        dict → enriches the record dict with language fields (pipeline mode)
    """
    if isinstance(text_or_record, dict):
        return process(text_or_record)
    # Plain string — return detection result dict directly (used by unit tests)
    return _run_language_detection(text_or_record)
