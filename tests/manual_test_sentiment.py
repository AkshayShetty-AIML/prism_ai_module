"""
PRISM — Manual LLM Test Script (Day 1, Eng C)

Sends 5 diverse sample records through the sentiment prompt to verify
that the LLM returns valid, parseable JSON responses.

Usage:
    # Test with Groq (free tier, default)
    LLM_PROVIDER=groq GROQ_API_KEY=your_key python tests/manual_test_sentiment.py

    # Test with Gemini
    LLM_PROVIDER=gemini GEMINI_API_KEY=your_key python tests/manual_test_sentiment.py

    # Dry run (just print prompts, no API call)
    python tests/manual_test_sentiment.py --dry-run
"""

import json
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from prompts.sentiment_prompt import (
    build_sentiment_prompt,
    VALID_EMOTIONS,
    VALID_SENTIMENTS,
    VALID_CRISIS_SEVERITIES,
)

# ──────────────────────────────────────────────────────────────────────
# 5 diverse test records covering key scenarios
# ──────────────────────────────────────────────────────────────────────
TEST_RECORDS = [
    {
        "item_id": "test-001",
        "keyword": "Leo movie",
        "platform": "youtube",
        "language_detected": "tanglish",
        "content": "Semma padam da! Thalapathy vera level acting 🔥🔥 BGM mass ah iruku",
        "normalised_text": "Excellent film! Thalapathy outstanding acting 🔥🔥 BGM impressive",
        "description": "Clear positive — Tanglish praise",
    },
    {
        "item_id": "test-002",
        "keyword": "Leo movie",
        "platform": "twitter",
        "language_detected": "english",
        "content": "Oh wow what a masterpiece. Didn't fall asleep at all during the second half 🙄",
        "normalised_text": "Oh wow what a masterpiece. Didn't fall asleep at all during the second half 🙄",
        "description": "Sarcastic — surface positive, actually negative",
    },
    {
        "item_id": "test-003",
        "keyword": "Leo movie",
        "platform": "reddit",
        "language_detected": "english",
        "content": "The film releases on October 19th across 5000 screens worldwide. Pre-booking starts tomorrow.",
        "normalised_text": "The film releases on October 19th across 5000 screens worldwide. Pre-booking starts tomorrow.",
        "description": "Neutral — factual announcement",
    },
    {
        "item_id": "test-004",
        "keyword": "Leo movie",
        "platform": "twitter",
        "language_detected": "hinglish",
        "content": "Bakwaas movie hai yaar. Paisa waste. Director ne kya socha tha? Bekar screenplay, bekar acting. #Boycott",
        "normalised_text": "Nonsense movie. Money waste. What was the director thinking? Useless screenplay, useless acting. #Boycott",
        "description": "Strong negative — Hinglish anger with boycott hashtag (crisis signal)",
    },
    {
        "item_id": "test-005",
        "keyword": "Leo movie",
        "platform": "youtube",
        "language_detected": "english",
        "content": "The cinematography is decent but the pacing in the second half drags. Mixed feelings overall.",
        "normalised_text": "The cinematography is decent but the pacing in the second half drags. Mixed feelings overall.",
        "description": "Mixed/Neutral — balanced review",
    },
]


def validate_response(response_dict: dict) -> list[str]:
    """Validate the LLM JSON response against expected schema. Returns list of errors."""
    errors = []

    # Required fields
    required_fields = [
        "sentiment", "positive_score", "neutral_score", "negative_score",
        "confidence", "dominant_emotion", "emotion_tags", "is_sarcastic",
        "crisis_severity", "reasoning",
    ]
    for field in required_fields:
        if field not in response_dict:
            errors.append(f"Missing field: {field}")

    # Sentiment value
    sentiment = response_dict.get("sentiment")
    if sentiment and sentiment not in VALID_SENTIMENTS:
        errors.append(f"Invalid sentiment: {sentiment}")

    # Score sum check (tolerance ±0.05)
    try:
        pos = float(response_dict.get("positive_score", 0))
        neu = float(response_dict.get("neutral_score", 0))
        neg = float(response_dict.get("negative_score", 0))
        score_sum = pos + neu + neg
        if abs(score_sum - 1.0) > 0.05:
            errors.append(f"Scores sum to {score_sum:.3f}, expected ~1.0")
    except (TypeError, ValueError) as e:
        errors.append(f"Score conversion error: {e}")

    # Confidence range
    try:
        conf = float(response_dict.get("confidence", 0))
        if not (0.0 <= conf <= 1.0):
            errors.append(f"Confidence {conf} out of range [0.0, 1.0]")
    except (TypeError, ValueError):
        errors.append("Confidence is not a valid number")

    # Dominant emotion
    dominant = response_dict.get("dominant_emotion")
    if dominant and dominant not in VALID_EMOTIONS:
        errors.append(f"Invalid dominant_emotion: {dominant}")

    # Emotion tags
    tags = response_dict.get("emotion_tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if tag not in VALID_EMOTIONS:
                errors.append(f"Invalid emotion_tag: {tag}")
    else:
        errors.append(f"emotion_tags is not a list: {type(tags)}")

    # Crisis severity
    severity = response_dict.get("crisis_severity")
    if severity and severity not in VALID_CRISIS_SEVERITIES:
        errors.append(f"Invalid crisis_severity: {severity}")

    # Sarcasm type check
    sarcastic = response_dict.get("is_sarcastic")
    if sarcastic is not None and not isinstance(sarcastic, bool):
        errors.append(f"is_sarcastic should be bool, got {type(sarcastic)}")

    return errors


def call_llm_raw(system: str, prompt: str) -> str | None:
    """
    Make a raw LLM call based on the LLM_PROVIDER env var.
    Returns the raw text response or None on failure.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    try:
        if provider == "groq":
            return _call_groq(system, prompt)
        elif provider == "gemini":
            return _call_gemini(system, prompt)
        elif provider == "claude":
            return _call_claude(system, prompt)
        else:
            print(f"  ❌ Unknown provider: {provider}")
            return None
    except Exception as e:
        print(f"  ❌ API call failed: {e}")
        return None


def _call_groq(system: str, prompt: str) -> str | None:
    """Call Groq API (Llama-3 70B)."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  ❌ GROQ_API_KEY not set")
        return None

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    return response.choices[0].message.content


def _call_gemini(system: str, prompt: str) -> str | None:
    """Call Google Gemini API."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ❌ GEMINI_API_KEY not set")
        return None

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system,
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )

    return response.text


def _call_claude(system: str, prompt: str) -> str | None:
    """Call Anthropic Claude API."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ❌ ANTHROPIC_API_KEY not set")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.content[0].text


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


def run_tests(dry_run: bool = False):
    """Run all 5 test records through the sentiment prompt."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    print(f"\n{'='*70}")
    print(f"PRISM Sentiment Prompt — Manual Test (Provider: {provider.upper()})")
    print(f"{'='*70}")

    if dry_run:
        print("\n⚠️  DRY RUN MODE — printing prompts only, no API calls\n")

    passed = 0
    failed = 0
    results = []

    for i, record in enumerate(TEST_RECORDS, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/5: {record['description']}")
        print(f"Text: \"{record['content'][:80]}...\"")
        print(f"{'─'*70}")

        system, prompt = build_sentiment_prompt(record)

        if dry_run:
            print(f"\n[SYSTEM PROMPT]\n{system[:200]}...\n")
            print(f"[USER PROMPT]\n{prompt[:300]}...\n")
            continue

        # Call LLM
        print("  ⏳ Calling LLM...")
        start = time.time()
        raw_response = call_llm_raw(system, prompt)
        elapsed = time.time() - start
        print(f"  ⏱️  Response in {elapsed:.2f}s")

        if raw_response is None:
            print("  ❌ FAILED — No response from LLM")
            failed += 1
            results.append({"record": record["item_id"], "status": "NO_RESPONSE"})
            continue

        # Strip markdown fences and parse JSON
        cleaned = strip_markdown_fences(raw_response)
        print(f"\n  Raw response (cleaned):\n  {cleaned[:500]}\n")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  ❌ FAILED — JSON parse error: {e}")
            failed += 1
            results.append({"record": record["item_id"], "status": "JSON_PARSE_ERROR"})
            continue

        # Validate schema
        errors = validate_response(parsed)
        if errors:
            print(f"  ⚠️  VALIDATION WARNINGS ({len(errors)}):")
            for err in errors:
                print(f"     - {err}")

        # Print parsed result
        print(f"  📊 Result:")
        print(f"     Sentiment:  {parsed.get('sentiment')} (conf: {parsed.get('confidence')})")
        print(f"     Scores:     pos={parsed.get('positive_score')} neu={parsed.get('neutral_score')} neg={parsed.get('negative_score')}")
        print(f"     Emotion:    {parsed.get('dominant_emotion')} | tags={parsed.get('emotion_tags')}")
        print(f"     Sarcastic:  {parsed.get('is_sarcastic')}")
        print(f"     Crisis:     {parsed.get('crisis_severity')}")
        print(f"     Reasoning:  {parsed.get('reasoning')}")

        if not errors:
            print(f"  ✅ PASSED — Valid JSON, schema OK")
            passed += 1
            results.append({"record": record["item_id"], "status": "PASSED"})
        else:
            print(f"  ⚠️  PASSED WITH WARNINGS")
            passed += 1  # Still counts as parsed
            results.append({"record": record["item_id"], "status": "WARNINGS", "errors": errors})

        # Rate limit delay between calls
        if i < len(TEST_RECORDS):
            time.sleep(1)

    if not dry_run:
        print(f"\n{'='*70}")
        print(f"RESULTS: {passed}/{len(TEST_RECORDS)} passed, {failed}/{len(TEST_RECORDS)} failed")
        print(f"Target: ≥4/5 must parse correctly")
        if passed >= 4:
            print("✅ DAY 1 TARGET MET — Prompt produces valid JSON ≥4/5 times")
        else:
            print("❌ DAY 1 TARGET NOT MET — Iterate on prompt template")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_tests(dry_run=dry_run)
