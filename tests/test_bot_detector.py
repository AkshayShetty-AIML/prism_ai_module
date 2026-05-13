"""
test_bot_detector.py — Unit tests for Phase 2a: Bot/Human Classification
Engineer B | PRISM AI Processing Pipeline

Tests for: synthetic bot (age + frequency), verified human override,
each signal independently, missing/null author, composite rule,
protected account discount, classify_bot() pipeline wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone, timedelta
from pipeline.bot_detector import detect_bot, classify_bot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_ago(n: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.isoformat()


def _base_author(**overrides) -> dict:
    """Returns a clean 'human' baseline author, overrides applied."""
    author = {
        "author_id": "test-user",
        "username": "genuine_user",
        "account_created_at": _days_ago(400),   # old account
        "verified": False,
        "follower_count": 1500,
        "following_count": 200,
        "post_count": 300,
        "profile_picture_present": True,
        "bio_present": True,
        "account_protected": False,
    }
    author.update(overrides)
    return author


# ---------------------------------------------------------------------------
# Test 1: Synthetic bot — very new + extreme frequency → "bot"
# ---------------------------------------------------------------------------
def test_synthetic_bot_new_account_extreme_frequency():
    author = _base_author(
        account_created_at=_days_ago(5),   # 5 days old
        post_count=600,                    # 600/5 = 120 posts/day
        profile_picture_present=False,
        bio_present=False,
    )
    result = detect_bot(author)
    assert result["bot_flag"] == "bot"
    assert result["bot_confidence"] >= 0.65
    assert "very_new_account" in result["bot_flags"] or "new_account" in result["bot_flags"]
    assert "extreme_frequency" in result["bot_flags"]


# ---------------------------------------------------------------------------
# Test 2: Verified account → always "human", confidence 0.0, no flags
# ---------------------------------------------------------------------------
def test_verified_account_always_human():
    author = _base_author(
        verified=True,
        account_created_at=_days_ago(2),    # would normally be suspicious
        post_count=5000,
    )
    result = detect_bot(author)
    assert result["bot_flag"] == "human"
    assert result["bot_confidence"] == 0.0
    assert result["bot_flags"] == []


# ---------------------------------------------------------------------------
# Test 3: Null author → human with no_author_data flag
# ---------------------------------------------------------------------------
def test_null_author_returns_human():
    result = detect_bot(None)
    assert result["bot_flag"] == "human"
    assert result["bot_confidence"] == 0.0
    assert "no_author_data" in result["bot_flags"]


# ---------------------------------------------------------------------------
# Test 4: Old account, normal activity → "human"
# ---------------------------------------------------------------------------
def test_genuine_old_account_human():
    author = _base_author(
        account_created_at=_days_ago(730),  # 2 years old
        post_count=400,                     # ~0.5 posts/day
        follower_count=3000,
        following_count=400,
    )
    result = detect_bot(author)
    assert result["bot_flag"] == "human"
    assert result["bot_confidence"] < 0.65


# ---------------------------------------------------------------------------
# Test 5: Suspicious follower ratio signal fires
# ---------------------------------------------------------------------------
def test_suspicious_follower_ratio_signal():
    author = _base_author(
        follower_count=10,
        following_count=5000,           # ratio = 10/5000 = 0.002 < 0.05
        account_created_at=_days_ago(200),
        post_count=100,
    )
    result = detect_bot(author)
    assert "suspicious_follower_ratio" in result["bot_flags"]


# ---------------------------------------------------------------------------
# Test 6: No profile picture + no bio → signals fire
# ---------------------------------------------------------------------------
def test_no_profile_pic_and_no_bio_signals():
    author = _base_author(
        profile_picture_present=False,
        bio_present=False,
    )
    result = detect_bot(author)
    assert "no_profile_pic" in result["bot_flags"]
    assert "no_bio" in result["bot_flags"]


# ---------------------------------------------------------------------------
# Test 7: Protected account reduces score
# ---------------------------------------------------------------------------
def test_protected_account_reduces_score():
    # Create a borderline author and check protected reduces score
    author_unprotected = _base_author(
        account_created_at=_days_ago(25),   # new account → +0.20
        post_count=100,                     # 4 posts/day → no frequency signal
        profile_picture_present=False,      # +0.08
        bio_present=False,                  # +0.07
        account_protected=False,
    )
    author_protected = dict(author_unprotected)
    author_protected["account_protected"] = True

    result_unprotected = detect_bot(author_unprotected)
    result_protected = detect_bot(author_protected)

    # Protected should score lower
    assert result_protected["bot_confidence"] < result_unprotected["bot_confidence"]


# ---------------------------------------------------------------------------
# Test 8: Composite rule — age < 30 AND > 100 posts/day → score >= 0.70
# ---------------------------------------------------------------------------
def test_composite_rule_minimum_score():
    author = _base_author(
        account_created_at=_days_ago(15),   # 15 days old → "new_account"
        post_count=2000,                    # 2000/15 ≈ 133 posts/day → "extreme_frequency"
        verified=False,
    )
    result = detect_bot(author)
    assert result["bot_confidence"] >= 0.70
    assert result["bot_flag"] == "bot"
    assert "new_account_high_frequency" in result["bot_flags"]


# ---------------------------------------------------------------------------
# Test 9: classify_bot() pipeline wrapper enriches record
# ---------------------------------------------------------------------------
def test_classify_bot_pipeline_wrapper():
    record = {
        "item_id": "test-001",
        "content": "This is a test comment about Leo movie",
        "author": _base_author(
            account_created_at=_days_ago(3),
            post_count=450,
            profile_picture_present=False,
            bio_present=False,
        ),
    }
    result = classify_bot(record)
    assert result is record  # in-place enrichment
    assert "bot_flag" in result
    assert "bot_confidence" in result
    assert "bot_flags" in result
    assert result["bot_flag"] in ("bot", "human")
    assert 0.0 <= result["bot_confidence"] <= 1.0
    assert isinstance(result["bot_flags"], list)
