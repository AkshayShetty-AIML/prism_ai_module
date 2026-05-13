"""
test_credibility.py — Unit tests for Phase 2c: Credibility Tier Scoring
Engineer B | PRISM AI Processing Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone, timedelta
from pipeline.credibility_scorer import score_credibility, assess_credibility


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _author(**overrides) -> dict:
    base = {
        "account_created_at": _days_ago(500),
        "verified": False,
        "follower_count": 2000,
        "following_count": 300,
        "post_count": 400,
        "profile_picture_present": True,
        "bio_present": True,
    }
    base.update(overrides)
    return base


# Test 1: Bot → always "bot" tier
def test_bot_always_returns_bot_tier():
    author = _author(follower_count=100000, verified=True)
    result = score_credibility(author, bot_flag="bot")
    assert result == "bot"


# Test 2: Null author → "low"
def test_null_author_returns_low():
    result = score_credibility(None, bot_flag="human")
    assert result == "low"


# Test 3: Verified + old + high followers → "high"
def test_high_credibility_verified_user():
    author = _author(
        account_created_at=_days_ago(800),
        verified=True,
        follower_count=50000,
        profile_picture_present=True,
        bio_present=True,
    )
    result = score_credibility(author, bot_flag="human")
    assert result == "high"


# Test 4: New account, low followers, no bio → "low"
def test_low_credibility_new_sparse_account():
    author = _author(
        account_created_at=_days_ago(10),
        follower_count=5,
        following_count=50,
        profile_picture_present=False,
        bio_present=False,
        verified=False,
    )
    result = score_credibility(author, bot_flag="human")
    assert result == "low"


# Test 5: Mid-tier account → "medium"
def test_medium_credibility_account():
    author = _author(
        account_created_at=_days_ago(200),   # > 180 days → +2
        follower_count=500,                   # > 100 → +1
        profile_picture_present=True,         # +1
        bio_present=True,                     # +1
        verified=False,
    )
    # Total = 2+1+1+1 = 5 → medium
    result = score_credibility(author, bot_flag="human")
    assert result == "medium"


# Test 6: assess_credibility() wrapper enriches record
def test_assess_credibility_pipeline_wrapper():
    record = {
        "item_id": "test-001",
        "content": "Great movie!",
        "bot_flag": "human",
        "author": _author(
            account_created_at=_days_ago(400),
            follower_count=5000,
            verified=False,
        ),
        "engagement": {"likes": 100, "replies": 20, "shares": 5, "views": 3000},
    }
    result = assess_credibility(record)
    assert result is record
    assert "credibility_tier" in result
    assert result["credibility_tier"] in ("high", "medium", "low", "bot")


# Test 7: Bot flag in record → assess_credibility returns "bot" tier
def test_assess_credibility_bot_record():
    record = {
        "item_id": "test-002",
        "content": "Spam comment",
        "bot_flag": "bot",
        "author": _author(follower_count=100000),
        "engagement": {"likes": 0, "replies": 0, "shares": 0, "views": 0},
    }
    result = assess_credibility(record)
    assert result["credibility_tier"] == "bot"


# Test 8: Engagement snapshot does not persist on author dict
def test_engagement_snapshot_not_leaked():
    author = _author()
    record = {
        "item_id": "test-003",
        "content": "Nice film",
        "bot_flag": "human",
        "author": author,
        "engagement": {"likes": 50, "replies": 10, "shares": 2, "views": 1000},
    }
    assess_credibility(record)
    # The temp key must be cleaned up
    assert "_engagement_snapshot" not in author
