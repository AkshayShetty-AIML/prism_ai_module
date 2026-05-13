"""
test_promo_classifier.py — Unit tests for Phase 2b: Promotional Classification
Engineer B | PRISM AI Processing Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.promo_classifier import classify, classify_promo


def test_ad_hashtag_is_promotional():
    result = classify("Watch Leo movie NOW! #ad")
    assert result["is_promotional"] is True
    assert result["content_type"] == "promotional"
    assert "promo_hashtag" in result["promo_signals"]


def test_sponsored_hashtag_is_promotional():
    result = classify("Got tickets for Leo this weekend #sponsored #collab!")
    assert result["is_promotional"] is True
    assert "promo_hashtag" in result["promo_signals"]


def test_cta_book_tickets_is_promotional():
    result = classify("Leo movie releasing Oct 19. Book tickets now at BookMyShow!")
    assert result["is_promotional"] is True
    assert "cta_language" in result["promo_signals"]


def test_cta_link_in_bio_is_promotional():
    result = classify("Get 20% off Leo tickets! Use code LEOFAN — link in bio!")
    assert result["is_promotional"] is True
    assert "cta_language" in result["promo_signals"]


def test_organic_comment_not_promotional():
    result = classify("Semma padam! Leo movie was absolutely amazing.")
    assert result["is_promotional"] is False
    assert result["content_type"] == "organic"
    assert result["promo_signals"] == []


def test_verified_brand_account_is_promotional():
    author = {
        "verified": True,
        "username": "thenandal_studio_official",
        "bio_text": "Thenandal Studio Ltd — Official Entertainment Production House",
    }
    result = classify("Leo movie — the blockbuster of the year!", author)
    assert result["is_promotional"] is True
    assert "verified_brand" in result["promo_signals"]


def test_verified_non_brand_is_organic():
    author = {
        "verified": True,
        "username": "vijay_actor",
        "bio_text": "Actor | Fan of cinema | Chennai",
    }
    result = classify("Grateful for all the love for Leo movie!", author)
    assert result["is_promotional"] is False
    assert result["content_type"] == "organic"


def test_multiple_promo_signals():
    result = classify("Leo is OUT NOW! Book tickets — link in bio! #ad #gifted")
    assert result["is_promotional"] is True
    assert "promo_hashtag" in result["promo_signals"]
    assert "cta_language" in result["promo_signals"]
    assert len(result["promo_signals"]) >= 2


def test_classify_promo_pipeline_wrapper():
    record = {
        "item_id": "test-001",
        "content": "Leo movie #ad Book now!",
        "normalised_text": "Leo movie #ad Book now!",
        "author": {"verified": False, "username": "random_user", "bio_text": "fan"},
    }
    result = classify_promo(record)
    assert result is record
    assert "is_promotional" in result
    assert "content_type" in result
    assert "promo_signals" in result
    assert result["content_type"] in ("promotional", "organic")


def test_no_author_still_classifies():
    result = classify("Watch Leo movie now! #ad Click here for tickets.", author=None)
    assert result["is_promotional"] is True
    assert "promo_hashtag" in result["promo_signals"]
