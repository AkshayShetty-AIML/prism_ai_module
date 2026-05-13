"""
llm/llm_cache.py — In-memory LLM response cache using MD5 hashing.

Prevents duplicate LLM API calls for identical prompts.
Cache lives in memory only — resets on server restart.

Usage:
    from llm.llm_cache import get_cached, set_cached, build_cache_key

    key = build_cache_key(system_prompt, user_prompt)
    cached = get_cached(key)
    if cached:
        return cached

    result = call_llm(...)
    set_cached(key, result)
"""

import hashlib

# ── In-memory cache store ───────────────────────────────────────────
_cache: dict[str, dict] = {}


def build_cache_key(system: str, prompt: str) -> str:
    """
    Generate an MD5 hash from system prompt + user prompt.
    Same input always produces the same key.
    """
    combined = (system + prompt).encode("utf-8")
    return hashlib.md5(combined).hexdigest()


def get_cached(key: str) -> dict | None:
    """
    Look up a cache key. Returns the stored LLM result dict if found,
    None if not cached.
    """
    return _cache.get(key)


def set_cached(key: str, result: dict) -> None:
    """
    Store an LLM result dict under the given cache key.
    """
    _cache[key] = result


def clear_cache() -> None:
    """Wipe the entire cache. Useful for testing."""
    _cache.clear()


def cache_stats() -> dict:
    """Return basic cache stats for debugging/logging."""
    return {
        "total_entries": len(_cache),
    }