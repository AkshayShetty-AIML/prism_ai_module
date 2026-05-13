"""
PRISM — LLM Client Router (Eng C)

Central entry point for all LLM calls in the PRISM pipeline.
Routes to the correct provider (Gemini, Claude, Groq), handles
automatic fallback on failure, and implements MD5-based response caching
to avoid duplicate API calls.

Usage:
    from llm.llm_client import call_llm
    result = call_llm(prompt, system="...", use_cache=True)
"""

import hashlib
import logging
import os

from .gemini_client import call_gemini
from .groq_client import call_groq

logger = logging.getLogger("prism.llm.router")

# ──────────────────────────────────────────────────────────────────────
# In-memory response cache (MD5 hash of system+prompt → parsed dict)
# Prevents duplicate LLM calls for identical text within the same
# process lifetime. Cleared on restart.
# ──────────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}


def get_cache_stats() -> dict:
    """Return cache statistics for monitoring/debugging."""
    return {"cache_size": len(_cache)}


def clear_cache() -> None:
    """Clear the response cache. Useful for testing."""
    _cache.clear()
    logger.info("LLM response cache cleared")


def call_llm(
    prompt: str,
    system: str = "",
    use_cache: bool = True,
) -> dict | None:
    """
    Route an LLM call to the appropriate provider with fallback and caching.

    Provider priority (based on LLM_PROVIDER env var):
        - "gemini" (default): Gemini → Groq fallback
        - "groq": Groq only
        - "claude": Claude (placeholder for Day 6)

    Parameters
    ----------
    prompt : str
        The user prompt to send to the LLM.
    system : str
        The system instruction / role prompt.
    use_cache : bool
        If True, check cache before calling and store results after.
        Set to False for on-demand features (engagement, crisis) where
        responses should always be fresh.

    Returns
    -------
    dict | None
        Parsed JSON response from the LLM, or None if all providers
        and retries fail. Never raises exceptions.
    """
    # ── Cache lookup ──
    cache_key = hashlib.md5((system + prompt).encode()).hexdigest()

    if use_cache and cache_key in _cache:
        logger.debug("Cache HIT for key %s", cache_key[:12])
        return _cache[cache_key]

    # ── Provider routing ──
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    result = None
    model_used = None

    if provider == "gemini":
        result = call_gemini(system, prompt)
        model_used = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not result:
            logger.info("Gemini failed, falling back to Groq")
            result = call_groq(system, prompt)
            model_used = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    elif provider == "groq":
        result = call_groq(system, prompt)
        model_used = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    elif provider == "claude":
        # Placeholder — claude_client.py will be implemented on Day 6
        logger.warning("Claude provider selected but not yet implemented")
        pass

    else:
        logger.error("Unknown LLM_PROVIDER: '%s'", provider)

    # ── Cache store ──
    if result and use_cache:
        _cache[cache_key] = result
        logger.debug("Cache STORE for key %s", cache_key[:12])

    if not result:
        logger.error("All LLM providers failed for provider='%s'", provider)

    return result
