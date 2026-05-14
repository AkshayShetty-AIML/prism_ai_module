"""
PRISM — Crisis Advisory Generator (Eng C · Flow 3 On-Demand)

Generates a plain-text crisis management advisory from a free-form description.

Signature (from Eng A):
    async def generate_crisis_advisory(crisis_description: str, keyword: str | None) -> str
"""

import logging
from typing import Optional

from prompts.crisis_prompt import build_crisis_prompt
from llm.llm_client import call_llm

logger = logging.getLogger("prism.pipeline.crisis")


async def generate_crisis_advisory(
    crisis_description: str,
    keyword: Optional[str] = None,
) -> str:
    """
    Flow 3: Generate a crisis management advisory from a free-form description.

    Parameters
    ----------
    crisis_description : str
        Free-form description of the crisis situation from the dashboard.
    keyword : str | None
        Optional tracked keyword for context (e.g. "Leo movie").

    Returns
    -------
    str
        Plain-text advisory covering immediate actions, 24-hour plan,
        common mistakes, and suggested statement tone. Returns an error
        message string on failure (never raises).
    """
    if not crisis_description or not crisis_description.strip():
        return "Error: crisis description is required."

    system, prompt = build_crisis_prompt(
        crisis_description=crisis_description,
        keyword=keyword,
    )

    result = call_llm(prompt, system=system, use_cache=False)

    if not result:
        logger.error("LLM failed to generate crisis advisory")
        return "Error: the AI service is temporarily unavailable. Please try again."

    advisory = result.get("advisory", "")
    if not advisory or not advisory.strip():
        logger.warning("LLM returned empty advisory field: %s", result)
        return "Error: received an empty response from the AI. Please try again."

    logger.info("Crisis advisory generated (%d chars)", len(advisory))
    return advisory.strip()
