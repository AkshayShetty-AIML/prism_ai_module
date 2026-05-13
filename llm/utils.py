"""
PRISM — LLM Utility Functions (Eng C)

Shared helper functions used across all LLM provider clients.
Centralised here to avoid code duplication (DRY principle).
"""

import re
import logging

logger = logging.getLogger("prism.llm")


def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM responses.

    LLMs frequently wrap JSON output in ```json ... ``` blocks despite
    explicit instructions not to. This function strips those fences to
    allow clean JSON parsing.

    Handles formats:
        ```json\n{...}\n```
        ```\n{...}\n```
        Mixed whitespace variants

    Parameters
    ----------
    text : str
        Raw text response from the LLM.

    Returns
    -------
    str
        Cleaned text with markdown fences removed.
    """
    if not text:
        return ""

    text = text.strip()

    # Pattern 1: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence line
        text = "\n".join(lines).strip()

    # Pattern 2: Stray backticks at start/end (edge case with some models)
    text = text.strip("`").strip()

    return text
