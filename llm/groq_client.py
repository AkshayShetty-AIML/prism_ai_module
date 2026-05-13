"""
PRISM — Groq LLM Client (Eng C)

Wrapper around the Groq SDK for calling Llama-3 models.
Handles JSON parsing, markdown fence stripping, and retry logic with
exponential backoff.

Provider: Groq / Llama-3 70B (dev/testing fallback — free tier)
"""

import json
import logging
import os
import time

from groq import Groq

from .utils import strip_markdown_fences

logger = logging.getLogger("prism.llm.groq")


def call_groq(system: str, prompt: str) -> dict | None:
    """
    Call the Groq API (Llama-3) and return a parsed JSON dict.

    Retries up to 3 times with exponential backoff on transient failures
    (JSON parse errors, rate limits, network timeouts).

    Parameters
    ----------
    system : str
        The system instruction / role prompt.
    prompt : str
        The user prompt containing the text to classify.

    Returns
    -------
    dict | None
        Parsed JSON response from the LLM, or None if all retries fail.
        Never raises exceptions — returns None on any unrecoverable error.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set")
        return None

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            raw_text = response.choices[0].message.content
            cleaned = strip_markdown_fences(raw_text)
            parsed = json.loads(cleaned)
            logger.debug("Groq call succeeded on attempt %d", attempt + 1)
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(
                "Groq returned non-JSON on attempt %d/3: %s", attempt + 1, e
            )
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.warning(
                "Groq API error on attempt %d/3: %s", attempt + 1, e
            )
            time.sleep(2 ** attempt)

    logger.error("Groq: all 3 retries exhausted for model '%s'", model)
    return None
