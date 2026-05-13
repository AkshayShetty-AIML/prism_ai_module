"""
PRISM — Gemini LLM Client (Eng C)

Wrapper around Google's Generative AI SDK for calling Gemini models.
Handles JSON parsing, markdown fence stripping, and retry logic with
exponential backoff.

Provider: Google Gemini 1.5/2.0 Flash (primary production LLM)
"""

import json
import logging
import os
import time
from typing import Optional

import google.generativeai as genai

from .utils import strip_markdown_fences

logger = logging.getLogger("prism.llm.gemini")


def call_gemini(system: str, prompt: str) -> Optional[dict]:
    """
    Call the Google Gemini API and return a parsed JSON dict.

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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set")
        return None

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system,
        )
    except Exception as e:
        logger.error("Failed to initialize Gemini model '%s': %s", model_name, e)
        return None

    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
            raw_text = response.text
            cleaned = strip_markdown_fences(raw_text)
            parsed = json.loads(cleaned)
            logger.debug("Gemini call succeeded on attempt %d", attempt + 1)
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(
                "Gemini returned non-JSON on attempt %d/3: %s", attempt + 1, e
            )
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.warning(
                "Gemini API error on attempt %d/3: %s", attempt + 1, e
            )
            time.sleep(2 ** attempt)

    logger.error("Gemini: all 3 retries exhausted for model '%s'", model_name)
    return None
