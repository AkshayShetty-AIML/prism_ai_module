"""
PRISM — Crisis Prompt Template (Eng C)

Prompt for Flow 3 (On-Demand Crisis Advisory). Builds a crisis management
advisory prompt from a free-form crisis description.

Output: Plain text advisory (3-5 paragraphs).
"""

import os

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "1.0")

CRISIS_SYSTEM = (
    "You are a senior PR and crisis communications strategist specialising in "
    "Indian film and entertainment. Your advisories are calm, structured, and "
    "actionable. You respond ONLY in valid JSON. No markdown fences, no preamble."
)

CRISIS_TEMPLATE = """A crisis situation has emerged{keyword_context}. Here is the description:

{crisis_description}

Write a crisis management advisory covering exactly these four sections as flowing paragraphs:
1. Immediate actions (next 2-4 hours)
2. 24-hour communications plan
3. Common mistakes to avoid in this type of crisis
4. Suggested tone and framing for public statements

Be specific to the situation. Do not use generic platitudes.

Return ONLY this JSON:
{{
  "advisory": "full advisory text as a single string with the four sections written as prose paragraphs"
}}"""


def build_crisis_prompt(
    crisis_description: str,
    keyword: str | None = None,
) -> tuple[str, str]:
    """
    Build the system and user prompt for crisis advisory generation.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    keyword_context = f" related to '{keyword}'" if keyword else ""
    user_prompt = CRISIS_TEMPLATE.format(
        keyword_context=keyword_context,
        crisis_description=crisis_description.strip(),
    )
    return CRISIS_SYSTEM, user_prompt
