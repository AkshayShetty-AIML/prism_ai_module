"""
PRISM — Engagement Prompt Template (Eng C)

Prompt for Flow 2 (On-Demand Comment Engagement). Builds multi-tone
reply prompts by fetching comments from DB2 for a given theme group.

Output: JSON with theme_summary, suggested_replies (3-5), confidence_note.
"""

import logging
import os

logger = logging.getLogger("prism.prompts.engagement")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "1.0")

ENGAGEMENT_SYSTEM = (
    "You are a professional social media manager for an Indian film "
    "production and PR company. Your job is to draft thoughtful, "
    "platform-appropriate replies to clusters of audience comments. "
    "You respond ONLY in valid JSON. No markdown fences, no preamble."
)

ENGAGEMENT_TEMPLATE = """Generate multiple reply drafts for the following cluster of
comments about: {keyword}

CLUSTER THEME: {theme_summary}
PLATFORMS: {platforms}

SAMPLE COMMENTS FROM THIS CLUSTER:
{sample_comments}

INSTRUCTIONS:
- Generate exactly 3-5 draft replies, each in a DIFFERENT tone
- Available tones: empathetic, professional, apologetic, informative, clarifying
- Each reply must directly address the core concern in the cluster
- Twitter replies: max 280 characters. YouTube/Reddit: max 500 characters.
- Never argue with users, never be defensive, never use corporate jargon
- If the cluster is Tanglish/Hindi, at least one reply should mirror that register
- suitable_for should list only platforms where that tone/length works

Return ONLY this JSON:
{{
  "theme_summary": "short description of what the cluster is broadly about",
  "suggested_replies": [
    {{
      "reply_id": "r1",
      "tone": "empathetic",
      "text": "the actual reply text",
      "suitable_for": ["twitter", "youtube"],
      "target_intent": "what concern this reply addresses"
    }}
  ],
  "confidence_note": "any caveats e.g. limited sample size or ambiguous theme"
}}"""


def build_engagement_prompt(
    theme_group_id: str, keyword: str, db
) -> tuple[str, str]:
    """
    Fetch comments for a theme group from DB2 and construct the prompt.

    Parameters
    ----------
    theme_group_id : str
        The cluster UUID to fetch comments for.
    keyword : str
        The tracked keyword/entity.
    db : object
        MongoDB database object (or mock). Must support
        db.analyzed_records.find({...}).limit(n).

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt). Returns ("", "") if no comments found.
    """
    cursor = db.analyzed_records.find(
        {"crisis_theme_group": theme_group_id},
        {"content": 1, "platform": 1, "sentiment": 1},
    ).limit(10)

    records_list = list(cursor)

    if not records_list:
        logger.warning("No comments found in DB for theme_group_id: %s",
                       theme_group_id)
        return "", ""

    sample_comments = "\n".join(
        f"[{r.get('platform', 'unknown').upper()}] {r.get('content', '')}"
        for r in records_list
    )

    platforms = sorted({r.get("platform", "unknown") for r in records_list})

    user_prompt = ENGAGEMENT_TEMPLATE.format(
        keyword=keyword,
        theme_summary="(derived from comments below)",
        platforms=", ".join(platforms),
        sample_comments=sample_comments,
    )

    logger.debug(
        "Built engagement prompt for theme '%s' with %d comments",
        theme_group_id, len(records_list),
    )

    return ENGAGEMENT_SYSTEM, user_prompt
