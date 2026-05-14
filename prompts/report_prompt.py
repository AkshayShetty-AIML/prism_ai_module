"""
PRISM — Report Prompt Template (Eng C)

Prompt for Flow 4 (On-Demand Insight Report). Builds an analytical report
prompt from pre-aggregated metrics for a keyword and date range.

Output: JSON with narrative_summary, key_insights, platform_breakdown,
sentiment_breakdown, top_posts, recommendations.
"""

import os

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "1.0")

REPORT_SYSTEM = (
    "You are a senior entertainment industry analyst for an Indian film PR platform. "
    "You write sharp, data-driven insight reports for production houses and PR teams. "
    "You respond ONLY in valid JSON. No markdown fences, no preamble."
)

REPORT_TEMPLATE = """Generate an insight report for the keyword: {keyword}
Date range: {date_from} to {date_to}

AGGREGATED METRICS:
- Total records analysed: {total_records}
- Sentiment: {positive_pct}% positive, {neutral_pct}% neutral, {negative_pct}% negative
- Bot activity: {bot_pct}% of posts flagged as bot/inauthentic
- Platform breakdown: {platform_breakdown}
- Crisis severity distribution: {crisis_breakdown}
- Top 5 high-impact posts (by impact_score):
{top_posts}

REQUESTED SEGMENTS: {segments}

Write a concise, actionable report. Focus on what is notable or unusual in the data.

Return ONLY this JSON:
{{
  "narrative_summary": "2-3 paragraph executive summary of the keyword's social media performance",
  "key_insights": [
    {{"insight": "specific finding from the data", "significance": "why this matters"}}
  ],
  "sentiment_breakdown": {{
    "positive_pct": {positive_pct},
    "neutral_pct": {neutral_pct},
    "negative_pct": {negative_pct},
    "interpretation": "one sentence on what this sentiment profile means"
  }},
  "bot_activity_note": "one sentence on the bot activity level and its implication",
  "platform_notes": "one sentence on which platform is driving most conversation",
  "recommendations": [
    "actionable recommendation 1",
    "actionable recommendation 2",
    "actionable recommendation 3"
  ]
}}"""


def build_report_prompt(
    keyword: str,
    date_from: str,
    date_to: str,
    metrics: dict,
    segments: list,
) -> tuple[str, str]:
    """
    Build the system and user prompt for report generation.

    Parameters
    ----------
    keyword : str
        The tracked keyword.
    date_from, date_to : str
        ISO8601 date range.
    metrics : dict
        Pre-aggregated metrics from MongoDB (sentiment, bot, platform, top posts).
    segments : list
        Requested report segments from the API request.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    platform_breakdown = ", ".join(
        f"{p}: {c}" for p, c in metrics.get("platforms", {}).items()
    ) or "no platform data"

    crisis_breakdown = ", ".join(
        f"{k}: {v}" for k, v in metrics.get("crisis_severity", {}).items() if v > 0
    ) or "none"

    top_posts_lines = []
    for i, post in enumerate(metrics.get("top_posts", [])[:5], 1):
        top_posts_lines.append(
            f"  {i}. [{post.get('platform', '?').upper()}] "
            f"impact={post.get('impact_score', 0)} "
            f"sentiment={post.get('sentiment', '?')} — "
            f"{str(post.get('content', ''))[:120]}"
        )
    top_posts = "\n".join(top_posts_lines) or "  (no high-impact posts found)"

    user_prompt = REPORT_TEMPLATE.format(
        keyword=keyword,
        date_from=date_from,
        date_to=date_to,
        total_records=metrics.get("total_records", 0),
        positive_pct=metrics.get("positive_pct", 0),
        neutral_pct=metrics.get("neutral_pct", 0),
        negative_pct=metrics.get("negative_pct", 0),
        bot_pct=metrics.get("bot_pct", 0),
        platform_breakdown=platform_breakdown,
        crisis_breakdown=crisis_breakdown,
        top_posts=top_posts,
        segments=", ".join(segments) if segments else "all",
    )

    return REPORT_SYSTEM, user_prompt
