"""
PRISM — Sentiment Classification Prompt (Eng C)

System prompt, user template, and prompt builder for the core sentiment
classification pipeline (Phase 3). Designed for Gemini 1.5 Flash as primary,
with fallback compatibility for Claude Sonnet and Groq/Llama-3.

Output: strict JSON with sentiment, scores, emotion tags, sarcasm flag,
and crisis severity. No markdown, no preamble.
"""

import os

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "1.0")

# ──────────────────────────────────────────────────────────────────────
# System Prompt — injected as the system/instruction role
# ──────────────────────────────────────────────────────────────────────
SENTIMENT_SYSTEM = (
    "You are PRISM Sentiment Engine — a structured sentiment classification "
    "system for the Indian entertainment intelligence platform, built by Zynthora.AI.\n"
    "You MUST return ONLY valid JSON. No markdown fences, no explanation, no preamble.\n"
    "If you add anything outside the JSON object, the parser will crash."
)

# ──────────────────────────────────────────────────────────────────────
# User Prompt Template
# ──────────────────────────────────────────────────────────────────────
SENTIMENT_TEMPLATE = """Analyse this social media post about: {keyword}
Platform: {platform}
Language detected: {language_detected}

POST TEXT:
\"\"\"{normalised_text}\"\"\"

TANGLISH/HINGLISH CONTEXT:
'semma'=excellent, 'vera level'=outstanding, 'mass'=impressive, 'nalla'=good,
'mokkai'=boring, 'waste'=terrible, 'ayyo'=dismay, 'padam'=film,
'bakwaas'=nonsense, 'zabardast'=outstanding, 'bekar'=useless, 'mast'=great,
'theri'=awesome, 'khatarnak'=dangerous/amazing, 'jhakaas'=fantastic,
'varutha padatha'=don't worry, 'oru vaat'=not great, 'kalakkal'=brilliant,
'marana mass'=extremely impressive, 'mosam'=fraud/terrible

CLASSIFICATION RULES:
1. SENTIMENT: exactly one of "positive"|"neutral"|"negative"
2. SCORES: three floats 0.0-1.0, MUST sum to exactly 1.0
3. CONFIDENCE: 0.8-1.0 clear cases, 0.5-0.79 mixed/short, 0.3-0.49 suspected sarcasm
4. EMOTION: pick the single strongest emotion from this fixed list:
   [excited, happy, disappointed, angry, neutral, surprised, sad, sarcastic, confused, praise]
5. EMOTION_TAGS: list ALL detected emotions (1-3 max) from the same fixed list
6. SARCASM: if detected, set is_sarcastic=true, INVERT the apparent sentiment, confidence ≤ 0.55
7. CRISIS SEVERITY:
   - "none" for positive/neutral posts
   - "low" for mild negative (disappointment, mild criticism)
   - "medium" for strong negative (anger, calls for boycott)
   - "severe" for targeted attacks, coordinated negativity, or threats

FEW-SHOT EXAMPLES:

Text: "Semma padam da! BGM vera level 🔥"
→ {{"sentiment":"positive","positive_score":0.92,"neutral_score":0.05,"negative_score":0.03,"confidence":0.92,"dominant_emotion":"excited","emotion_tags":["excited","praise"],"is_sarcastic":false,"crisis_severity":"none","reasoning":"Tanglish praise — 'semma' and 'vera level' are strong positive markers"}}

Text: "Oh wow masterpiece. Didn't fall asleep halfway 🙄"
→ {{"sentiment":"negative","positive_score":0.10,"neutral_score":0.05,"negative_score":0.85,"confidence":0.52,"dominant_emotion":"sarcastic","emotion_tags":["sarcastic","disappointed"],"is_sarcastic":true,"crisis_severity":"low","reasoning":"Sarcastic praise — surface positive but the sleeping comment and eye-roll emoji invert to negative"}}

Text: "Film releases Dec 22 in 3000 screens across TN"
→ {{"sentiment":"neutral","positive_score":0.05,"neutral_score":0.90,"negative_score":0.05,"confidence":0.95,"dominant_emotion":"neutral","emotion_tags":["neutral"],"is_sarcastic":false,"crisis_severity":"none","reasoning":"Factual announcement with no opinion expressed"}}

Text: "Great movie great movie great movie buy tickets now #ad"
→ {{"sentiment":"positive","positive_score":0.55,"neutral_score":0.35,"negative_score":0.10,"confidence":0.30,"dominant_emotion":"neutral","emotion_tags":["neutral"],"is_sarcastic":false,"crisis_severity":"none","reasoning":"Repetitive text with promotional markers — low confidence spam-like content"}}

Text: "Boycott this movie! Director is anti-national. Share and spread!"
→ {{"sentiment":"negative","positive_score":0.02,"neutral_score":0.03,"negative_score":0.95,"confidence":0.90,"dominant_emotion":"angry","emotion_tags":["angry"],"is_sarcastic":false,"crisis_severity":"severe","reasoning":"Call for boycott with coordinated action language — severe crisis signal"}}

Now analyse the POST TEXT above. Return ONLY this JSON:
{{
  "sentiment": "positive|neutral|negative",
  "positive_score": 0.0,
  "neutral_score": 0.0,
  "negative_score": 0.0,
  "confidence": 0.0,
  "dominant_emotion": "from fixed list",
  "emotion_tags": ["emotion1"],
  "is_sarcastic": false,
  "crisis_severity": "none|low|medium|severe",
  "reasoning": "one sentence explaining classification"
}}"""

# ──────────────────────────────────────────────────────────────────────
# Fixed emotion list — used for validation in sentiment_tagger.py
# ──────────────────────────────────────────────────────────────────────
VALID_EMOTIONS = frozenset([
    "excited", "happy", "disappointed", "angry", "neutral",
    "surprised", "sad", "sarcastic", "confused", "praise",
])

VALID_SENTIMENTS = frozenset(["positive", "neutral", "negative"])

VALID_CRISIS_SEVERITIES = frozenset(["none", "low", "medium", "severe"])


def build_sentiment_prompt(record: dict) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) tuple for sentiment classification.

    Parameters
    ----------
    record : dict
        A pre-processed record dict containing at minimum:
        - keyword (str): the tracked keyword/entity
        - platform (str): youtube|twitter|reddit|external
        - language_detected (str): detected language code
        - normalised_text (str): slang-normalised text (or raw content fallback)
        - content (str): original raw text (used as fallback)

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt) ready to send to the LLM.
    """
    # Use normalised_text if available, fall back to raw content
    text = record.get("normalised_text") or record.get("content", "")

    user_prompt = SENTIMENT_TEMPLATE.format(
        keyword=record.get("keyword", "unknown"),
        platform=record.get("platform", "unknown"),
        language_detected=record.get("language_detected", "unknown"),
        normalised_text=text,
    )

    return SENTIMENT_SYSTEM, user_prompt
