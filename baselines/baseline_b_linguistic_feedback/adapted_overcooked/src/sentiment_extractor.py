"""VADER-based sentiment / valence extraction (English only).

Faithful port of the paper's ``science/observations/text_analysis.py``:

- ``vader_observation``  -> raw VADER compound score in ``[-1, 1]``;
- ``modified_vader_observation`` -> the variant actually used by the paper's
  experiment learners:
    * phrases containing ``zero / worthless / nothing / " 0"`` return exactly 0
      (these really mean "no value", so no positive bias);
    * otherwise, if VADER detects any valence, return the compound score;
    * if VADER is neutral, fall back to a positive default (+0.5), the paper's
      "positive implicature".

VADER is English-only, and per project decision the feedback corpus is now
English-only as well. The ``valence_scale`` (paper default 30) is applied later
by the learner / observation builder, not here.
"""

from __future__ import annotations

from functools import lru_cache


DEFAULT_SENTIMENT = 0.5
ZERO_MARKERS = ("zero", "worthless", "nothing", " 0")


@lru_cache(maxsize=1)
def _analyzer():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_compound(text: str | None) -> float:
    """Raw VADER compound score in [-1, 1]."""

    if not text:
        return 0.0
    return float(_analyzer().polarity_scores(str(text))["compound"])


def modified_vader_observation(
    text: str | None, *, default_sentiment: float = DEFAULT_SENTIMENT
) -> float:
    """Reproduce the paper's ``modified_vader_observation`` valence."""

    phrase = (text or "").lower()
    if any(marker in phrase for marker in ZERO_MARKERS):
        return 0.0

    compound = vader_compound(text)
    if compound != 0.0:
        return compound

    return default_sentiment


def extract_sentiment(text: str | None, *, scalar_value: int | None = None) -> dict:
    """Return a sentiment label + continuous score for one feedback utterance.

    ``scalar_value`` (an explicit numeric rating, if any) takes precedence and
    is mapped to a unit valence, mirroring an explicit evaluative signal.
    Otherwise the continuous ``modified_vader_observation`` score is used.
    """

    if scalar_value is not None:
        if scalar_value > 0:
            return {"sentiment": "positive", "sentiment_score": 1.0}
        if scalar_value < 0:
            return {"sentiment": "negative", "sentiment_score": -1.0}
        return {"sentiment": "neutral", "sentiment_score": 0.0}

    score = modified_vader_observation(text)
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"sentiment": label, "sentiment_score": float(score)}


def desired_action_sentiment(feedback_type: str, target_features: dict[str, float]) -> float:
    """Commands grounded to desired actions should reinforce those action features."""

    if feedback_type == "imperative" and target_features:
        return 1.0
    return 0.0
