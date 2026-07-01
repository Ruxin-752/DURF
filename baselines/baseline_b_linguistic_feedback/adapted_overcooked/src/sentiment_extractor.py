"""Small deterministic sentiment extractor for linguistic feedback."""

from __future__ import annotations


POSITIVE_MARKERS = (
    "good",
    "nice",
    "great",
    "correct",
    "right",
    "thanks",
    "useful",
    "好",
    "不错",
    "对",
    "棒",
    "有用",
)

NEGATIVE_MARKERS = (
    "bad",
    "wrong",
    "do not",
    "don't",
    "not good",
    "useless",
    "block",
    "blocking",
    "duplicate",
    "repeat",
    "不好",
    "错",
    "别",
    "不要",
    "堵",
    "挡",
    "抢",
    "重复",
)


def extract_sentiment(text: str | None, *, scalar_value: int | None = None) -> dict:
    if scalar_value is not None:
        if scalar_value > 0:
            return {"sentiment": "positive", "sentiment_score": 1.0}
        if scalar_value < 0:
            return {"sentiment": "negative", "sentiment_score": -1.0}
        return {"sentiment": "neutral", "sentiment_score": 0.0}

    lowered = (text or "").strip().lower()
    positive = sum(marker in lowered for marker in POSITIVE_MARKERS)
    negative = sum(marker in lowered for marker in NEGATIVE_MARKERS)
    if positive > negative:
        return {"sentiment": "positive", "sentiment_score": 1.0}
    if negative > positive:
        return {"sentiment": "negative", "sentiment_score": -1.0}
    return {"sentiment": "neutral", "sentiment_score": 0.0}


def desired_action_sentiment(feedback_type: str, target_features: dict[str, float]) -> float:
    """Commands grounded to desired actions should reinforce those action features."""

    if feedback_type == "imperative" and target_features:
        return 1.0
    return 0.0
