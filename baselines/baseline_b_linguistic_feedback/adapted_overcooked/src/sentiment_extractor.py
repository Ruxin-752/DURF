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
    "helpful",
    "好",
    "不错",
    "对",
    "棒",
    "有用",
    "有帮助",
    "帮助",
    "帮我",
)

POSITIVE_PHRASES = (
    "did well",
    "well by",
    "good job",
    "that was smoother",
    "much better",
    "better if",
    "without blocking",
    "not blocking",
    "did not block",
    "helping is good",
    "没有挡",
    "没挡",
    "绕开",
    "好多了",
    "更顺",
    "舒服",
    "留出路",
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
    "很差",
    "差",
    "没用",
    "添乱",
    "碍事",
    "挤",
    "乱",
)

NEGATIVE_PHRASES = (
    "not helpful",
    "not useful",
    "not working",
    "not smooth",
    "too crowded",
    "too messy",
    "made me wait",
    "makes me wait",
    "in my way",
    "cut in front",
    "interrupted my plan",
    "breaking my flow",
    "不太行",
    "不太舒服",
    "有点乱",
    "有点挤",
    "打乱",
    "卡住我",
    "没必要",
)


def extract_sentiment(text: str | None, *, scalar_value: int | None = None) -> dict:
    if scalar_value is not None:
        if scalar_value > 0:
            return {"sentiment": "positive", "sentiment_score": 1.0}
        if scalar_value < 0:
            return {"sentiment": "negative", "sentiment_score": -1.0}
        return {"sentiment": "neutral", "sentiment_score": 0.0}

    lowered = (text or "").strip().lower()
    if any(phrase in lowered for phrase in NEGATIVE_PHRASES):
        return {"sentiment": "negative", "sentiment_score": -1.0}
    if any(phrase in lowered for phrase in POSITIVE_PHRASES):
        return {"sentiment": "positive", "sentiment_score": 1.0}

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
