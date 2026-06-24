"""Minimal feedback type router.

This is deliberately simple and deterministic. It gives us a baseline and a
stable field before we add LLM parsing.
"""

from __future__ import annotations


IMPERATIVE_MARKERS = (
    "should",
    "need to",
    "go ",
    "take ",
    "get ",
    "put ",
    "serve",
    "你应该",
    "应该",
    "去",
    "拿",
    "放",
    "送",
)

EVALUATIVE_MARKERS = (
    "good",
    "bad",
    "nice",
    "great",
    "wrong",
    "好",
    "不好",
    "不错",
    "错",
    "棒",
)

DESCRIPTIVE_MARKERS = (
    "block",
    "blocking",
    "stuck",
    "in my way",
    "抢",
    "堵",
    "挡",
    "卡",
    "别",
)


def route_feedback_type(text: str | None, scalar_value: int | None = None) -> str:
    if scalar_value is not None and not text:
        return "evaluative"

    lowered = (text or "").strip().lower()
    if not lowered:
        return "unknown"

    if any(marker in lowered for marker in DESCRIPTIVE_MARKERS):
        return "descriptive"
    if any(marker in lowered for marker in IMPERATIVE_MARKERS):
        return "imperative"
    if any(marker in lowered for marker in EVALUATIVE_MARKERS):
        return "evaluative"
    return "descriptive"


def polarity_from_feedback(text: str | None, scalar_value: int | None = None) -> str:
    if scalar_value is not None:
        if scalar_value > 0:
            return "positive"
        if scalar_value < 0:
            return "negative"
        return "neutral"

    lowered = (text or "").strip().lower()
    negative_markers = ("bad", "wrong", "don't", "do not", "别", "不要", "不好", "错", "堵", "挡")
    positive_markers = ("good", "nice", "great", "thanks", "好", "不错", "棒")
    if any(marker in lowered for marker in negative_markers):
        return "negative"
    if any(marker in lowered for marker in positive_markers):
        return "positive"
    return "unknown"

