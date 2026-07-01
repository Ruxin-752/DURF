"""Rule-based feedback form classifier.

This mirrors the paper's high-level split in a deterministic Overcooked-specific
form: evaluative, imperative, and descriptive feedback.
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
    "move",
    "clear",
    "你应该",
    "应该",
    "去",
    "拿",
    "放",
    "让开",
)

EVALUATIVE_MARKERS = (
    "good",
    "bad",
    "nice",
    "great",
    "wrong",
    "correct",
    "thanks",
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
    "duplicate",
    "repeat",
    "shortest path",
    "crowd",
    "抢",
    "堵",
    "挡",
    "卡",
    "重复",
    "最短路径",
    "插",
)


def classify_feedback(text: str | None, *, scalar_value: int | None = None) -> str:
    if scalar_value is not None and not text:
        return "evaluative"

    lowered = (text or "").strip().lower()
    if not lowered:
        return "unknown"

    imperative_score = sum(marker in lowered for marker in IMPERATIVE_MARKERS)
    descriptive_score = sum(marker in lowered for marker in DESCRIPTIVE_MARKERS)
    evaluative_score = sum(marker in lowered for marker in EVALUATIVE_MARKERS)

    if imperative_score:
        return "imperative"
    if descriptive_score:
        return "descriptive"
    if evaluative_score:
        return "evaluative"
    return "descriptive"
