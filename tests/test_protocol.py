"""Unit tests for FeedbackEvent protocol (Phase 1 验收标准)."""
import pytest
from src.feedback.protocol import FeedbackEvent


def test_create_positive_event():
    e = FeedbackEvent(timestamp_ms=1000, event_type="positive", signal_value=1.0, source="keyboard")
    assert e.event_type == "positive"
    assert e.signal_value == 1.0
    assert e.source == "keyboard"
    assert e.timestamp_ms == 1000


def test_create_negative_event():
    e = FeedbackEvent(timestamp_ms=2000, event_type="negative", signal_value=-1.0, source="keyboard")
    assert e.signal_value == -1.0
    assert e.event_type == "negative"


def test_coach_pause_zero_signal():
    e = FeedbackEvent(timestamp_ms=3000, event_type="coach_pause", signal_value=0.0, source="keyboard")
    assert e.signal_value == 0.0
    assert e.event_type == "coach_pause"


def test_frozen_immutable():
    """frozen=True: 不允许修改任何字段。"""
    e = FeedbackEvent(timestamp_ms=1000, event_type="positive", signal_value=1.0, source="keyboard")
    with pytest.raises((AttributeError, TypeError)):
        e.signal_value = 2.0  # type: ignore[misc]


def test_metadata_default_none():
    e = FeedbackEvent(timestamp_ms=1000, event_type="positive", signal_value=1.0, source="keyboard")
    assert e.metadata is None


def test_metadata_extensible():
    """metadata dict 为 LLM source 预留扩展空间。"""
    e = FeedbackEvent(
        timestamp_ms=1000,
        event_type="positive",
        signal_value=1.0,
        source="llm",
        metadata={"raw_text": "good robot", "confidence": 0.95},
    )
    assert e.metadata["raw_text"] == "good robot"
    assert e.metadata["confidence"] == 0.95


def test_sources_keyboard_random_null_llm():
    """source 字段接受所有预期值。"""
    for src in ("keyboard", "random", "null", "llm"):
        e = FeedbackEvent(timestamp_ms=0, event_type="positive", signal_value=1.0, source=src)
        assert e.source == src


def test_hashable_because_frozen():
    """frozen dataclass 可以放进 set / 作为 dict key。"""
    e1 = FeedbackEvent(timestamp_ms=1000, event_type="positive", signal_value=1.0, source="keyboard")
    e2 = FeedbackEvent(timestamp_ms=2000, event_type="negative", signal_value=-1.0, source="random")
    s = {e1, e2}
    assert len(s) == 2
