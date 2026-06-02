"""Feedback event protocol — 反馈消息的契约定义。

所有 feedback source（keyboard / random / null / future LLM）都产生这种事件。
frozen=True 保证跨进程传递时不会被篡改。
metadata 用 dict 留给未来 LLM source 附加 raw text，不改 protocol 本体。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EventType = Literal["positive", "negative", "coach_pause"]


@dataclass(frozen=True)
class FeedbackEvent:
    timestamp_ms: int        # 自 session 开始的 monotonic 毫秒
    event_type: EventType
    signal_value: float      # +1.0 / -1.0 / 0.0（coach_pause 当前不影响 reward）
    source: str              # 'keyboard' / 'random' / 'null' / 'llm'
    metadata: dict | None = field(default=None, hash=False)
