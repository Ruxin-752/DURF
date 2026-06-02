"""Random feedback source — 子进程自动随机注入 ±1 信号。

用途:
  - 验证 HumanFeedbackEnvWrapper 的奖励注入逻辑是否正确（不依赖真人）
  - 提供 random-feedback 训练模式作为对照实验
  - 在没有显示器的服务器上运行 smoke test

rate_hz 控制每秒注入频率（默认 2.0），实际间隔有轻微随机抖动以模拟真人节奏。
"""
from __future__ import annotations

import random
import time

from src.feedback.protocol import FeedbackEvent


def run_random_source(
    queue,                    # multiprocessing.Queue
    stop_event,               # multiprocessing.Event
    rate_hz: float = 2.0,
    session_start_ms: int = 0,
) -> None:
    """每秒约 rate_hz 次随机向 queue 推送 +1 或 -1 FeedbackEvent。

    Parameters
    ----------
    queue           : 接收端轮询的 multiprocessing.Queue
    stop_event      : set 后立即退出
    rate_hz         : 平均注入频率（Hz），默认 2.0
    session_start_ms: session 开始的 monotonic 毫秒基准（用于 timestamp_ms）
    """
    base_interval = 1.0 / max(rate_hz, 0.01)

    while not stop_event.is_set():
        # 轻微随机抖动：±20%，模拟真人按键节奏
        jitter = base_interval * 0.2 * (random.random() * 2 - 1)
        sleep_time = max(0.05, base_interval + jitter)
        time.sleep(sleep_time)

        if stop_event.is_set():
            break

        signal = 1.0 if random.random() < 0.5 else -1.0
        event_type = "positive" if signal > 0 else "negative"
        now_ms = int(time.monotonic() * 1000) - session_start_ms

        event = FeedbackEvent(
            timestamp_ms=now_ms,
            event_type=event_type,
            signal_value=signal,
            source="random",
        )
        queue.put(event)
