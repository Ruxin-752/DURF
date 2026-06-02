"""Null feedback source — 永不发送任何反馈，用于 no-feedback baseline 对照。

与 random_source / keyboard_listener 共用同一启动接口，trainer.py 可以
只换这里的 process target 而不改其他逻辑。
"""
from __future__ import annotations

import time


def run_null_source(
    queue,          # multiprocessing.Queue（接口一致，此处从不写入）
    stop_event,     # multiprocessing.Event，set 后退出
) -> None:
    """什么都不发。每 100ms 轮询一次 stop_event，收到信号后干净退出。"""
    while not stop_event.is_set():
        time.sleep(0.1)
