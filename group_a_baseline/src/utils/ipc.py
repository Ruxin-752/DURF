"""进程间通信辅助 — feedback queue 的创建与清理。

封装 multiprocessing.Queue 和 Event 的生命周期管理，
避免 trainer.py 中重复的样板代码。
"""
from __future__ import annotations

import multiprocessing
from typing import Any


def create_feedback_queue() -> multiprocessing.Queue:
    """创建 feedback 队列（无大小限制）。"""
    return multiprocessing.Queue()


def create_stop_event() -> multiprocessing.Event:
    """创建停止事件。"""
    return multiprocessing.Event()


def cleanup_process(proc: multiprocessing.Process | None, timeout: float = 3.0) -> None:
    """安全终止子进程。

    Parameters
    ----------
    proc   : 子进程对象
    timeout: 等待进程结束的超时秒数
    """
    if proc is not None and proc.is_alive():
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
