"""Keyboard listener — pygame 子进程，捕获按键转为 FeedbackEvent。

按键映射:
  Q  → positive  (+1.0)  "Good job"
  E  → negative  (-1.0)  "Wrong move"
  C  → coach_pause (0.0) 触发 Coach-Time（Group B 用）
  ESC → 请求退出训练

运行要求:
  - 必须在子进程中调用（Windows 的 multiprocessing 默认 spawn 模式）
  - pygame 在子进程内初始化，不影响主进程
  - 每次按键同时写入 CSV audit log（供 IRB / 回放审计）

接口:
  run_keyboard_listener(feedback_queue, participant_id, log_path, stop_event)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Union

from src.feedback.protocol import FeedbackEvent

# 按键 → (event_type, signal_value, 显示文字)
_KEY_MAP = {
    ord("q"): ("positive",    +1.0, "Q  +1  ✓"),
    ord("e"): ("negative",    -1.0, "E  -1  ✗"),
    ord("c"): ("coach_pause",  0.0, "C  PAUSE"),
}
_WINDOW_W, _WINDOW_H = 420, 220
_BG_COLOR   = (30, 30, 30)
_TEXT_COLOR = (220, 220, 220)
_FLASH_POS  = (30, 140)
_FLASH_MS   = 600   # 按键提示显示时长（毫秒）


def run_keyboard_listener(
    feedback_queue,
    participant_id: str,
    log_path: Union[str, Path],
    stop_event,
) -> None:
    """子进程入口：启动 pygame 窗口，捕获按键，写入 queue + CSV。

    Parameters
    ----------
    feedback_queue : multiprocessing.Queue
    participant_id : 参与者 ID，写入 CSV 用于实验溯源
    log_path       : CSV 日志文件路径（目录需提前存在）
    stop_event     : multiprocessing.Event，set 后退出
    """
    import pygame  # 在子进程内 import，避免主进程被污染

    pygame.init()
    screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H))
    pygame.display.set_caption(f"Feedback — {participant_id}")
    font_big   = pygame.font.Font(None, 28)   # 内置字体，headless 模式也可用
    font_small = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp_ms", "participant_id", "event_type",
                     "signal_value", "source"])

    session_start_ms = int(time.monotonic() * 1000)
    flash_msg  = ""
    flash_until = 0

    try:
        while not stop_event.is_set():
            now = pygame.time.get_ticks()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stop_event.set()
                    break

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        stop_event.set()
                        break

                    if ev.key in _KEY_MAP:
                        etype, sval, label = _KEY_MAP[ev.key]
                        ts_ms = int(time.monotonic() * 1000) - session_start_ms
                        event = FeedbackEvent(
                            timestamp_ms=ts_ms,
                            event_type=etype,
                            signal_value=sval,
                            source="keyboard",
                        )
                        feedback_queue.put(event)
                        writer.writerow([ts_ms, participant_id,
                                         etype, sval, "keyboard"])
                        csv_file.flush()
                        flash_msg   = label
                        flash_until = now + _FLASH_MS

            # ── 绘制 UI ──────────────────────────────────────────
            screen.fill(_BG_COLOR)
            lines = [
                "Human Feedback",
                "",
                "  Q  →  +1  (Good)",
                "  E  →  -1  (Bad)",
                "  C  →  Pause",
                " ESC →  Quit",
            ]
            for i, line in enumerate(lines):
                surf = font_small.render(line, True, _TEXT_COLOR)
                screen.blit(surf, (30, 10 + i * 22))

            if now < flash_until:
                flash_surf = font_big.render(flash_msg, True, (80, 220, 120))
                screen.blit(flash_surf, _FLASH_POS)

            pygame.display.flip()
            clock.tick(30)  # 30 FPS 足够响应，不浪费 CPU
    finally:
        csv_file.close()
        pygame.quit()
