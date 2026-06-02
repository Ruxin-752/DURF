"""日志字段定义 + JSONL 写入器。

所有轨迹日志的字段在这里统一定义，
方便以后扩展（新增字段不破坏已有日志）。
"""
from __future__ import annotations

import json
from pathlib import Path

# 每个 step 必须包含的字段；episode 结束时追加 episode_human_reward
TRAJECTORY_FIELDS = [
    "run_id",
    "timestamp_iso",
    "monotonic_ms",
    "global_step",
    "episode_count",
    "env_reward",
    "human_reward",
    "total_reward",
    "action",
]


class TrajectoryWriter:
    """行式 JSONL 写入器，每次 flush 保证崩溃时不丢数据。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._path, "a", encoding="utf-8", buffering=1)

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
