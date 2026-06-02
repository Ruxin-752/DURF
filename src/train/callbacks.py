"""SB3 Callbacks — 轨迹日志 + 配置快照。

TrajectoryLoggerCallback : 每 step 写一行 JSONL
ConfigSnapshotCallback   : 训练开始时把完整配置存到 run_metadata.json（一次性）
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from src.train.logging_schema import TrajectoryWriter


class TrajectoryLoggerCallback(BaseCallback):
    """每个 env step 把 reward 明细 + action 写入 JSONL。

    info 字段来自 HumanFeedbackEnvWrapper 注入的 env_reward / human_reward /
    total_reward；若 wrapper 未启用则回退到 SB3 Monitor 记录的 reward。
    """

    def __init__(
        self,
        run_id: str,
        writer: TrajectoryWriter,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.run_id = run_id
        self.writer = writer
        self._start_ms = int(time.monotonic() * 1000)
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [{}])
        info    = infos[0] if infos else {}
        actions = self.locals.get("actions", [None])
        action  = int(actions[0]) if actions[0] is not None else -1

        record: dict[str, Any] = {
            "run_id":        self.run_id,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "monotonic_ms":  int(time.monotonic() * 1000) - self._start_ms,
            "global_step":   self.num_timesteps,
            "episode_count": self._episode_count,
            "env_reward":    info.get("env_reward",   0.0),
            "human_reward":  info.get("human_reward", 0.0),
            "total_reward":  info.get("total_reward",
                                      info.get("reward", 0.0)),
            "action":        action,
        }

        # episode 结束时追加汇总字段
        if "episode_human_reward" in info:
            record["episode_human_reward"] = info["episode_human_reward"]
            self._episode_count += 1

        self.writer.write(record)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()


class ConfigSnapshotCallback(BaseCallback):
    """训练开始时把 config + args + git hash 写入 run_metadata.json（一次性）。"""

    def __init__(
        self,
        snapshot: dict,
        log_dir: str | Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.snapshot = snapshot
        self.log_dir  = Path(log_dir)

    def _on_training_start(self) -> None:
        path = self.log_dir / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, indent=2, ensure_ascii=False)

    def _on_step(self) -> bool:
        return True
