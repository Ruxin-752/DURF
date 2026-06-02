"""Smoke test — 5000 timesteps no-feedback 模式，验证 pipeline 端到端可用。

验收标准:
  1. 5分钟内完成
  2. logs/{run_id}/ 目录存在
  3. 至少一个 checkpoint .zip 文件存在
  4. trajectories.jsonl 存在且每行可 json.loads()
  5. run_metadata.json 存在

用法:
  conda activate pantheonrl_env
  cd group_a_baseline
  python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 确保在 group_a_baseline/ 目录下运行
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.train.trainer import train

SMOKE_STEPS = 5000


def main() -> None:
    print("=" * 60)
    print("  Smoke Test — 5000 timesteps, no-feedback")
    print("=" * 60)

    t0 = time.time()
    train([
        "--mode",            "no-feedback",
        "--total-timesteps", str(SMOKE_STEPS),
        "--participant",     "SMOKE",
        "--seed",            "42",
        "--config",          str(ROOT / "config/default.yaml"),
        "--run-name",        "smoke",
    ])
    elapsed = time.time() - t0

    # ── 找到最新的 smoke run ──────────────────────────────────
    log_dir = ROOT / "logs"
    smoke_dirs = sorted(log_dir.glob("no-feedback_SMOKE_*smoke"),
                        key=lambda d: d.stat().st_mtime, reverse=True)
    assert smoke_dirs, "找不到 smoke run 目录"
    run_dir = smoke_dirs[0]
    print(f"\n检查目录: {run_dir}")

    # 1. checkpoint
    ckpts = list((run_dir / "checkpoints").glob("*.zip"))
    assert ckpts, f"没有找到 checkpoint .zip 文件 in {run_dir / 'checkpoints'}"
    print(f"  checkpoint : {ckpts[0].name}  ✓")

    # 2. trajectories.jsonl 可解析
    traj = run_dir / "trajectories.jsonl"
    assert traj.exists(), f"trajectories.jsonl 不存在"
    lines = traj.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "trajectories.jsonl 为空"
    for line in lines:
        json.loads(line)   # 任何一行解析失败都会抛异常
    print(f"  trajectories: {len(lines)} lines  ✓")

    # 3. run_metadata.json
    meta = run_dir / "run_metadata.json"
    assert meta.exists(), "run_metadata.json 不存在"
    with open(meta) as f:
        m = json.load(f)
    assert m["mode"] == "no-feedback"
    print(f"  run_metadata: mode={m['mode']}  ✓")

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST PASSED  ({elapsed:.1f}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
