"""Group A Baseline — 训练主入口。

CLI:
  python -m src.train.trainer \\
      --mode {no-feedback|random-feedback|human-feedback} \\
      --total-timesteps 100000 \\
      --participant P00 \\
      --seed 42 \\
      --config config/default.yaml \\
      --run-name optional_tag

三种模式通过替换 feedback source 子进程实现，env wrapper 逻辑不变。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.env.factory import make_overcooked_env
from src.feedback.null_source import run_null_source
from src.feedback.random_source import run_random_source
from src.feedback.keyboard_listener import run_keyboard_listener
from src.train.callbacks import ConfigSnapshotCallback, TrajectoryLoggerCallback
from src.train.logging_schema import TrajectoryWriter


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Group A Baseline Trainer")
    p.add_argument("--mode", choices=["no-feedback", "random-feedback", "human-feedback"],
                   default="no-feedback")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="覆盖 config 中的 total_timesteps")
    p.add_argument("--participant", default="P00")
    p.add_argument("--seed", type=int, default=None,
                   help="覆盖 config 中的 seed")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--run-name", default="",
                   help="可选标签，附加到 run_id 末尾")
    return p.parse_args(argv)


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def train(argv=None) -> None:
    args   = _parse_args(argv)
    config = _load_config(args.config)

    # CLI 参数覆盖 config
    seed             = args.seed           if args.seed            is not None else config.get("seed", 42)
    total_timesteps  = args.total_timesteps if args.total_timesteps is not None else config["train"]["total_timesteps"]
    checkpoint_freq  = config["train"]["checkpoint_freq"]
    layout_name      = config["env"]["layout_name"]
    alpha            = config["feedback"]["alpha"]
    random_rate_hz   = config["feedback"]["random_source_rate_hz"]
    ppo_kwargs       = {k: v for k, v in config["ppo"].items() if k != "policy"}
    ppo_policy       = config["ppo"]["policy"]
    log_dir_base     = Path(config["logging"]["log_dir"])
    device           = config.get("device", "auto")

    # run_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_name}" if args.run_name else ""
    run_id = f"{args.mode}_{args.participant}_{ts}{tag}"

    run_dir  = log_dir_base / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  run_id : {run_id}")
    print(f"  mode   : {args.mode}")
    print(f"  steps  : {total_timesteps:,}")
    print(f"  seed   : {seed}")
    print(f"{'='*60}\n")

    # ── Feedback source 子进程 ────────────────────────────────
    feedback_queue = None
    source_process = None
    stop_event     = multiprocessing.Event()
    session_start_ms = int(time.monotonic() * 1000)

    if args.mode == "random-feedback":
        feedback_queue = multiprocessing.Queue()
        source_process = multiprocessing.Process(
            target=run_random_source,
            args=(feedback_queue, stop_event),
            kwargs={"rate_hz": random_rate_hz,
                    "session_start_ms": session_start_ms},
            daemon=True,
        )
        source_process.start()
        print(f"random_source 启动 (rate={random_rate_hz} Hz)")

    elif args.mode == "human-feedback":
        feedback_queue = multiprocessing.Queue()
        audit_csv = run_dir / "keyboard_audit.csv"
        source_process = multiprocessing.Process(
            target=run_keyboard_listener,
            args=(feedback_queue, args.participant, str(audit_csv), stop_event),
            daemon=True,
        )
        source_process.start()
        print(f"keyboard_listener 启动 → {audit_csv}")

    # ── Ctrl+C 优雅退出 ──────────────────────────────────────
    def _shutdown(sig, frame):
        print("\n[trainer] 收到中断信号，清理中…")
        stop_event.set()
        if source_process and source_process.is_alive():
            source_process.join(timeout=3)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # ── 构造 env ─────────────────────────────────────────
        env = make_overcooked_env(
            layout_name=layout_name,
            feedback_queue=feedback_queue,
            alpha=alpha,
            seed=seed,
            monitor_path=run_dir / "monitor",
        )

        # ── 元数据快照 ────────────────────────────────────────
        snapshot = {
            "run_id":          run_id,
            "mode":            args.mode,
            "participant":     args.participant,
            "seed":            seed,
            "total_timesteps": total_timesteps,
            "config_path":     str(args.config),
            "config":          config,
            "git_hash":        _get_git_hash(),
            "start_time_iso":  datetime.now().isoformat(),
        }

        # ── Callbacks ─────────────────────────────────────────
        writer = TrajectoryWriter(run_dir / "trajectories.jsonl")

        callbacks = CallbackList([
            ConfigSnapshotCallback(snapshot, run_dir),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(ckpt_dir),
                name_prefix="rl_model",
                verbose=0,
            ),
            TrajectoryLoggerCallback(run_id, writer),
        ])

        # ── PPO ───────────────────────────────────────────────
        model = PPO(
            ppo_policy,
            env,
            seed=seed,
            device=device,
            tensorboard_log=str(run_dir / "tb") if config["logging"]["tensorboard"] else None,
            verbose=1,
            **ppo_kwargs,
        )

        # ── 训练 ─────────────────────────────────────────────
        t0 = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        elapsed = time.time() - t0

        # ── 保存最终模型 ──────────────────────────────────────
        final_path = run_dir / "final_model"
        model.save(str(final_path))

        # ── 训练结束摘要 ──────────────────────────────────────
        summary = {
            "run_id":           run_id,
            "elapsed_seconds":  round(elapsed, 1),
            "final_model_path": str(final_path) + ".zip",
            "checkpoint_dir":   str(ckpt_dir),
        }
        with open(run_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  训练完成！耗时 {elapsed:.1f}s")
        print(f"  run_dir : {run_dir}")
        print(f"{'='*60}\n")

    finally:
        stop_event.set()
        if source_process and source_process.is_alive():
            source_process.join(timeout=3)


if __name__ == "__main__":
    # Windows 多进程需要保护入口
    multiprocessing.freeze_support()
    train()
