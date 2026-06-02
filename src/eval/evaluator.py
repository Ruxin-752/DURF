"""Evaluator — 加载训练好的模型，在 Overcooked 环境中跑 N 个 episode 评估。

用法:
    from src.eval.evaluator import evaluate_model
    metrics = evaluate_model("logs/run_dir/final_model.zip", n_episodes=10)

输出:
    {
        "mean_episode_reward": 12.3,
        "std_episode_reward": 2.1,
        "mean_episode_length": 380.5,
        "std_episode_length": 45.2,
        "success_rate": 0.7,
        "n_episodes": 10,
        "total_steps": 3805,
    }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from src.env.factory import make_overcooked_env


def evaluate_model(
    model_path: str | Path,
    layout_name: str = "cramped_room",
    n_episodes: int = 10,
    seed: int = 42,
    render: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """评估训练好的 PPO 模型。

    Parameters
    ----------
    model_path   : 模型文件路径（.zip）
    layout_name  : Overcooked 地图名
    n_episodes   : 评估 episode 数量
    seed         : 随机种子
    render       : 是否渲染画面（需要显示器）
    deterministic: 是否使用确定性策略（True = 更稳定，False = 更探索）
    verbose      : 是否打印进度

    Returns
    -------
    metrics : dict 包含 mean/std reward、length、success_rate 等
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 构造评估环境（no-feedback 模式）
    env = make_overcooked_env(
        layout_name=layout_name,
        feedback_queue=None,
        alpha=1.0,
        seed=seed,
        monitor_path=None,
    )

    # 加载模型
    model = PPO.load(str(model_path), device="auto")
    if verbose:
        print(f"模型加载完成: {model_path.name}")

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_success: list[bool] = []

    for ep in range(n_episodes):
        obs, _ = env.reset() if hasattr(env, "reset") and hasattr(env.reset, "__code__") else env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        # 如果 reward > 0 视为成功（Overcooked 中完成一道菜得 20 分）
        episode_success.append(total_reward > 0)

        if verbose:
            print(f"  Episode {ep+1:2d}/{n_episodes} | "
                  f"reward={total_reward:+7.2f} | length={step_count:4d} | "
                  f"success={'Y' if total_reward > 0 else 'N'}")

    env.close()

    metrics = {
        "model_path": str(model_path),
        "layout_name": layout_name,
        "n_episodes": n_episodes,
        "seed": seed,
        "deterministic": deterministic,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "min_episode_reward": float(np.min(episode_rewards)),
        "max_episode_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(episode_success)),
        "total_steps": int(np.sum(episode_lengths)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  评估结果 ({n_episodes} episodes)")
        print(f"{'='*50}")
        print(f"  mean_episode_reward : {metrics['mean_episode_reward']:>8.2f}")
        print(f"  std_episode_reward  : {metrics['std_episode_reward']:>8.2f}")
        print(f"  mean_episode_length : {metrics['mean_episode_length']:>8.1f}")
        print(f"  success_rate        : {metrics['success_rate']:>8.1%}")
        print(f"{'='*50}\n")

    return metrics


def evaluate_checkpoints(
    checkpoint_dir: str | Path,
    layout_name: str = "cramped_room",
    n_episodes: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """评估 checkpoints 目录下的所有模型，返回每个 checkpoint 的指标。

    用于观察训练过程中模型性能的变化趋势。
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_files = sorted(checkpoint_dir.glob("rl_model_*.zip"))

    if not model_files:
        print(f"未找到 checkpoint 文件: {checkpoint_dir}")
        return []

    results = []
    for model_file in model_files:
        if verbose:
            print(f"\n评估 {model_file.name} ...")
        metrics = evaluate_model(
            model_path=model_file,
            layout_name=layout_name,
            n_episodes=n_episodes,
            seed=seed,
            deterministic=True,
            verbose=verbose,
        )
        results.append(metrics)

    return results


def save_evaluation_results(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """将评估结果保存为 JSON 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 移除大型列表以保持 JSON 简洁
    save_metrics = {k: v for k, v in metrics.items()
                    if not isinstance(v, list)}
    save_metrics["episode_rewards_summary"] = {
        "values": metrics.get("episode_rewards", []),
        "lengths": metrics.get("episode_lengths", []),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存: {output_path}")
