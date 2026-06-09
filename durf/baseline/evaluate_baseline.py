"""Evaluate an SB3 PPO baseline over a fixed set of seeded episodes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from stable_baselines3 import PPO

from durf.baseline.runtime import (
    make_baseline_env,
    reset_env,
    resolve_model_path,
    step_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="SB3 PPO .zip; defaults to the archived 1M model")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    model_path = resolve_model_path(args.model)
    model = PPO.load(model_path, device="cpu")
    rewards: list[float] = []
    lengths: list[int] = []

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        env = make_baseline_env(args.layout, episode_seed)
        obs = reset_env(env)
        done = False
        reward_sum = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _ = step_env(env, action)
            reward_sum += reward
            steps += 1

        env.close()
        rewards.append(reward_sum)
        lengths.append(steps)
        print(
            f"Episode {episode_index + 1:02d}/{args.episodes}: "
            f"seed={episode_seed}, reward={reward_sum:.1f}, steps={steps}"
        )

    result = {
        "model": str(model_path),
        "layout": args.layout,
        "episodes": args.episodes,
        "base_seed": args.seed,
        "deterministic": args.deterministic,
        "mean_reward": statistics.fmean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "success_rate": sum(reward > 0 for reward in rewards) / len(rewards),
        "mean_episode_length": statistics.fmean(lengths),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
