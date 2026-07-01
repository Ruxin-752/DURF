"""Evaluate the archived RLlib PPO baseline over seeded episodes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from durf.baseline.runtime import (
    DEFAULT_AGENT_NAME,
    ensure_agent_layout,
    load_rllib_agent,
    make_baseline_env,
    resolve_agent_dir,
    rllib_action_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="RLlib agent name or path; defaults to RllibCrampedRoomSP",
    )
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-layout-check",
        action="store_true",
        help="Allow diagnostic cross-layout evaluation for same-sized curriculum maps.",
    )
    return parser.parse_args()


def count_event_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        return sum(count_event_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, bool) for item in value):
            return sum(1 for item in value if item)
        if all(isinstance(item, (int, float)) for item in value):
            return len(value)
        return sum(count_event_value(item) for item in value)
    return 0


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    if not args.skip_layout_check:
        ensure_agent_layout(args.agent, args.layout)
    agent_dir = resolve_agent_dir(args.agent)
    rewards: list[float] = []
    lengths: list[int] = []
    event_counts: Counter[str] = Counter()
    agent0 = load_rllib_agent(args.agent, agent_index=0)
    agent1 = load_rllib_agent(args.agent, agent_index=1)

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        env = make_baseline_env(args.layout, episode_seed)
        agent0.reset()
        agent1.reset()
        env.multi_reset()
        done = False
        reward_sum = 0.0
        steps = 0

        while not done:
            state = env.base_env.state
            action0 = rllib_action_index(agent0, state)
            action1 = rllib_action_index(agent1, state)
            _, reward, done, info = env.multi_step(action0, action1)
            reward_sum += float(reward[0])
            steps += 1

        episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
        episode_stats = episode_info.get("ep_game_stats", {})
        for event_name, value in episode_stats.items():
            event_counts[event_name] += count_event_value(value)

        env.close()
        rewards.append(reward_sum)
        lengths.append(steps)
        print(
            f"Episode {episode_index + 1:02d}/{args.episodes}: "
            f"seed={episode_seed}, reward={reward_sum:.1f}, steps={steps}"
        )

    result = {
        "agent": str(agent_dir),
        "layout": args.layout,
        "episodes": args.episodes,
        "base_seed": args.seed,
        "mean_reward": statistics.fmean(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "success_rate": sum(reward > 0 for reward in rewards) / len(rewards),
        "mean_episode_length": statistics.fmean(lengths),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "event_counts": dict(event_counts),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
