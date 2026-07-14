"""Evaluate a short scripted prefix followed by archived RLlib PPO agents."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from durf.baseline.action_prior import (
    COUNTER_ONION_TO_POT_TOP,
    prefix_action_chars,
    prefix_action_indices,
)
from durf.baseline.evaluate_baseline import count_event_value
from durf.baseline.runtime import (
    load_rllib_agent,
    make_baseline_env,
    resolve_agent_dir,
    rllib_action_index,
)
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--prior", default=COUNTER_ONION_TO_POT_TOP)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Evaluate PPO suffix with argmax actions instead of stochastic sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for PPO suffix actions. Values below 1 sharpen the policy distribution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent_dir = resolve_agent_dir(args.agent)
    prefix_indices = prefix_action_indices(args.prior)
    stay = Action.ACTION_TO_INDEX[Action.STAY]

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
        info = {}

        for action0 in prefix_indices:
            if done:
                break
            _, reward, done, info = env.multi_step(action0, stay)
            reward_sum += float(reward[0])
            steps += 1

        while not done:
            state = env.base_env.state
            action0 = rllib_action_index(
                agent0,
                state,
                deterministic=args.deterministic,
                temperature=args.temperature,
            )
            action1 = rllib_action_index(
                agent1,
                state,
                deterministic=args.deterministic,
                temperature=args.temperature,
            )
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
        "prior": args.prior,
        "prefix_actions": prefix_action_chars(args.prior),
        "episodes": args.episodes,
        "base_seed": args.seed,
        "deterministic": args.deterministic,
        "temperature": args.temperature,
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
