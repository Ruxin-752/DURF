"""Collect successful PPO rollouts as full-chain imitation data.

This script samples an installed RLlib PPO agent from the real reset
distribution, keeps only episodes whose sparse reward indicates at least one
successful delivery, and writes both:

* human-readable JSONL trajectory records for inspection;
* per-policy NPZ files that can be used by the existing supervised update tools.

The purpose is to turn rare stochastic successes into explicit full-chain
examples instead of hoping PPO rediscovers the same action sequence later.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import dill
import numpy as np

from durf.baseline.evaluate_baseline import count_event_value
from durf.baseline.runtime import (
    load_rllib_agent,
    make_direct_multi_env,
    resolve_agent_dir,
    rllib_action_index,
)
from human_aware_rl.rllib.rllib import load_trainer
from overcooked_ai_py.mdp.actions import Action


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "success_trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-successes", type=int, default=50)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Keep episodes whose total sparse reward is at least this value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for PPO actions.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions. Usually not useful for the current stochastic ring agents.",
    )
    parser.add_argument("--policy-id-0", default="ppo_0")
    parser.add_argument("--policy-id-1", default="ppo_1")
    return parser.parse_args()


def load_params_for_agent(agent_dir: Path) -> dict:
    config_path = agent_dir.parent / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("rb") as handle:
        return dill.load(handle)


def object_name(obj) -> str | None:
    if obj is None:
        return None
    return getattr(obj, "name", str(obj))


def state_summary(state) -> dict:
    players = []
    for player in state.players:
        players.append(
            {
                "position": list(player.position),
                "orientation": list(player.orientation),
                "held_object": object_name(player.held_object),
            }
        )
    objects = []
    for position, obj in sorted(state.objects.items()):
        objects.append(
            {
                "position": list(position),
                "name": object_name(obj),
                "is_ready": bool(getattr(obj, "is_ready", False)),
                "is_cooking": bool(getattr(obj, "is_cooking", False)),
            }
        )
    return {
        "timestep": int(state.timestep),
        "players": players,
        "objects": objects,
    }


def featurize_state(featurize_fn, state, agent_index: int) -> np.ndarray:
    try:
        obs_pair = featurize_fn(state, debug=False)
    except TypeError:
        obs_pair = featurize_fn(state)
    return np.asarray(obs_pair[agent_index], dtype=np.float32)


def episode_event_counts(info: dict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
    episode_stats = episode_info.get("ep_game_stats", {})
    for event_name, value in episode_stats.items():
        counts[event_name] += count_event_value(value)
    return dict(counts)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_successes <= 0:
        raise ValueError("--max-successes must be positive")

    agent_dir = resolve_agent_dir(args.agent)
    output_dir = args.output_dir or (
        OUTPUT_ROOT
        / f"{agent_dir.parent.name}_{args.layout}_seed{args.seed}_successes{args.max_successes}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer(str(agent_dir))
    agent0 = load_rllib_agent(args.agent, agent_index=0, policy_id=args.policy_id_0)
    agent1 = load_rllib_agent(args.agent, agent_index=1, policy_id=args.policy_id_1)
    try:
        dummy_env = trainer.env_creator(trainer.config["env_config"])
        featurize0 = dummy_env._get_featurize_fn(args.policy_id_0)
        featurize1 = dummy_env._get_featurize_fn(args.policy_id_1)

        policy_observations = {0: [], 1: []}
        policy_labels = {0: [], 1: []}
        trajectory_steps: list[dict] = []
        successful_episodes: list[dict] = []
        attempted = 0

        for episode_index in range(args.episodes):
            if len(successful_episodes) >= args.max_successes:
                break

            episode_seed = args.seed + episode_index
            attempted += 1
            env = make_direct_multi_env(args.layout, episode_seed)
            agent0.reset()
            agent1.reset()
            env.multi_reset()
            done = False
            reward_sum = 0.0
            episode_records: list[dict] = []
            episode_obs0: list[np.ndarray] = []
            episode_obs1: list[np.ndarray] = []
            episode_labels0: list[int] = []
            episode_labels1: list[int] = []
            info: dict = {}

            try:
                while not done:
                    state = copy.deepcopy(env.base_env.state)
                    obs0 = featurize_state(featurize0, state, 0)
                    obs1 = featurize_state(featurize1, state, 1)
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
                    reward_value = float(reward[0])
                    reward_sum += reward_value
                    step = len(episode_records)
                    episode_obs0.append(obs0)
                    episode_obs1.append(obs1)
                    episode_labels0.append(int(action0))
                    episode_labels1.append(int(action1))
                    episode_records.append(
                        {
                            "record_type": "success_trajectory_step",
                            "episode_seed": episode_seed,
                            "step": step,
                            "action0": int(action0),
                            "action1": int(action1),
                            "action0_name": str(Action.INDEX_TO_ACTION[int(action0)]),
                            "action1_name": str(Action.INDEX_TO_ACTION[int(action1)]),
                            "reward": reward_value,
                            "state_before": state_summary(state),
                        }
                    )
            finally:
                env.close()

            success = reward_sum >= args.success_threshold
            event_counts = episode_event_counts(info)
            print(
                f"Episode {episode_index + 1:04d}/{args.episodes}: "
                f"seed={episode_seed}, reward={reward_sum:.1f}, success={success}"
            )
            if not success:
                continue

            success_id = len(successful_episodes)
            for record in episode_records:
                record["success_id"] = success_id
            trajectory_steps.extend(episode_records)
            policy_observations[0].extend(episode_obs0)
            policy_observations[1].extend(episode_obs1)
            policy_labels[0].extend(episode_labels0)
            policy_labels[1].extend(episode_labels1)
            successful_episodes.append(
                {
                    "record_type": "successful_episode",
                    "success_id": success_id,
                    "episode_seed": episode_seed,
                    "reward": reward_sum,
                    "steps": len(episode_records),
                    "event_counts": event_counts,
                }
            )

        write_jsonl(output_dir / "successful_episodes.jsonl", successful_episodes)
        write_jsonl(output_dir / "trajectory_steps.jsonl", trajectory_steps)
        for agent_index in (0, 1):
            if not policy_observations[agent_index]:
                continue
            observations = np.stack(policy_observations[agent_index]).astype(np.float32)
            labels = np.asarray(policy_labels[agent_index], dtype=np.int32)
            np.savez_compressed(
                output_dir / f"policy_{agent_index}.npz",
                observations=observations,
                labels=labels,
            )

        metadata = {
            "agent": str(agent_dir),
            "layout": args.layout,
            "episodes_requested": args.episodes,
            "episodes_attempted": attempted,
            "successes_collected": len(successful_episodes),
            "success_threshold": args.success_threshold,
            "base_seed": args.seed,
            "temperature": args.temperature,
            "deterministic": args.deterministic,
            "policy_id_0": args.policy_id_0,
            "policy_id_1": args.policy_id_1,
            "steps_collected": len(trajectory_steps),
            "label_counts_policy_0": np.bincount(
                np.asarray(policy_labels[0], dtype=np.int32),
                minlength=Action.NUM_ACTIONS,
            ).tolist(),
            "label_counts_policy_1": np.bincount(
                np.asarray(policy_labels[1], dtype=np.int32),
                minlength=Action.NUM_ACTIONS,
            ).tolist(),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"Saved success dataset: {output_dir}")
    finally:
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
