"""Build BC data from successful human-played pygame sessions.

The Group A pygame interface logs joint actions in ``trajectory.csv``.  This
script selects successful episodes and replays their joint action sequence from
the environment reset state, saving observations paired with the human player's
actions as an imitation dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from durf.baseline.runtime import make_direct_multi_env, resolve_agent_dir
from human_aware_rl.rllib.rllib import load_trainer
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--reference-agent",
        required=True,
        help="Installed RLlib agent whose featurizer defines the BC observation space.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument(
        "--human-player-index",
        type=int,
        choices=(0, 1),
        default=1,
        help="The player slot controlled by the human in the session.",
    )
    parser.add_argument(
        "--policy-id",
        default="ppo_1",
        help="Policy id whose featurizer should be used for the human slot.",
    )
    parser.add_argument(
        "--drop-stay-fraction",
        type=float,
        default=0.0,
        help=(
            "Optionally downsample STAY labels by dropping this fraction. "
            "Use 0.0 to keep the raw demonstration."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def featurize_state(featurize_fn, state, agent_index: int) -> np.ndarray:
    try:
        obs_pair = featurize_fn(state, debug=False)
    except TypeError:
        obs_pair = featurize_fn(state)
    return np.asarray(obs_pair[agent_index], dtype=np.float32)


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def episode_key(row: dict[str, str]) -> tuple[str, str]:
    return row["layout"], row["episode"]


def session_horizon(session_dir: Path) -> int:
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        return 400
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("horizon") or 400)


def main() -> int:
    args = parse_args()
    trajectory_path = args.session / "trajectory.csv"
    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)
    if not 0.0 <= args.drop_stay_fraction < 1.0:
        raise ValueError("--drop-stay-fraction must be in [0, 1)")

    rows = read_csv(trajectory_path)
    if not rows:
        raise RuntimeError(f"No trajectory rows found: {trajectory_path}")

    episodes: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        episodes[episode_key(row)].append(row)

    reference_dir = resolve_agent_dir(args.reference_agent)
    trainer = load_trainer(str(reference_dir))
    output_dir = args.output_dir or (args.session / "bc_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    horizon = session_horizon(args.session)

    observations: list[np.ndarray] = []
    labels: list[int] = []
    kept_episodes: list[dict] = []
    skipped_episodes: list[dict] = []
    label_counts: Counter[int] = Counter()

    try:
        dummy_env = trainer.env_creator(trainer.config["env_config"])
        featurize_fn = dummy_env._get_featurize_fn(args.policy_id)

        for (layout, episode), episode_rows in sorted(
            episodes.items(),
            key=lambda item: (as_int(item[0][1]), item[0][0]),
        ):
            episode_rows = sorted(
                episode_rows,
                key=lambda row: as_int(row["episode_step"]),
            )
            final_reward = as_float(episode_rows[-1]["episode_reward"])
            if final_reward < args.success_threshold:
                skipped_episodes.append(
                    {
                        "layout": layout,
                        "episode": as_int(episode),
                        "final_reward": final_reward,
                        "steps": len(episode_rows),
                    }
                )
                continue

            env = make_direct_multi_env(layout, seed=0, horizon=horizon)
            env.multi_reset()
            episode_examples = 0
            try:
                for row in episode_rows:
                    state = env.base_env.state
                    label = as_int(
                        row["human_action"]
                        if args.human_player_index == 1
                        else row["ai_action"]
                    )
                    if (
                        label == Action.ACTION_TO_INDEX[Action.STAY]
                        and args.drop_stay_fraction > 0
                        and rng.random() < args.drop_stay_fraction
                    ):
                        pass
                    else:
                        observations.append(
                            featurize_state(
                                featurize_fn,
                                state,
                                args.human_player_index,
                            )
                        )
                        labels.append(label)
                        label_counts[label] += 1
                        episode_examples += 1
                    env.multi_step(as_int(row["ai_action"]), as_int(row["human_action"]))
            finally:
                env.close()

            kept_episodes.append(
                {
                    "layout": layout,
                    "episode": as_int(episode),
                    "final_reward": final_reward,
                    "steps": len(episode_rows),
                    "examples": episode_examples,
                }
            )

        if not observations:
            raise RuntimeError(
                "No successful demonstration examples found. "
                "Play until at least one episode has reward >= success threshold."
            )

        observation_array = np.stack(observations).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        np.savez_compressed(
            output_dir / "human_demo_policy.npz",
            observations=observation_array,
            labels=label_array,
        )
        metadata = {
            "session": str(args.session),
            "reference_agent": str(reference_dir),
            "policy_id": args.policy_id,
            "human_player_index": args.human_player_index,
            "success_threshold": args.success_threshold,
            "drop_stay_fraction": args.drop_stay_fraction,
            "horizon": horizon,
            "episodes_total": len(episodes),
            "episodes_kept": len(kept_episodes),
            "examples": int(label_array.shape[0]),
            "observation_shape": list(observation_array.shape),
            "label_counts": {
                str(Action.INDEX_TO_ACTION[index]): int(label_counts[index])
                for index in range(Action.NUM_ACTIONS)
            },
            "kept_episodes": kept_episodes,
            "skipped_episodes": skipped_episodes,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"Saved BC dataset: {output_dir / 'human_demo_policy.npz'}")
    finally:
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
