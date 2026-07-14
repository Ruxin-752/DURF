"""Combine subgoal-conditioned executor datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from durf.baseline.collect_rule_teacher_dataset import SUBGOALS
from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("datasets", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations = []
    labels = []
    subgoals = []
    included = []
    expected_shape = None

    for dataset in args.datasets:
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        data = np.load(dataset)
        obs = data["observations"].astype(np.float32)
        y = data["labels"].astype(np.int32)
        sg = data["subgoals"].astype(np.int32)
        if not (obs.shape[0] == y.shape[0] == sg.shape[0]):
            raise ValueError(f"Length mismatch in {dataset}")
        if expected_shape is None:
            expected_shape = tuple(obs.shape[1:])
        elif tuple(obs.shape[1:]) != expected_shape:
            raise ValueError(
                f"Observation shape mismatch in {dataset}: {obs.shape[1:]} != {expected_shape}"
            )
        valid = (
            np.isfinite(obs).all(axis=tuple(range(1, obs.ndim)))
            & (y >= 0)
            & (y < Action.NUM_ACTIONS)
            & (sg >= 0)
            & (sg < len(SUBGOALS))
        )
        dropped = int((~valid).sum())
        obs = obs[valid]
        y = y[valid]
        sg = sg[valid]
        observations.append(obs)
        labels.append(y)
        subgoals.append(sg)
        included.append(
            {
                "dataset": str(dataset),
                "examples": int(y.shape[0]),
                "dropped_invalid_examples": dropped,
            }
        )

    obs_all = np.concatenate(observations, axis=0).astype(np.float32)
    y_all = np.concatenate(labels, axis=0).astype(np.int32)
    sg_all = np.concatenate(subgoals, axis=0).astype(np.int32)
    rng = np.random.default_rng(args.shuffle_seed)
    order = rng.permutation(y_all.shape[0])
    obs_all = obs_all[order]
    y_all = y_all[order]
    sg_all = sg_all[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dataset.npz",
        observations=obs_all,
        labels=y_all,
        subgoals=sg_all,
    )
    label_counts = Counter(y_all.tolist())
    subgoal_counts = Counter(sg_all.tolist())
    metadata = {
        "output_dataset": str(args.output_dir / "dataset.npz"),
        "examples": int(y_all.shape[0]),
        "observation_shape": list(obs_all.shape),
        "included": included,
        "shuffle_seed": args.shuffle_seed,
        "subgoals": list(SUBGOALS),
        "subgoal_counts": {
            name: int(subgoal_counts[index])
            for index, name in enumerate(SUBGOALS)
        },
        "label_counts": {
            str(Action.INDEX_TO_ACTION[index]): int(label_counts[index])
            for index in range(Action.NUM_ACTIONS)
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
