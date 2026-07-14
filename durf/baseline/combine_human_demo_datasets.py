"""Combine and lightly clean human demonstration BC datasets.

Each Group A session can be converted into a ``bc_dataset/human_demo_policy.npz``
with ``build_human_demo_dataset``.  This script concatenates those per-session
files into one training dataset while dropping obviously invalid examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from overcooked_ai_py.mdp.actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("outputs") / "human_ai_sessions",
        help="Directory containing Group A session folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the combined dataset.npz and metadata.json.",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1,
        help="Skip per-session datasets with fewer examples than this.",
    )
    parser.add_argument(
        "--max-stay-fraction",
        type=float,
        default=0.35,
        help="Skip a per-session dataset if STAY labels exceed this fraction.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Shuffle examples after concatenation. Use a negative value to disable.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if not args.sessions_root.exists():
        raise FileNotFoundError(args.sessions_root)
    if not 0.0 <= args.max_stay_fraction <= 1.0:
        raise ValueError("--max-stay-fraction must be in [0, 1]")

    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    included: list[dict] = []
    skipped: list[dict] = []
    expected_shape: tuple[int, ...] | None = None

    for session_dir in sorted(p for p in args.sessions_root.iterdir() if p.is_dir()):
        dataset_path = session_dir / "bc_dataset" / "human_demo_policy.npz"
        metadata_path = session_dir / "bc_dataset" / "metadata.json"
        if not dataset_path.exists():
            continue

        metadata = load_json(metadata_path)
        data = np.load(dataset_path)
        obs = data["observations"].astype(np.float32)
        y = data["labels"].astype(np.int32)
        reason = None
        if obs.shape[0] != y.shape[0]:
            reason = f"length mismatch: {obs.shape[0]} observations vs {y.shape[0]} labels"
        elif obs.shape[0] < args.min_examples:
            reason = f"too few examples: {obs.shape[0]}"
        elif obs.ndim < 2:
            reason = f"unexpected observation ndim: {obs.ndim}"
        elif expected_shape is not None and tuple(obs.shape[1:]) != expected_shape:
            reason = f"observation shape {tuple(obs.shape[1:])} != {expected_shape}"

        if reason is not None:
            skipped.append({"session": session_dir.name, "reason": reason})
            continue

        valid = (
            np.isfinite(obs).all(axis=tuple(range(1, obs.ndim)))
            & np.isfinite(y)
            & (y >= 0)
            & (y < Action.NUM_ACTIONS)
        )
        dropped_invalid = int((~valid).sum())
        obs = obs[valid]
        y = y[valid]
        if obs.shape[0] == 0:
            skipped.append(
                {"session": session_dir.name, "reason": "all examples invalid"}
            )
            continue

        counts = np.bincount(y, minlength=Action.NUM_ACTIONS)
        stay_fraction = float(counts[Action.ACTION_TO_INDEX[Action.STAY]] / y.shape[0])
        if stay_fraction > args.max_stay_fraction:
            skipped.append(
                {
                    "session": session_dir.name,
                    "reason": f"STAY fraction {stay_fraction:.3f} > {args.max_stay_fraction:.3f}",
                }
            )
            continue

        if expected_shape is None:
            expected_shape = tuple(obs.shape[1:])
        arrays.append(obs)
        labels.append(y)
        included.append(
            {
                "session": session_dir.name,
                "dataset": str(dataset_path),
                "examples": int(y.shape[0]),
                "episodes_kept": metadata.get("episodes_kept"),
                "horizon": metadata.get("horizon"),
                "dropped_invalid_examples": dropped_invalid,
                "stay_fraction": stay_fraction,
                "label_counts": {
                    str(Action.INDEX_TO_ACTION[index]): int(counts[index])
                    for index in range(Action.NUM_ACTIONS)
                },
            }
        )

    if not arrays:
        raise RuntimeError("No usable per-session BC datasets were found.")

    observations = np.concatenate(arrays, axis=0).astype(np.float32)
    combined_labels = np.concatenate(labels, axis=0).astype(np.int32)
    if args.shuffle_seed >= 0:
        rng = np.random.default_rng(args.shuffle_seed)
        order = rng.permutation(combined_labels.shape[0])
        observations = observations[order]
        combined_labels = combined_labels[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "dataset.npz",
        observations=observations,
        labels=combined_labels,
    )
    counts = np.bincount(combined_labels, minlength=Action.NUM_ACTIONS)
    metadata = {
        "sessions_root": str(args.sessions_root),
        "output_dataset": str(args.output_dir / "dataset.npz"),
        "examples": int(combined_labels.shape[0]),
        "observation_shape": list(observations.shape),
        "label_counts": {
            str(Action.INDEX_TO_ACTION[index]): int(counts[index])
            for index in range(Action.NUM_ACTIONS)
        },
        "included_sessions": included,
        "skipped_sessions": skipped,
        "settings": {
            "min_examples": args.min_examples,
            "max_stay_fraction": args.max_stay_fraction,
            "shuffle_seed": args.shuffle_seed,
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
