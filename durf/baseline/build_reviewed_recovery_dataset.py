"""Build a supervised recovery dataset from human-reviewed labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from overcooked_ai_py.mdp.actions import Action


ACTION_TEXT_TO_INDEX = {
    "north": Action.ACTION_TO_INDEX[(0, -1)],
    "south": Action.ACTION_TO_INDEX[(0, 1)],
    "east": Action.ACTION_TO_INDEX[(1, 0)],
    "west": Action.ACTION_TO_INDEX[(-1, 0)],
    "stay": Action.ACTION_TO_INDEX[(0, 0)],
    "interact": Action.ACTION_TO_INDEX["interact"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-reasonable-scaffold",
        action="store_true",
        help=(
            "If a review marks the scaffold label as reasonable but leaves "
            "preferred_ai_action blank, keep the original scaffold label."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def key_for(record: dict) -> tuple[int, int, int]:
    return (
        int(record["episode_index"]),
        int(record["seed"]),
        int(record["step"]),
    )


def normalize_action(action: str | None) -> str | None:
    if action is None:
        return None
    normalized = str(action).strip().lower()
    return normalized or None


def scaffold_action_from_review(record: dict) -> str | None:
    return normalize_action(record.get("scaffold_action0"))


def preferred_action_from_review(record: dict, include_reasonable_scaffold: bool) -> str | None:
    review = record.get("human_review") or {}
    preferred = normalize_action(review.get("preferred_ai_action"))
    if preferred:
        return preferred
    if include_reasonable_scaffold and review.get("is_scaffold_label_reasonable") is True:
        return scaffold_action_from_review(record)
    return None


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.source_jsonl)
    reviewed_records = read_jsonl(args.reviewed_jsonl)
    source_data = np.load(args.source_npz)
    observations = source_data["observations"].astype(np.float32)
    if len(source_records) != observations.shape[0]:
        raise ValueError(
            "source-jsonl and source-npz are not aligned: "
            f"{len(source_records)} records vs {observations.shape[0]} observations"
        )
    source_index = {key_for(record): idx for idx, record in enumerate(source_records)}

    selected_observations: list[np.ndarray] = []
    selected_labels: list[int] = []
    exported_records: list[dict] = []
    skipped: list[dict] = []

    for review_record in reviewed_records:
        action = preferred_action_from_review(
            review_record,
            args.include_reasonable_scaffold,
        )
        if action is None:
            skipped.append({"review_id": review_record.get("review_id"), "reason": "no_action"})
            continue
        if action not in ACTION_TEXT_TO_INDEX:
            skipped.append(
                {
                    "review_id": review_record.get("review_id"),
                    "reason": f"unknown_action:{action}",
                }
            )
            continue
        key = key_for(review_record)
        if key not in source_index:
            skipped.append(
                {
                    "review_id": review_record.get("review_id"),
                    "reason": f"source_state_not_found:{key}",
                }
            )
            continue
        source_i = source_index[key]
        selected_observations.append(observations[source_i])
        selected_labels.append(ACTION_TEXT_TO_INDEX[action])
        exported = dict(review_record)
        exported["compiled_action"] = action
        exported["compiled_action_index"] = ACTION_TEXT_TO_INDEX[action]
        exported_records.append(exported)

    if not selected_labels:
        raise RuntimeError("No reviewed labels were compiled.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(selected_labels, dtype=np.int32)
    np.savez_compressed(
        args.output_dir / "dataset.npz",
        observations=np.stack(selected_observations).astype(np.float32),
        labels=labels,
    )
    with (args.output_dir / "compiled_reviewed_labels.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in exported_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "source_npz": str(args.source_npz),
        "reviewed_jsonl": str(args.reviewed_jsonl),
        "compiled_examples": len(selected_labels),
        "label_counts": np.bincount(labels, minlength=Action.NUM_ACTIONS).tolist(),
        "skipped": skipped,
        "output_dataset": str(args.output_dir / "dataset.npz"),
    }
    (args.output_dir / "compile_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
