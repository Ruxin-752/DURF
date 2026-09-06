"""Train Hu-v0, a condition-aware pairwise subgoal reranker.

Input files are `hu_subgoal_preferences.jsonl` files produced by:

    python -m durf.feedback_attribution.demo_offline_attribution ...

You may pass either JSONL files or session directories containing that file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from durf.hu.subgoal_reranker import (
    DECISION_LEVELS,
    HierarchicalHu,
    load_pairwise_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="One or more hu_subgoal_preferences.jsonl files or session dirs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--test-dataset",
        type=Path,
        nargs="+",
        default=[],
        help=(
            "Held-out session(s) for per-participant evaluation: a later play "
            "session of the SAME participant who contributed --dataset. "
            "Enforced: every test user must have training data, and samples "
            "whose source_feedback_id already appears in the training data are "
            "skipped (a reused feedback can never leak into test metrics). "
            "Per-user test metrics are written to metadata.json."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def split_samples(samples, validation_fraction: float, seed: int):
    """Split each decision domain independently.

    Coordination feedback is usually much rarer than task feedback. A global
    random split can therefore leave the coordination head with no training
    examples even when coordination labels exist.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if not samples:
        return [], []
    rng = np.random.default_rng(seed)
    train = []
    val = []
    for decision_level in DECISION_LEVELS:
        domain_samples = [
            sample
            for sample in samples
            if sample.decision_level == decision_level
        ]
        order = rng.permutation(len(domain_samples))
        # Keep at least one training sample in every represented domain.
        val_size = min(
            int(round(len(domain_samples) * validation_fraction)),
            max(0, len(domain_samples) - 1),
        )
        val_indices = set(int(index) for index in order[:val_size])
        train.extend(
            sample
            for index, sample in enumerate(domain_samples)
            if index not in val_indices
        )
        val.extend(
            sample
            for index, sample in enumerate(domain_samples)
            if index in val_indices
        )
    return train, val


def dedupe_against_train(
    test_samples,
    train_samples,
) -> list[PairwiseSample]:
    """Drop test samples that share a source_feedback_id with training data.

    In the per-participant protocol the same participant contributes a
    collection session (train) and a later play session (test).  A feedback
    id appearing in both means the identical labelled pair was reused, which
    would inflate the test metrics; those samples are skipped.
    """
    train_ids = {
        sample.source_feedback_id
        for sample in train_samples
        if sample.source_feedback_id
    }
    if not train_ids:
        return test_samples
    return [
        sample
        for sample in test_samples
        if sample.source_feedback_id not in train_ids
    ]


def dedupe_samples(samples) -> list[PairwiseSample]:
    """Drop exact duplicate pairs inside one dataset.

    One feedback expands into several pairs (same source_feedback_id, different
    preferred/rejected), and those are all legitimate.  Only an identical
    (feedback, pair) tuple -- e.g. the same file passed twice -- is dropped.
    Samples without a source_feedback_id are kept as-is.
    """
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for sample in samples:
        if not sample.source_feedback_id:
            unique.append(sample)
            continue
        key = (
            sample.source_feedback_id,
            sample.preferred_subgoal,
            sample.rejected_subgoal,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def _feedback_timestamp(sample: PairwiseSample) -> datetime | None:
    """Recover the ISO timestamp embedded in a source_feedback_id.

    Feedback ids look like ``chat_messages.csv:49:2026-07-22T13:18:02.836+00:00``;
    the timestamp lets us verify the test session really came after training.
    """
    if not sample.source_feedback_id:
        return None
    for part in sample.source_feedback_id.split(":"):
        try:
            return datetime.fromisoformat(part)
        except ValueError:
            continue
    return None


def check_participant_protocol(
    train_samples,
    test_samples,
) -> dict[str, Any]:
    """Enforce the per-participant protocol on a train/test split.

    Each test session must be a *later play session of the same participant*
    who contributed the training data.  This checks:

    - every test user has training data (hard failure otherwise), and
    - test feedback does not predate the participant's latest training
      feedback (warning: --dataset/--test-dataset may be swapped).
    """
    train_users = {sample.user_id for sample in train_samples}
    test_users = {sample.user_id for sample in test_samples}
    unseen = sorted(test_users - train_users)
    checks: dict[str, Any] = {
        "test_users_without_training_data": unseen,
        "protocol_ok": not unseen,
        "temporal_warnings": [],
    }
    if unseen:
        raise ValueError(
            "Per-participant protocol violated: test data contains users with "
            f"no training data: {unseen}. Each test session must be a later "
            "play session of the same participant who contributed training data."
        )
    for user in sorted(test_users):
        train_times = [
            timestamp
            for sample in train_samples
            if sample.user_id == user
            for timestamp in [_feedback_timestamp(sample)]
            if timestamp is not None
        ]
        test_times = [
            timestamp
            for sample in test_samples
            if sample.user_id == user
            for timestamp in [_feedback_timestamp(sample)]
            if timestamp is not None
        ]
        if train_times and test_times and min(test_times) < max(train_times):
            checks["temporal_warnings"].append(
                f"user {user}: test feedback predates the latest training "
                "feedback; --dataset and --test-dataset may be swapped"
            )
    return checks


def evaluate_by_user(
    model: HierarchicalHu,
    samples: list[PairwiseSample],
) -> dict[str, dict[str, Any]]:
    """Per-participant test metrics, the unit the protocol is evaluated on."""
    return {
        user: model.evaluate(
            [sample for sample in samples if sample.user_id == user]
        )
        for user in sorted({sample.user_id for sample in samples})
    }


def main() -> int:
    args = parse_args()
    samples = load_pairwise_samples(args.dataset)
    if not samples:
        raise ValueError(
            "No Hu pairwise samples found. Run offline attribution on a session "
            "with attributed natural-language feedback first."
        )
    samples = dedupe_samples(samples)

    train_samples, val_samples = split_samples(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not train_samples:
        raise ValueError("Validation split left no training samples")

    test_samples = []
    protocol_checks: dict[str, Any] = {}
    test_metrics_by_user: dict[str, Any] = {}
    if args.test_dataset:
        loaded_test = load_pairwise_samples(args.test_dataset)
        test_samples = dedupe_against_train(loaded_test, train_samples)
        protocol_checks = check_participant_protocol(train_samples, test_samples)

    model = HierarchicalHu.from_samples(samples, seed=args.seed)
    history = model.train(
        train_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hierarchical_hu.json"
    model.save(model_path)
    if test_samples:
        test_metrics_by_user = evaluate_by_user(model, test_samples)
    summary = {
        "model": str(model_path),
        "datasets": [str(path) for path in args.dataset],
        "test_datasets": [str(path) for path in args.test_dataset],
        "samples_total": len(samples),
        "samples_train": len(train_samples),
        "samples_validation": len(val_samples),
        "samples_test": len(test_samples),
        "train_users": sorted({sample.user_id for sample in train_samples}),
        "test_users": sorted({sample.user_id for sample in test_samples}),
        "protocol_checks": protocol_checks,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "train_metrics_by_level": model.evaluate(train_samples),
        "validation_metrics_by_level": model.evaluate(val_samples),
        "test_metrics_by_level": model.evaluate(test_samples),
        "test_metrics_by_user": test_metrics_by_user,
        "history": history,
        "top_condition_weights_by_level": model.top_condition_weights(limit=30),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
