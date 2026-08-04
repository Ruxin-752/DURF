"""Train Hu-v0, a condition-aware pairwise subgoal reranker.

Input files are `hu_subgoal_preferences.jsonl` files produced by:

    python -m durf.feedback_attribution.demo_offline_attribution ...

You may pass either JSONL files or session directories containing that file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> int:
    args = parse_args()
    samples = load_pairwise_samples(args.dataset)
    if not samples:
        raise ValueError(
            "No Hu pairwise samples found. Run offline attribution on a session "
            "with attributed natural-language feedback first."
        )

    train_samples, val_samples = split_samples(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not train_samples:
        raise ValueError("Validation split left no training samples")

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
    summary = {
        "model": str(model_path),
        "datasets": [str(path) for path in args.dataset],
        "samples_total": len(samples),
        "samples_train": len(train_samples),
        "samples_validation": len(val_samples),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "train_metrics_by_level": model.evaluate(train_samples),
        "validation_metrics_by_level": model.evaluate(val_samples),
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
