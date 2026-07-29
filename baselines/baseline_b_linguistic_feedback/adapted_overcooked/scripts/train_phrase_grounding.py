"""Train phrase-to-feature TF-IDF + OneVsRest LogisticRegression grounding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from joblib import dump


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.overcooked_grounding import (  # noqa: E402
    DEFAULT_GROUNDING_MODEL_PATH,
    build_grounding_rows,
    train_phrase_grounding,
)
from src.evaluation_splits import make_train_dev_test_split  # noqa: E402


def _evaluate_and_tune(artifact: dict, rows: list[dict], *, tune: bool) -> dict:
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    if not rows:
        return {}
    matrix = artifact["vectorizer"].transform([row["processed"] for row in rows])
    probabilities = np.asarray(artifact["classifier"].predict_proba(matrix))
    classes = [str(label) for label in artifact["label_binarizer"].classes_]
    label_index = {label: index for index, label in enumerate(classes)}
    targets = np.zeros_like(probabilities, dtype=int)
    for row_index, row in enumerate(rows):
        for label in row["labels"]:
            if label in label_index:
                targets[row_index, label_index[label]] = 1
    thresholds = dict(artifact.get("feature_thresholds") or {})
    if tune:
        for column, label in enumerate(classes):
            best = (float("-inf"), 0.5)
            for threshold in np.linspace(0.2, 0.8, 25):
                predicted = probabilities[:, column] >= threshold
                _p, _r, f1, _support = precision_recall_fscore_support(
                    targets[:, column],
                    predicted,
                    average="binary",
                    zero_division=0,
                )
                candidate = (float(f1), float(threshold))
                if candidate > best:
                    best = candidate
            thresholds[label] = best[1]
        artifact["feature_thresholds"] = thresholds
    predicted = np.asarray(
        [
            probabilities[:, column] >= float(thresholds.get(label, 0.5))
            for column, label in enumerate(classes)
        ]
    ).T
    micro = precision_recall_fscore_support(
        targets, predicted, average="micro", zero_division=0
    )
    macro = precision_recall_fscore_support(
        targets, predicted, average="macro", zero_division=0
    )
    return {
        "rows": len(rows),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "synthetic_feedback.validated.json",
    )
    parser.add_argument("--augment", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_GROUNDING_MODEL_PATH)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    examples: list[dict] = []
    for path in [args.input, *args.augment]:
        examples.extend(json.loads(path.read_text(encoding="utf-8")))
    if any(example.get("split") in {"train", "dev", "test"} for example in examples):
        split_indices = {
            name: [
                index
                for index, example in enumerate(examples)
                if (example.get("split") or "train") == name
            ]
            for name in ("train", "dev", "test")
        }
        split_policy = "generator_declared_split"
    else:
        split = make_train_dev_test_split(
            [example.get("group_id") for example in examples],
            seed=args.seed,
            dev_fraction=0.15,
            test_fraction=0.15,
        )
        split_indices = {
            name: list(split[f"{name}_indices"])
            for name in ("train", "dev", "test")
        }
        split_policy = "scenario_grouped_legacy"
    split_examples = {
        name: [examples[index] for index in split_indices[name]]
        for name in ("train", "dev", "test")
    }
    artifact = train_phrase_grounding(
        split_examples["train"], min_df=args.min_df, seed=args.seed
    )
    rows = build_grounding_rows(split_examples["train"])
    dev_metrics = _evaluate_and_tune(
        artifact, build_grounding_rows(split_examples["dev"]), tune=True
    )
    test_metrics = _evaluate_and_tune(
        artifact, build_grounding_rows(split_examples["test"]), tune=False
    )
    artifact["manifest"] = {
        "input": str(args.input),
        "augment": [str(path) for path in args.augment],
        "min_df": args.min_df,
        "seed": args.seed,
        "split_policy": split_policy,
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "dev": dev_metrics,
        "untouched_test": test_metrics,
        "annotation_sources": {
            source: sum(row["annotation_source"] == source for row in rows)
            for source in sorted({row["annotation_source"] for row in rows})
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, args.output)
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(artifact["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved phrase grounding model: {args.output}")
    print(f"Training rows: {len(rows)}")
    print(f"Untouched test: {test_metrics}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
