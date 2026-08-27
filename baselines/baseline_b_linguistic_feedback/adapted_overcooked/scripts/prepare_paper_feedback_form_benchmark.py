"""Freeze the original-paper human reference labels as a three-class benchmark.

The source corpus is already split by original participant/task UUID and has no
normalized-text overlap across train, dev, and test.  This conversion follows
the paper's main-experiment operational mapping:

* trajectory -> Evaluative
* object/action behavior -> Evaluative
* object/action spatial -> Imperative
* feature -> Descriptive
* other -> excluded (no reward update)

The original notebook's strict three-way analysis excluded behavioral actions,
but the paper appendix explicitly says the main experiment treated object
behavior as Evaluative.  The manifest records both facts.

The emitted manifest contains only hashes for frozen-test membership.  Training
can therefore prove that it did not ingest a test utterance without opening the
test file or exposing its text.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_SOURCE = ROOT / "data" / "reference_classifier_paper_strict.v1.json"
DEFAULT_TRAIN = ROOT / "data" / "paper_feedback_form_human_train.v1.json"
DEFAULT_DEV = ROOT / "data" / "paper_feedback_form_human_dev.v1.json"
DEFAULT_FROZEN_TEST = (
    ROOT / "data" / "paper_feedback_form_human_frozen_test.v1.json"
)
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "feedback_form_classifier" / "paper_human_benchmark.v1.manifest.json"
)
BENCHMARK_VERSION = "paper-human-feedback-form-task-heldout-exposed-v2"
MAPPING_VERSION = "paper-main-experiment-reference-collapse-v1"
DIAGNOSTIC_ACCURACY_THRESHOLD = 0.87
SPLITS = ("train", "dev", "test")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
REFERENCE_TO_FEEDBACK = {
    "trajectory": "evaluative",
    "action_behavioral": "evaluative",
    "action_spatial": "imperative",
    "feature": "descriptive",
}
EXCLUDED_REFERENCE_TYPES = {"other"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def prepare_benchmark(source_rows: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    """Collapse five-class human labels without consulting any classifier."""

    converted: dict[str, list[dict]] = {split: [] for split in SPLITS}
    excluded = Counter()
    for index, raw in enumerate(source_rows, start=1):
        split = str(raw.get("split") or "")
        if split not in SPLITS:
            raise ValueError(f"source row {index}: invalid split {split!r}")
        reference_type = str(raw.get("reference_type") or "")
        if reference_type in EXCLUDED_REFERENCE_TYPES:
            excluded[reference_type] += 1
            continue
        if reference_type not in REFERENCE_TO_FEEDBACK:
            raise ValueError(
                f"source row {index}: unsupported reference_type {reference_type!r}"
            )
        text = str(raw.get("text") or "").strip()
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError(f"source row {index}: empty text")
        task_uuid = str(raw.get("paper_task_uuid") or "").strip()
        group_id = str(raw.get("group_id") or "").strip()
        if not task_uuid or not group_id:
            raise ValueError(f"source row {index}: missing paper task identity")
        feedback_type = REFERENCE_TO_FEEDBACK[reference_type]
        converted[split].append(
            {
                "feedback_id": str(
                    raw.get("feedback_id") or f"paper_feedback_form_{split}_{index}"
                ),
                "group_id": group_id,
                "paper_task_uuid": task_uuid,
                "paper_source_index": int(raw.get("paper_source_index", index - 1)),
                "text": text,
                "normalized_text": normalized,
                "reference_type": reference_type,
                "expected_feedback_type": feedback_type,
                "classification_label": CANONICAL_LABELS[feedback_type],
                "label_source": (
                    "original_paper_human_reference_main_experiment_mapping"
                ),
                "mapping_version": MAPPING_VERSION,
                "split": split,
            }
        )

    for split in SPLITS:
        converted[split].sort(
            key=lambda row: (int(row["paper_source_index"]), row["feedback_id"])
        )
        if not converted[split]:
            raise ValueError(f"paper feedback-form {split} split is empty")
        labels = {row["expected_feedback_type"] for row in converted[split]}
        if labels != set(CANONICAL_LABELS):
            raise ValueError(
                f"paper feedback-form {split} does not contain all labels: {sorted(labels)}"
            )

    text_sets = {
        split: {row["normalized_text"] for row in rows}
        for split, rows in converted.items()
    }
    group_sets = {
        split: {row["paper_task_uuid"] for row in rows}
        for split, rows in converted.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if text_sets[left] & text_sets[right]:
                raise ValueError(f"normalized-text leakage between {left} and {right}")
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"task leakage between {left} and {right}")

    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "mapping_version": MAPPING_VERSION,
        "mapping": dict(REFERENCE_TO_FEEDBACK),
        "strict_original_notebook_mapping": {
            "trajectory": "evaluative",
            "feature": "descriptive",
            "action_spatial": "imperative",
        },
        "paper_main_experiment_mapping": {
            "action_behavioral": "evaluative",
            "paper_notebook_exact": False,
            "paper_main_experiment_exact": True,
        },
        "excluded_reference_types": sorted(EXCLUDED_REFERENCE_TYPES),
        "excluded_counts": dict(sorted(excluded.items())),
        "source_rows": len(source_rows),
        "selection_uses_model_predictions": False,
        "benchmark_status": "previously_exposed_task_held_out_proxy_regression",
        "independent_sealed_test": False,
        "prior_exposure": (
            "the source paper-v1 test was evaluated before this three-class conversion"
        ),
        "diagnostic_accuracy_threshold": DIAGNOSTIC_ACCURACY_THRESHOLD,
        "split_unit": "original_paper_task_uuid",
        "split_sizes": {split: len(converted[split]) for split in SPLITS},
        "split_group_counts": {split: len(group_sets[split]) for split in SPLITS},
        "split_label_counts": {
            split: dict(
                sorted(Counter(row["expected_feedback_type"] for row in rows).items())
            )
            for split, rows in converted.items()
        },
        "group_overlap_count": 0,
        "normalized_text_overlap_count": 0,
        "frozen_test_membership": {
            "normalized_text_sha256": sorted(
                _digest(row["normalized_text"]) for row in converted["test"]
            ),
            "paper_task_uuid_sha256": sorted(
                {_digest(row["paper_task_uuid"]) for row in converted["test"]}
            ),
        },
    }
    return converted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--frozen-test", type=Path, default=DEFAULT_FROZEN_TEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    source_rows = _read_json_list(args.source)
    converted, report = prepare_benchmark(source_rows)
    outputs = {
        "train": args.train,
        "dev": args.dev,
        "frozen_test": args.frozen_test,
    }
    for split, path in (
        ("train", args.train),
        ("dev", args.dev),
        ("test", args.frozen_test),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, converted[split])

    report["source"] = {"path": str(args.source), "sha256": _sha256(args.source)}
    report["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items()
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.manifest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
