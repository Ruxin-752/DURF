"""Evaluate a frozen feedback-form classifier on its locked synthetic test split.

The target is derived only from ``reference_type``:

* trajectory -> evaluative
* feature -> descriptive
* action_spatial -> imperative

Behavioral-action and other rows are excluded.  This evaluator never trains or
selects a model.  It follows the synthetic corpus path embedded in the model
artifact, verifies its SHA256, rejects train/dev text or group overlap, and
checks that the model bytes are unchanged after evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_PATH,
    FEEDBACK_TYPES,
)


DEFAULT_REPORT = (
    ROOT / "outputs" / "feedback_form_classifier" / "synthetic_test.report.json"
)
REFERENCE_TYPE_TO_FEEDBACK_TYPE = {
    "trajectory": "evaluative",
    "feature": "descriptive",
    "action_spatial": "imperative",
}
EXCLUDED_REFERENCE_TYPES = frozenset(
    {"action_behavioral", "behavioral", "other"}
)
ALLOWED_SPLITS = frozenset({"train", "dev", "test"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _normalized_digest(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"synthetic corpus must be a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _manifest_corpus_path(model_path: Path, manifest: dict) -> Path:
    supplied = manifest.get("synthetic_input")
    if not isinstance(supplied, str) or not supplied.strip():
        raise ValueError("artifact manifest is missing synthetic_input")
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = model_path.parent / candidate
    return candidate.resolve(strict=True)


def _synthetic_group_id(row: dict, normalized: str, *, row_number: int) -> str:
    raw_group = row.get("group_id") or row.get("paraphrase_family") or row.get(
        "feedback_id"
    )
    if raw_group is None or not str(raw_group).strip():
        raw_group = f"synthetic:{_normalized_digest(normalized)}"
    # Match the training preparer's source namespace exactly.
    return f"synthetic:{raw_group}"


def _validated_membership_hashes(artifact: dict) -> tuple[set[str], set[str]]:
    membership = artifact.get("data_membership")
    if not isinstance(membership, dict):
        raise ValueError("artifact is missing data_membership")

    def load(name: str) -> set[str]:
        values = membership.get(name)
        if not isinstance(values, list):
            raise ValueError(f"artifact data_membership.{name} must be a list")
        output = {
            _require_sha256(value, label=f"data_membership.{name}") for value in values
        }
        if len(output) != len(values):
            raise ValueError(f"artifact data_membership.{name} contains duplicates")
        return output

    train = load("train_normalized_text_sha256")
    dev = load("dev_normalized_text_sha256")
    overlap = train & dev
    if overlap:
        raise ValueError(
            f"artifact train/dev normalized-text membership overlaps: {len(overlap)}"
        )
    return train, dev


def _prepare_corpus_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    membership = {
        "train": {"texts": set(), "groups": set()},
        "dev": {"texts": set(), "groups": set()},
    }
    test_rows: list[dict] = []
    excluded_test = Counter()
    eligible_counts = Counter()
    seen_test_texts: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        split = row.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"synthetic row {row_number}: split must be one of {sorted(ALLOWED_SPLITS)}"
            )
        reference_type = row.get("reference_type")
        if reference_type in EXCLUDED_REFERENCE_TYPES:
            if split == "test":
                excluded_test[str(reference_type)] += 1
            continue
        label = REFERENCE_TYPE_TO_FEEDBACK_TYPE.get(str(reference_type))
        if label is None:
            raise ValueError(
                f"synthetic row {row_number}: unsupported reference_type "
                f"{reference_type!r}"
            )
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"synthetic row {row_number}: missing non-empty text")
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError(
                f"synthetic row {row_number}: text is empty after normalization"
            )
        digest = _normalized_digest(normalized)
        group_id = _synthetic_group_id(row, normalized, row_number=row_number)
        eligible_counts[str(split)] += 1

        if split in {"train", "dev"}:
            membership[str(split)]["texts"].add(digest)
            membership[str(split)]["groups"].add(group_id)
            continue

        if normalized in seen_test_texts:
            raise ValueError(
                f"duplicate normalized synthetic test text at row {row_number}"
            )
        seen_test_texts.add(normalized)
        origin = row.get("classifier_corpus_origin")
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError(
                f"synthetic test row {row_number}: classifier_corpus_origin is required"
            )
        test_rows.append(
            {
                "text": text.strip(),
                "normalized_text": normalized,
                "text_sha256": digest,
                "group_id": group_id,
                "label": label,
                "reference_type": str(reference_type),
                "classifier_corpus_origin": origin.strip(),
            }
        )

    if not test_rows:
        raise ValueError("locked synthetic corpus contains no eligible test rows")
    missing_labels = sorted(set(FEEDBACK_TYPES) - {row["label"] for row in test_rows})
    if missing_labels:
        raise ValueError(
            f"synthetic test requires all three feedback classes; missing={missing_labels}"
        )
    train_dev_group_overlap = (
        membership["train"]["groups"] & membership["dev"]["groups"]
    )
    if train_dev_group_overlap:
        raise ValueError(
            "synthetic train/dev group membership overlaps: "
            f"{len(train_dev_group_overlap)}"
        )
    return test_rows, {
        "membership": membership,
        "eligible_counts": dict(sorted(eligible_counts.items())),
        "excluded_test_reference_counts": dict(sorted(excluded_test.items())),
    }


def _metrics(labels: list[str], predictions: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    details = classification_report(
        labels,
        predictions,
        labels=list(FEEDBACK_TYPES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(FEEDBACK_TYPES),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "canonical_label": CANONICAL_FEEDBACK_LABELS[label],
                "precision": float(details[label]["precision"]),
                "recall": float(details[label]["recall"]),
                "f1": float(details[label]["f1-score"]),
                "support": int(details[label]["support"]),
            }
            for label in FEEDBACK_TYPES
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(FEEDBACK_TYPES)
        ).tolist(),
        "label_order": list(FEEDBACK_TYPES),
        "canonical_label_order": [
            CANONICAL_FEEDBACK_LABELS[label] for label in FEEDBACK_TYPES
        ],
    }


def _validate_artifact(artifact: Any) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("feedback-form model artifact must be a dictionary")
    if artifact.get("model_type") != "feedback_form_tfidf_logistic_regression":
        raise ValueError("not a feedback-form classifier artifact")
    if artifact.get("input_mode") != "raw_text":
        raise ValueError("feedback-form artifact input_mode must be raw_text")
    if tuple(artifact.get("labels") or ()) != FEEDBACK_TYPES:
        raise ValueError("feedback-form artifact label contract mismatch")
    if "vectorizer" not in artifact or "classifier" not in artifact:
        raise ValueError("feedback-form artifact is missing vectorizer/classifier")
    classes = {str(value) for value in artifact["classifier"].classes_}
    if classes != set(FEEDBACK_TYPES):
        raise ValueError("feedback-form classifier classes do not match label contract")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("feedback-form artifact is missing its manifest")
    if manifest.get("human_test_loaded") is not False:
        raise ValueError("artifact manifest does not prove test was excluded from training")
    return manifest


def evaluate(
    model_path: str | Path,
    *,
    training_report_path: str | Path | None = None,
    expected_model_sha256: str | None = None,
) -> dict:
    """Evaluate only the locked synthetic test rows and return a report object."""

    from joblib import load

    model = Path(model_path).resolve(strict=True)
    model_sha_before = _sha256(model)
    artifact = load(model)
    manifest = _validate_artifact(artifact)

    training_report = Path(
        training_report_path
        if training_report_path is not None
        else model.with_suffix(".report.json")
    ).resolve(strict=True)
    report_value = json.loads(training_report.read_text(encoding="utf-8"))
    if not isinstance(report_value, dict):
        raise ValueError("feedback-form training report must be a JSON object")
    report_model_sha = _require_sha256(
        report_value.get("artifact_model_sha256"),
        label="training report artifact_model_sha256",
    )
    if report_model_sha != model_sha_before:
        raise ValueError("model SHA256 does not match the frozen training report")
    if report_value.get("artifact_manifest") != manifest:
        raise ValueError("training report and model artifact manifests differ")
    training_report_sha = _sha256(training_report)

    manifest_model_sha = manifest.get("model_sha256")
    if manifest_model_sha is not None:
        manifest_model_sha = _require_sha256(
            manifest_model_sha, label="artifact manifest model_sha256"
        )
    if expected_model_sha256 is not None:
        expected_model_sha256 = _require_sha256(
            expected_model_sha256, label="expected_model_sha256"
        )
    frozen_model_sha = expected_model_sha256 or manifest_model_sha or report_model_sha
    if (
        expected_model_sha256 is not None
        and manifest_model_sha is not None
        and expected_model_sha256 != manifest_model_sha
    ):
        raise ValueError("external and artifact-manifest model SHA256 disagree")
    if frozen_model_sha is not None and model_sha_before != frozen_model_sha:
        raise ValueError("model SHA256 does not match the frozen expected value")

    corpus_path = _manifest_corpus_path(model, manifest)
    expected_input_sha = _require_sha256(
        manifest.get("synthetic_input_sha256"),
        label="artifact manifest synthetic_input_sha256",
    )
    input_sha_before = _sha256(corpus_path)
    if input_sha_before != expected_input_sha:
        raise ValueError("synthetic input SHA256 does not match the artifact manifest")

    artifact_train_hashes, artifact_dev_hashes = _validated_membership_hashes(artifact)
    test_rows, corpus_audit = _prepare_corpus_rows(_read_json_rows(corpus_path))
    corpus_membership = corpus_audit.pop("membership")
    test_text_hashes = {row["text_sha256"] for row in test_rows}
    test_groups = {row["group_id"] for row in test_rows}
    overlaps = {
        "artifact_train_normalized_text": len(test_text_hashes & artifact_train_hashes),
        "artifact_dev_normalized_text": len(test_text_hashes & artifact_dev_hashes),
        "corpus_train_normalized_text": len(
            test_text_hashes & corpus_membership["train"]["texts"]
        ),
        "corpus_dev_normalized_text": len(
            test_text_hashes & corpus_membership["dev"]["texts"]
        ),
        "corpus_train_group": len(test_groups & corpus_membership["train"]["groups"]),
        "corpus_dev_group": len(test_groups & corpus_membership["dev"]["groups"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"synthetic test membership leakage detected: {overlaps}")

    texts = [row["text"] for row in test_rows]
    labels = [row["label"] for row in test_rows]
    matrix = artifact["vectorizer"].transform(texts)
    predictions = [str(value) for value in artifact["classifier"].predict(matrix)]
    if len(predictions) != len(labels):
        raise ValueError("feedback-form classifier returned the wrong prediction count")
    unknown_predictions = sorted(set(predictions) - set(FEEDBACK_TYPES))
    if unknown_predictions:
        raise ValueError(
            f"feedback-form classifier returned unknown labels: {unknown_predictions}"
        )

    by_origin: dict[str, dict] = {}
    origins = sorted({row["classifier_corpus_origin"] for row in test_rows})
    for origin in origins:
        indices = [
            index
            for index, row in enumerate(test_rows)
            if row["classifier_corpus_origin"] == origin
        ]
        origin_labels = [labels[index] for index in indices]
        origin_predictions = [predictions[index] for index in indices]
        by_origin[origin] = {
            "label_counts": dict(sorted(Counter(origin_labels).items())),
            "metrics": _metrics(origin_labels, origin_predictions),
        }

    model_sha_after = _sha256(model)
    if model_sha_after != model_sha_before:
        raise RuntimeError("feedback-form model changed during evaluation")
    input_sha_after = _sha256(corpus_path)
    if input_sha_after != input_sha_before:
        raise RuntimeError("synthetic input changed during evaluation")

    return {
        "schema_version": "feedback-form-synthetic-three-class-test-v1",
        "benchmark_role": "locked_synthetic_test_only",
        "read_only_evaluator": True,
        "used_for_training_or_hyperparameter_selection": False,
        "target_source": "reference_type_fixed_mapping",
        "reference_type_mapping": dict(REFERENCE_TYPE_TO_FEEDBACK_TYPE),
        "excluded_reference_types": sorted(EXCLUDED_REFERENCE_TYPES),
        "generator_expected_feedback_type_used_as_target": False,
        "model": str(model),
        "training_report": str(training_report),
        "training_report_sha256": training_report_sha,
        "model_sha256_before": model_sha_before,
        "model_sha256_after": model_sha_after,
        "model_sha256_unchanged": True,
        "model_sha256_verified_against": (
            "training_report_and_optional_expected_sha256"
        ),
        "synthetic_input": str(corpus_path),
        "synthetic_input_sha256_expected": expected_input_sha,
        "synthetic_input_sha256_before": input_sha_before,
        "synthetic_input_sha256_after": input_sha_after,
        "synthetic_input_sha256_unchanged": True,
        "test_rows": len(test_rows),
        "test_label_counts": dict(sorted(Counter(labels).items())),
        "membership_overlap_audit": overlaps,
        "corpus_audit": corpus_audit,
        "metrics": _metrics(labels, predictions),
        "by_classifier_corpus_origin": by_origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--training-report",
        type=Path,
        help="Frozen training report; defaults to model.report.json beside the model.",
    )
    parser.add_argument(
        "--expected-model-sha256",
        help="Optional externally frozen model SHA256; checked before evaluation.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    result = evaluate(
        model,
        training_report_path=args.training_report,
        expected_model_sha256=args.expected_model_sha256,
    )
    report = args.report.resolve()
    corpus = Path(result["synthetic_input"]).resolve()
    training_report = Path(result["training_report"]).resolve()
    if report in {model, corpus, training_report}:
        raise ValueError(
            "report path must be independent from model, training report, and synthetic input"
        )
    write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
