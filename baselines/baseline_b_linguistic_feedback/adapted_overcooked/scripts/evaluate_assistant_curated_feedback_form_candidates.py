"""Read-only model regression on frozen assistant-curated feedback-form text.

This benchmark is useful for classifier regression only.  Its suggested labels
were authored by an assistant, so no metric produced here is human accuracy.
The evaluator never trains, selects, calibrates, or writes model state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_feedback_form_review_candidates import (  # noqa: E402
    REVIEW_STATUS,
)
from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    template_family_signature,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_PATH,
    FEEDBACK_TYPES,
)


DEFAULT_CANDIDATES = (
    ROOT / "data" / "assistant_curated_feedback_form_candidates.pending_review.jsonl"
)
DEFAULT_CANDIDATE_REPORT = (
    ROOT / "outputs" / "assistant_curated_feedback_form_candidates.report.json"
)
DEFAULT_REPORT = (
    ROOT
    / "outputs"
    / "feedback_form_classifier"
    / "assistant_curated_regression.report.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = frozenset(
    {"language", "classification_label", "review_status"}
)
CANONICAL_TO_INTERNAL = {
    canonical: internal for internal, canonical in CANONICAL_FEEDBACK_LABELS.items()
}


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
    if {str(value) for value in artifact["classifier"].classes_} != set(
        FEEDBACK_TYPES
    ):
        raise ValueError("feedback-form classifier classes do not match label contract")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("feedback-form artifact is missing its manifest")
    if manifest.get("human_test_loaded") is not False:
        raise ValueError("artifact manifest does not prove test was excluded from training")
    return manifest


def _membership_hashes(artifact: dict) -> tuple[set[str], set[str]]:
    membership = artifact.get("data_membership")
    if not isinstance(membership, dict):
        raise ValueError("artifact is missing data_membership")

    def load(name: str) -> set[str]:
        values = membership.get(name)
        if not isinstance(values, list):
            raise ValueError(f"artifact data_membership.{name} must be a list")
        hashes = {
            _require_sha256(value, label=f"data_membership.{name}")
            for value in values
        }
        if len(hashes) != len(values):
            raise ValueError(f"artifact data_membership.{name} contains duplicates")
        return hashes

    train = load("train_normalized_text_sha256")
    dev = load("dev_normalized_text_sha256")
    if train & dev:
        raise ValueError("artifact train/dev normalized-text membership overlaps")
    return train, dev


def _load_json_object(path: Path, *, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_candidates(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"candidate row {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"candidate row {line_number}: expected an object")
        if set(value) != REQUIRED_FIELDS:
            raise ValueError(
                f"candidate row {line_number}: fields must be exactly "
                f"{sorted(REQUIRED_FIELDS)}"
            )
        language = value.get("language")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                f"candidate row {line_number}: language must be non-empty"
            )
        if value.get("classification_label") not in CANONICAL_LABELS:
            raise ValueError(
                f"candidate row {line_number}: invalid classification_label"
            )
        if value.get("review_status") != REVIEW_STATUS:
            raise ValueError(
                f"candidate row {line_number}: review_status must be {REVIEW_STATUS!r}"
            )
        normalized = normalize_text(language)
        rows.append(
            {
                "text": language.strip(),
                "normalized_text": normalized,
                "text_sha256": _normalized_digest(normalized),
                "template_family": template_family_signature(normalized),
                "canonical_label": value["classification_label"],
                "label": CANONICAL_TO_INTERNAL[value["classification_label"]],
            }
        )
    if not rows:
        raise ValueError("candidate JSONL contains no rows")
    normalized = [row["normalized_text"] for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate JSONL contains duplicate normalized text")
    families = [row["template_family"] for row in rows]
    if len(set(families)) != len(families):
        raise ValueError("candidate JSONL contains duplicate template families")
    missing = sorted(set(FEEDBACK_TYPES) - {row["label"] for row in rows})
    if missing:
        raise ValueError(f"candidate JSONL is missing feedback classes: {missing}")
    return rows


def _validate_candidate_report(
    report: dict,
    *,
    report_path: Path,
    candidate_path: Path,
    candidate_sha256: str,
    rows: list[dict],
) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("candidate report schema_version mismatch")
    if report.get("provenance") != REVIEW_STATUS:
        raise ValueError("candidate report provenance mismatch")
    for key in (
        "human_authorship_claimed",
        "human_review_completed",
        "training_eligible",
        "evaluation_eligible",
        "model_or_classifier_used_to_assign_labels",
    ):
        if report.get(key) is not False:
            raise ValueError(f"candidate report {key} must be false")
    schema = report.get("candidate_schema")
    if not isinstance(schema, dict):
        raise ValueError("candidate report is missing candidate_schema")
    if set(schema.get("exact_fields") or ()) != REQUIRED_FIELDS:
        raise ValueError("candidate report field contract mismatch")
    if schema.get("required_review_status") != REVIEW_STATUS:
        raise ValueError("candidate report review-status contract mismatch")
    _validate_candidate_file_binding(
        report,
        report_path=report_path,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
    )
    if report.get("candidate_count") != len(rows):
        raise ValueError("candidate report row count does not match candidate data")
    if report.get("distinct_surface_family_count") != len(rows):
        raise ValueError("candidate report surface-family count mismatch")
    counts = Counter(row["canonical_label"] for row in rows)
    if report.get("counts_by_label") != {
        label: counts[label] for label in CANONICAL_LABELS
    }:
        raise ValueError("candidate report label counts do not match candidate data")
    novelty = report.get("novelty_guard")
    if not isinstance(novelty, dict):
        raise ValueError("candidate report is missing novelty_guard")
    for key in (
        "exact_collision_in_emitted_pool",
        "template_family_collision_in_emitted_pool",
        "near_duplicate_in_emitted_pool",
    ):
        if novelty.get(key) != 0:
            raise ValueError(f"candidate novelty guard failed: {key}")
    boundary = report.get("suggested_label_boundary")
    if not isinstance(boundary, dict) or boundary.get("never_valid_as") != "human gold":
        raise ValueError("candidate report does not forbid a human-gold claim")


def _validate_candidate_file_binding(
    report: dict,
    *,
    report_path: Path,
    candidate_path: Path,
    candidate_sha256: str,
) -> None:
    """Verify the frozen file binding before parsing any candidate row."""

    declared_output = report.get("output")
    if not isinstance(declared_output, str) or not declared_output.strip():
        raise ValueError("candidate report is missing output path")
    declared_path = Path(declared_output)
    if not declared_path.is_absolute():
        declared_path = report_path.parent / declared_path
    if declared_path.resolve(strict=True) != candidate_path:
        raise ValueError("candidate report output path does not match candidate input")
    expected_sha = _require_sha256(
        report.get("output_sha256"), label="candidate report output_sha256"
    )
    if expected_sha != candidate_sha256:
        raise ValueError("candidate data SHA256 does not match candidate report")


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


def evaluate(
    model_path: str | Path,
    candidate_path: str | Path,
    candidate_report_path: str | Path,
    *,
    training_report_path: str | Path | None = None,
    expected_model_sha256: str | None = None,
    expected_training_report_sha256: str | None = None,
    expected_candidate_sha256: str | None = None,
    expected_candidate_report_sha256: str | None = None,
) -> dict:
    """Evaluate a frozen model without invoking runtime fallback or mutation."""

    from joblib import load

    model = Path(model_path).resolve(strict=True)
    candidates = Path(candidate_path).resolve(strict=True)
    candidate_report = Path(candidate_report_path).resolve(strict=True)
    training_report = Path(
        training_report_path
        if training_report_path is not None
        else model.with_suffix(".report.json")
    ).resolve(strict=True)

    paths = (model, candidates, candidate_report, training_report)
    if len(set(paths)) != len(paths):
        raise ValueError("model and regression inputs must be distinct files")
    hashes_before = {str(path): _sha256(path) for path in paths}
    model_sha_before = hashes_before[str(model)]
    training_report_sha = hashes_before[str(training_report)]
    candidate_sha = hashes_before[str(candidates)]
    candidate_report_sha = hashes_before[str(candidate_report)]

    expected_values = {
        "model": expected_model_sha256,
        "training report": expected_training_report_sha256,
        "candidate data": expected_candidate_sha256,
        "candidate report": expected_candidate_report_sha256,
    }
    actual_values = {
        "model": model_sha_before,
        "training report": training_report_sha,
        "candidate data": candidate_sha,
        "candidate report": candidate_report_sha,
    }
    for name, expected in expected_values.items():
        if expected is None:
            continue
        expected = _require_sha256(expected, label=f"expected {name} SHA256")
        if expected != actual_values[name]:
            raise ValueError(f"{name} SHA256 does not match externally frozen value")

    artifact = load(model)
    manifest = _validate_artifact(artifact)
    training_value = _load_json_object(training_report, label="training report")
    frozen_model_sha = _require_sha256(
        training_value.get("artifact_model_sha256"),
        label="training report artifact_model_sha256",
    )
    if frozen_model_sha != model_sha_before:
        raise ValueError("model SHA256 does not match the frozen training report")
    if training_value.get("artifact_manifest") != manifest:
        raise ValueError("training report and model artifact manifests differ")

    candidate_value = _load_json_object(candidate_report, label="candidate report")
    _validate_candidate_file_binding(
        candidate_value,
        report_path=candidate_report,
        candidate_path=candidates,
        candidate_sha256=candidate_sha,
    )
    rows = _load_candidates(candidates)
    _validate_candidate_report(
        candidate_value,
        report_path=candidate_report,
        candidate_path=candidates,
        candidate_sha256=candidate_sha,
        rows=rows,
    )

    train_hashes, dev_hashes = _membership_hashes(artifact)
    candidate_hashes = {row["text_sha256"] for row in rows}
    overlaps = {
        "artifact_train_normalized_text": len(candidate_hashes & train_hashes),
        "artifact_dev_normalized_text": len(candidate_hashes & dev_hashes),
    }
    if any(overlaps.values()):
        raise ValueError(f"candidate membership leakage detected: {overlaps}")

    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    matrix = artifact["vectorizer"].transform(texts)
    predictions = [str(value) for value in artifact["classifier"].predict(matrix)]
    if len(predictions) != len(labels):
        raise ValueError("feedback-form classifier returned the wrong prediction count")
    unknown = sorted(set(predictions) - set(FEEDBACK_TYPES))
    if unknown:
        raise ValueError(f"feedback-form classifier returned unknown labels: {unknown}")

    hashes_after = {str(path): _sha256(path) for path in paths}
    if hashes_after != hashes_before:
        changed = [
            path for path in hashes_before if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"read-only regression input changed: {changed}")

    return {
        "schema_version": "feedback-form-assistant-curated-regression-v1",
        "benchmark_role": "assistant_curated_regression",
        "read_only_evaluator": True,
        "prediction_path": "model_only_vectorizer_and_classifier_predict",
        "runtime_low_confidence_fallback_used": False,
        "used_for_training_or_hyperparameter_selection": False,
        "used_for_model_selection": False,
        "human_accuracy_claim": False,
        "natural_human_player_language_claim": False,
        "target_source": "assistant_curated_suggested_labels",
        "model": str(model),
        "training_report": str(training_report),
        "candidates": str(candidates),
        "candidate_report": str(candidate_report),
        "model_sha256_before": model_sha_before,
        "model_sha256_after": hashes_after[str(model)],
        "model_sha256_unchanged": True,
        "training_report_sha256": training_report_sha,
        "candidate_sha256": candidate_sha,
        "candidate_report_sha256": candidate_report_sha,
        "all_read_only_input_sha256_unchanged": True,
        "external_expected_hashes_supplied": {
            name.replace(" ", "_"): expected is not None
            for name, expected in expected_values.items()
        },
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(labels).items())),
        "membership_overlap_audit": overlaps,
        "metrics": _metrics(labels, predictions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT
    )
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-training-report-sha256")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-candidate-report-sha256")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    result = evaluate(
        args.model,
        args.candidates,
        args.candidate_report,
        training_report_path=args.training_report,
        expected_model_sha256=args.expected_model_sha256,
        expected_training_report_sha256=args.expected_training_report_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_candidate_report_sha256=args.expected_candidate_report_sha256,
    )
    report = args.report.resolve()
    protected = {
        Path(result[key]).resolve()
        for key in ("model", "training_report", "candidates", "candidate_report")
    }
    if report in protected:
        raise ValueError("output report path must be separate from all read-only inputs")
    write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
