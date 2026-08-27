"""Evaluate the previously exposed task-held-out paper-human proxy.

The receipt is a local procedural guard claimed before the test JSON is hashed
or parsed. It is not an external immutable ledger and must not be represented as
a new blind or independently sealed test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_feedback_form_classifier import (  # noqa: E402
    PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE,
    classification_metrics,
    probability_metrics,
    temperature_scaled_probabilities,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import DEFAULT_MODEL_PATH, FEEDBACK_TYPES  # noqa: E402


DEFAULT_TEST = ROOT / "data" / "paper_feedback_form_human_frozen_test.v1.json"
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "feedback_form_classifier" / "paper_human_benchmark.v1.manifest.json"
)
DEFAULT_TRAINING_REPORT = DEFAULT_MODEL_PATH.with_suffix(".report.json")
DEFAULT_REPORT = (
    ROOT / "outputs" / "feedback_form_classifier" / "paper_human_frozen_test.report.json"
)
DEFAULT_RECEIPT = (
    ROOT / "outputs" / "feedback_form_classifier" / "paper_human_frozen_test.receipt.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _claim_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evaluate_once(
    *,
    model_path: Path,
    training_report_path: Path,
    manifest_path: Path,
    test_path: Path,
    report_path: Path,
    receipt_path: Path,
) -> dict:
    """Evaluate a locked candidate on the exposed task-held-out regression."""

    from joblib import load
    import numpy as np

    if report_path.exists():
        raise FileExistsError(f"frozen-test report already exists: {report_path}")
    if receipt_path.exists():
        raise FileExistsError(f"frozen-test receipt already exists: {receipt_path}")
    if not model_path.exists() or not training_report_path.exists() or not test_path.exists():
        raise FileNotFoundError("model, training report, and frozen test must all exist")

    model_hash_before = _sha256(model_path)
    manifest_hash = _sha256(manifest_path)
    manifest = _read_json_object(manifest_path)
    training_report = _read_json_object(training_report_path)
    expected_test = (manifest.get("outputs") or {}).get("frozen_test") or {}
    expected_test_hash = expected_test.get("sha256")
    if not expected_test_hash:
        raise ValueError("benchmark manifest has no frozen-test hash")
    if manifest.get("independent_sealed_test") is not False:
        raise ValueError("benchmark must disclose that this regression is not sealed")
    diagnostic_threshold = float(manifest.get("diagnostic_accuracy_threshold", -1.0))
    if not 0.0 <= diagnostic_threshold <= 1.0:
        raise ValueError("benchmark has invalid diagnostic accuracy threshold")
    artifact_manifest = training_report.get("artifact_manifest") or {}
    if training_report.get("artifact_model_sha256") != model_hash_before:
        raise ValueError("training report does not match the current model hash")
    if artifact_manifest.get("paper_benchmark_manifest_sha256") != manifest_hash:
        raise ValueError("training report does not match the benchmark manifest")
    if artifact_manifest.get("frozen_test_loaded") is not False:
        raise ValueError("training report does not prove frozen_test_loaded=false")
    if artifact_manifest.get("frozen_test_sha256_bound_not_loaded") != expected_test_hash:
        raise ValueError("training artifact is not bound to this frozen test")
    if training_report.get("test_metrics") is not None:
        raise ValueError("training report already contains test metrics")
    if (
        (training_report.get("selection_protocol") or {}).get(
            "test_examples_evaluated_during_selection"
        )
        != 0
    ):
        raise ValueError("test examples were used during model selection")

    artifact = load(model_path)
    if _sha256(model_path) != model_hash_before:
        raise RuntimeError("model changed while being loaded")
    if artifact.get("model_type") != "feedback_form_tfidf_logistic_regression":
        raise ValueError("unexpected model artifact type")
    embedded_manifest = artifact.get("manifest") or {}
    if embedded_manifest.get("paper_benchmark_manifest_sha256") != manifest_hash:
        raise ValueError("model artifact does not match the benchmark manifest")
    if embedded_manifest.get("frozen_test_sha256_bound_not_loaded") != expected_test_hash:
        raise ValueError("model artifact is not bound to this regression test")

    receipt = {
        "schema_version": "paper-human-feedback-form-frozen-receipt-v1",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_sha256": model_hash_before,
        "training_report": str(training_report_path),
        "training_report_sha256": _sha256(training_report_path),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": manifest_hash,
        "frozen_test": str(test_path),
        "expected_frozen_test_sha256": expected_test_hash,
        "diagnostic_accuracy_threshold": diagnostic_threshold,
        "report": str(report_path),
        "policy": "local_path_guard_before_test_hash_or_parse; not_external_ledger",
    }
    _claim_receipt(receipt_path, receipt)

    # No frozen-test bytes are read before the atomic receipt above.
    test_bytes = test_path.read_bytes()
    test_hash_before = hashlib.sha256(test_bytes).hexdigest()
    if test_hash_before != expected_test_hash:
        raise ValueError("frozen-test hash mismatch after receipt claim")
    raw_rows = json.loads(test_bytes.decode("utf-8-sig"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("frozen test must be a non-empty JSON list")
    rows = [dict(row) for row in raw_rows]
    labels: list[str] = []
    texts: list[str] = []
    for index, row in enumerate(rows, start=1):
        if row.get("split") != "test":
            raise ValueError(f"frozen-test row {index} is not split=test")
        reference_type = str(row.get("reference_type") or "")
        label = PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE.get(reference_type)
        if label is None or row.get("expected_feedback_type") != label:
            raise ValueError(f"frozen-test row {index} has invalid proxy mapping")
        if row.get("label_source") != (
            "original_paper_human_reference_main_experiment_mapping"
        ):
            raise ValueError(f"frozen-test row {index} has invalid provenance")
        text = str(row.get("text") or "").strip()
        if not normalize_text(text):
            raise ValueError(f"frozen-test row {index} has empty text")
        texts.append(text)
        labels.append(label)

    membership = artifact.get("data_membership") or {}
    development_text_hashes = set(
        membership.get("train_normalized_text_sha256") or []
    ) | set(membership.get("dev_normalized_text_sha256") or [])
    test_text_hashes = {_digest(normalize_text(text)) for text in texts}
    text_overlap = development_text_hashes & test_text_hashes
    if text_overlap:
        raise ValueError(f"frozen test has {len(text_overlap)} development text overlaps")
    development_group_hashes = set(membership.get("train_group_sha256") or []) | set(
        membership.get("dev_group_sha256") or []
    )
    test_group_hashes = {
        _digest(f"original_paper_human:{row['group_id']}") for row in rows
    }
    group_overlap = development_group_hashes & test_group_hashes
    if group_overlap:
        raise ValueError(f"frozen test has {len(group_overlap)} development group overlaps")

    matrix = artifact["vectorizer"].transform(texts)
    classifier = artifact["classifier"]
    metrics = classification_metrics(classifier, matrix, labels)
    calibration = artifact.get("probability_calibration") or {}
    if calibration.get("method") == "temperature_scaling":
        probabilities = temperature_scaled_probabilities(
            classifier, matrix, float(calibration["temperature"])
        )
        calibration_status = "exposed_task_heldout_evaluation_of_dev_temperature"
    else:
        probabilities = np.asarray(classifier.predict_proba(matrix), dtype=float)
        calibration_status = "uncalibrated_predict_proba"
    metrics["calibration"] = probability_metrics(
        probabilities, labels, classifier.classes_
    )

    predictions = [str(value) for value in classifier.predict(matrix)]
    by_reference_type = {}
    for reference_type in PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE:
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("reference_type") == reference_type
        ]
        if indices:
            by_reference_type[reference_type] = {
                "rows": len(indices),
                "accuracy": sum(predictions[index] == labels[index] for index in indices)
                / len(indices),
            }

    result = {
        "schema_version": "paper-human-feedback-form-exposed-regression-v1",
        "benchmark_role": (
            "previously_exposed_task_held_out_original_paper_human_reference_proxy"
        ),
        "claim_scope": (
            "human-authored original-paper task language with reference labels "
            "collapsed by the paper main-experiment operational mapping"
        ),
        "claim_limit": (
            "the source test was previously exposed; labels are not independently "
            "annotated speech-form gold and are not current Overcooked-player language"
        ),
        "natural_human_authored_text": True,
        "independent_human_speech_form_gold": False,
        "current_overcooked_player_accuracy_claim": False,
        "independent_sealed_test": False,
        "previously_exposed_source_test": True,
        "mapping": dict(PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE),
        "mapping_is_exact_original_notebook": False,
        "mapping_is_exact_paper_main_experiment": True,
        "label_counts": dict(sorted(Counter(labels).items())),
        "reference_type_counts": dict(
            sorted(Counter(str(row["reference_type"]) for row in rows).items())
        ),
        "rows": len(rows),
        "metrics": metrics,
        "by_reference_type": by_reference_type,
        "diagnostic_accuracy_threshold": diagnostic_threshold,
        "diagnostic_threshold_met": metrics["accuracy"] >= diagnostic_threshold,
        "probability_status": calibration_status,
        "model": str(model_path),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": _sha256(model_path),
        "model_sha256_unchanged": model_hash_before == _sha256(model_path),
        "training_report": str(training_report_path),
        "training_report_sha256": _sha256(training_report_path),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": manifest_hash,
        "frozen_test": str(test_path),
        "frozen_test_sha256_before": test_hash_before,
        "frozen_test_sha256_after": _sha256(test_path),
        "frozen_test_sha256_unchanged": test_hash_before == _sha256(test_path),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "development_normalized_text_overlap": 0,
        "development_task_group_overlap": 0,
        "training_pipeline_human_dev_used_in_late_tie_breakers": True,
        "regression_rows_used_for_training_or_selection": False,
        "test_file_mutated": False,
        "local_receipt_is_external_immutable_ledger": False,
    }
    if _sha256(model_path) != model_hash_before:
        raise RuntimeError("model changed during evaluation")
    if _sha256(test_path) != test_hash_before:
        raise RuntimeError("test changed during evaluation")
    if _sha256(manifest_path) != manifest_hash:
        raise RuntimeError("benchmark manifest changed during evaluation")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--training-report", type=Path, default=DEFAULT_TRAINING_REPORT
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frozen-test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = evaluate_once(
        model_path=args.model,
        training_report_path=args.training_report,
        manifest_path=args.manifest,
        test_path=args.frozen_test,
        report_path=args.report,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
