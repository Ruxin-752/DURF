"""Evaluate a frozen feedback-form model on held-out human text labels.

This command is read-only with respect to the model and annotation data.  It
does not select hyperparameters or retrain.  With the minimal two-field public
schema, the supported claim is automatically derived template-family-disjoint
generalization only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_PATH,
    FEEDBACK_TYPES,
)
from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    template_family_signature,
)
from scripts.prepare_human_feedback_form_holdout import (  # noqa: E402
    DEFAULT_SIDECAR,
    DEFAULT_TEST,
    HOLDOUT_SOURCE,
    LABEL_SOURCE_BY_TEXT_ORIGIN,
    PROTOCOL_VERSION,
    TEXT_ORIGINS,
)


DEFAULT_REPORT = ROOT / "outputs" / "human_feedback_form_holdout.evaluation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _value_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics(classifier, matrix, labels: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    predictions = classifier.predict(matrix)
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
            CANONICAL_FEEDBACK_LABELS[label]: {
                metric: float(details[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in FEEDBACK_TYPES
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(FEEDBACK_TYPES)
        ).tolist(),
        "label_order": [CANONICAL_FEEDBACK_LABELS[label] for label in FEEDBACK_TYPES],
    }


def _read_sidecar(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"frozen holdout sidecar is required: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object sidecar: {path}")
    return dict(value)


def _validate_sidecar_paths_and_hashes(
    sidecar: dict,
    *,
    sidecar_path: Path,
    model_path: Path,
    test_path: Path,
) -> None:
    if sidecar.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("holdout sidecar protocol mismatch")
    if sidecar.get("frozen_model_sha256") != _sha256(model_path):
        raise ValueError("holdout sidecar frozen-model hash mismatch")
    if sidecar.get("prepared_test_sha256") != _sha256(test_path):
        raise ValueError("holdout sidecar prepared-test hash mismatch")
    recorded_model = sidecar.get("frozen_model")
    recorded_test = sidecar.get("prepared_test")
    if (
        not isinstance(recorded_model, str)
        or Path(recorded_model).resolve() != model_path.resolve()
    ):
        raise ValueError("holdout sidecar frozen-model path mismatch")
    if (
        not isinstance(recorded_test, str)
        or Path(recorded_test).resolve() != test_path.resolve()
    ):
        raise ValueError("holdout sidecar prepared-test path mismatch")
    input_path_value = sidecar.get("public_input")
    input_hash = sidecar.get("public_input_sha256")
    if not isinstance(input_path_value, str) or not isinstance(input_hash, str):
        raise ValueError("holdout sidecar lacks public-input provenance")
    input_path = Path(input_path_value)
    if not input_path.exists() or _sha256(input_path) != input_hash:
        raise ValueError("holdout public input changed after it was sealed")
    if sidecar.get("sidecar") != str(sidecar_path.resolve()):
        raise ValueError("holdout sidecar path mismatch")


def evaluate(
    model_path: Path,
    test_path: Path,
    sidecar_path: Path | None = None,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not test_path.exists():
        rows: list[dict] = []
        test_hash = None
    else:
        value = json.loads(test_path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"expected a JSON list of objects: {test_path}")
        rows = [dict(row) for row in value]
        test_hash = _sha256(test_path)

    # Preserve the old read-only empty-file diagnostic used by unit tests, but
    # require a sealed sidecar before inspecting any non-empty benchmark.
    if sidecar_path is None and not rows:
        from joblib import load

        artifact = load(model_path)
        if artifact.get("model_type") != "feedback_form_tfidf_logistic_regression":
            raise ValueError("not a feedback-form classifier artifact")
        if tuple(artifact.get("labels") or ()) != FEEDBACK_TYPES:
            raise ValueError("feedback-form artifact label contract mismatch")
        return {
            "benchmark_role": "human_template_family_disjoint_test",
            "read_only_evaluator": True,
            "used_for_training_or_hyperparameter_selection": False,
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "test_data": str(test_path),
            "test_data_sha256": test_hash,
            "test_rows": 0,
            "label_counts": {},
            "status": "no_human_test_data",
            "claim_ready": False,
            "human_language_claim_ready": False,
            "metrics": None,
        }
    if sidecar_path is None:
        raise ValueError("non-empty human holdout requires a frozen sidecar")
    sidecar_path = Path(sidecar_path)
    sidecar = _read_sidecar(sidecar_path)
    _validate_sidecar_paths_and_hashes(
        sidecar,
        sidecar_path=sidecar_path,
        model_path=model_path,
        test_path=test_path,
    )
    provenance = sidecar.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("holdout sidecar lacks provenance")
    text_origin = provenance.get("text_origin")
    if text_origin not in TEXT_ORIGINS:
        raise ValueError("holdout sidecar text-origin contract mismatch")
    expected_label_source = LABEL_SOURCE_BY_TEXT_ORIGIN[str(text_origin)]
    if provenance.get("label_source") != expected_label_source:
        raise ValueError("holdout sidecar label-source contract mismatch")
    if rows and provenance.get("labels_confirmed_by_human") is not True:
        raise ValueError("holdout labels were not explicitly confirmed by a human")

    # Verify the pinned bytes before loading the joblib artifact.
    from joblib import load

    artifact = load(model_path)
    if artifact.get("model_type") != "feedback_form_tfidf_logistic_regression":
        raise ValueError("not a feedback-form classifier artifact")
    if tuple(artifact.get("labels") or ()) != FEEDBACK_TYPES:
        raise ValueError("feedback-form artifact label contract mismatch")

    texts: list[str] = []
    labels: list[str] = []
    template_families: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("split") != "test":
            raise ValueError(f"test row {index}: split must be 'test'")
        if row.get("source") != HOLDOUT_SOURCE:
            raise ValueError(f"test row {index}: invalid frozen-holdout source")
        if row.get("label_source") != expected_label_source:
            raise ValueError(f"test row {index}: label provenance mismatch")
        if row.get("text_origin") != text_origin:
            raise ValueError(f"test row {index}: text provenance mismatch")
        if row.get("holdout_protocol") != PROTOCOL_VERSION:
            raise ValueError(f"test row {index}: holdout protocol mismatch")
        text = row.get("text")
        label = row.get("expected_feedback_type")
        if not isinstance(text, str) or not text.strip() or label not in FEEDBACK_TYPES:
            raise ValueError(f"test row {index}: invalid text or expected_feedback_type")
        if row.get("classification_label") != CANONICAL_FEEDBACK_LABELS[str(label)]:
            raise ValueError(f"test row {index}: canonical label mismatch")
        template_family = row.get("template_family")
        if not isinstance(template_family, str) or not template_family.strip():
            raise ValueError(f"test row {index}: missing template_family")
        normalized = normalize_text(text)
        if row.get("normalized_text") != normalized:
            raise ValueError(f"test row {index}: normalized text mismatch")
        if template_family != template_family_signature(normalized):
            raise ValueError(f"test row {index}: derived template family mismatch")
        if normalized in seen:
            raise ValueError(f"duplicate normalized human test text: {normalized!r}")
        seen.add(normalized)
        texts.append(text.strip())
        labels.append(str(label))
        template_families.append(template_family.strip())

    membership = artifact.get("data_membership")
    required_membership = (
        "train_normalized_text_sha256",
        "dev_normalized_text_sha256",
        "human_train_template_family_sha256",
        "human_dev_template_family_sha256",
    )
    if not isinstance(membership, dict) or any(
        not isinstance(membership.get(key), list) for key in required_membership
    ):
        raise ValueError("frozen model lacks auditable train/dev membership")
    trained_hashes = set(membership.get("train_normalized_text_sha256") or ())
    dev_hashes = set(membership.get("dev_normalized_text_sha256") or ())
    train_family_hashes = set(
        membership.get("human_train_template_family_sha256") or ()
    )
    dev_family_hashes = set(
        membership.get("human_dev_template_family_sha256") or ()
    )
    test_hashes = {_text_digest(text) for text in texts}
    test_family_hashes = {_value_digest(value) for value in template_families}
    train_overlap = test_hashes & trained_hashes
    dev_overlap = test_hashes & dev_hashes
    train_family_overlap = test_family_hashes & train_family_hashes
    dev_family_overlap = test_family_hashes & dev_family_hashes
    if train_overlap or dev_overlap or train_family_overlap or dev_family_overlap:
        raise ValueError(
            "human test text/template family was exposed to training/selection: "
            f"train_text={len(train_overlap)}, dev_text={len(dev_overlap)}, "
            f"train_family={len(train_family_overlap)}, "
            f"dev_family={len(dev_family_overlap)}"
        )

    canonical_label_counts = {
        label: labels.count(label.lower()) for label in CANONICAL_LABELS
    }
    family_counts = {
        label: len(
            {
                family
                for family, internal_label in zip(template_families, labels)
                if internal_label == label.lower()
            }
        )
        for label in CANONICAL_LABELS
    }
    minimum_families = sidecar.get("minimum_families_per_label")
    if not isinstance(minimum_families, int) or minimum_families < 1:
        raise ValueError("holdout sidecar has invalid family quota")
    missing_labels = [label for label, count in canonical_label_counts.items() if not count]
    below_quota = {
        label: count
        for label, count in family_counts.items()
        if count < minimum_families
    }
    claim_ready = not missing_labels and not below_quota
    expected_metric_scope = (
        "human_authored_language"
        if text_origin == "human_authored"
        else "human_verified_ai_candidate_language"
    )
    expected_human_language_ready = claim_ready and text_origin == "human_authored"
    sidecar_contract = {
        "test_rows": len(rows),
        "label_counts": canonical_label_counts,
        "template_family_counts_by_label": family_counts,
        "missing_labels": missing_labels,
        "labels_below_family_quota": below_quota,
        "claim_ready": claim_ready,
        "human_language_claim_ready": expected_human_language_ready,
        "metric_scope": expected_metric_scope,
    }
    for key, expected in sidecar_contract.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"holdout sidecar content mismatch: {key}")

    result = {
        "benchmark_role": "frozen_feedback_form_holdout",
        "read_only_evaluator": True,
        "used_for_training_or_hyperparameter_selection": False,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "test_data": str(test_path),
        "test_data_sha256": test_hash,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "test_rows": len(rows),
        "label_counts": canonical_label_counts,
        "template_family_counts_by_label": family_counts,
        "minimum_families_per_label": minimum_families,
        "missing_labels": missing_labels,
        "labels_below_family_quota": below_quota,
        "claim_ready": claim_ready,
        "human_language_claim_ready": expected_human_language_ready,
        "metric_scope": expected_metric_scope,
        "provenance": dict(provenance),
        "train_normalized_text_overlap": len(train_overlap),
        "dev_normalized_text_overlap": len(dev_overlap),
        "train_template_family_overlap": len(train_family_overlap),
        "dev_template_family_overlap": len(dev_family_overlap),
        "teacher_session_identity_available": False,
        "claim_limit": "template-family-disjoint only; not teacher/session-held-out",
    }
    if not claim_ready:
        result.update(
            {"status": "insufficient_human_test_data", "metrics": None}
        )
        return result
    matrix = artifact["vectorizer"].transform(texts)
    result.update(
        {
            "status": "evaluated",
            "metrics": _metrics(artifact["classifier"], matrix, labels),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = evaluate(args.model, args.test, args.sidecar)
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
