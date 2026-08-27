"""Read-only evaluation of the frozen reference classifier on human gold.

This evaluator never trains, tunes, or rewrites the classifier.  It accepts
only independently adjudicated dev/test rows and fails closed if their
normalized text or teacher/session group appears in any corpus recorded by the
model manifest.  An absent or empty gold file produces an explicit ``unknown``
report with no accuracy claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import read_json, write_json  # noqa: E402
from src.phrase_reference_classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    REFERENCE_TYPES,
)
from scripts.train_phrase_reference_classifier import (  # noqa: E402
    _safe_preprocess,
    build_phrase_rows,
)


DEFAULT_GOLD = ROOT / "data" / "human_reference_gold_benchmark.json"
DEFAULT_REPORT = ROOT / "outputs" / "human_reference_gold_classifier.report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_manifest_path(value: str, *, model_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (
        (model_path.parent / path).resolve(),
        (ROOT / path).resolve(),
        (ROOT.parents[2] / path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _manifest_sources(artifact: dict, *, model_path: Path) -> list[dict]:
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("frozen model has no artifact manifest; exposure audit unavailable")
    primary = manifest.get("input")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("frozen model manifest has no input corpus path")

    sources = [
        {
            "role": "input",
            "path": _resolve_manifest_path(primary, model_path=model_path),
            "expected_sha256": manifest.get("input_sha256"),
        }
    ]
    raw_augments = manifest.get("augment") or []
    if not isinstance(raw_augments, list):
        raise ValueError("frozen model manifest augment must be a list")
    augment_hashes = manifest.get("augment_sha256") or {}
    for index, raw in enumerate(raw_augments):
        if isinstance(raw, dict):
            raw_path = raw.get("path")
            expected = raw.get("sha256")
        else:
            raw_path = raw
            if isinstance(augment_hashes, dict):
                expected = augment_hashes.get(str(raw_path))
            elif isinstance(augment_hashes, list) and index < len(augment_hashes):
                expected = augment_hashes[index]
            else:
                expected = None
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"frozen model manifest augment[{index}] has no path")
        sources.append(
            {
                "role": "augment",
                "path": _resolve_manifest_path(raw_path, model_path=model_path),
                "expected_sha256": expected,
            }
        )
    return sources


def _load_model_exposure(artifact: dict, *, model_path: Path) -> tuple[set[str], set[tuple[str, str]], list[dict]]:
    """Load every labelled source recorded in the model manifest.

    Auditing the whole manifest source, including declared dev rows, is more
    conservative than checking fitted train rows alone: a gold item inspected
    during model selection is not treated as an untouched external example.
    """

    exposed_texts: set[str] = set()
    exposed_groups: set[tuple[str, str]] = set()
    source_reports: list[dict] = []
    for source in _manifest_sources(artifact, model_path=model_path):
        path = Path(source["path"])
        if not path.is_file():
            raise ValueError(f"model exposure corpus is missing: {path}")
        actual_sha = _sha256(path)
        expected_sha = source.get("expected_sha256")
        if expected_sha and str(expected_sha) != actual_sha:
            raise ValueError(
                f"model exposure corpus hash mismatch: {path}; "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        rows = read_json(path)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"model exposure corpus must be a JSON object list: {path}")
        phrase_count = 0
        for row in rows:
            teacher = str(row.get("teacher_id") or "").strip()
            session = str(row.get("session_id") or "").strip()
            if teacher and session:
                exposed_groups.add((teacher, session))
            for phrase_row in build_phrase_rows([row]):
                normalized = normalize_text(phrase_row.get("text"))
                if normalized:
                    exposed_texts.add(normalized)
                    phrase_count += 1
        source_reports.append(
            {
                "role": source["role"],
                "path": str(path),
                "sha256": actual_sha,
                "sha256_bound_in_model": bool(expected_sha),
                "rows": len(rows),
                "classifier_phrase_rows": phrase_count,
            }
        )
    return exposed_texts, exposed_groups, source_reports


def _validate_gold(rows: Iterable[dict]) -> list[dict]:
    validated: list[dict] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_text_labels: dict[str, str] = {}
    for index, raw in enumerate(rows):
        label = f"gold[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{label}: expected an object")
            continue
        feedback_id = str(raw.get("feedback_id") or "").strip()
        text = str(raw.get("text") or "").strip()
        teacher = str(raw.get("teacher_id") or "").strip()
        session = str(raw.get("session_id") or "").strip()
        annotator = str(raw.get("annotator_id") or "").strip()
        adjudicator = str(raw.get("adjudicator_id") or "").strip()
        reference_type = raw.get("reference_type")
        split = raw.get("split")
        if not feedback_id:
            errors.append(f"{label}: feedback_id must be non-empty")
        elif feedback_id in seen_ids:
            errors.append(f"{label}: duplicate feedback_id {feedback_id!r}")
        else:
            seen_ids.add(feedback_id)
        if not text or not normalize_text(text):
            errors.append(f"{label}: text must be non-empty")
        if reference_type not in REFERENCE_TYPES:
            errors.append(f"{label}: invalid reference_type {reference_type!r}")
        if raw.get("annotation_status") != "adjudicated":
            errors.append(f"{label}: gold accepts annotation_status=adjudicated only")
        if split not in {"dev", "test"}:
            errors.append(f"{label}: gold split must be dev or test")
        assigned = raw.get("assigned_split")
        if assigned is not None and assigned != split:
            errors.append(
                f"{label}: assigned_split={assigned!r} disagrees with split={split!r}"
            )
        if not teacher or not session:
            errors.append(f"{label}: teacher_id and session_id are required")
        if not annotator or not adjudicator or annotator == adjudicator:
            errors.append(
                f"{label}: independent annotator_id and adjudicator_id are required"
            )
        normalized = normalize_text(text)
        previous_label = seen_text_labels.get(normalized)
        if previous_label is not None and previous_label != reference_type:
            errors.append(
                f"{label}: normalized text has conflicting labels "
                f"{previous_label!r}/{reference_type!r}"
            )
        elif normalized:
            seen_text_labels[normalized] = str(reference_type)
        validated.append(dict(raw))
    if errors:
        raise ValueError("Invalid human gold benchmark:\n- " + "\n- ".join(errors))
    return validated


def _split_metrics(artifact: dict, rows: list[dict]) -> dict:
    labels = list(REFERENCE_TYPES)
    zero_confusion = [[0 for _ in labels] for _ in labels]
    if not rows:
        return {
            "status": "unknown_no_rows",
            "rows": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": zero_confusion,
            "labels": labels,
            "support_by_label": {label: 0 for label in labels},
            "accuracy_claim_allowed": False,
        }

    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    vectorizer = artifact.get("vectorizer")
    classifier = artifact.get("classifier")
    if vectorizer is None or classifier is None:
        raise ValueError("frozen model artifact lacks vectorizer/classifier")
    input_mode = (
        artifact.get("input_mode")
        or (artifact.get("manifest") or {}).get("preprocessing")
        or "paper_preprocess_phrase"
    )
    raw_texts = [str(row["text"]) for row in rows]
    model_texts = raw_texts if input_mode == "raw_phrase" else [_safe_preprocess(text) for text in raw_texts]
    matrix = vectorizer.transform(model_texts)
    predictions = [str(value) for value in np.asarray(classifier.predict(matrix))]
    gold_labels = [str(row["reference_type"]) for row in rows]
    return {
        "status": "evaluated",
        "rows": len(rows),
        "accuracy": float(accuracy_score(gold_labels, predictions)),
        "macro_f1": float(
            f1_score(
                gold_labels,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            gold_labels, predictions, labels=labels
        ).tolist(),
        "labels": labels,
        "support_by_label": dict(Counter(gold_labels)),
        "prediction_counts": dict(Counter(predictions)),
        "accuracy_claim_allowed": True,
    }


def evaluate_human_gold(rows: Iterable[dict], *, model_path: str | Path) -> dict:
    """Evaluate validated gold without changing the model or source corpora."""

    from joblib import load

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ValueError(f"frozen classifier does not exist: {model_path}")
    model_sha_before = _sha256(model_path)
    artifact = load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("frozen classifier artifact must be a dictionary")
    classifier = artifact.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_"):
        raise ValueError("frozen classifier artifact lacks a fitted classifier")
    model_labels = {str(value) for value in classifier.classes_}
    if model_labels != set(REFERENCE_TYPES):
        raise ValueError(
            f"frozen classifier classes are not the required five labels: {sorted(model_labels)}"
        )

    gold = _validate_gold(rows)
    exposed_texts, exposed_groups, source_reports = _load_model_exposure(
        artifact, model_path=model_path
    )
    gold_texts = {normalize_text(row["text"]) for row in gold}
    gold_groups = {
        (str(row["teacher_id"]), str(row["session_id"])) for row in gold
    }
    text_overlap = sorted(gold_texts & exposed_texts)
    group_overlap = sorted(gold_groups & exposed_groups)
    if text_overlap or group_overlap:
        raise ValueError(
            "human gold overlaps frozen-model exposure; refusing evaluation: "
            f"normalized_text={len(text_overlap)}, teacher_session={len(group_overlap)}; "
            f"text_examples={text_overlap[:5]}, group_examples={group_overlap[:5]}"
        )

    split_metrics = {
        split: _split_metrics(
            artifact, [row for row in gold if row.get("split") == split]
        )
        for split in ("dev", "test")
    }
    if not gold:
        status = "unknown_no_human_gold"
    elif split_metrics["test"]["rows"] == 0:
        status = "development_only_no_final_test"
    else:
        status = "evaluated_with_held_out_human_test"

    model_sha_after = _sha256(model_path)
    if model_sha_after != model_sha_before:
        raise RuntimeError("frozen model bytes changed during read-only evaluation")
    return {
        "schema_version": "human-reference-gold-evaluation-v1",
        "evaluation_status": status,
        "benchmark_role": "held_out_human_reference_type_gold",
        "gold_rows": len(gold),
        "dev": split_metrics["dev"],
        "test": split_metrics["test"],
        "final_human_accuracy_claim_allowed": split_metrics["test"][
            "accuracy_claim_allowed"
        ],
        "empty_gold_policy": "accuracy and macro_f1 are null/unknown; row count is zero",
        "gold_acceptance_policy": "adjudicated dev/test only",
        "leakage_audit": {
            "scope": (
                "all labelled input/augment rows named by the model manifest, "
                "including declared dev exposure"
            ),
            "normalized_text_overlap_count": 0,
            "teacher_session_overlap_count": 0,
            "fail_closed": True,
            "model_exposure_sources": source_reports,
        },
        "model": {
            "path": str(model_path),
            "sha256_before": model_sha_before,
            "sha256_after": model_sha_after,
            "unchanged": True,
            "read_only_evaluation": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gold.exists():
        gold_sha = _sha256(args.gold)
        rows = read_json(args.gold)
        if not isinstance(rows, list):
            raise ValueError("human gold benchmark must be a JSON row list")
    else:
        gold_sha = None
        rows = []
    report = evaluate_human_gold(rows, model_path=args.model)
    report["gold"] = {
        "path": str(args.gold),
        "exists": args.gold.exists(),
        "sha256": gold_sha,
    }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "status": report["evaluation_status"],
                "dev": report["dev"],
                "test": report["test"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
