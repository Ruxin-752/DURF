"""Seal a two-field, human-confirmed feedback-form holdout.

The public JSONL input remains deliberately minimal: every non-blank line must
contain exactly ``language`` and ``classification_label``.  Provenance and
integrity metadata live in a generated sidecar, never in the user's file.

This command never trains or evaluates a model.  It fails closed if any text
or derived template family was exposed to the frozen model's train/dev data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    INTERNAL_LABELS,
    PUBLIC_FIELDS,
    read_public_jsonl,
    stable_feedback_id,
    template_family_signature,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FEEDBACK_TYPES,
)


DEFAULT_INPUT = ROOT / "data" / "human_feedback_form_holdout.jsonl"
DEFAULT_TEST = ROOT / "data" / "human_feedback_form_holdout_test.json"
DEFAULT_SIDECAR = ROOT / "outputs" / "human_feedback_form_holdout.sidecar.json"

PROTOCOL_VERSION = "human-feedback-form-frozen-holdout-v1"
TEXT_ORIGINS = ("human_authored", "ai_candidate_human_confirmed")
DEFAULT_MINIMUM_FAMILIES_PER_LABEL = 10
HOLDOUT_SOURCE = "human_holdout"
LABEL_SOURCE_BY_TEXT_ORIGIN = {
    "human_authored": "human_explicit",
    "ai_candidate_human_confirmed": "human_confirmed",
}
_MEMBERSHIP_KEYS = (
    "train_normalized_text_sha256",
    "dev_normalized_text_sha256",
    "human_train_template_family_sha256",
    "human_dev_template_family_sha256",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def value_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_frozen_model(path: str | Path) -> dict:
    from joblib import load

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = load(path)
    if artifact.get("model_type") != "feedback_form_tfidf_logistic_regression":
        raise ValueError("not a feedback-form classifier artifact")
    if tuple(artifact.get("labels") or ()) != FEEDBACK_TYPES:
        raise ValueError("feedback-form artifact label contract mismatch")
    _membership_sets(artifact)
    return artifact


def _membership_sets(artifact: dict) -> dict[str, set[str]]:
    membership = artifact.get("data_membership")
    if not isinstance(membership, dict):
        raise ValueError("frozen model lacks auditable train/dev membership")
    result: dict[str, set[str]] = {}
    for key in _MEMBERSHIP_KEYS:
        values = membership.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"frozen model has invalid or missing membership: {key}")
        result[key] = set(values)
    return result


def prepare_holdout(
    rows: list[dict],
    artifact: dict,
    *,
    text_origin: str,
    minimum_families_per_label: int = DEFAULT_MINIMUM_FAMILIES_PER_LABEL,
    labels_confirmed_by_human: bool = False,
) -> tuple[list[dict], dict]:
    """Validate public rows and construct an all-test frozen holdout."""

    if text_origin not in TEXT_ORIGINS:
        raise ValueError(f"text_origin must be one of {list(TEXT_ORIGINS)}")
    if minimum_families_per_label < 1:
        raise ValueError("minimum_families_per_label must be at least 1")
    if rows and not labels_confirmed_by_human:
        raise ValueError(
            "non-empty holdout requires an explicit human label confirmation"
        )

    label_source = LABEL_SOURCE_BY_TEXT_ORIGIN[text_origin]
    prepared: list[dict] = []
    seen_text: dict[str, str] = {}
    family_labels: dict[str, set[str]] = {}
    errors: list[str] = []
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            errors.append(f"row {index}: expected an object")
            continue
        if set(raw) != PUBLIC_FIELDS:
            errors.append(
                f"row {index}: schema must contain exactly {sorted(PUBLIC_FIELDS)}"
            )
            continue
        language = raw.get("language")
        canonical_label = raw.get("classification_label")
        if not isinstance(language, str) or not language.strip():
            errors.append(f"row {index}: language must be a non-empty string")
            continue
        if canonical_label not in CANONICAL_LABELS:
            errors.append(
                f"row {index}: classification_label must be exactly one of "
                f"{list(CANONICAL_LABELS)}"
            )
            continue
        text = language.strip()
        normalized = normalize_text(text)
        if not normalized:
            errors.append(f"row {index}: language is empty after normalization")
            continue
        if normalized in seen_text:
            errors.append(
                f"row {index}: duplicate normalized language also appears as "
                f"{seen_text[normalized]}"
            )
            continue
        seen_text[normalized] = f"row {index}"
        family = template_family_signature(normalized)
        family_labels.setdefault(family, set()).add(str(canonical_label))
        prepared.append(
            {
                "feedback_id": stable_feedback_id(normalized),
                "text": text,
                "normalized_text": normalized,
                "classification_label": canonical_label,
                "expected_feedback_type": INTERNAL_LABELS[str(canonical_label)],
                "template_family": family,
                "split": "test",
                "source": HOLDOUT_SOURCE,
                "label_source": label_source,
                "text_origin": text_origin,
                "holdout_protocol": PROTOCOL_VERSION,
            }
        )
    conflicts = sorted(
        family for family, labels in family_labels.items() if len(labels) > 1
    )
    if conflicts:
        errors.append(
            f"{len(conflicts)} template families have conflicting labels"
        )
    if errors:
        raise ValueError("Invalid frozen holdout:\n- " + "\n- ".join(errors))

    membership = _membership_sets(artifact)
    test_text_hashes = {text_digest(row["normalized_text"]) for row in prepared}
    test_family_hashes = {value_digest(row["template_family"]) for row in prepared}
    overlaps = {
        "train_normalized_text": len(
            test_text_hashes & membership["train_normalized_text_sha256"]
        ),
        "dev_normalized_text": len(
            test_text_hashes & membership["dev_normalized_text_sha256"]
        ),
        "human_train_template_family": len(
            test_family_hashes
            & membership["human_train_template_family_sha256"]
        ),
        "human_dev_template_family": len(
            test_family_hashes
            & membership["human_dev_template_family_sha256"]
        ),
    }
    if any(overlaps.values()):
        raise ValueError(
            "holdout was exposed to frozen model train/dev data: "
            + ", ".join(f"{key}={value}" for key, value in overlaps.items())
        )

    prepared.sort(key=lambda row: (row["normalized_text"], row["feedback_id"]))
    label_counts = Counter(row["classification_label"] for row in prepared)
    family_counts = {
        label: len(
            {
                row["template_family"]
                for row in prepared
                if row["classification_label"] == label
            }
        )
        for label in CANONICAL_LABELS
    }
    missing_labels = [label for label in CANONICAL_LABELS if not label_counts[label]]
    below_quota = {
        label: count
        for label, count in family_counts.items()
        if count < minimum_families_per_label
    }
    claim_ready = not missing_labels and not below_quota
    metric_scope = (
        "human_authored_language"
        if text_origin == "human_authored"
        else "human_verified_ai_candidate_language"
    )
    report = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "frozen_feedback_form_holdout",
        "status": "ready" if claim_ready else "insufficient_human_test_data",
        "claim_ready": claim_ready,
        "human_language_claim_ready": (
            claim_ready and text_origin == "human_authored"
        ),
        "metric_scope": metric_scope,
        "public_input_schema": {
            "allowed_fields": ["language", "classification_label"],
            "additional_fields_allowed": False,
            "canonical_labels": list(CANONICAL_LABELS),
        },
        "provenance": {
            "text_origin": text_origin,
            "labels_confirmed_by_human": bool(labels_confirmed_by_human),
            "label_source": label_source,
            "ai_candidate_is_not_claimed_as_human_authored": (
                text_origin != "ai_candidate_human_confirmed"
                or metric_scope != "human_authored_language"
            ),
        },
        "all_rows_assigned_to": "test",
        "test_rows": len(prepared),
        "label_counts": {
            label: int(label_counts[label]) for label in CANONICAL_LABELS
        },
        "template_family_counts_by_label": family_counts,
        "minimum_families_per_label": minimum_families_per_label,
        "missing_labels": missing_labels,
        "labels_below_family_quota": below_quota,
        "model_exposure_check": overlaps,
        "used_for_training_or_hyperparameter_selection": False,
        "teacher_session_identity_available": False,
        "claim_limit": (
            "template-family-disjoint only; not teacher/session-held-out"
        ),
    }
    return prepared, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--text-origin", choices=TEXT_ORIGINS, required=True)
    parser.add_argument(
        "--minimum-families-per-label",
        type=int,
        default=DEFAULT_MINIMUM_FAMILIES_PER_LABEL,
    )
    parser.add_argument(
        "--confirm-human-labels",
        action="store_true",
        help="Assert that a human assigned or explicitly confirmed every label.",
    )
    args = parser.parse_args()

    rows, blank_lines = read_public_jsonl(args.input)
    artifact = load_frozen_model(args.model)
    prepared, report = prepare_holdout(
        rows,
        artifact,
        text_origin=args.text_origin,
        minimum_families_per_label=args.minimum_families_per_label,
        labels_confirmed_by_human=args.confirm_human_labels,
    )
    write_json(args.test_output, prepared)
    report.update(
        {
            "public_input": str(args.input.resolve()),
            "public_input_sha256": sha256_file(args.input),
            "blank_lines_ignored": blank_lines,
            "frozen_model": str(args.model.resolve()),
            "frozen_model_sha256": sha256_file(args.model),
            "prepared_test": str(args.test_output.resolve()),
            "prepared_test_sha256": sha256_file(args.test_output),
            "sidecar": str(args.sidecar.resolve()),
        }
    )
    write_json(args.sidecar, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
