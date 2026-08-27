"""Validate and split minimal human feedback-form annotations.

The public JSONL schema is intentionally limited to exactly two fields:
``language`` and ``classification_label``.  Labels must use the paper-facing
Title Case names ``Evaluative``, ``Imperative``, or ``Descriptive``.

Because the public schema contains no player or session identity, this script
can guarantee normalized-text and automatically derived template-family
disjointness only.  It cannot construct or claim a teacher/session-held-out
evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "human_feedback_form_annotations.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "human_feedback_form_train.json"
DEFAULT_DEV = ROOT / "data" / "human_feedback_form_dev.json"
DEFAULT_TEST = ROOT / "data" / "human_feedback_form_test.json"
DEFAULT_REPORT = ROOT / "outputs" / "human_feedback_form_annotations.report.json"

PUBLIC_FIELDS = frozenset({"language", "classification_label"})
CANONICAL_LABELS = ("Evaluative", "Imperative", "Descriptive")
INTERNAL_LABELS = {label: label.lower() for label in CANONICAL_LABELS}
SPLIT_POLICY_VERSION = "human-feedback-form-template-family-hash-v2"
SPLIT_BUCKETS = {"train": (0, 8000), "dev": (8000, 9000), "test": (9000, 10000)}

_TYPO_NORMALIZATIONS = {
    "impressice": "impressive",
    "jut": "just",
    "tomatos": "tomatoes",
    "unneccessary": "unnecessary",
}
_ITEM_PATTERN = re.compile(
    r"\b(?:tomato(?:es)?|onions?|plates?|dishes?|soup|food|ingredients?)\b"
)
_QUALITY_PATTERN = re.compile(
    r"\b(?:not\s+helpful\s+at\s+all|not\s+the\s+best\s+choice|"
    r"really\s+(?:bad|good|helpful)|quite\s+impressive|"
    r"unnecessary|efficient|helpful|impressive|fine|good|bad|okay|ok|better)\b"
)


def template_family_signature(normalized: str) -> str:
    """Collapse obvious lexical slots without using a model or a label.

    This intentionally small abstraction keeps near-clones such as
    ``fetch an onion`` / ``fetch a tomato`` and ``good`` / ``really good`` in
    one partition.  It is not used as a training target.
    """

    signature = normalize_text(normalized)
    for typo, corrected in _TYPO_NORMALIZATIONS.items():
        signature = re.sub(rf"\b{re.escape(typo)}\b", corrected, signature)
    signature = _ITEM_PATTERN.sub("<item>", signature)
    signature = _QUALITY_PATTERN.sub("<quality>", signature)
    signature = re.sub(r"\b(?:a|an|the)\s+(?=<item>)", "", signature)
    signature = re.sub(r"\s+", " ", signature).strip()
    return signature


def stable_text_split(normalized: str) -> str:
    """Backward-compatible helper using the derived template family."""

    family = template_family_signature(normalized)
    payload = f"{SPLIT_POLICY_VERSION}\0{family}"
    bucket = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % 10000
    for split, (lower, upper) in SPLIT_BUCKETS.items():
        if lower <= bucket < upper:
            return split
    raise AssertionError(f"unreachable split bucket: {bucket}")


def stable_feedback_id(normalized: str) -> str:
    digest = hashlib.sha256(
        f"human-feedback-form\0{normalized}".encode("utf-8")
    ).hexdigest()
    return f"human_feedback_form_{digest[:24]}"


def read_public_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """Read strict two-field JSONL; missing and blank input are valid."""

    path = Path(path)
    if not path.exists():
        return [], 0
    rows: list[dict] = []
    blank_lines = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                blank_lines += 1
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            actual_fields = set(value)
            if actual_fields != PUBLIC_FIELDS:
                missing = sorted(PUBLIC_FIELDS - actual_fields)
                extra = sorted(actual_fields - PUBLIC_FIELDS)
                raise ValueError(
                    f"{path}:{line_number}: schema must contain exactly "
                    f"{sorted(PUBLIC_FIELDS)}; missing={missing}, extra={extra}"
                )
            rows.append(dict(value))
    return rows, blank_lines


def prepare_annotations(rows: Iterable[dict]) -> tuple[dict[str, list[dict]], dict]:
    """Validate, de-duplicate, and split explicitly human-labelled rows."""

    input_rows = list(rows)
    normalized_rows: list[dict] = []
    errors: list[str] = []
    for index, raw in enumerate(input_rows, start=1):
        if not isinstance(raw, dict):
            errors.append(f"row {index}: expected an object")
            continue
        actual_fields = set(raw)
        if actual_fields != PUBLIC_FIELDS:
            errors.append(
                f"row {index}: schema must contain exactly {sorted(PUBLIC_FIELDS)}"
            )
            continue
        language = raw.get("language")
        if not isinstance(language, str) or not language.strip():
            errors.append(f"row {index}: language must be a non-empty string")
            continue
        canonical_label = raw.get("classification_label")
        if canonical_label not in CANONICAL_LABELS:
            errors.append(
                f"row {index}: classification_label must be exactly one of "
                f"{list(CANONICAL_LABELS)}"
            )
            continue
        cleaned = language.strip()
        normalized = normalize_text(cleaned)
        if not normalized:
            errors.append(f"row {index}: language is empty after normalization")
            continue
        normalized_rows.append(
            {
                "feedback_id": stable_feedback_id(normalized),
                "text": cleaned,
                "normalized_text": normalized,
                "classification_label": canonical_label,
                "expected_feedback_type": INTERNAL_LABELS[canonical_label],
                "template_family": template_family_signature(normalized),
                "split": stable_text_split(normalized),
                "source": "human",
                "label_source": "human_explicit",
            }
        )
    if errors:
        raise ValueError("Invalid human feedback-form annotations:\n- " + "\n- ".join(errors))

    by_text: dict[str, dict] = {}
    duplicate_rows_dropped = 0
    for row in normalized_rows:
        key = row["normalized_text"]
        previous = by_text.get(key)
        if previous is None:
            by_text[key] = row
            continue
        if previous["expected_feedback_type"] != row["expected_feedback_type"]:
            raise ValueError(
                "exact normalized language has conflicting explicit labels: "
                f"{key!r} -> {previous['classification_label']!r} and "
                f"{row['classification_label']!r}"
            )
        duplicate_rows_dropped += 1

    split_rows = {split: [] for split in SPLIT_BUCKETS}
    for row in by_text.values():
        split_rows[row["split"]].append(row)
    for rows_for_split in split_rows.values():
        rows_for_split.sort(key=lambda row: (row["normalized_text"], row["feedback_id"]))

    text_sets = {
        split: {row["normalized_text"] for row in rows_for_split}
        for split, rows_for_split in split_rows.items()
    }
    overlap = {
        f"{left}_{right}": len(text_sets[left] & text_sets[right])
        for index, left in enumerate(("train", "dev", "test"))
        for right in ("train", "dev", "test")[index + 1 :]
    }
    if any(overlap.values()):
        raise AssertionError(f"normalized-text split leakage: {overlap}")

    family_sets = {
        split: {row["template_family"] for row in rows_for_split}
        for split, rows_for_split in split_rows.items()
    }
    family_overlap = {
        f"{left}_{right}": len(family_sets[left] & family_sets[right])
        for index, left in enumerate(("train", "dev", "test"))
        for right in ("train", "dev", "test")[index + 1 :]
    }
    if any(family_overlap.values()):
        raise AssertionError(f"template-family split leakage: {family_overlap}")

    family_labels: dict[str, set[str]] = {}
    for row in by_text.values():
        family_labels.setdefault(row["template_family"], set()).add(
            row["classification_label"]
        )
    family_label_conflicts = sum(
        len(labels) > 1 for labels in family_labels.values()
    )
    independent_families_by_label = {
        label: len(
            {
                row["template_family"]
                for row in by_text.values()
                if row["classification_label"] == label
            }
        )
        for label in CANONICAL_LABELS
    }
    all_labels_have_three_families = all(
        count >= 3 for count in independent_families_by_label.values()
    )
    test_has_all_labels = {
        row["classification_label"] for row in split_rows["test"]
    } == set(CANONICAL_LABELS)

    report = {
        "schema_version": 1,
        "public_input_schema": {
            "allowed_fields": ["language", "classification_label"],
            "additional_fields_allowed": False,
            "canonical_labels": list(CANONICAL_LABELS),
            "canonical_label_case_sensitive": True,
        },
        "label_provenance": "human_explicit_only",
        "model_predictions_used_as_labels": False,
        "input_rows": len(input_rows),
        "unique_rows": len(by_text),
        "duplicate_same_label_rows_dropped": duplicate_rows_dropped,
        "split_counts": {split: len(rows_for_split) for split, rows_for_split in split_rows.items()},
        "split_label_counts": {
            split: dict(sorted(Counter(row["classification_label"] for row in rows_for_split).items()))
            for split, rows_for_split in split_rows.items()
        },
        "template_family_quality": {
            "independent_family_counts_by_label": independent_families_by_label,
            "cross_label_family_conflicts": family_label_conflicts,
            "all_labels_have_at_least_three_families": all_labels_have_three_families,
            "test_has_all_three_labels": test_has_all_labels,
            "human_test_accuracy_claim_ready": (
                all_labels_have_three_families and test_has_all_labels
            ),
            "minimum_collection_advice": (
                "add independent surface families until every label can appear "
                "in train, dev, and test"
            ),
        },
        "split_policy": {
            "version": SPLIT_POLICY_VERSION,
            "unit": "automatically_derived_template_family",
            "buckets": "stable_sha256_80_10_10",
            "normalized_text_overlap": overlap,
            "template_family_overlap": family_overlap,
            "template_family_uses_labels_or_model_predictions": False,
            "teacher_session_identity_available": False,
            "valid_claim": "template-family-disjoint evaluation only",
            "invalid_claims": [
                "teacher-held-out generalization",
                "session-held-out generalization",
                "independently adjudicated gold labels",
            ],
        },
    }
    return split_rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows, blank_lines = read_public_jsonl(args.input)
    split_rows, report = prepare_annotations(rows)
    report["input"] = str(args.input)
    report["blank_lines_ignored"] = blank_lines
    report["outputs"] = {
        "train": str(args.train_output),
        "dev": str(args.dev_output),
        "test": str(args.test_output),
        "report": str(args.report),
    }
    write_json(args.train_output, split_rows["train"])
    write_json(args.dev_output, split_rows["dev"])
    write_json(args.test_output, split_rows["test"])
    write_json(args.report, report)
    print(
        "Prepared human feedback-form annotations: "
        + ", ".join(f"{split}={len(rows)}" for split, rows in split_rows.items())
    )
    print("Evaluation boundary: template-family-disjoint only; no teacher/session IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
