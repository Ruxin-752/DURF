"""Validate and partition append-only human reference-type annotations.

The output policy is intentionally conservative:

* the split is a stable hash of ``teacher_id + session_id``;
* ``single_annotated`` rows may become train candidates only;
* dev/test gold rows must be independently ``adjudicated``;
* model predictions are never accepted as labels; and
* an exact normalized phrase cannot carry two different labels.

The default input may be absent or empty.  In that case the command writes
empty, valid outputs so that the rest of the project can run before human data
has been collected.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "human_reference_annotations.jsonl"
DEFAULT_NORMALIZED = ROOT / "data" / "human_reference_annotations.normalized.json"
DEFAULT_TRAIN = ROOT / "data" / "human_reference_train_candidates.json"
DEFAULT_GOLD = ROOT / "data" / "human_reference_gold_benchmark.json"
DEFAULT_REPORT = ROOT / "outputs" / "human_reference_annotations.report.json"

REFERENCE_TYPES = (
    "trajectory",
    "feature",
    "action_spatial",
    "action_behavioral",
    "other",
)
ANNOTATION_STATUSES = ("single_annotated", "adjudicated")
SPLIT_POLICY_VERSION = "human-reference-teacher-session-hash-v1"
SPLIT_BUCKETS = {"train": (0, 8000), "dev": (8000, 9000), "test": (9000, 10000)}


def stable_teacher_session_split(teacher_id: str, session_id: str) -> str:
    """Map one teacher/session group to a stable 80/10/10 split.

    This is a direct hash threshold, not a rank among the currently available
    groups.  Appending a new group therefore cannot move an existing group.
    """

    payload = f"{SPLIT_POLICY_VERSION}\0{teacher_id}\0{session_id}"
    bucket = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % 10000
    for split, (lower, upper) in SPLIT_BUCKETS.items():
        if lower <= bucket < upper:
            return split
    raise AssertionError(f"unreachable split bucket: {bucket}")


def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """Read JSON objects from JSONL; absent and blank files are valid."""

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
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            value = dict(value)
            value["_source_line"] = line_number
            rows.append(value)
    return rows, blank_lines


def _required_string(row: dict, field: str, label: str, errors: list[str]) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: {field} must be a non-empty string")
        return ""
    return value.strip()


def _validate_row(raw: dict, index: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    source_line = raw.get("_source_line", index + 1)
    label = f"line {source_line}"
    feedback_id = _required_string(raw, "feedback_id", label, errors)
    text = _required_string(raw, "text", label, errors)
    teacher_id = _required_string(raw, "teacher_id", label, errors)
    session_id = _required_string(raw, "session_id", label, errors)
    annotator_id = _required_string(raw, "annotator_id", label, errors)

    reference_type = raw.get("reference_type")
    if reference_type not in REFERENCE_TYPES:
        errors.append(
            f"{label}: reference_type must be one of {list(REFERENCE_TYPES)}"
        )

    annotation_status = raw.get("annotation_status")
    if annotation_status not in ANNOTATION_STATUSES:
        errors.append(
            f"{label}: annotation_status must be one of {list(ANNOTATION_STATUSES)}"
        )

    scenario_id = str(raw.get("scenario_id") or "").strip()
    group_id = str(raw.get("group_id") or "").strip()
    if not scenario_id and not group_id:
        errors.append(f"{label}: provide scenario_id or group_id for provenance")

    adjudicator_id = str(raw.get("adjudicator_id") or "").strip()
    if annotation_status == "adjudicated":
        if not adjudicator_id:
            errors.append(f"{label}: adjudicated rows require adjudicator_id")
        elif adjudicator_id == annotator_id:
            errors.append(
                f"{label}: adjudicator_id must differ from annotator_id for gold data"
            )

    revision = raw.get("annotation_revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append(f"{label}: annotation_revision must be a positive integer")
        revision = 1

    normalized = normalize_text(text)
    if text and not normalized:
        errors.append(f"{label}: text is empty after normalization")

    assigned_split = (
        stable_teacher_session_split(teacher_id, session_id)
        if teacher_id and session_id
        else ""
    )
    provenance_group_id = (
        f"scenario:{scenario_id}" if scenario_id else f"group:{group_id}"
    )
    normalized_row = {
        "feedback_id": feedback_id,
        "text": text,
        "normalized_text": normalized,
        "reference_type": reference_type,
        "annotation_status": annotation_status,
        "annotation_revision": revision,
        "annotator_id": annotator_id,
        "adjudicator_id": adjudicator_id or None,
        "teacher_id": teacher_id,
        "session_id": session_id,
        "scenario_id": scenario_id or None,
        "group_id": group_id or None,
        "provenance_group_id": provenance_group_id,
        "split_group_id": f"teacher_session:{teacher_id}:{session_id}",
        "assigned_split": assigned_split,
        "source": "human",
    }
    if raw.get("annotation_notes") is not None:
        normalized_row["annotation_notes"] = str(raw["annotation_notes"])
    return normalized_row, errors


def prepare_human_annotations(
    rows: Iterable[dict],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Validate rows and return normalized, train, gold, and report objects."""

    validated: list[tuple[int, dict]] = []
    errors: list[str] = []
    input_rows = list(rows)
    for index, raw in enumerate(input_rows):
        if not isinstance(raw, dict):
            errors.append(f"line {index + 1}: expected an object")
            continue
        row, row_errors = _validate_row(raw, index)
        errors.extend(row_errors)
        validated.append((index, row))
    if errors:
        raise ValueError("Invalid human reference annotations:\n- " + "\n- ".join(errors))

    # An append-only correction retains feedback_id and increments revision.
    # Identity/provenance and text cannot silently change across revisions.
    histories: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in validated:
        histories[row["feedback_id"]].append((index, row))

    active: list[tuple[int, dict]] = []
    identical_revisions_dropped = 0
    superseded_revisions = 0
    for feedback_id, versions in histories.items():
        identity_fields = (
            "normalized_text",
            "teacher_id",
            "session_id",
            "scenario_id",
            "group_id",
        )
        identity = {field: versions[0][1][field] for field in identity_fields}
        by_revision: dict[int, tuple[int, dict]] = {}
        status_by_revision: list[tuple[int, str]] = []
        for index, row in versions:
            if any(row[field] != identity[field] for field in identity_fields):
                raise ValueError(
                    f"feedback_id {feedback_id!r} changed text or provenance across revisions"
                )
            revision = row["annotation_revision"]
            previous = by_revision.get(revision)
            if previous is not None:
                comparable = {k: v for k, v in row.items() if k != "annotation_notes"}
                prior_comparable = {
                    k: v for k, v in previous[1].items() if k != "annotation_notes"
                }
                if comparable != prior_comparable:
                    raise ValueError(
                        f"feedback_id {feedback_id!r} reuses revision {revision} "
                        "with different annotation data"
                    )
                identical_revisions_dropped += 1
                continue
            by_revision[revision] = (index, row)
            status_by_revision.append((revision, row["annotation_status"]))
        status_by_revision.sort()
        seen_adjudicated = False
        for revision, status in status_by_revision:
            if seen_adjudicated and status != "adjudicated":
                raise ValueError(
                    f"feedback_id {feedback_id!r} regresses from adjudicated at revision {revision}"
                )
            seen_adjudicated = seen_adjudicated or status == "adjudicated"
        selected = by_revision[max(by_revision)]
        superseded_revisions += len(by_revision) - 1
        active.append(selected)

    active.sort(key=lambda item: item[0])

    # Reject contradictory gold/labels before any output can be used.
    by_text: dict[str, list[dict]] = defaultdict(list)
    for _, row in active:
        by_text[row["normalized_text"]].append(row)
    conflicts = {
        text: sorted({str(row["reference_type"]) for row in text_rows})
        for text, text_rows in by_text.items()
        if len({row["reference_type"] for row in text_rows}) > 1
    }
    if conflicts:
        examples = "; ".join(
            f"{text!r} -> {labels}" for text, labels in list(conflicts.items())[:10]
        )
        raise ValueError(f"exact normalized text has cross-label conflicts: {examples}")

    # Same-label exact duplicates add no information. Keep the first active row;
    # because the source is append-only, adding later rows cannot replace it.
    unique: list[dict] = []
    seen_text: set[str] = set()
    same_label_duplicates_dropped = 0
    for _, row in active:
        text_key = row["normalized_text"]
        if text_key in seen_text:
            same_label_duplicates_dropped += 1
            continue
        seen_text.add(text_key)
        unique.append(dict(row))

    train: list[dict] = []
    gold: list[dict] = []
    withheld: Counter[str] = Counter()
    for row in unique:
        split = row["assigned_split"]
        status = row["annotation_status"]
        if split == "train":
            release_status = "train_candidate"
            released = dict(row)
            released["split"] = "train"
            train.append(released)
        elif status == "adjudicated":
            release_status = f"{split}_gold"
            released = dict(row)
            released["split"] = split
            gold.append(released)
        else:
            release_status = f"withheld_{split}_pending_adjudication"
            withheld[split] += 1
        row["release_status"] = release_status

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in unique:
        group_splits[row["split_group_id"]].add(row["assigned_split"])
    leaking_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    if leaking_groups:
        raise AssertionError(f"teacher/session groups cross splits: {leaking_groups}")
    if any(row["annotation_status"] != "adjudicated" for row in gold):
        raise AssertionError("non-adjudicated row reached gold benchmark")
    if any(row["split"] not in {"dev", "test"} for row in gold):
        raise AssertionError("gold benchmark contains a non-held-out split")

    report = {
        "schema_version": "human-reference-annotations-v1",
        "split_policy_version": SPLIT_POLICY_VERSION,
        "split_policy": {
            "group_key": ["teacher_id", "session_id"],
            "method": "sha256 fixed threshold",
            "buckets": {key: list(value) for key, value in SPLIT_BUCKETS.items()},
            "append_stable": True,
        },
        "input_rows": len(input_rows),
        "active_feedback_ids": len(active),
        "normalized_rows": len(unique),
        "identical_revisions_dropped": identical_revisions_dropped,
        "superseded_revisions": superseded_revisions,
        "same_label_exact_duplicates_dropped": same_label_duplicates_dropped,
        "cross_label_exact_duplicate_conflicts": 0,
        "counts_by_reference_type": dict(sorted(Counter(row["reference_type"] for row in unique).items())),
        "counts_by_annotation_status": dict(sorted(Counter(row["annotation_status"] for row in unique).items())),
        "counts_by_assigned_split": dict(sorted(Counter(row["assigned_split"] for row in unique).items())),
        "train_candidate_rows": len(train),
        "gold_dev_rows": sum(row["split"] == "dev" for row in gold),
        "gold_test_rows": sum(row["split"] == "test" for row in gold),
        "withheld_single_annotated": dict(sorted(withheld.items())),
        "teacher_session_group_overlap_count": 0,
        "model_predictions_used_as_labels": False,
        "gold_policy": "dev/test require adjudicated; single_annotated is never gold",
        "empty_input_is_valid": True,
    }
    return unique, train, gold, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and split append-only human reference annotations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--normalized-output", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--gold-output", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, blank_lines = read_jsonl(args.input)
    normalized, train, gold, report = prepare_human_annotations(rows)
    report["input_path"] = str(args.input)
    report["input_exists"] = args.input.exists()
    report["blank_lines_ignored"] = blank_lines
    report["outputs"] = {
        "normalized": str(args.normalized_output),
        "train_candidates": str(args.train_output),
        "gold_benchmark": str(args.gold_output),
    }
    write_json(args.normalized_output, normalized)
    write_json(args.train_output, train)
    write_json(args.gold_output, gold)
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "normalized": len(normalized),
                "train_candidates": len(train),
                "gold_dev": report["gold_dev_rows"],
                "gold_test": report["gold_test_rows"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
