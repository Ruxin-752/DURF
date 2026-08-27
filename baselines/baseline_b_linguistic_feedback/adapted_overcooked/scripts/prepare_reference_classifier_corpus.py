"""Build the five-class Overcooked reference-classifier corpus.

The paper-style classifier needs phrases whose reference scope is recoverable
from the text alone.  The stricter v7 generator supplies trajectory, feature,
and spatial phrases; the existing v6 corpus supplies the already reliable
behavioral class; and the dedicated hard-negative corpus supplies ``other``.

This script never uses classifier predictions to select examples.  It removes
only deterministic label conflicts, exact cross-split leakage, and duplicate
rows, so the declared test split remains independent of model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import read_json, write_json  # noqa: E402


DEFAULT_V7 = ROOT / "outputs" / "synth" / "synthetic_feedback.reference_v7.raw.json"
DEFAULT_BEHAVIOR = ROOT / "data" / "synthetic_feedback.deepseek_12k.validated.json"
DEFAULT_OTHER = ROOT / "outputs" / "synth" / "deepseek_other_reference_v2.json"
DEFAULT_CONTRASTIVE = ROOT / "data" / "reference_classifier_contrastive.v1.json"
DEFAULT_PAPER_LABELS = (
    ROOT.parent
    / "original_rewards_repo"
    / "notebooks"
    / "data"
    / "pilot_chat_messages_labels.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "reference_classifier_feedback.v8.json"
DEFAULT_REPORT = ROOT / "outputs" / "synth" / "reference_classifier_corpus.v8.report.json"
DEFAULT_PAPER_STRICT_OUTPUT = ROOT / "data" / "reference_classifier_paper_strict.v1.json"
DEFAULT_PAPER_STRICT_REPORT = (
    ROOT / "outputs" / "synth" / "reference_classifier_paper_strict.v1.report.json"
)
PAPER_STRICT_VERSION = "paper-task-text-disjoint-v1"

REFERENCE_TYPES = (
    "trajectory",
    "feature",
    "action_spatial",
    "action_behavioral",
    "other",
)
SPLITS = ("train", "dev", "test")
SPLIT_RANK = {name: index for index, name in enumerate(SPLITS)}
PAPER_LABEL_MAP = {
    "trajectory": "trajectory",
    "features": "feature",
    "object_spatial": "action_spatial",
    "object_behavior": "action_behavioral",
    "other": "other",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_rows(
    v7_rows: list[dict],
    behavior_rows: list[dict],
    other_rows: list[dict],
    contrastive_rows: list[dict] | None = None,
    paper_development_rows: list[dict] | None = None,
) -> list[dict]:
    selected: list[dict] = []
    sources = (
        (
            "deepseek_reference_v7",
            v7_rows,
            {"trajectory", "feature", "action_spatial"},
        ),
        ("deepseek_behavior_v6", behavior_rows, {"action_behavioral"}),
        ("deepseek_other_v2", other_rows, {"other"}),
        (
            "deterministic_contrastive_v1",
            contrastive_rows or [],
            set(REFERENCE_TYPES),
        ),
        (
            "original_paper_strict_development",
            paper_development_rows or [],
            set(REFERENCE_TYPES),
        ),
    )
    for origin, rows, allowed_labels in sources:
        for raw in rows:
            label = str(raw.get("reference_type") or "")
            if label not in allowed_labels:
                continue
            row = dict(raw)
            row["classifier_corpus_origin"] = origin
            selected.append(row)
    return selected


def _strict_paper_candidate(
    eligible: list[dict],
    *,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> dict:
    """Build one task split and remove every normalized-text bridge."""

    group_ids = sorted({str(row["paper_task_uuid"]) for row in eligible})
    ordered_groups = sorted(
        group_ids,
        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest(),
    )
    test_groups = max(1, round(len(ordered_groups) * test_fraction))
    dev_groups = max(1, round(len(ordered_groups) * dev_fraction))
    train_groups = len(ordered_groups) - dev_groups - test_groups
    if train_groups < 1:
        raise ValueError("paper strict split needs at least three task groups")
    group_splits = {
        "train": set(ordered_groups[:train_groups]),
        "dev": set(ordered_groups[train_groups : train_groups + dev_groups]),
        "test": set(ordered_groups[train_groups + dev_groups :]),
    }
    assigned = {
        split: [
            row for row in eligible if str(row["paper_task_uuid"]) in groups
        ]
        for split, groups in group_splits.items()
    }

    text_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in assigned.items():
        for row in rows:
            text_splits[str(row["normalized_text"])].add(split)
    cross_split_texts = {
        text for text, splits in text_splits.items() if len(splits) > 1
    }

    kept: dict[str, list[dict]] = {}
    within_split_duplicates = 0
    within_split_conflicts = 0
    for split, rows in assigned.items():
        by_text: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["normalized_text"] not in cross_split_texts:
                by_text[str(row["normalized_text"])].append(row)
        split_rows: list[dict] = []
        for duplicates in by_text.values():
            labels = {str(row["reference_type"]) for row in duplicates}
            if len(labels) > 1:
                within_split_conflicts += len(duplicates)
                continue
            split_rows.append(min(duplicates, key=lambda row: int(row["source_index"])))
            within_split_duplicates += len(duplicates) - 1
        kept[split] = sorted(split_rows, key=lambda row: int(row["source_index"]))

    counts = {
        split: Counter(row["reference_type"] for row in rows)
        for split, rows in kept.items()
    }
    return {
        "seed": seed,
        "kept": kept,
        "counts": counts,
        "group_splits": group_splits,
        "cross_split_texts": cross_split_texts,
        "cross_split_rows_dropped": sum(
            row["normalized_text"] in cross_split_texts for row in eligible
        ),
        "within_split_duplicates_dropped": within_split_duplicates,
        "within_split_conflicts_dropped": within_split_conflicts,
    }


def paper_strict_partition(
    source_rows: list[dict],
    *,
    candidate_seeds: int = 4096,
    dev_fraction: float = 0.15,
    test_fraction: float = 0.15,
    minimum_per_label: int = 5,
) -> tuple[list[dict], dict]:
    """Create a fixed task- and normalized-text-disjoint paper benchmark.

    Candidate partitions are ranked only by label coverage/distribution and split
    size. No classifier is trained or queried while choosing the split.
    """

    if candidate_seeds < 1:
        raise ValueError("candidate_seeds must be positive")
    eligible: list[dict] = []
    excluded_empty = 0
    excluded_missing_task = 0
    for source_index, row in enumerate(source_rows):
        source_label = str(row.get("reference_type") or "")
        if source_label not in PAPER_LABEL_MAP:
            continue
        text = str(row.get("phrase") or row.get("chat") or "").strip()
        normalized = normalize_text(text)
        task_uuid = str(row.get("task_uuid") or "").strip()
        if not normalized:
            excluded_empty += 1
            continue
        if not task_uuid:
            excluded_missing_task += 1
            continue
        eligible.append(
            {
                "source_index": source_index,
                "paper_task_uuid": task_uuid,
                "text": text,
                "normalized_text": normalized,
                "reference_type": PAPER_LABEL_MAP[source_label],
            }
        )
    if not eligible:
        raise ValueError("no eligible original-paper reference rows")

    global_counts = Counter(row["reference_type"] for row in eligible)
    global_distribution = {
        label: global_counts[label] / len(eligible) for label in REFERENCE_TYPES
    }
    best: tuple[tuple[float, float, int, int], dict] | None = None
    for seed in range(candidate_seeds):
        candidate = _strict_paper_candidate(
            eligible,
            seed=seed,
            dev_fraction=dev_fraction,
            test_fraction=test_fraction,
        )
        counts = candidate["counts"]
        if any(
            counts[split][label] < minimum_per_label
            for split in SPLITS
            for label in REFERENCE_TYPES
        ):
            continue
        kept = candidate["kept"]
        label_divergence = sum(
            abs(counts[split][label] / len(kept[split]) - global_distribution[label])
            for split in SPLITS
            for label in REFERENCE_TYPES
        )
        total_kept = sum(len(kept[split]) for split in SPLITS)
        fraction_error = sum(
            abs(
                len(kept[split]) / total_kept
                - {
                    "train": 1.0 - dev_fraction - test_fraction,
                    "dev": dev_fraction,
                    "test": test_fraction,
                }[split]
            )
            for split in SPLITS
        )
        minimum_count = min(
            counts[split][label] for split in SPLITS for label in REFERENCE_TYPES
        )
        score = (label_divergence, fraction_error, -minimum_count, seed)
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        raise ValueError(
            "could not build a paper strict split with all labels; "
            f"candidate_seeds={candidate_seeds}, minimum_per_label={minimum_per_label}"
        )

    score, selected = best
    converted: list[dict] = []
    for split in SPLITS:
        for row in selected["kept"][split]:
            text = str(row["text"])
            task_uuid = str(row["paper_task_uuid"])
            label = str(row["reference_type"])
            converted.append(
                {
                    "feedback_id": f"original_paper_strict_{row['source_index']}",
                    "group_id": f"paper_task_uuid:{task_uuid}",
                    "paper_task_uuid": task_uuid,
                    "paper_source_index": int(row["source_index"]),
                    "paraphrase_family": f"paper_task_uuid:{task_uuid}",
                    "phrase_annotations": [
                        {
                            "start": 0,
                            "end": len(text),
                            "text": text,
                            "reference_type": label,
                        }
                    ],
                    "reference_type": label,
                    "role": "original_paper_phrase",
                    "source": "original_paper_human",
                    "split": split,
                    "text": text,
                }
            )

    split_groups = {
        split: sorted(
            str(group) for group in selected["group_splits"][split]
        )
        for split in SPLITS
    }
    split_texts = {
        split: {
            normalize_text(row["text"])
            for row in converted
            if row["split"] == split
        }
        for split in SPLITS
    }
    if any(
        set(split_groups[left]) & set(split_groups[right])
        or split_texts[left] & split_texts[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise AssertionError("paper strict partition is not task/text disjoint")
    report = {
        "version": PAPER_STRICT_VERSION,
        "selection_uses_model_predictions": False,
        "selection_criterion": (
            "all labels meet fixed floor, then label-distribution L1, "
            "split-size error, minimum label count, seed"
        ),
        "candidate_seeds": candidate_seeds,
        "selected_seed": int(selected["seed"]),
        "selected_score": {
            "label_distribution_l1": float(score[0]),
            "split_fraction_error": float(score[1]),
            "minimum_split_label_count": int(-score[2]),
        },
        "source_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "excluded_empty_text": excluded_empty,
        "excluded_missing_task_uuid": excluded_missing_task,
        "cross_split_normalized_texts_dropped": len(selected["cross_split_texts"]),
        "cross_split_rows_dropped": int(selected["cross_split_rows_dropped"]),
        "within_split_duplicates_dropped": int(
            selected["within_split_duplicates_dropped"]
        ),
        "within_split_conflicts_dropped": int(
            selected["within_split_conflicts_dropped"]
        ),
        "split_sizes": {
            split: sum(row["split"] == split for row in converted) for split in SPLITS
        },
        "split_group_counts": {
            split: len(split_groups[split]) for split in SPLITS
        },
        "split_label_counts": {
            split: {
                label: sum(
                    row["split"] == split and row["reference_type"] == label
                    for row in converted
                )
                for label in REFERENCE_TYPES
            }
            for split in SPLITS
        },
        "group_overlap_count": 0,
        "normalized_text_overlap_count": 0,
    }
    return converted, report


def prepare_reference_corpus(
    v7_rows: list[dict],
    behavior_rows: list[dict],
    other_rows: list[dict],
    *,
    contrastive_rows: list[dict] | None = None,
    paper_development_rows: list[dict] | None = None,
    reserved_external_texts: set[str] | None = None,
    minimum_per_class_per_split: int = 1,
) -> tuple[list[dict], dict]:
    """Merge sources and remove only deterministic conflicts/leakage."""

    candidates = _selected_rows(
        v7_rows,
        behavior_rows,
        other_rows,
        contrastive_rows,
        paper_development_rows,
    )
    reserved = {normalize_text(text) for text in (reserved_external_texts or set())}
    reserved_rows = [
        row for row in candidates if normalize_text(row.get("text")) in reserved
    ]
    candidates = [
        row for row in candidates if normalize_text(row.get("text")) not in reserved
    ]
    invalid = [
        row
        for row in candidates
        if row.get("reference_type") not in REFERENCE_TYPES
        or row.get("split") not in SPLITS
        or not normalize_text(row.get("text"))
    ]
    if invalid:
        raise ValueError(f"invalid reference rows: {len(invalid)}")

    locations: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"labels": set(), "splits": set()}
    )
    for row in candidates:
        normalized = normalize_text(row.get("text"))
        locations[normalized]["labels"].add(str(row["reference_type"]))
        locations[normalized]["splits"].add(str(row["split"]))

    cross_label = {
        text for text, metadata in locations.items() if len(metadata["labels"]) > 1
    }
    cross_split = {
        text for text, metadata in locations.items() if len(metadata["splits"]) > 1
    }
    forbidden = cross_label | cross_split

    kept_by_key: dict[tuple[str, str, str], dict] = {}
    duplicate_rows = 0
    dropped_rows = Counter()
    conflict_samples: list[dict] = []
    for row in candidates:
        normalized = normalize_text(row.get("text"))
        if normalized in cross_label:
            dropped_rows["cross_label_exact_conflict"] += 1
            if len(conflict_samples) < 20:
                conflict_samples.append(
                    {
                        "text": row.get("text"),
                        "label": row.get("reference_type"),
                        "split": row.get("split"),
                    }
                )
            continue
        if normalized in cross_split:
            dropped_rows["cross_split_exact_leakage"] += 1
            continue
        key = (str(row["split"]), normalized, str(row["reference_type"]))
        if key in kept_by_key:
            duplicate_rows += 1
            dropped_rows["within_split_exact_duplicate"] += 1
            continue
        kept_by_key[key] = row

    kept = sorted(
        kept_by_key.values(),
        key=lambda row: (
            SPLIT_RANK[str(row["split"])],
            str(row.get("reference_type")),
            str(row.get("group_id") or ""),
            str(row.get("feedback_id") or ""),
            normalize_text(row.get("text")),
        ),
    )

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in kept:
        if row.get("group_id") is not None:
            group_splits[str(row["group_id"])].add(str(row["split"]))
    leaking_groups = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    if leaking_groups:
        raise ValueError(f"group ids cross declared splits: {leaking_groups[:5]}")

    split_label_counts = {
        split: {
            label: sum(
                row["split"] == split and row["reference_type"] == label
                for row in kept
            )
            for label in REFERENCE_TYPES
        }
        for split in SPLITS
    }
    underfilled = [
        f"{split}:{label}={count}"
        for split, counts in split_label_counts.items()
        for label, count in counts.items()
        if count < minimum_per_class_per_split
    ]
    if underfilled:
        raise ValueError("underfilled reference buckets: " + ", ".join(underfilled))

    report = {
        "policy": (
            "v7_scope_plus_v6_behavior_plus_hard_other_plus_"
            "deterministic_contrastive_v1"
        ),
        "selection_uses_model_predictions": False,
        "reserved_external_text_rows_dropped": len(reserved_rows),
        "candidate_rows": len(candidates),
        "kept_rows": len(kept),
        "dropped_rows": dict(sorted(dropped_rows.items())),
        "duplicate_rows": duplicate_rows,
        "cross_label_conflicting_texts": len(cross_label),
        "cross_split_leaking_texts": len(cross_split),
        "forbidden_normalized_texts": len(forbidden),
        "split_label_counts": split_label_counts,
        "split_sizes": {
            split: sum(row["split"] == split for row in kept) for split in SPLITS
        },
        "origin_counts": dict(
            sorted(Counter(row["classifier_corpus_origin"] for row in kept).items())
        ),
        "group_counts": {
            split: len(
                {
                    str(row["group_id"])
                    for row in kept
                    if row["split"] == split and row.get("group_id") is not None
                }
            )
            for split in SPLITS
        },
        "conflict_samples": conflict_samples,
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--other", type=Path, default=DEFAULT_OTHER)
    parser.add_argument("--contrastive", type=Path, default=DEFAULT_CONTRASTIVE)
    parser.add_argument("--paper-labels", type=Path, default=DEFAULT_PAPER_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--paper-strict-output", type=Path, default=DEFAULT_PAPER_STRICT_OUTPUT)
    parser.add_argument("--paper-strict-report", type=Path, default=DEFAULT_PAPER_STRICT_REPORT)
    parser.add_argument("--paper-strict-candidate-seeds", type=int, default=4096)
    parser.add_argument("--minimum-per-class-per-split", type=int, default=1)
    args = parser.parse_args()

    v7_rows = read_json(args.v7)
    behavior_rows = read_json(args.behavior)
    other_rows = read_json(args.other)
    contrastive_rows = read_json(args.contrastive)
    with args.paper_labels.open(encoding="utf-8-sig", newline="") as handle:
        paper_source_rows = list(csv.DictReader(handle))
    paper_strict_rows, paper_strict_report = paper_strict_partition(
        paper_source_rows,
        candidate_seeds=args.paper_strict_candidate_seeds,
    )
    paper_development_rows = [
        row for row in paper_strict_rows if row["split"] in {"train", "dev"}
    ]
    paper_external_texts = {
        normalize_text(row["text"])
        for row in paper_strict_rows
        if row["split"] == "test"
    }
    rows, report = prepare_reference_corpus(
        v7_rows,
        behavior_rows,
        other_rows,
        contrastive_rows=contrastive_rows,
        paper_development_rows=paper_development_rows,
        reserved_external_texts=paper_external_texts,
        minimum_per_class_per_split=args.minimum_per_class_per_split,
    )
    report["paper_strict_development_selection"] = {
        "version": paper_strict_report["version"],
        "train_rows": paper_strict_report["split_sizes"]["train"],
        "dev_rows": paper_strict_report["split_sizes"]["dev"],
        "reserved_test_rows": paper_strict_report["split_sizes"]["test"],
        "selected_seed": paper_strict_report["selected_seed"],
    }
    report["inputs"] = {
        "v7": {"path": str(args.v7), "sha256": _sha256(args.v7)},
        "behavior": {
            "path": str(args.behavior),
            "sha256": _sha256(args.behavior),
        },
        "other": {"path": str(args.other), "sha256": _sha256(args.other)},
        "contrastive": {
            "path": str(args.contrastive),
            "sha256": _sha256(args.contrastive),
        },
        "paper_labels": {
            "path": str(args.paper_labels),
            "sha256": _sha256(args.paper_labels),
        },
    }
    report["output"] = str(args.output)
    args.paper_strict_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.paper_strict_output, paper_strict_rows)
    paper_strict_report["source"] = {
        "path": str(args.paper_labels),
        "sha256": _sha256(args.paper_labels),
    }
    paper_strict_report["output"] = str(args.paper_strict_output)
    paper_strict_report["output_sha256"] = _sha256(args.paper_strict_output)
    write_json(args.paper_strict_report, paper_strict_report)
    write_json(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
