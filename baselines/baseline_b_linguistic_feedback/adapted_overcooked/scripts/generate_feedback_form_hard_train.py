"""Generate deterministic train-only hard examples for feedback-form classification.

The corpus contrasts the paper-facing feedback forms without copying held-out
sentences: descriptive state/value statements, imperative current commands,
and evaluative judgments of completed behavior.  The v8 dev/test partitions
are consulted only through normalized-text hashes and group identifiers as a
leakage gate.  Their contents, labels, and metrics are never emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


VERSION = "feedback-form-hard-train-v2"
DEFAULT_REFERENCE = ROOT / "data" / "reference_classifier_feedback.v8.json"
DEFAULT_OUTPUT = ROOT / "data" / "feedback_form_hard_train.v2.json"
DEFAULT_REPORT = (
    ROOT / "outputs" / "synth" / "feedback_form_hard_train.v2.report.json"
)

FEEDBACK_TYPES = ("evaluative", "imperative", "descriptive")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
SCHEMA_FIELDS = frozenset(
    {
        "text",
        "normalized",
        "expected_feedback_type",
        "classification_label",
        "split",
        "source",
        "label_source",
        "group_id",
        "template_family",
    }
)

# Eight held-in variants cross different objects, locations, and task contexts.
# Applying every variant to twelve independent families yields 96 rows per class.
VARIANTS = (
    {
        "item": "clean plate",
        "location": "dish return",
        "context": "before the next handoff",
    },
    {
        "item": "chopped tomato",
        "location": "serving window",
        "context": "during a busy recipe",
    },
    {
        "item": "raw onion",
        "location": "lower counter",
        "context": "while both cooks are crossing",
    },
    {
        "item": "empty dish",
        "location": "prep table",
        "context": "for the upcoming delivery",
    },
    {
        "item": "spare tomato",
        "location": "sink corner",
        "context": "when the kitchen is crowded",
    },
    {
        "item": "unused onion",
        "location": "tomato crate",
        "context": "between two orders",
    },
    {
        "item": "extra plate",
        "location": "central aisle",
        "context": "while a soup is cooking",
    },
    {
        "item": "unneeded vegetable",
        "location": "upper worktop",
        "context": "before the current timer expires",
    },
)

TEMPLATES = {
    "descriptive": (
        "A clear route beside the {location} makes coordination easier {context}.",
        "Open space around the {location} is valuable for smooth handoffs {context}.",
        "Easy access to the {location} is helpful for both cooks {context}.",
        "The unused {item} by the {location} creates avoidable clutter {context}.",
        "A spare {item} left near the {location} is unnecessary {context}.",
        "The extra {item} at the {location} is an obstacle rather than useful preparation {context}.",
        # The following six families are short, human-like minimal contrasts.
        # They deliberately share action words and modal words with the other
        # classes, so the classifier must learn sentence function rather than
        # a single keyword.  They are generated training-only and are not
        # copied from any human evaluation partition.
        "You are standing beside the {location} with the {item} {context}.",
        "The timer should finish while the {item} stays by the {location} {context}.",
        "The next order needs the {item} near the {location} {context}.",
        "You can reach the {location} while carrying the {item} {context}.",
        "The {location} could hold the {item} {context}.",
        "The route to the {location} is open for the {item} {context}.",
    ),
    "imperative": (
        "Clear the route beside the {location} now {context}.",
        "Keep the space around the {location} open right now {context}.",
        "Move away from the {location} and leave the lane clear {context}.",
        "Remove the unused {item} from the {location} now {context}.",
        "Take the spare {item} away from the {location} {context}.",
        "Please move the extra {item} off the {location} immediately {context}.",
        "Don't stand beside the {location} with the {item} {context}.",
        "You should carry the {item} to the {location} {context}.",
        "You need to bring the {item} to the {location} {context}.",
        "Can you go to the {location} with the {item} {context}?",
        "Could you leave the {item} at the {location} {context}?",
        "Why don't you take the {item} to the {location} {context}?",
    ),
    "evaluative": (
        "That detour to fetch the {item} slowed our completed handoff {context}.",
        "The trip you just made for the {item} cost the team time {context}.",
        "Our last loop past the {location} delayed the order {context}.",
        "Going back again for the {item} put the team behind {context}.",
        "Your previous round trip near the {location} made that delivery slower {context}.",
        "The route you took to the {location} wasted time on the last task {context}.",
        "Standing beside the {location} with the {item} was a bad choice on the last handoff {context}.",
        "You should not have carried the {item} to the {location} on the last handoff {context}.",
        "Bringing the {item} to the {location} was the right choice on the last order {context}.",
        "You reached the {location} with the {item} efficiently on the last trip {context}.",
        "Leaving the {item} at the {location} was unhelpful on the previous order {context}.",
        "Taking the {item} to the {location} was unnecessary on the last delivery {context}.",
    ),
}


def _text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _read_json_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _holdout_gate_tokens(reference_path: Path) -> tuple[set[str], set[str]]:
    """Return opaque hashes/groups for v8 dev/test leakage checks only."""

    text_hashes: set[str] = set()
    group_tokens: set[str] = set()
    for row in _read_json_rows(reference_path):
        if row.get("split") not in {"dev", "test"}:
            continue
        text = row.get("text")
        if isinstance(text, str) and normalize_text(text):
            text_hashes.add(_text_hash(text))
        for field in ("group_id", "paraphrase_family"):
            value = row.get(field)
            if isinstance(value, str) and value:
                group_tokens.add(value)
    return text_hashes, group_tokens


def generate_rows() -> list[dict]:
    """Build the deterministic 288-row train-only corpus before leakage gates."""

    if set(TEMPLATES) != set(FEEDBACK_TYPES):
        raise ValueError("template labels do not match the feedback-form labels")
    family_counts = {label: len(TEMPLATES[label]) for label in FEEDBACK_TYPES}
    if set(family_counts.values()) != {12}:
        raise ValueError(f"each class requires twelve template families: {family_counts}")

    rows: list[dict] = []
    for family_index in range(12):
        for variant_index, variant in enumerate(VARIANTS):
            group_id = (
                f"{VERSION}:contrast_{family_index:02d}_{variant_index:02d}"
            )
            for label in FEEDBACK_TYPES:
                text = TEMPLATES[label][family_index].format(**variant)
                rows.append(
                    {
                        "text": text,
                        "normalized": normalize_text(text),
                        "expected_feedback_type": label,
                        "classification_label": CANONICAL_LABELS[label],
                        "split": "train",
                        "source": "feedback_form_hard_train",
                        "label_source": "paper_mapping_contrastive_train_only",
                        "group_id": group_id,
                        "template_family": (
                            f"{VERSION}:{label}:family_{family_index:02d}"
                        ),
                    }
                )
    return rows


def build_hard_train_corpus(
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> tuple[list[dict], dict]:
    """Build, validate, and leakage-gate the train-only hard corpus."""

    rows = generate_rows()
    normalized_rows: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if set(row) != SCHEMA_FIELDS:
            raise AssertionError(f"unexpected generated schema: {sorted(row)}")
        normalized_rows[row["normalized"]].append(row)
        groups[row["group_id"]].add(row["expected_feedback_type"])
        families[row["expected_feedback_type"]].add(row["template_family"])

    duplicate_count = sum(len(values) - 1 for values in normalized_rows.values())
    cross_label_conflicts = sum(
        len({row["expected_feedback_type"] for row in values}) > 1
        for values in normalized_rows.values()
    )
    if duplicate_count:
        raise ValueError(f"normalized duplicate train phrases: {duplicate_count}")
    if cross_label_conflicts:
        raise ValueError(f"cross-label exact conflicts: {cross_label_conflicts}")
    if any(labels != set(FEEDBACK_TYPES) for labels in groups.values()):
        raise ValueError("each hard contrast group must contain all three labels")
    if any(len(values) != 12 for values in families.values()):
        raise ValueError("each class must contain twelve independent template families")

    holdout_hashes, holdout_groups = _holdout_gate_tokens(Path(reference_path))
    generated_hashes = {_text_hash(row["text"]) for row in rows}
    generated_group_tokens = {
        value
        for row in rows
        for value in (row["group_id"], row["template_family"])
    }
    exact_overlap_count = len(generated_hashes & holdout_hashes)
    group_overlap_count = len(generated_group_tokens & holdout_groups)
    if exact_overlap_count or group_overlap_count:
        # Deliberately report counts only; never expose held-out text or tokens.
        raise ValueError(
            "v8 holdout leakage gate failed: "
            f"exact_overlap_count={exact_overlap_count}, "
            f"group_overlap_count={group_overlap_count}"
        )

    label_counts = Counter(row["expected_feedback_type"] for row in rows)
    report = {
        "version": VERSION,
        "rows": len(rows),
        "split": "train",
        "train_only": True,
        "generation_is_deterministic": True,
        "generation_uses_llm": False,
        "selection_uses_model_predictions": False,
        "schema_fields": sorted(SCHEMA_FIELDS),
        "label_counts": dict(sorted(label_counts.items())),
        "canonical_label_counts": dict(
            sorted(Counter(row["classification_label"] for row in rows).items())
        ),
        "template_families_per_class": 12,
        "variants_per_template_family": len(VARIANTS),
        "contrast_groups": len(groups),
        "normalized_duplicate_count": 0,
        "cross_label_exact_conflict_count": 0,
        "holdout_leakage_gate": {
            "reference": "reference_classifier_feedback.v8.json",
            "partitions": ["dev", "test"],
            "usage": "normalized_text_hash_and_group_gate_only",
            "heldout_contents_output": False,
            "heldout_labels_accessed": False,
            "heldout_metrics_computed": False,
            "exact_overlap_count": 0,
            "group_overlap_count": 0,
        },
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows, report = build_hard_train_corpus(args.reference)
    write_json(args.output, rows)
    report["output"] = str(args.output)
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
