"""Create deterministic, train-only variants of explicit human form labels.

The prepared human train file is read-only.  Development and test files are
loaded only to reject exact, template-family, and near-duplicate leakage.  No
holdout text, label distribution, or model metric is written to the report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_human_feedback_form_annotations import (  # noqa: E402
    CANONICAL_LABELS,
    INTERNAL_LABELS,
    template_family_signature,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_TRAIN = ROOT / "data" / "human_feedback_form_train.json"
DEFAULT_DEV = ROOT / "data" / "human_feedback_form_dev.json"
DEFAULT_TEST = ROOT / "data" / "human_feedback_form_test.json"
DEFAULT_OUTPUT = ROOT / "data" / "human_feedback_form_train_augmentation.json"
DEFAULT_REPORT = ROOT / "outputs" / "human_feedback_form_train_augmentation.report.json"

POLICY_VERSION = "human-feedback-form-train-augmentation-v1"
SOURCE = "human_train_augmentation"
LABEL_SOURCE = "deterministic_label_preserving_transform"
DEFAULT_MAX_VARIANTS_PER_PARENT = 6
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.88

_SPELLING_REPLACEMENTS = {
    "impressice": "impressive",
    "jut": "just",
    "tomatos": "tomatoes",
    "unneccessary": "unnecessary",
}

_ENTITY_SWAPS = {
    "ingredient": {
        "tomato": "onion",
        "tomatoes": "onions",
        "onion": "tomato",
        "onions": "tomatoes",
    },
    "dishware": {
        "plate": "dish",
        "plates": "dishes",
        "dish": "plate",
        "dishes": "plates",
    },
    "serving_location": {
        "serving window": "serving counter",
        "serving counter": "serving window",
    },
}


def _read_json_list(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_words(text: str, replacements: dict[str, str]) -> str:
    pattern = re.compile(
        r"\b(?:" + "|".join(
            re.escape(key) for key in sorted(replacements, key=len, reverse=True)
        ) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        return _preserve_case(source, replacements[source.casefold()])

    return pattern.sub(replace, text)


def correct_spelling(text: str) -> str:
    """Correct only the small, audited spelling list; never mutate the input row."""

    return _replace_words(text, _SPELLING_REPLACEMENTS)


def _fix_swapped_articles(text: str) -> str:
    repairs = (
        (r"\ba onion\b", "an onion"),
        (r"\bA onion\b", "An onion"),
        (r"\ban tomato\b", "a tomato"),
        (r"\bAn tomato\b", "A tomato"),
        (r"\ba ingredient\b", "an ingredient"),
        (r"\bA ingredient\b", "An ingredient"),
    )
    for pattern, replacement in repairs:
        text = re.sub(pattern, replacement, text)
    return text


def entity_variants(text: str) -> list[tuple[str, str]]:
    """Swap all mentions within one ontology class in a single pass."""

    variants: list[tuple[str, str]] = []
    for group, mapping in _ENTITY_SWAPS.items():
        changed = _fix_swapped_articles(_replace_words(text, mapping))
        if normalize_text(changed) != normalize_text(text):
            variants.append((f"entity_swap:{group}", changed))
    return variants


def _sub_variant(
    text: str,
    rule: str,
    pattern: str,
    replacement: str,
    *,
    flags: int = re.IGNORECASE,
) -> tuple[str, str] | None:
    changed, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if text[:1].isupper() and changed[:1].islower():
        changed = changed[:1].upper() + changed[1:]
    if count and normalize_text(changed) != normalize_text(text):
        return rule, changed
    return None


def _evaluative_variants(text: str) -> list[tuple[str, str]]:
    rules = (
        ("evaluative_scope:add_last", r"\bThat move\b", "That last move"),
        ("evaluative_scope:remove_last", r"\bThat last move\b", "That move"),
        ("evaluative_synonym:helpful_to_useful", r"\bhelpful\b", "useful"),
        ("evaluative_synonym:good_to_helpful", r"\bgood\b", "helpful"),
        ("evaluative_synonym:bad_to_poor", r"\bbad\b", "poor"),
        ("evaluative_synonym:efficient_to_effective", r"\befficient\b", "effective"),
        ("evaluative_synonym:fine_to_acceptable", r"\bfine\b", "acceptable"),
        ("evaluative_synonym:ok_to_acceptable", r"\bok\b", "acceptable"),
        (
            "evaluative_synonym:quite_impressive_to_excellent",
            r"\bquite impressive\b",
            "excellent",
        ),
        (
            "evaluative_synonym:impressive_to_excellent",
            r"(?<!quite )\bimpressive\b",
            "excellent",
        ),
        ("evaluative_surface:unnecessary_to_not_needed", r"\bunnecessary\b", "not needed"),
    )
    return [
        result
        for rule, pattern, replacement in rules
        if (result := _sub_variant(text, rule, pattern, replacement)) is not None
    ]


def _imperative_variants(text: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    if re.match(r"^please\s+", text, flags=re.IGNORECASE):
        result = _sub_variant(
            text,
            "imperative_politeness:remove_please",
            r"^please\s+",
            "",
        )
        if result:
            variants.append(result)
    elif re.match(
        r"^(?:fetch|grab|help|put|take|bring|serve|move|go|get|pick|do not)\b",
        text,
        flags=re.IGNORECASE,
    ):
        variants.append(("imperative_politeness:add_please", f"Please {text[:1].lower()}{text[1:]}"))

    rules = (
        (
            "imperative_synonym:fetch_to_grab",
            r"^((?:please\s+)?)fetch\b",
            r"\1grab",
        ),
        (
            "imperative_synonym:grab_to_pick_up",
            r"^((?:please\s+)?(?:help me\s+)?)grab\b",
            r"\1pick up",
        ),
        (
            "imperative_synonym:put_to_place",
            r"^((?:please\s+)?)put\b",
            r"\1place",
        ),
        (
            "imperative_synonym:take_to_bring",
            r"^((?:please\s+)?)take\b",
            r"\1bring",
        ),
        (
            "imperative_surface:go_to_grab_to_get",
            r"^((?:please\s+)?)go to grab\b",
            r"\1get",
        ),
    )
    variants.extend(
        result
        for rule, pattern, replacement in rules
        if (result := _sub_variant(text, rule, pattern, replacement)) is not None
    )
    return variants


def _descriptive_variants(text: str) -> list[tuple[str, str]]:
    rules = (
        (
            "descriptive_grammar:gerund_grabbing",
            r"^Go to grab another\b",
            "Grabbing another",
        ),
        (
            "descriptive_surface:unnecessary_to_not_needed",
            r"\bis unnecessary\b",
            "is not needed",
        ),
        (
            "descriptive_synonym:good_to_helpful",
            r"\bis good\b(?!\s+not blocking)",
            "is helpful",
        ),
        (
            "descriptive_synonym:bad_to_harmful",
            r"\bis bad\b(?!\s+blocking)",
            "is harmful",
        ),
        (
            "descriptive_grammar:will_be_not",
            r"\bwill be not helpful\b",
            "will not be helpful",
        ),
        (
            "descriptive_grammar:not_blocking",
            r"^It is good not blocking my way\.?$",
            "Not blocking my way is good.",
        ),
        (
            "descriptive_grammar:blocking",
            r"^It is bad blocking my way\.?$",
            "Blocking my way is bad.",
        ),
    )
    return [
        result
        for rule, pattern, replacement in rules
        if (result := _sub_variant(text, rule, pattern, replacement)) is not None
    ]


_LABEL_VARIANTS = {
    "Evaluative": _evaluative_variants,
    "Imperative": _imperative_variants,
    "Descriptive": _descriptive_variants,
}


def generate_parent_candidates(parent: dict) -> list[tuple[str, str]]:
    """Return a deterministic, bounded source list before global safety filters."""

    raw_text = str(parent["text"])
    corrected = correct_spelling(raw_text)
    prefix = "spelling_correction+" if corrected != raw_text else ""
    candidates: list[tuple[str, str]] = []
    if normalize_text(corrected) != normalize_text(raw_text):
        candidates.append(("spelling_correction", corrected))
    candidates.extend(
        (prefix + rule, text) for rule, text in entity_variants(corrected)
    )
    candidates.extend(
        (prefix + rule, text)
        for rule, text in _LABEL_VARIANTS[str(parent["classification_label"])](corrected)
    )

    unique: dict[str, tuple[str, str]] = {}
    raw_normalized = normalize_text(raw_text)
    for rule, text in candidates:
        normalized = normalize_text(text)
        if normalized and normalized != raw_normalized and normalized not in unique:
            unique[normalized] = (rule, text.strip())
    return list(unique.values())


def _validate_train_parent(row: dict, index: int) -> dict:
    required = {
        "feedback_id",
        "text",
        "classification_label",
        "expected_feedback_type",
        "split",
        "source",
        "label_source",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"train row {index}: missing fields {missing}")
    if row["split"] != "train":
        raise ValueError(f"train row {index}: split must be 'train'")
    if row["source"] != "human" or row["label_source"] != "human_explicit":
        raise ValueError(f"train row {index}: parent must be explicit human data")
    canonical = row["classification_label"]
    if canonical not in CANONICAL_LABELS:
        raise ValueError(f"train row {index}: invalid canonical label")
    if row["expected_feedback_type"] != INTERNAL_LABELS[canonical]:
        raise ValueError(f"train row {index}: canonical/internal label mismatch")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError(f"train row {index}: text must be non-empty")
    if not isinstance(row["feedback_id"], str) or not row["feedback_id"].strip():
        raise ValueError(f"train row {index}: feedback_id must be non-empty")
    return dict(row)


def _holdout_texts(rows: Iterable[dict], name: str) -> list[str]:
    normalized: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{name} row {index}: text must be non-empty")
        normalized.append(normalize_text(text))
    return normalized


def _augmentation_id(parent_id: str, normalized: str) -> str:
    payload = f"{POLICY_VERSION}\0{parent_id}\0{normalized}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"human_feedback_form_aug_{digest[:24]}"


def augment_training_rows(
    train_rows: Iterable[dict],
    dev_rows: Iterable[dict],
    test_rows: Iterable[dict],
    *,
    max_variants_per_parent: int = DEFAULT_MAX_VARIANTS_PER_PARENT,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Generate safe train variants without exposing holdout contents."""

    if not 1 <= max_variants_per_parent <= 12:
        raise ValueError("max_variants_per_parent must be between 1 and 12")
    if not 0.75 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0.75 and 1.0")

    parents = [
        _validate_train_parent(row, index)
        for index, row in enumerate(train_rows, start=1)
    ]
    parents.sort(key=lambda row: row["feedback_id"])
    parent_ids = [row["feedback_id"] for row in parents]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("train parent feedback_id values must be unique")

    train_normalized = {normalize_text(row["text"]) for row in parents}
    holdout_normalized = _holdout_texts(dev_rows, "development") + _holdout_texts(
        test_rows, "test"
    )
    holdout_exact = set(holdout_normalized)
    holdout_templates = {
        template_family_signature(normalized) for normalized in holdout_normalized
    }

    rejections = Counter(
        {
            "train_exact_duplicate": 0,
            "holdout_exact": 0,
            "holdout_template_family": 0,
            "holdout_near_duplicate": 0,
            "global_same_label_duplicate": 0,
            "global_cross_label_conflict": 0,
            "per_parent_cap": 0,
        }
    )
    provisional: list[dict] = []
    generated_candidates = 0
    for parent in parents:
        for order, (rule, text) in enumerate(generate_parent_candidates(parent)):
            generated_candidates += 1
            normalized = normalize_text(text)
            if normalized in train_normalized:
                rejections["train_exact_duplicate"] += 1
                continue
            if normalized in holdout_exact:
                rejections["holdout_exact"] += 1
                continue
            if template_family_signature(normalized) in holdout_templates:
                rejections["holdout_template_family"] += 1
                continue
            if any(
                SequenceMatcher(None, normalized, held_out).ratio()
                >= near_duplicate_threshold
                for held_out in holdout_normalized
            ):
                rejections["holdout_near_duplicate"] += 1
                continue
            provisional.append(
                {
                    "parent": parent,
                    "order": order,
                    "text": text,
                    "normalized_text": normalized,
                    "augmentation_rule": rule,
                }
            )

    by_normalized: dict[str, list[dict]] = defaultdict(list)
    for candidate in provisional:
        by_normalized[candidate["normalized_text"]].append(candidate)

    unique_candidates: list[dict] = []
    for normalized in sorted(by_normalized):
        same_text = sorted(
            by_normalized[normalized],
            key=lambda row: (row["parent"]["feedback_id"], row["order"]),
        )
        labels = {row["parent"]["classification_label"] for row in same_text}
        if len(labels) > 1:
            rejections["global_cross_label_conflict"] += len(same_text)
            continue
        unique_candidates.append(same_text[0])
        rejections["global_same_label_duplicate"] += len(same_text) - 1

    accepted_per_parent = Counter()
    augmented: list[dict] = []
    for candidate in sorted(
        unique_candidates,
        key=lambda row: (row["parent"]["feedback_id"], row["order"]),
    ):
        parent = candidate["parent"]
        parent_id = parent["feedback_id"]
        if accepted_per_parent[parent_id] >= max_variants_per_parent:
            rejections["per_parent_cap"] += 1
            continue
        accepted_per_parent[parent_id] += 1
        normalized = candidate["normalized_text"]
        family = parent.get("template_family") or template_family_signature(
            normalize_text(parent["text"])
        )
        augmented.append(
            {
                "feedback_id": _augmentation_id(parent_id, normalized),
                "parent_feedback_id": parent_id,
                "text": candidate["text"],
                "normalized_text": normalized,
                "classification_label": parent["classification_label"],
                "expected_feedback_type": parent["expected_feedback_type"],
                "augmentation_rule": candidate["augmentation_rule"],
                "template_family": family,
                "group_id": f"human_train_augmentation:{parent_id}",
                "split": "train",
                "source": SOURCE,
                "label_source": LABEL_SOURCE,
            }
        )

    if len({row["normalized_text"] for row in augmented}) != len(augmented):
        raise AssertionError("augmentation normalized text must be unique")
    if {row["normalized_text"] for row in augmented} & train_normalized:
        raise AssertionError("augmentation duplicates prepared train text")
    if {row["normalized_text"] for row in augmented} & holdout_exact:
        raise AssertionError("augmentation exact holdout leakage")

    report = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "source": SOURCE,
        "label_source": LABEL_SOURCE,
        "raw_human_rows_modified": False,
        "train_parent_rows": len(parents),
        "generated_candidates": generated_candidates,
        "augmentation_rows": len(augmented),
        "max_variants_per_parent": max_variants_per_parent,
        "augmentation_label_counts": dict(
            sorted(Counter(row["classification_label"] for row in augmented).items())
        ),
        "augmentation_rule_counts": dict(
            sorted(Counter(row["augmentation_rule"] for row in augmented).items())
        ),
        "rejections": dict(sorted(rejections.items())),
        "holdout_guard": {
            "dev_and_test_loaded_only_for_leakage_rejection": True,
            "checks": ["exact_text", "template_family", "near_duplicate"],
            "near_duplicate_threshold": near_duplicate_threshold,
            "holdout_content_in_report": False,
            "holdout_row_or_label_counts_in_report": False,
            "model_loaded": False,
            "performance_metrics_computed": False,
        },
    }
    return augmented, report


def _assert_separate_outputs(paths: Iterable[Path], output: Path, report: Path) -> None:
    protected = {path.resolve() for path in paths}
    if output.resolve() in protected or report.resolve() in protected:
        raise ValueError("augmentation outputs must not overwrite prepared human inputs")
    if output.resolve() == report.resolve():
        raise ValueError("augmentation data and report paths must be different")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--max-variants-per-parent",
        type=int,
        default=DEFAULT_MAX_VARIANTS_PER_PARENT,
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    args = parser.parse_args()

    _assert_separate_outputs(
        (args.train, args.dev, args.test), args.output, args.report
    )
    augmented, report = augment_training_rows(
        _read_json_list(args.train),
        _read_json_list(args.dev),
        _read_json_list(args.test),
        max_variants_per_parent=args.max_variants_per_parent,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    report["input_train"] = str(args.train)
    report["output_data"] = str(args.output)
    report["output_report"] = str(args.report)
    write_json(args.output, augmented)
    write_json(args.report, report)
    print(f"Prepared {len(augmented)} deterministic train-only augmentation rows.")
    print("Holdout data was used only for leakage rejection; no metrics were computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
