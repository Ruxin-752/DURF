"""Train an isolated direct-fG v2 candidate from audited train-only inputs.

This entry point never opens a mixed test container, a human test file, or a
frozen benchmark.  Targets come from explicit three-class fields.  Five-class
``reference_type`` metadata is never converted into a target.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_feedback_form_classifier as shared  # noqa: E402
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
    FEEDBACK_TYPES,
)


VERSION = "direct-fg-classifier-candidate-v2"
SEED = 1
MIN_DF = 1
NEAR_SEQUENCE_THRESHOLD = 0.88
NEAR_TOKEN_JACCARD_THRESHOLD = 0.70
NEAR_LENGTH_RATIO_THRESHOLD = 0.88
DEV_COMPONENT_FRACTION = 0.25
C_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CLASS_WEIGHT_GRID = (None, "balanced")

DEFAULT_HARD_V2 = ROOT / "data" / "feedback_form_hard_train.v2.json"
DEFAULT_HUMAN_TRAIN = ROOT / "data" / "human_feedback_form_train.json"
DEFAULT_HUMAN_AUGMENTATION = (
    ROOT / "data" / "human_feedback_form_train_augmentation.json"
)
DEFAULT_BOUNDARY = ROOT / "data" / "feedback_form_boundary_train.v1.json"
DEFAULT_HUMAN_DEV = ROOT / "data" / "human_feedback_form_dev.json"
DEFAULT_PAPER_PROXY = ROOT / "data" / "paper_feedback_form_human_dev.v1.json"
DEFAULT_BASELINE_MODEL = ROOT / "outputs" / "feedback_form_classifier" / "model.joblib"
DEFAULT_CORPUS_OUTPUT = ROOT / "data" / "direct_fg_classifier_corpus.v2.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "feedback_form_classifier_candidates" / "direct_fg_v2"
)

PINNED_SHA256 = {
    "hard_v2": "9ccec3cd8c15c1234f4f34cd0e4cee7c668fb84e150c5a818777cc9ae73bf35e",
    "human_train": "7f0973399940ab4b8541f3b27c73802d319084d3be1b8071229c130dc1cda3af",
    "human_augmentation": "309e092b0e3219843668c441aa8db732aaeb1d7fd80cb3339e4667c0f83bccab",
    "boundary": "0d52e35a096b1dc502f47653d9090f206a0ac6af732e5f8fd48cfbd35566e4ca",
}

HUMAN_EXCLUSIONS = {
    "human_feedback_form_495d0340592fedcde8edce83": (
        "30d84a2c661b23b0cdc817cb280c3f445968ee3016a8201db56515eb71a4d68e",
        "broken_action_reason_relation",
    ),
    "human_feedback_form_c7f4da491314fc42181cdefc": (
        "846cd33af78f4a2b3bd264689de5da034ab4d7bbacc8d7107c9199faf68285b2",
        "indirect_request_descriptive_ambiguity",
    ),
    "human_feedback_form_81756fc88f06b277d3436871": (
        "09c385f7e2b1a1f11018842d92a0405803d408c645cd37ae0dd199ddc29a86d9",
        "indirect_request_descriptive_ambiguity",
    ),
}
AUGMENTATION_EXCLUSIONS = {
    "human_feedback_form_aug_ea8e8a20fb8d991ac9ea40f5": (
        "9b01433ed22d1f2d33214cbc04b080855abf2e8b1ccc7979e8cf1e0a51b1fd98",
        "inherits_excluded_broken_parent",
        "human_feedback_form_495d0340592fedcde8edce83",
    )
}

HARD_LABEL_SOURCE = "paper_mapping_contrastive_train_only"
BOUNDARY_LABEL_SOURCE = "direct_speech_act_contrastive_templates_train_only"
HUMAN_LABEL_SOURCE = "human_explicit"
AUGMENTATION_LABEL_SOURCE = "deterministic_label_preserving_transform"

# AI-assisted, manually checked targeted diagnostics.  These are intentionally
# not an independent benchmark and cannot affect model selection or pass/fail.
NATURAL_PROBES = {
    "evaluative": (
        "That move was very good.",
        "You did a great job just then.",
        "That was the wrong thing to do.",
        "Thanks, that really helped.",
        "Your previous move slowed us down.",
        "You should have brought the plate sooner.",
        "Why did you leave the soup there?",
        "That handoff was excellent.",
        "I did not like that decision.",
        "The last delivery was a mistake.",
    ),
    "imperative": (
        "Could you grab an onion?",
        "Why don't you chop the onion?",
        "Please move the plate to the counter.",
        "Would you mind watching the pot?",
        "Get a clean dish next.",
        "Do not block the middle lane.",
        "I need you to serve the soup.",
        "How about taking the lower route?",
        "You should fetch another tomato.",
        "Let's clear this counter.",
    ),
    "descriptive": (
        "The left path is blocked.",
        "The soup is ready to serve.",
        "There is one onion beside the pot.",
        "The next order needs three tomatoes.",
        "The upper chopping board is free.",
        "A clean plate is on the table.",
        "The side corridor is the quickest route.",
        "We have ten seconds remaining.",
        "Orders leave through the service hatch.",
        "Carrying a plate uses one hand.",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON list of objects: {path}")
    return [dict(row) for row in value]


def _verify_pinned(path: Path, key: str) -> str:
    actual = _sha256(path)
    expected = PINNED_SHA256[key]
    if actual != expected:
        raise ValueError(f"{key} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _explicit_target(raw: dict, *, source: str, index: int) -> tuple[str, str]:
    """Resolve an explicit three-class target without reading reference_type."""

    expected = raw.get("expected_feedback_type")
    canonical = raw.get("classification_label")
    canonical_target = None
    if isinstance(canonical, str):
        canonical_target = {
            value: key for key, value in CANONICAL_FEEDBACK_LABELS.items()
        }.get(canonical)
    if expected in FEEDBACK_TYPES:
        if canonical is not None and canonical_target != expected:
            raise ValueError(
                f"{source} row {index}: expected/canonical direct labels disagree"
            )
        return str(expected), "expected_feedback_type"
    if raw.get("label_source") == HUMAN_LABEL_SOURCE and canonical_target:
        return canonical_target, "classification_label"
    raise ValueError(f"{source} row {index}: no admissible explicit direct-fG label")


def _natural_probe_rows() -> list[dict]:
    return [
        {"text": text, "label": label}
        for label in FEEDBACK_TYPES
        for text in NATURAL_PROBES[label]
    ]


def _load_secondary_rows(path: Path, *, required_split: str) -> list[dict]:
    rows = []
    for index, raw in enumerate(_read_json_list(path), start=1):
        if raw.get("split") != required_split:
            raise ValueError(f"{path.name} row {index}: expected {required_split}")
        label, target_field = _explicit_target(
            raw, source=path.name, index=index
        )
        text = raw.get("text")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"{path.name} row {index}: missing text")
        rows.append(
            {
                "text": text.strip(),
                "label": label,
                "target_field": target_field,
                "reference_type": raw.get("reference_type"),
            }
        )
    return rows


def _evaluate_rows(artifact: dict, rows: list[dict]) -> dict:
    matrix = artifact["vectorizer"].transform([row["text"] for row in rows])
    labels = [row["label"] for row in rows]
    classifier = artifact["classifier"]
    raw = classifier.predict_proba(matrix)
    calibrated = shared.temperature_scaled_probabilities(
        classifier,
        matrix,
        float(artifact["probability_calibration"]["temperature"]),
    )
    return {
        "classification": shared.classification_metrics(
            classifier, matrix, labels
        ),
        "raw_probability": shared.probability_metrics(
            raw, labels, classifier.classes_
        ),
        "calibrated_probability": shared.probability_metrics(
            calibrated, labels, classifier.classes_
        ),
    }


def _baseline_probe_report(path: Path, probes: list[dict]) -> dict:
    if not path.exists():
        return {"available": False, "path": str(path)}
    from joblib import load

    artifact = load(path)
    matrix = artifact["vectorizer"].transform([row["text"] for row in probes])
    return {
        "available": True,
        "path": str(path),
        "sha256": _sha256(path),
        "classification": shared.classification_metrics(
            artifact["classifier"], matrix, [row["label"] for row in probes]
        ),
    }


def _requested_phrase_report(artifact: dict, text: str) -> dict:
    classifier = artifact["classifier"]
    matrix = artifact["vectorizer"].transform([text])
    raw = classifier.predict_proba(matrix)[0]
    calibrated = shared.temperature_scaled_probabilities(
        classifier,
        matrix,
        float(artifact["probability_calibration"]["temperature"]),
    )[0]
    raw_values = {
        str(label): float(value) for label, value in zip(classifier.classes_, raw)
    }
    calibrated_values = {
        str(label): float(value)
        for label, value in zip(classifier.classes_, calibrated)
    }
    return {
        "text": text,
        "raw_probabilities": raw_values,
        "raw_prediction": max(raw_values, key=raw_values.get),
        "calibrated_probabilities": calibrated_values,
        "calibrated_prediction": max(calibrated_values, key=calibrated_values.get),
        "confidence_interpretation": "model score, not measured correctness",
    }


_LABELED_FAMILY = re.compile(
    r"^(?P<prefix>.*?):(?:evaluative|imperative|descriptive):(?P<surface>.*)$"
)


def _surface_family(template_family: str, namespace: str) -> str:
    match = _LABELED_FAMILY.fullmatch(template_family)
    if not match:
        raise ValueError(f"invalid labeled template_family: {template_family!r}")
    return f"{namespace}:{match.group('surface')}"


def _row(
    raw: dict,
    *,
    index: int,
    source: str,
    family: str,
    group: str,
    fixed_train: bool,
) -> dict:
    label, target_field = _explicit_target(raw, source=source, index=index)
    text = raw.get("text")
    if not isinstance(text, str) or not normalize_text(text):
        raise ValueError(f"{source} row {index}: missing text")
    return {
        "text": text.strip(),
        "normalized_text": normalize_text(text),
        "label": label,
        "classification_label": CANONICAL_FEEDBACK_LABELS[label],
        "split": "train" if fixed_train else None,
        "source": source,
        "label_source": str(raw.get("label_source")),
        "target_field": target_field,
        "group_id": group,
        "template_family": str(raw.get("template_family") or ""),
        "split_family": family,
        "fixed_train": fixed_train,
        "reference_type": raw.get("reference_type"),
    }


def _load_hard(path: Path) -> tuple[list[dict], dict]:
    _verify_pinned(path, "hard_v2")
    raw_rows = _read_json_list(path)
    if len(raw_rows) != 288:
        raise ValueError(f"hard v2 must contain 288 rows, got {len(raw_rows)}")
    rows = []
    for index, raw in enumerate(raw_rows, start=1):
        if (
            raw.get("split") != "train"
            or raw.get("source") != "feedback_form_hard_train"
            or raw.get("label_source") != HARD_LABEL_SOURCE
        ):
            raise ValueError(f"hard v2 row {index}: provenance mismatch")
        family_raw = str(raw.get("template_family") or "")
        group_raw = str(raw.get("group_id") or "")
        if not family_raw.startswith("feedback-form-hard-train-v2:"):
            raise ValueError(f"hard v2 row {index}: family version mismatch")
        if not group_raw.startswith("feedback-form-hard-train-v2:"):
            raise ValueError(f"hard v2 row {index}: group version mismatch")
        rows.append(
            _row(
                raw,
                index=index,
                source="hard_v2_direct_explicit",
                family=_surface_family(family_raw, "hard"),
                group=f"hard:{group_raw.split(':', 1)[1]}",
                fixed_train=False,
            )
        )
    return rows, {"input_rows": 288, "accepted_rows": 288, "excluded_rows": 0}


def _load_human(path: Path) -> tuple[list[dict], dict]:
    _verify_pinned(path, "human_train")
    raw_rows = _read_json_list(path)
    if len(raw_rows) != 44:
        raise ValueError(f"human train must contain 44 rows, got {len(raw_rows)}")
    rows = []
    excluded = {}
    for index, raw in enumerate(raw_rows, start=1):
        if (
            raw.get("split") != "train"
            or raw.get("source") != "human"
            or raw.get("label_source") != HUMAN_LABEL_SOURCE
        ):
            raise ValueError(f"human row {index}: provenance mismatch")
        feedback_id = str(raw.get("feedback_id") or "")
        if feedback_id in HUMAN_EXCLUSIONS:
            expected_hash, reason = HUMAN_EXCLUSIONS[feedback_id]
            actual_hash = _digest(normalize_text(str(raw.get("text") or "")))
            if actual_hash != expected_hash:
                raise ValueError(f"human exclusion {feedback_id}: text hash changed")
            excluded[feedback_id] = reason
            continue
        family = str(raw.get("template_family") or "")
        if not feedback_id or not family:
            raise ValueError(f"human row {index}: missing id/family")
        rows.append(
            _row(
                raw,
                index=index,
                source="human_explicit_reviewed",
                family=f"human:{family}",
                group=f"human:{feedback_id}",
                fixed_train=True,
            )
        )
    if set(excluded) != set(HUMAN_EXCLUSIONS) or len(rows) != 41:
        raise ValueError("human quality exclusions did not resolve to exactly 41 rows")
    return rows, {
        "input_rows": 44,
        "accepted_rows": 41,
        "excluded_rows": 3,
        "exclusions": excluded,
    }


def _load_augmentation(path: Path) -> tuple[list[dict], dict]:
    _verify_pinned(path, "human_augmentation")
    raw_rows = _read_json_list(path)
    if len(raw_rows) != 85:
        raise ValueError(f"augmentation must contain 85 rows, got {len(raw_rows)}")
    rows = []
    excluded = {}
    for index, raw in enumerate(raw_rows, start=1):
        if (
            raw.get("split") != "train"
            or raw.get("source") != "human_train_augmentation"
            or raw.get("label_source") != AUGMENTATION_LABEL_SOURCE
        ):
            raise ValueError(f"augmentation row {index}: provenance mismatch")
        feedback_id = str(raw.get("feedback_id") or "")
        parent = str(raw.get("parent_feedback_id") or "")
        if feedback_id in AUGMENTATION_EXCLUSIONS:
            expected_hash, reason, expected_parent = AUGMENTATION_EXCLUSIONS[feedback_id]
            actual_hash = _digest(normalize_text(str(raw.get("text") or "")))
            if actual_hash != expected_hash or parent != expected_parent:
                raise ValueError(f"augmentation exclusion {feedback_id}: audit changed")
            excluded[feedback_id] = reason
            continue
        family = str(raw.get("template_family") or "")
        if not feedback_id or not parent or not family:
            raise ValueError(f"augmentation row {index}: missing id/parent/family")
        rows.append(
            _row(
                raw,
                index=index,
                source="human_augmentation_reviewed",
                family=f"human:{family}",
                group=f"human:{parent}",
                fixed_train=True,
            )
        )
    if set(excluded) != set(AUGMENTATION_EXCLUSIONS) or len(rows) != 84:
        raise ValueError("augmentation quality exclusion did not resolve to 84 rows")
    return rows, {
        "input_rows": 85,
        "accepted_rows": 84,
        "excluded_rows": 1,
        "exclusions": excluded,
    }


def _load_boundary(path: Path, occupied: set[str]) -> tuple[list[dict], dict]:
    _verify_pinned(path, "boundary")
    raw_rows = _read_json_list(path)
    if len(raw_rows) != 648:
        raise ValueError(f"boundary must contain 648 rows, got {len(raw_rows)}")
    rows = []
    overlap_rows = []
    prefix_rows = []
    for index, raw in enumerate(raw_rows, start=1):
        normalized = normalize_text(str(raw.get("text") or ""))
        family_raw = str(raw.get("template_family") or "")
        if family_raw.startswith("feedback-form-hard-train-v1:"):
            prefix_rows.append(normalized)
        if normalized in occupied:
            overlap_rows.append(normalized)
            continue
        if (
            raw.get("split") != "train"
            or raw.get("source") != "feedback_form_hard_train"
            or raw.get("label_source") != BOUNDARY_LABEL_SOURCE
        ):
            raise ValueError(f"boundary retained row {index}: provenance mismatch")
        group_raw = str(raw.get("group_id") or "")
        if not family_raw.startswith("feedback-form-boundary-train-v1:"):
            raise ValueError(f"boundary row {index}: family version mismatch")
        if not group_raw.startswith("feedback-form-boundary-train-v1:"):
            raise ValueError(f"boundary row {index}: group version mismatch")
        rows.append(
            _row(
                raw,
                index=index,
                source="boundary_direct_explicit",
                family=_surface_family(family_raw, "boundary"),
                group=f"boundary:{group_raw.split(':', 1)[1]}",
                fixed_train=False,
            )
        )
    if len(rows) != 504 or len(overlap_rows) != 144:
        raise ValueError("boundary de-duplication did not resolve to 504 + 144")
    if set(overlap_rows) != set(prefix_rows) or len(prefix_rows) != 144:
        raise ValueError("boundary exact de-dup no longer matches hard-v1 provenance")
    return rows, {
        "input_rows": 648,
        "accepted_rows": 504,
        "excluded_exact_core_overlap": 144,
        "overlap_equals_hard_v1_prefix_rows": True,
    }


def _near_values(left: str, right: str) -> tuple[bool, float, float, float]:
    length_ratio = min(len(left), len(right)) / max(len(left), len(right))
    if length_ratio < NEAR_LENGTH_RATIO_THRESHOLD:
        return False, 0.0, 0.0, length_ratio
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    token_jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    if token_jaccard < NEAR_TOKEN_JACCARD_THRESHOLD:
        return False, 0.0, token_jaccard, length_ratio
    sequence_ratio = SequenceMatcher(
        None, left, right, autojunk=False
    ).ratio()
    return (
        sequence_ratio >= NEAR_SEQUENCE_THRESHOLD,
        sequence_ratio,
        token_jaccard,
        length_ratio,
    )


class _UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _component_category(families: set[str]) -> str:
    categories = set()
    for family in families:
        if family.startswith("hard:"):
            categories.add("hard")
        elif family.startswith("boundary:short_"):
            categories.add("short")
        elif family.startswith("boundary:contrast_"):
            categories.add("contrast")
        elif family.startswith("human:"):
            categories.add("fixed_human")
        else:
            raise ValueError(f"unknown split family: {family}")
    if len(categories) != 1:
        raise ValueError(f"near component crosses source categories: {categories}")
    return next(iter(categories))


def _choose_component_subset(components: list[dict], target_rows: int) -> set[str]:
    ordered = sorted(components, key=lambda item: item["component_id"])
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for index, component in enumerate(ordered):
        for total, selected in list(choices.items()):
            next_total = total + component["rows"]
            next_selected = selected + (index,)
            previous = choices.get(next_total)
            next_key = tuple(ordered[item]["component_id"] for item in next_selected)
            previous_key = (
                tuple(ordered[item]["component_id"] for item in previous)
                if previous is not None
                else None
            )
            if previous is None or next_key < previous_key:
                choices[next_total] = next_selected
    total, selected = min(
        choices.items(),
        key=lambda item: (
            abs(item[0] - target_rows),
            item[0] == 0,
            tuple(ordered[index]["component_id"] for index in item[1]),
        ),
    )
    if total == 0:
        raise ValueError("component selection produced an empty development split")
    return {
        family
        for index in selected
        for family in ordered[index]["families"]
    }


def _assign_component_split(rows: list[dict]) -> tuple[list[dict], dict]:
    families = {row["split_family"] for row in rows}
    union_find = _UnionFind(families)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split_family"])
    group_edges = 0
    for group_families in groups.values():
        ordered = sorted(group_families)
        for family in ordered[1:]:
            union_find.union(ordered[0], family)
            group_edges += 1

    near_edges = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["split_family"] == right["split_family"]:
                continue
            is_near, sequence_ratio, token_jaccard, length_ratio = _near_values(
                left["normalized_text"], right["normalized_text"]
            )
            if not is_near:
                continue
            union_find.union(left["split_family"], right["split_family"])
            near_edges.append(
                {
                    "left_text_sha256": _digest(left["normalized_text"]),
                    "right_text_sha256": _digest(right["normalized_text"]),
                    "sequence_ratio": sequence_ratio,
                    "token_jaccard": token_jaccard,
                    "length_ratio": length_ratio,
                }
            )

    component_families: dict[str, set[str]] = defaultdict(set)
    for family in families:
        component_families[union_find.find(family)].add(family)
    family_row_counts = Counter(row["split_family"] for row in rows)
    fixed_families = {
        row["split_family"] for row in rows if row["fixed_train"]
    }
    components = []
    for family_set in component_families.values():
        component_id = _digest("|".join(sorted(family_set)))
        components.append(
            {
                "component_id": component_id,
                "families": sorted(family_set),
                "rows": sum(family_row_counts[family] for family in family_set),
                "category": _component_category(family_set),
                "fixed_train": bool(family_set & fixed_families),
            }
        )

    dev_families: set[str] = set()
    category_report = {}
    for category in ("hard", "contrast", "short"):
        candidates = [
            component
            for component in components
            if component["category"] == category and not component["fixed_train"]
        ]
        total_rows = sum(component["rows"] for component in candidates)
        target_rows = round(total_rows * DEV_COMPONENT_FRACTION)
        selected_families = _choose_component_subset(candidates, target_rows)
        dev_families.update(selected_families)
        category_report[category] = {
            "eligible_rows": total_rows,
            "target_dev_rows": target_rows,
            "selected_dev_rows": sum(
                family_row_counts[family] for family in selected_families
            ),
            "selected_family_sha256": sorted(
                _digest(family) for family in selected_families
            ),
        }

    for row in rows:
        row["split"] = "dev" if row["split_family"] in dev_families else "train"
    cross_split_near = 0
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    for train_row in train_rows:
        for dev_row in dev_rows:
            if _near_values(
                train_row["normalized_text"], dev_row["normalized_text"]
            )[0]:
                cross_split_near += 1
    if cross_split_near:
        raise ValueError(f"near-component split leaked {cross_split_near} pairs")
    return rows, {
        "method": "group_and_near_duplicate_family_connected_components",
        "component_count": len(components),
        "fixed_train_component_count": sum(
            component["fixed_train"] for component in components
        ),
        "near_edge_count": len(near_edges),
        "group_edge_count": group_edges,
        "near_thresholds": {
            "sequence_matcher": NEAR_SEQUENCE_THRESHOLD,
            "token_jaccard": NEAR_TOKEN_JACCARD_THRESHOLD,
            "length_ratio": NEAR_LENGTH_RATIO_THRESHOLD,
        },
        "near_edge_hashes": near_edges,
        "cross_split_near_overlap": 0,
        "categories": category_report,
    }


def _integrity_audit(rows: list[dict]) -> dict:
    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_text[row["normalized_text"]].append(row)
    duplicates = sum(len(group) - 1 for group in by_text.values())
    conflicts = sum(
        len({row["label"] for row in group}) > 1 for group in by_text.values()
    )
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    text_overlap = len(
        {row["normalized_text"] for row in train}
        & {row["normalized_text"] for row in dev}
    )
    group_overlap = len(
        {row["group_id"] for row in train} & {row["group_id"] for row in dev}
    )
    family_overlap = len(
        {row["split_family"] for row in train}
        & {row["split_family"] for row in dev}
    )
    if any((duplicates, conflicts, text_overlap, group_overlap, family_overlap)):
        raise ValueError(
            "v2 integrity failed: "
            f"duplicates={duplicates}, conflicts={conflicts}, text={text_overlap}, "
            f"group={group_overlap}, family={family_overlap}"
        )
    return {
        "normalized_duplicate_rows": 0,
        "cross_label_text_conflicts": 0,
        "train_dev_text_overlap": 0,
        "train_dev_group_overlap": 0,
        "train_dev_family_overlap": 0,
    }


def build_direct_fg_v2_corpus(
    hard_path: Path = DEFAULT_HARD_V2,
    human_path: Path = DEFAULT_HUMAN_TRAIN,
    augmentation_path: Path = DEFAULT_HUMAN_AUGMENTATION,
    boundary_path: Path = DEFAULT_BOUNDARY,
) -> tuple[list[dict], dict]:
    hard, hard_audit = _load_hard(hard_path)
    human, human_audit = _load_human(human_path)
    augmentation, augmentation_audit = _load_augmentation(augmentation_path)
    core = [*hard, *human, *augmentation]
    boundary, boundary_audit = _load_boundary(
        boundary_path, {row["normalized_text"] for row in core}
    )
    rows, component_audit = _assign_component_split([*core, *boundary])
    integrity = _integrity_audit(rows)
    if len(rows) != 917:
        raise ValueError(f"v2 MVP must contain exactly 917 rows, got {len(rows)}")
    split_counts = Counter(row["split"] for row in rows)
    label_counts = {
        split: Counter(row["label"] for row in rows if row["split"] == split)
        for split in ("train", "dev")
    }
    if set(label_counts["train"]) != set(FEEDBACK_TYPES) or set(
        label_counts["dev"]
    ) != set(FEEDBACK_TYPES):
        raise ValueError("all three direct-fG labels are required in train/dev")
    if min(label_counts["dev"].values()) < 30:
        raise ValueError("direct synthetic dev requires at least 30 rows per class")
    return rows, {
        "version": VERSION,
        "target": "direct_fG_surface_speech_act",
        "rows": len(rows),
        "split_rows": dict(sorted(split_counts.items())),
        "split_label_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in label_counts.items()
        },
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "input_audits": {
            "hard_v2": hard_audit,
            "human_train": human_audit,
            "human_augmentation": augmentation_audit,
            "boundary": boundary_audit,
        },
        "component_split": component_audit,
        "split_integrity": integrity,
        "reference_type_policy": {
            "used_as_target": False,
            "mapping_applied": False,
            "retained_as_provenance_only": True,
        },
        "data_scope": {
            "deepseek_rows": 0,
            "mixed_test_containers_opened": 0,
            "human_test_files_opened": 0,
            "frozen_files_opened": 0,
        },
    }


def _metric_bundle(classifier, matrix, labels: list[str]) -> dict:
    return {
        "classification": shared.classification_metrics(classifier, matrix, labels),
        "raw_probability": shared.probability_metrics(
            classifier.predict_proba(matrix), labels, classifier.classes_
        ),
    }


def train_direct_fg_v2_candidate(
    rows: list[dict], *, seed: int = SEED, min_df: int = MIN_DF
) -> tuple[dict, dict]:
    import joblib
    import numpy
    import sklearn
    from sklearn.linear_model import LogisticRegression

    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    vectorizer = shared._build_vectorizer(min_df)
    train_matrix = vectorizer.fit_transform([row["text"] for row in train_rows])
    dev_matrix = vectorizer.transform([row["text"] for row in dev_rows])
    train_labels = [row["label"] for row in train_rows]
    dev_labels = [row["label"] for row in dev_rows]
    candidates = []
    for c_value in C_GRID:
        for class_weight in CLASS_WEIGHT_GRID:
            classifier = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=3000,
                random_state=seed,
            )
            classifier.fit(train_matrix, train_labels)
            candidates.append(
                {
                    "config": {"C": c_value, "class_weight": class_weight},
                    "classifier": classifier,
                    "metrics": _metric_bundle(classifier, dev_matrix, dev_labels),
                }
            )

    def selection_key(candidate: dict) -> tuple:
        metrics = candidate["metrics"]
        classification = metrics["classification"]
        recalls = {
            label: classification["per_class"][label]["recall"]
            for label in FEEDBACK_TYPES
        }
        return (
            recalls["evaluative"] >= 0.85,
            min(recalls.values()) >= 0.80,
            classification["macro_f1"],
            classification["balanced_accuracy"],
            classification["accuracy"],
            -metrics["raw_probability"]["negative_log_likelihood"],
            -float(candidate["config"]["C"]),
            candidate["config"]["class_weight"] is None,
        )

    selected = max(candidates, key=selection_key)
    classifier = selected["classifier"]
    temperature, calibration = shared.fit_temperature(
        classifier, dev_matrix, dev_labels
    )
    environment = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    }
    script_path = Path(__file__).resolve()
    artifact = {
        "model_type": "feedback_form_tfidf_logistic_regression",
        "model_version": 4,
        "candidate_version": VERSION,
        "candidate_only": True,
        "promotion_status": "not_promoted",
        "target": "direct_fG_surface_speech_act",
        "input_mode": "raw_text",
        "labels": list(FEEDBACK_TYPES),
        "canonical_labels": dict(CANONICAL_FEEDBACK_LABELS),
        "vectorizer": vectorizer,
        "classifier": classifier,
        "selected_hyperparameters": selected["config"],
        "minimum_model_confidence": DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
        "probability_calibration": {
            "method": "temperature_scaling",
            "temperature": temperature,
            "version": "direct-fg-v2-temperature-selection-dev-v1",
            "fit_split": "direct_fg_near_component_disjoint_dev",
            "fit_split_used_for_model_selection": True,
            "independently_validated": False,
        },
        "reference_type_mapping": None,
        "reference_type_used_as_target": False,
        "reproducibility": {
            "seed": seed,
            "min_df": min_df,
            "script_sha256": _sha256(script_path),
            "environment": environment,
        },
    }
    return artifact, {
        "version": VERSION,
        "target": "direct_fG_surface_speech_act",
        "candidate_only": True,
        "production_artifact_overwritten": False,
        "selected_hyperparameters": selected["config"],
        "temperature": temperature,
        "reproducibility": artifact["reproducibility"],
        "selection_protocol": {
            "primary": "direct_fG_near_component_disjoint_dev",
            "selection_order": [
                "evaluative_recall_at_least_0_85",
                "all_class_recall_at_least_0_80",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "raw_negative_log_likelihood",
            ],
            "natural_probe_used_for_selection": False,
            "natural_probe_used_for_pass_gate": False,
            "secondary_dev_used_for_selection": False,
            "test_or_frozen_rows_used": 0,
        },
        "direct_fg_near_component_disjoint_dev": {
            **selected["metrics"],
            "calibration": calibration,
        },
        "candidate_grid": [
            {"config": item["config"], "metrics": item["metrics"]}
            for item in candidates
        ],
        "confidence_limit": (
            "temperature is fitted and measured on selection dev; values are model "
            "scores, not independently measured correctness"
        ),
        "reference_type_policy": {
            "mapping": None,
            "used_as_target": False,
            "five_class_role": "provenance_or_downstream_routing_only",
        },
        "test_and_frozen_access": {
            "mixed_test_containers_opened": False,
            "human_test_files_opened": False,
            "frozen_files_opened": False,
            "metrics_computed": False,
        },
        "test_metrics": None,
    }


def compute_candidate_status(required_gates: dict[str, bool]) -> str:
    if not required_gates or not all(required_gates.values()):
        return "failed_research_candidate_not_promoted"
    return "passed_research_candidate_not_promoted"


def run_pipeline(
    hard_path: Path,
    human_path: Path,
    augmentation_path: Path,
    boundary_path: Path,
    human_dev_path: Path,
    paper_proxy_path: Path,
    baseline_model_path: Path,
    corpus_output: Path,
    output_dir: Path,
) -> dict:
    from joblib import dump

    rows, corpus_audit = build_direct_fg_v2_corpus(
        hard_path, human_path, augmentation_path, boundary_path
    )
    write_json(corpus_output, rows)
    corpus_audit["output"] = str(corpus_output)
    corpus_audit["output_sha256"] = _sha256(corpus_output)
    artifact, report = train_direct_fg_v2_candidate(rows)
    report["corpus_audit"] = corpus_audit

    probes = _natural_probe_rows()
    if {normalize_text(row["text"]) for row in probes} & {
        row["normalized_text"] for row in rows
    }:
        raise ValueError("targeted diagnostic probe has exact corpus overlap")
    human_dev = _load_secondary_rows(human_dev_path, required_split="dev")
    paper_proxy = _load_secondary_rows(paper_proxy_path, required_split="dev")
    report["secondary_evaluations"] = {
        "human_explicit_dev": {
            "selection_role": "none",
            "pass_gate_role": "none",
            "metrics": _evaluate_rows(artifact, human_dev),
        },
        "paper_reference_collapse_proxy": {
            "selection_role": "none",
            "pass_gate_role": "none",
            "target_semantics_match_direct_fG": False,
            "metrics": _evaluate_rows(artifact, paper_proxy),
        },
        "natural_probe": {
            "provenance": "ai_assisted_targeted_diagnostic",
            "independent_benchmark": False,
            "selection_role": "none",
            "pass_gate_role": "none",
            "candidate": _evaluate_rows(artifact, probes),
            "production_baseline": _baseline_probe_report(
                baseline_model_path, probes
            ),
        },
    }
    report["requested_phrase"] = _requested_phrase_report(
        artifact, "That move was very good."
    )
    primary = report["direct_fg_near_component_disjoint_dev"]["classification"]
    access_audit = report["test_and_frozen_access"]
    required_gates = {
        "row_count_is_917": corpus_audit["rows"] == 917,
        "exact_group_family_integrity": all(
            value == 0 for value in corpus_audit["split_integrity"].values()
        ),
        "near_overlap_is_zero": (
            corpus_audit["component_split"]["cross_split_near_overlap"] == 0
        ),
        "accuracy_at_least_0_87": primary["accuracy"] >= 0.87,
        "macro_f1_at_least_0_87": primary["macro_f1"] >= 0.87,
        "evaluative_recall_at_least_0_85": (
            primary["per_class"]["evaluative"]["recall"] >= 0.85
        ),
        "minimum_class_recall_at_least_0_80": min(
            primary["per_class"][label]["recall"] for label in FEEDBACK_TYPES
        )
        >= 0.80,
        "reference_type_not_used_as_target": not corpus_audit[
            "reference_type_policy"
        ]["used_as_target"],
        "no_test_or_frozen_access": not any(
            access_audit[key]
            for key in (
                "mixed_test_containers_opened",
                "human_test_files_opened",
                "frozen_files_opened",
                "metrics_computed",
            )
        ),
    }
    report["candidate_quality_gate"] = {
        "required": required_gates,
        "diagnostics_excluded_from_gate": [
            "natural_probe",
            "human_explicit_dev",
            "paper_reference_collapse_proxy",
        ],
        "status": compute_candidate_status(required_gates),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    report_path = output_dir / "model.report.json"
    manifest_path = output_dir / "manifest.json"
    dump(artifact, model_path)
    write_json(report_path, report)
    manifest = {
        "version": VERSION,
        "candidate_only": True,
        "promotion_status": "not_promoted",
        "quality_status": report["candidate_quality_gate"]["status"],
        "production_artifact_overwritten": False,
        "target": "direct_fG_surface_speech_act",
        "inputs": {
            "hard_v2": {"path": str(hard_path), "sha256": _sha256(hard_path)},
            "human_train": {"path": str(human_path), "sha256": _sha256(human_path)},
            "human_augmentation": {
                "path": str(augmentation_path),
                "sha256": _sha256(augmentation_path),
            },
            "boundary": {
                "path": str(boundary_path),
                "sha256": _sha256(boundary_path),
            },
            "human_dev_secondary": {
                "path": str(human_dev_path),
                "sha256": _sha256(human_dev_path),
            },
            "paper_proxy_secondary": {
                "path": str(paper_proxy_path),
                "sha256": _sha256(paper_proxy_path),
            },
        },
        "outputs": {
            "corpus": {"path": str(corpus_output), "sha256": _sha256(corpus_output)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        },
        "selected_hyperparameters": report["selected_hyperparameters"],
        "temperature": report["temperature"],
        "reproducibility": report["reproducibility"],
        "required_quality_gates": required_gates,
        "diagnostics_excluded_from_gate": report["candidate_quality_gate"][
            "diagnostics_excluded_from_gate"
        ],
        "test_and_frozen_access": report["test_and_frozen_access"],
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-v2", type=Path, default=DEFAULT_HARD_V2)
    parser.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN)
    parser.add_argument(
        "--human-augmentation", type=Path, default=DEFAULT_HUMAN_AUGMENTATION
    )
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--human-dev", type=Path, default=DEFAULT_HUMAN_DEV)
    parser.add_argument("--paper-proxy", type=Path, default=DEFAULT_PAPER_PROXY)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_pipeline(
        args.hard_v2,
        args.human_train,
        args.human_augmentation,
        args.boundary,
        args.human_dev,
        args.paper_proxy,
        args.baseline_model,
        args.corpus_output,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
