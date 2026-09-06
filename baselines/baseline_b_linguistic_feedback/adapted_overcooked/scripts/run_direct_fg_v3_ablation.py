"""Run in-memory A/B diagnostics for a future clean direct-fG v3.

No artifact or corpus is written.  Synthetic rows are regenerated from Python
template constants without invoking any holdout, membership, dev, test, or
frozen-data gate.  This diagnostic is AI-assisted and prior-error-informed; it
is not an independently human-audited benchmark.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_feedback_form_boundary_train as boundary_gen  # noqa: E402
from scripts import generate_feedback_form_hard_train as hard_gen  # noqa: E402
from scripts import train_direct_fg_classifier_candidate_v2 as v2  # noqa: E402
from scripts import train_feedback_form_classifier as shared_trainer  # noqa: E402
from src import evaluation_splits  # noqa: E402
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feedback_form_classifier import CANONICAL_FEEDBACK_LABELS, FEEDBACK_TYPES  # noqa: E402


VERSION = "direct-fg-v3-old-template-in-memory-ablation-v1"
SEQUENCE_THRESHOLD = 0.88
TOKEN_JACCARD_THRESHOLD = 0.70
SYNTHETIC_LABEL_SOURCE = "direct_surface_template_contract"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _synthetic_row(
    raw: dict, *, source: str, family: str, group: str
) -> dict:
    label = str(raw["expected_feedback_type"])
    if label not in FEEDBACK_TYPES:
        raise ValueError(f"invalid direct label: {label}")
    if raw.get("classification_label") != CANONICAL_FEEDBACK_LABELS[label]:
        raise ValueError("canonical label mismatch")
    text = str(raw["text"]).strip()
    return {
        "text": text,
        "normalized_text": normalize_text(text),
        "label": label,
        "classification_label": CANONICAL_FEEDBACK_LABELS[label],
        "split": None,
        "source": source,
        "label_source": SYNTHETIC_LABEL_SOURCE,
        "target_field": "expected_feedback_type",
        "group_id": group,
        "template_family": str(raw["template_family"]),
        "split_family": family,
        "fixed_train": False,
        "reference_type": None,
    }


def generate_pure_synthetic() -> tuple[list[dict], dict]:
    hard_raw = hard_gen.generate_rows()
    boundary_raw = []
    for family_index in range(len(boundary_gen.CONTRAST_TEMPLATES["evaluative"])):
        for scenario_index, scenario in enumerate(boundary_gen.SCENARIOS):
            values = {
                **scenario,
                "base_cap": scenario["base"][:1].upper() + scenario["base"][1:],
                "gerund_cap": scenario["gerund"][:1].upper()
                + scenario["gerund"][1:],
            }
            group_id = (
                f"{boundary_gen.VERSION}:contrast_"
                f"{family_index:02d}_{scenario_index:02d}"
            )
            for label in FEEDBACK_TYPES:
                boundary_raw.append(
                    boundary_gen._make_row(
                        boundary_gen.CONTRAST_TEMPLATES[label][family_index].format(
                            **values
                        ),
                        label,
                        group_id,
                        f"{boundary_gen.VERSION}:{label}:contrast_{family_index:02d}",
                    )
                )
    for index in range(len(boundary_gen.SHORT_UTTERANCES["evaluative"])):
        for label in FEEDBACK_TYPES:
            boundary_raw.append(
                boundary_gen._make_row(
                    boundary_gen.SHORT_UTTERANCES[label][index],
                    label,
                    f"{boundary_gen.VERSION}:short_{index:02d}",
                    f"{boundary_gen.VERSION}:{label}:short_{index:02d}",
                )
            )
    if len(hard_raw) != 288 or len(boundary_raw) != 504:
        raise ValueError("pure generators did not resolve to 288 + 504")

    hard = [
        _synthetic_row(
            raw,
            source="direct_fg_hard_template_v3_ablation",
            family=v2._surface_family(str(raw["template_family"]), "hard"),
            group=f"hard:{str(raw['group_id']).split(':', 1)[1]}",
        )
        for raw in hard_raw
    ]
    boundary = [
        _synthetic_row(
            raw,
            source="direct_fg_boundary_template_v3_ablation",
            family=v2._surface_family(str(raw["template_family"]), "boundary"),
            group=f"boundary:{str(raw['group_id']).split(':', 1)[1]}",
        )
        for raw in boundary_raw
    ]
    rows = [*hard, *boundary]
    if len({row["normalized_text"] for row in rows}) != len(rows):
        raise ValueError("pure synthetic rows contain exact duplicates")
    return rows, {
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "hard_rows": len(hard),
        "boundary_rows": len(boundary),
        "external_inputs_opened": 0,
        "dev_test_frozen_membership_opened": False,
        "generation": "pure_function_template_expansion",
        "ai_assisted": True,
        "prior_error_informed": True,
        "independent_human_semantic_audit": False,
        "ontology_warning": "old template ontology; not final clean-v3 data",
    }


class _UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _sequence_ratio(left: str, right: str) -> float:
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    if matcher.real_quick_ratio() < SEQUENCE_THRESHOLD:
        return 0.0
    if matcher.quick_ratio() < SEQUENCE_THRESHOLD:
        return 0.0
    return matcher.ratio()


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def strict_component_split(
    synthetic: list[dict], fixed_train: list[dict]
) -> tuple[set[str], dict]:
    rows = [*synthetic, *fixed_train]
    families = {row["split_family"] for row in rows}
    union_find = _UnionFind(families)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split_family"])
    for group_families in groups.values():
        ordered = sorted(group_families)
        for family in ordered[1:]:
            union_find.union(ordered[0], family)

    sequence_edges = 0
    jaccard_edges = 0
    both_edges = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["split_family"] == right["split_family"]:
                continue
            jaccard = _token_jaccard(
                left["normalized_text"], right["normalized_text"]
            )
            sequence = _sequence_ratio(
                left["normalized_text"], right["normalized_text"]
            )
            sequence_hit = sequence >= SEQUENCE_THRESHOLD
            jaccard_hit = jaccard >= TOKEN_JACCARD_THRESHOLD
            sequence_edges += sequence_hit
            jaccard_edges += jaccard_hit
            both_edges += sequence_hit and jaccard_hit
            if sequence_hit or jaccard_hit:
                union_find.union(left["split_family"], right["split_family"])

    component_families: dict[str, set[str]] = defaultdict(set)
    for family in families:
        component_families[union_find.find(family)].add(family)
    fixed_families = {row["split_family"] for row in fixed_train}
    family_counts = Counter(row["split_family"] for row in synthetic)
    candidates = []
    anchored_synthetic_families = set()
    for family_set in component_families.values():
        synthetic_families = {family for family in family_set if family in family_counts}
        if not synthetic_families:
            continue
        if family_set & fixed_families:
            anchored_synthetic_families.update(synthetic_families)
            continue
        candidates.append(
            {
                "component_id": _digest("|".join(sorted(family_set))),
                "families": sorted(synthetic_families),
                "rows": sum(family_counts[family] for family in synthetic_families),
            }
        )
    eligible_rows = sum(component["rows"] for component in candidates)
    dev_families = v2._choose_component_subset(
        candidates, round(eligible_rows * 0.25)
    )
    return dev_families, {
        "component_count": len(component_families),
        "eligible_component_count": len(candidates),
        "anchored_synthetic_families": len(anchored_synthetic_families),
        "sequence_edges_at_0_88": sequence_edges,
        "jaccard_edges_at_0_70": jaccard_edges,
        "both_edges": both_edges,
        "dev_family_sha256": sorted(_digest(value) for value in dev_families),
    }


def _cross_split_counts(rows: list[dict]) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    sequence = 0
    jaccard = 0
    exact = 0
    for left in train:
        for right in dev:
            exact += left["normalized_text"] == right["normalized_text"]
            sequence += (
                _sequence_ratio(left["normalized_text"], right["normalized_text"])
                >= SEQUENCE_THRESHOLD
            )
            jaccard += (
                _token_jaccard(left["normalized_text"], right["normalized_text"])
                >= TOKEN_JACCARD_THRESHOLD
            )
    return {
        "exact": exact,
        "sequence_matcher_at_least_0_88": sequence,
        "token_jaccard_at_least_0_70": jaccard,
    }


def _run_model(rows: list[dict]) -> dict:
    _artifact, report = v2.train_direct_fg_v2_candidate(rows)
    metrics = report["direct_fg_near_component_disjoint_dev"]["classification"]
    return {
        "selected_hyperparameters": report["selected_hyperparameters"],
        "temperature": report["temperature"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_recall": {
            label: metrics["per_class"][label]["recall"]
            for label in FEEDBACK_TYPES
        },
    }


def _import_hashes() -> dict:
    modules = {
        "hard_generator": hard_gen,
        "boundary_generator": boundary_gen,
        "v2_training_helper": v2,
        "shared_classifier_trainer": shared_trainer,
        "normalization": evaluation_splits,
    }
    values = {
        name: {"path": str(Path(module.__file__).resolve()), "sha256": _sha256(Path(module.__file__).resolve())}
        for name, module in modules.items()
    }
    script_path = Path(__file__).resolve()
    values["ablation_script"] = {
        "path": str(script_path),
        "sha256": _sha256(script_path),
    }
    return values


def run_ablation() -> dict:
    synthetic, generation = generate_pure_synthetic()
    human, human_audit = v2._load_human(v2.DEFAULT_HUMAN_TRAIN)
    augmentation, augmentation_audit = v2._load_augmentation(
        v2.DEFAULT_HUMAN_AUGMENTATION
    )
    fixed_train = [*human, *augmentation]
    dev_families, component_audit = strict_component_split(
        synthetic, fixed_train
    )

    def assign(rows: list[dict]) -> list[dict]:
        assigned = [dict(row) for row in rows]
        for row in assigned:
            row["split"] = (
                "dev" if row["split_family"] in dev_families else "train"
            )
        return assigned

    a_rows = assign(synthetic)
    b_rows = assign([*synthetic, *fixed_train])
    a_cross = _cross_split_counts(a_rows)
    b_cross = _cross_split_counts(b_rows)
    a_gate = (
        a_cross["sequence_matcher_at_least_0_88"] == 0
        and a_cross["token_jaccard_at_least_0_70"] == 0
    )
    b_gate = (
        b_cross["sequence_matcher_at_least_0_88"] == 0
        and b_cross["token_jaccard_at_least_0_70"] == 0
    )
    return {
        "version": VERSION,
        "execution": "in_memory_only_no_artifact_written",
        "generation_provenance": generation,
        "imported_script_hashes": _import_hashes(),
        "component_split": component_audit,
        "ablation_a_pure_synthetic_792": {
            "rows": len(a_rows),
            "split_counts": dict(sorted(Counter(row["split"] for row in a_rows).items())),
            "cross_split_similarity": a_cross,
            "similarity_quality_gate_passed": a_gate,
            "model": _run_model(a_rows),
        },
        "ablation_b_plus_filtered_human_125": {
            "rows": len(b_rows),
            "split_counts": dict(sorted(Counter(row["split"] for row in b_rows).items())),
            "human_audit": human_audit,
            "augmentation_audit": augmentation_audit,
            "independent_human_semantic_audit": False,
            "cross_split_similarity": b_cross,
            "similarity_quality_gate_passed": b_gate,
            "model": _run_model(b_rows),
        },
    }


def main() -> int:
    print(json.dumps(run_ablation(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
