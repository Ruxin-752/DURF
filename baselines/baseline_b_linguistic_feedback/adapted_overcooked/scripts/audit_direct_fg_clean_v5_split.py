"""Audit the clean-v5 five-bank surface split without training a model."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v5 as generator  # noqa: E402


VERSION = "direct-fg-clean-v5-five-bank-split-audit-v1"
SEQUENCE_THRESHOLD = 0.88
JACCARD_THRESHOLD = 0.70
TOP_PAIR_COUNT = 5
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v5"
)
V4_STRESS_REFERENCE = {
    "role": "frozen_action_and_scenario_ood_stress_test_only",
    "used_for_v5_generation_selection_or_gate": False,
    "status": "failed_research_candidate",
    "synthetic_accuracy": 0.9133333333333333,
    "synthetic_macro_f1": 0.9113866370736393,
    "imperative_recall": 0.74,
    "report_sha256": "a4669257f62bb144ce00c9cc416056fa247e990b21fee412001e0c4c91051d49",
    "manifest_sha256": "dac6db5b1cd0feaa4d1f1a71ef9d7bbfdfa604ab92a02642c5238a99e4e05be0",
}


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def sequence_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def similarity_flags(left: str, right: str) -> tuple[bool, bool, bool]:
    exact = left == right
    sequence = sequence_ratio(left, right) >= SEQUENCE_THRESHOLD
    jaccard = token_jaccard(left, right) >= JACCARD_THRESHOLD
    return exact, sequence, jaccard


def _offer_top_pair(
    heap: list[tuple[float, str, dict]], score: float, left: dict, right: dict
) -> None:
    left_hash = _text_hash(str(left["normalized_text"]))
    right_hash = _text_hash(str(right["normalized_text"]))
    tie_break = f"{left_hash}|{right_hash}"
    summary = {
        "score": score,
        "left_text_sha256": left_hash,
        "right_text_sha256": right_hash,
        "left_feedback_id": left["feedback_id"],
        "right_feedback_id": right["feedback_id"],
        "left_bank": left["bank_id"],
        "right_bank": right["bank_id"],
        "left_label": left["expected_feedback_type"],
        "right_label": right["expected_feedback_type"],
    }
    entry = (score, tie_break, summary)
    if len(heap) < TOP_PAIR_COUNT:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def _sequence_ratio_for_top_and_gate(
    left: str,
    right: str,
    top_heap: list[tuple[float, str, dict]],
) -> float | None:
    # SequenceMatcher's quick methods are upper bounds. Once top-k is full, a
    # pair below both the threshold and current top-k floor can be skipped.
    top_floor = top_heap[0][0] if len(top_heap) == TOP_PAIR_COUNT else 0.0
    required = min(SEQUENCE_THRESHOLD, top_floor)
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    if matcher.real_quick_ratio() < required:
        return None
    if matcher.quick_ratio() < required:
        return None
    return matcher.ratio()


def _partition_contract(rows: list[dict]) -> dict:
    result = {}
    expected_scenarios = {
        scenario.scenario_id for scenario in generator.ontology.SCENARIOS
    }
    expected_actions = set(generator.ontology.ALLOWED_ACTIONS)
    for split_role in ("train", "calibration", "final_eval"):
        selected = [row for row in rows if row["bank_split_role"] == split_role]
        labels = Counter(row["expected_feedback_type"] for row in selected)
        scenarios = {row["scenario_id"] for row in selected}
        actions = {row["semantic_payload"]["action"] for row in selected}
        action_label = Counter(
            (row["semantic_payload"]["action"], row["expected_feedback_type"])
            for row in selected
        )
        result[split_role] = {
            "rows": len(selected),
            "banks": sorted({row["bank_id"] for row in selected}),
            "label_counts": dict(sorted(labels.items())),
            "scenario_count": len(scenarios),
            "canonical_action_count": len(actions),
            "action_x_label_cells": len(action_label),
            "all_24_scenarios_present": scenarios == expected_scenarios,
            "all_10_actions_present": actions == expected_actions,
            "all_30_action_x_label_cells_present": len(action_label) == 30,
            "class_balanced": len(set(labels.values())) == 1
            and set(labels) == set(generator.LABELS),
        }
    return result


def _style_diagnostics(rows: list[dict]) -> dict:
    diagnostics = {}
    for bank in generator.BANKS:
        per_label = {}
        for label in generator.LABELS:
            selected = [
                row
                for row in rows
                if row["bank_id"] == bank.bank_id
                and row["expected_feedback_type"] == label
            ]
            token_lengths = [len(row["normalized_text"].split()) for row in selected]
            character_lengths = [len(row["text"]) for row in selected]
            per_label[label] = {
                "rows": len(selected),
                "token_length_min": min(token_lengths),
                "token_length_max": max(token_lengths),
                "token_length_mean": sum(token_lengths) / len(token_lengths),
                "character_length_min": min(character_lengths),
                "character_length_max": max(character_lengths),
                "character_length_mean": sum(character_lengths) / len(character_lengths),
                "initial_uppercase_fraction": sum(
                    bool(row["text"][:1].isupper()) for row in selected
                )
                / len(selected),
                "terminal_punctuation": dict(
                    Counter(row["text"][-1] for row in selected)
                ),
            }
        diagnostics[bank.bank_id] = per_label
    return {
        "included_in_quality_gate": False,
        "role": "length_punctuation_case_diagnostic_only",
        "by_bank_and_label": diagnostics,
    }


def audit_split(rows: list[dict]) -> dict:
    generation = generator.audit_rows(rows)
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["feedback_id"])
    union_find = UnionFind(len(ordered))

    by_surface_bank: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        by_surface_bank[str(row["surface_frame_triad_id"])].append(index)
    for surface_bank, members in by_surface_bank.items():
        labels = Counter(ordered[index]["expected_feedback_type"] for index in members)
        if len(members) != 432 or labels != Counter(
            {label: 144 for label in generator.LABELS}
        ):
            raise ValueError(f"invalid surface bank {surface_bank}: {labels}")
        for index in members[1:]:
            union_find.union(members[0], index)

    cross_bank = Counter(exact=0, sequence=0, jaccard=0, either=0)
    cross_partition = Counter(exact=0, sequence=0, jaccard=0, either=0)
    graph_edges = Counter(exact=0, sequence=0, jaccard=0, either=0)
    sequence_top: list[tuple[float, str, dict]] = []
    jaccard_top: list[tuple[float, str, dict]] = []
    pairwise_bank_counts: dict[str, Counter] = defaultdict(
        lambda: Counter(exact=0, sequence=0, jaccard=0, either=0)
    )
    pairwise_partition_counts: dict[str, Counter] = defaultdict(
        lambda: Counter(exact=0, sequence=0, jaccard=0, either=0)
    )

    for right_index, right in enumerate(ordered):
        right_text = str(right["normalized_text"])
        for left_index in range(right_index):
            left = ordered[left_index]
            left_text = str(left["normalized_text"])
            exact_hit = left_text == right_text
            jaccard = token_jaccard(left_text, right_text)
            jaccard_hit = jaccard >= JACCARD_THRESHOLD

            cross_bank_pair = left["bank_id"] != right["bank_id"]
            if cross_bank_pair:
                _offer_top_pair(jaccard_top, jaccard, left, right)
                sequence = _sequence_ratio_for_top_and_gate(
                    left_text, right_text, sequence_top
                )
                if sequence is not None:
                    _offer_top_pair(sequence_top, sequence, left, right)
                sequence_hit = bool(
                    sequence is not None and sequence >= SEQUENCE_THRESHOLD
                )
            else:
                matcher = SequenceMatcher(None, left_text, right_text, autojunk=False)
                sequence_hit = (
                    matcher.real_quick_ratio() >= SEQUENCE_THRESHOLD
                    and matcher.quick_ratio() >= SEQUENCE_THRESHOLD
                    and matcher.ratio() >= SEQUENCE_THRESHOLD
                )

            if exact_hit or sequence_hit or jaccard_hit:
                union_find.union(left_index, right_index)
                graph_edges["exact"] += int(exact_hit)
                graph_edges["sequence"] += int(sequence_hit)
                graph_edges["jaccard"] += int(jaccard_hit)
                graph_edges["either"] += 1

            if cross_bank_pair:
                pair_key = "__".join(sorted((left["bank_id"], right["bank_id"])))
                for target in (cross_bank, pairwise_bank_counts[pair_key]):
                    target["exact"] += int(exact_hit)
                    target["sequence"] += int(sequence_hit)
                    target["jaccard"] += int(jaccard_hit)
                    target["either"] += int(exact_hit or sequence_hit or jaccard_hit)
                if left["bank_split_role"] != right["bank_split_role"]:
                    partition_key = "__".join(
                        sorted((left["bank_split_role"], right["bank_split_role"]))
                    )
                    for target in (
                        cross_partition,
                        pairwise_partition_counts[partition_key],
                    ):
                        target["exact"] += int(exact_hit)
                        target["sequence"] += int(sequence_hit)
                        target["jaccard"] += int(jaccard_hit)
                        target["either"] += int(
                            exact_hit or sequence_hit or jaccard_hit
                        )

    component_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        component_members[union_find.find(index)].append(index)
    components = []
    for members in component_members.values():
        banks = sorted({str(ordered[index]["bank_id"]) for index in members})
        signature = _canonical_hash(
            sorted(str(ordered[index]["feedback_id"]) for index in members)
        )
        components.append(
            {
                "component_id": f"clean-v5-surface-component:{signature[:20]}",
                "rows": len(members),
                "banks": banks,
                "split_roles": sorted(
                    {str(ordered[index]["bank_split_role"]) for index in members}
                ),
                "label_counts": dict(
                    sorted(
                        Counter(
                            ordered[index]["expected_feedback_type"]
                            for index in members
                        ).items()
                    )
                ),
            }
        )
    components.sort(key=lambda value: value["component_id"])

    partition_contract = _partition_contract(ordered)
    examples = {}
    for bank in generator.BANKS:
        examples[bank.bank_id] = {
            row["expected_feedback_type"]: row["text"]
            for row in ordered
            if row["bank_id"] == bank.bank_id
            and row["scenario_id"] == "get_tomato_dispenser"
            and row["semantic_payload"]["grounding_variant_index"] == 0
        }

    checks = {
        "generation_contract_passed": generation["rows"] == 2160,
        "five_surface_components": len(components) == 5,
        "each_component_contains_one_bank": all(
            len(component["banks"]) == 1 for component in components
        ),
        "cross_bank_exact_zero": cross_bank["exact"] == 0,
        "cross_bank_sequence_zero": cross_bank["sequence"] == 0,
        "cross_bank_jaccard_zero": cross_bank["jaccard"] == 0,
        "cross_partition_exact_zero": cross_partition["exact"] == 0,
        "cross_partition_sequence_zero": cross_partition["sequence"] == 0,
        "cross_partition_jaccard_zero": cross_partition["jaccard"] == 0,
        "all_partitions_balanced_and_cover_ontology": all(
            details["class_balanced"]
            and details["all_24_scenarios_present"]
            and details["all_10_actions_present"]
            and details["all_30_action_x_label_cells_present"]
            for details in partition_contract.values()
        ),
        "metadata_only_baseline_is_chance": generation[
            "metadata_only_baseline_is_chance"
        ],
    }
    quality_gate_passed = all(checks.values())
    return {
        "version": VERSION,
        "status": "passed_split_audit" if quality_gate_passed else "failed_split_audit",
        "quality_gate_passed": quality_gate_passed,
        "checks": checks,
        "generation": generation,
        "split_protocol": {
            "train_banks": [
                bank.bank_id for bank in generator.BANKS if bank.split_role == "train"
            ],
            "calibration_banks": [
                bank.bank_id
                for bank in generator.BANKS
                if bank.split_role == "calibration"
            ],
            "final_eval_banks": [
                bank.bank_id
                for bank in generator.BANKS
                if bank.split_role == "final_eval"
            ],
            "scenario_content_intentionally_shared": True,
            "semantic_payload_intentionally_shared": True,
            "surface_frame_banks_exclusive": True,
            "benchmark_scope": "surface_heldout_synthetic_only",
            "independent_scenario_action_human_or_player_benchmark": False,
            "training_performed": False,
        },
        "partition_contract": partition_contract,
        "component_graph": {
            "edge_rule": "same_surface_bank_OR_exact_OR_sequence_gte_0.88_OR_token_jaccard_gte_0.70",
            "scenario_grounding_action_edges_used": False,
            "length_prefilter_used": False,
            "sequence_threshold": SEQUENCE_THRESHOLD,
            "token_jaccard_threshold": JACCARD_THRESHOLD,
            "component_count": len(components),
            "component_sizes_desc": sorted(
                (component["rows"] for component in components), reverse=True
            ),
            "components": components,
            "similarity_edge_counts": dict(graph_edges),
        },
        "cross_bank_similarity": {
            "counts": dict(cross_bank),
            "pairwise_counts": {
                key: dict(value)
                for key, value in sorted(pairwise_bank_counts.items())
            },
            "maximum_sequence_ratio": max(
                (entry[0] for entry in sequence_top), default=0.0
            ),
            "maximum_token_jaccard": max(
                (entry[0] for entry in jaccard_top), default=0.0
            ),
            "top_sequence_pair_hashes": [
                entry[2] for entry in sorted(sequence_top, reverse=True)
            ],
            "top_jaccard_pair_hashes": [
                entry[2] for entry in sorted(jaccard_top, reverse=True)
            ],
        },
        "cross_partition_similarity": {
            "counts": dict(cross_partition),
            "pairwise_counts": {
                key: dict(value)
                for key, value in sorted(pairwise_partition_counts.items())
            },
        },
        "examples": {
            "scenario_id": "get_tomato_dispenser",
            "grounding_variant_index": 0,
            "by_bank": examples,
        },
        "style_diagnostics": _style_diagnostics(ordered),
        "v4_stress_reference": V4_STRESS_REFERENCE,
        "provenance": {
            "ai_assisted": True,
            "human_gold": False,
            "independent_human_semantic_audit": False,
            "external_corpus_inputs_opened": 0,
            "old_human_dev_test_frozen_rows_read": 0,
            "five_class_reference_classifier_used_as_target": False,
            "training_performed": False,
            "production_promotion_eligible": False,
            "real_player_accuracy_claim": False,
            "style_limit": "five lexical banks with one synthetic held-out evaluation bank",
        },
        "script_hashes": {
            "generator_sha256": hashlib.sha256(
                Path(generator.__file__).read_bytes()
            ).hexdigest(),
            "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    rows = generator.generate_rows()
    report = audit_split(rows)
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        corpus_path = output_dir / "corpus.json"
        report_path = output_dir / "split_audit.json"
        _write_json(corpus_path, rows)
        report["output_artifacts"] = {
            "corpus": {
                "path": str(corpus_path),
                "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            },
            "split_audit": {"path": str(report_path)},
            "model_written": False,
        }
        _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["quality_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
