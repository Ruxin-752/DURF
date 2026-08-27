"""Deterministic, auditable corpus splits without ML dependencies."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator


SPLIT_MANIFEST_VERSION = "route2-split-v2"
LEGACY_SPLIT_MANIFEST_VERSION = "route2-split-v1"
GROUPING_POLICIES = frozenset({"joint", "scenario", "template"})
_FAMILY_SUFFIX = re.compile(r"_(?:template|llm)\d+$", re.IGNORECASE)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable serialization."""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(text: str | None) -> str:
    """Normalize feedback text for corpus-wide duplicate checks."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def _semantic_signature(example: dict) -> str:
    fields = {
        key: example.get(key)
        for key in (
            "target_features",
            "trajectory_features",
            "target_action",
            "attributed_sentiment_score",
            "expected_feedback_type",
            "reference_type",
            "referenced_subgoal",
            "expected_subgoal",
        )
        if example.get(key) is not None
    }
    return canonical_sha256(fields)


def _iter_near_duplicate_pairs(
    examples: list[dict],
    *,
    threshold: float = 0.92,
) -> Iterator[tuple[int, int, float]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in [0, 1]")
    normalized = [normalize_text(row.get("text")) for row in examples]
    token_sets = [set(text.split()) for text in normalized]
    postings: dict[str, list[int]] = defaultdict(list)
    for right, (text, tokens) in enumerate(zip(normalized, token_sets)):
        candidates: set[int] = set()
        for token in tokens:
            candidates.update(postings[token])
        for left in sorted(candidates):
            other = normalized[left]
            if not text or text == other:
                continue
            length_ratio = min(len(text), len(other)) / max(len(text), len(other))
            if length_ratio < threshold:
                continue
            union = tokens | token_sets[left]
            token_score = len(tokens & token_sets[left]) / len(union) if union else 0.0
            if token_score < max(0.5, threshold - 0.2):
                continue
            score = SequenceMatcher(None, other, text, autojunk=False).ratio()
            if score < threshold:
                continue
            yield left, right, score
        for token in tokens:
            postings[token].append(right)


def find_near_duplicates(
    examples: list[dict],
    *,
    threshold: float = 0.92,
    max_examples: int = 100,
) -> dict:
    """Find likely paraphrases using token blocking plus sequence similarity."""

    pairs: list[dict] = []
    total = 0
    pair_indices: list[list[float | int]] = []
    for left, right, score in _iter_near_duplicate_pairs(
        examples, threshold=threshold
    ):
        total += 1
        pair_indices.append([left, right, round(score, 6)])
        if len(pairs) < max_examples:
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "similarity": round(score, 6),
                    "left_text": examples[left].get("text"),
                    "right_text": examples[right].get("text"),
                }
            )
    return {
        "threshold": threshold,
        "count": total,
        "examples": pairs,
        "pair_indices": pair_indices,
    }


def deduplicate_corpus(
    examples: list[dict],
    *,
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[dict], dict]:
    """Globally deduplicate normalized text and report contradictory labels."""

    unique: list[dict] = []
    seen: dict[str, tuple[int, str]] = {}
    duplicate_examples: list[dict] = []
    conflicts: list[dict] = []
    for source_index, example in enumerate(examples):
        text_key = normalize_text(example.get("text"))
        if not text_key:
            unique.append(example)
            continue
        signature = _semantic_signature(example)
        previous = seen.get(text_key)
        if previous is None:
            seen[text_key] = (source_index, signature)
            unique.append(example)
            continue
        prior_index, prior_signature = previous
        duplicate = {
            "kept_source_index": prior_index,
            "dropped_source_index": source_index,
            "normalized_text": text_key,
        }
        duplicate_examples.append(duplicate)
        if signature != prior_signature:
            conflicts.append(duplicate)
    return unique, {
        "input_count": len(examples),
        "output_count": len(unique),
        "exact_duplicates_removed": len(duplicate_examples),
        "exact_duplicate_examples": duplicate_examples[:100],
        "supervision_conflict_count": len(conflicts),
        "supervision_conflicts": conflicts[:100],
        "near_duplicates": find_near_duplicates(
            unique, threshold=near_duplicate_threshold
        ),
    }


def template_family(example: dict, index: int) -> str:
    """Return the best available stable paraphrase/template family."""

    explicit = example.get("paraphrase_family") or example.get("template_family")
    if explicit:
        return str(explicit)
    intent = example.get("intent_id")
    if intent:
        return str(intent)
    feedback_id = str(example.get("feedback_id") or "")
    if feedback_id:
        return _FAMILY_SUFFIX.sub("", feedback_id)
    return f"record:{index}"


def make_train_dev_test_split(
    group_ids: list[str | None],
    *,
    seed: int = 0,
    dev_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[int] | list[str]]:
    if not 0 <= dev_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("dev_fraction and test_fraction must be in [0, 1)")
    if dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be < 1")
    grouped: dict[str, list[int]] = {}
    ungrouped: list[int] = []
    for index, group in enumerate(group_ids):
        if group is None:
            ungrouped.append(index)
        else:
            grouped.setdefault(str(group), []).append(index)
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    n_groups = len(groups)
    n_test = min(n_groups, max(1, round(n_groups * test_fraction))) if test_fraction else 0
    remaining = n_groups - n_test
    n_dev = min(remaining, max(1, round(n_groups * dev_fraction))) if dev_fraction else 0
    test_groups = groups[:n_test]
    dev_groups = groups[n_test : n_test + n_dev]
    train_groups = groups[n_test + n_dev :]

    def indices_for(selected: list[str]) -> list[int]:
        return [index for group in selected for index in grouped[group]]

    return {
        "train_groups": train_groups,
        "dev_groups": dev_groups,
        "test_groups": test_groups,
        "train_indices": sorted([*ungrouped, *indices_for(train_groups)]),
        "dev_indices": sorted(indices_for(dev_groups)),
        "test_indices": sorted(indices_for(test_groups)),
    }


def _component_group_ids(
    examples: list[dict],
    *,
    policy: str,
    near_duplicate_threshold: float,
    near_duplicate_pairs: list[tuple[int, int, float]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Build leakage-safe components for one declared holdout policy.

    ``joint`` preserves the strict v1 behavior. ``scenario`` and ``template``
    support larger, interpretable holdouts on corpora where reusable template
    families connect most scenarios into a few giant joint components.
    Near-duplicate utterances are always kept in the same component.
    """

    if policy not in GROUPING_POLICIES:
        raise ValueError(
            f"grouping policy must be one of {sorted(GROUPING_POLICIES)}, got {policy!r}"
        )

    parents: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    scenarios: list[str] = []
    families: list[str] = []
    row_nodes: list[str] = []
    for index, example in enumerate(examples):
        scenario = str(example.get("group_id") or example.get("probe_id") or f"record:{index}")
        family = template_family(example, index)
        row_node = f"row:{index}"
        if policy in {"joint", "scenario"}:
            union(row_node, f"scenario:{scenario}")
        if policy in {"joint", "template"}:
            union(row_node, f"family:{family}")
        scenarios.append(scenario)
        families.append(family)
        row_nodes.append(row_node)
    if policy == "joint":
        pairs = (
            list(_iter_near_duplicate_pairs(examples, threshold=near_duplicate_threshold))
            if near_duplicate_pairs is None
            else near_duplicate_pairs
        )
        for left, right, _score in pairs:
            union(f"row:{left}", f"row:{right}")

    members: dict[str, list[str]] = defaultdict(list)
    for node in parents:
        members[find(node)].append(node)
    component_name = {
        root: "component:" + canonical_sha256(sorted(nodes))[:16]
        for root, nodes in members.items()
    }
    return [component_name[find(node)] for node in row_nodes], scenarios, families


def _split_conflicts(
    examples: list[dict],
    split_names: list[str],
    *,
    near_duplicate_threshold: float,
    near_duplicate_pairs: list[tuple[int, int, float]] | None = None,
) -> dict:
    normalized_locations: dict[str, set[str]] = defaultdict(set)
    for example, split_name in zip(examples, split_names):
        if split_name == "excluded":
            continue
        text = normalize_text(example.get("text"))
        if text:
            normalized_locations[text].add(split_name)
    exact = [
        {"normalized_text": text, "splits": sorted(locations)}
        for text, locations in normalized_locations.items()
        if len(locations) > 1
    ]
    cross_near = []
    cross_near_count = 0
    pairs = (
        list(_iter_near_duplicate_pairs(examples, threshold=near_duplicate_threshold))
        if near_duplicate_pairs is None
        else near_duplicate_pairs
    )
    for left, right, score in pairs:
        left_split = split_names[left]
        right_split = split_names[right]
        if "excluded" not in {left_split, right_split} and left_split != right_split:
            cross_near_count += 1
            if len(cross_near) < 100:
                cross_near.append(
                    {
                        "left_index": left,
                        "right_index": right,
                        "similarity": round(score, 6),
                        "left_text": examples[left].get("text"),
                        "right_text": examples[right].get("text"),
                        "left_split": left_split,
                        "right_split": right_split,
                    }
                )
    if exact:
        raise ValueError(
            "Normalized text occurs across fixed splits: "
            + json.dumps(exact[:10], ensure_ascii=False)
        )
    return {
        "exact_normalized_text_conflicts": exact,
        "near_duplicate_threshold": near_duplicate_threshold,
        "cross_split_near_duplicate_count": cross_near_count,
        "cross_split_near_duplicates": cross_near,
    }


def _near_duplicate_exclusions(
    split_names: list[str],
    pairs: list[tuple[int, int, float]],
) -> tuple[set[int], list[dict]]:
    """Protect higher-priority evaluation rows by pruning similar lower splits."""

    priority = {"train": 0, "dev": 1, "test": 2}
    excluded: set[int] = set()
    decisions: list[dict] = []
    for left, right, score in pairs:
        left_split, right_split = split_names[left], split_names[right]
        if left_split == right_split:
            continue
        if priority[left_split] < priority[right_split]:
            dropped, kept = left, right
        else:
            dropped, kept = right, left
        excluded.add(dropped)
        decisions.append(
            {
                "dropped_index": dropped,
                "dropped_split": split_names[dropped],
                "protected_index": kept,
                "protected_split": split_names[kept],
                "similarity": round(score, 6),
            }
        )
    return excluded, decisions


def make_split_manifest(
    examples: list[dict],
    *,
    seed: int = 0,
    dev_fraction: float = 0.2,
    test_fraction: float = 0.2,
    near_duplicate_threshold: float = 0.92,
    grouping_policy: str = "joint",
    near_duplicate_pairs: list[tuple[int, int, float]] | None = None,
) -> dict:
    """Create a versioned split manifest for a declared holdout axis."""

    pairs = (
        list(_iter_near_duplicate_pairs(examples, threshold=near_duplicate_threshold))
        if near_duplicate_pairs is None
        else near_duplicate_pairs
    )
    component_groups, scenarios, families = _component_group_ids(
        examples,
        policy=grouping_policy,
        near_duplicate_threshold=near_duplicate_threshold,
        near_duplicate_pairs=pairs,
    )
    split = make_train_dev_test_split(
        component_groups,
        seed=seed,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
    )
    index_to_split = [""] * len(examples)
    sections: dict[str, dict] = {}
    for name in ("train", "dev", "test"):
        indices = list(split[f"{name}_indices"])
        for index in indices:
            index_to_split[index] = name
        sections[name] = {"indices": indices}

    excluded_indices: set[int] = set()
    exclusion_decisions: list[dict] = []
    if grouping_policy != "joint":
        excluded_indices, exclusion_decisions = _near_duplicate_exclusions(
            index_to_split, pairs
        )
        for index in excluded_indices:
            index_to_split[index] = "excluded"
        for name in ("train", "dev", "test"):
            sections[name]["indices"] = [
                index
                for index in sections[name]["indices"]
                if index not in excluded_indices
            ]

    for name in ("train", "dev", "test"):
        indices = sections[name]["indices"]
        sections[name]["groups"] = sorted({scenarios[index] for index in indices})
        sections[name]["template_families"] = sorted(
            {families[index] for index in indices}
        )
        sections[name]["components"] = sorted(
            {component_groups[index] for index in indices}
        )

    disjoint_fields = ["components"]
    if grouping_policy in {"joint", "scenario"}:
        disjoint_fields.append("groups")
    if grouping_policy in {"joint", "template"}:
        disjoint_fields.append("template_families")
    policy_description = {
        "joint": "connected_components_by_scenario_or_template_family",
        "scenario": "scenario_holdout_with_cross_split_near_duplicates_pruned",
        "template": "template_holdout_with_cross_split_near_duplicates_pruned",
    }[grouping_policy]
    manifest = {
        "version": SPLIT_MANIFEST_VERSION,
        "corpus_sha256": canonical_sha256(examples),
        "grouping": {
            "scenario_fields": ["group_id", "probe_id"],
            "template_family_fields": [
                "paraphrase_family",
                "template_family",
                "intent_id",
                "feedback_id",
            ],
            "policy": policy_description,
            "holdout_axis": grouping_policy,
            "disjoint_fields": disjoint_fields,
            "near_duplicate_grouping_threshold": near_duplicate_threshold,
        },
        "seed": seed,
        "dev_fraction": dev_fraction,
        "test_fraction": test_fraction,
        "splits": sections,
        "excluded_indices": sorted(excluded_indices),
        "near_duplicate_exclusions": exclusion_decisions,
        # Flat keys preserve compatibility with existing split consumers.
        **{
            f"{name}_indices": sections[name]["indices"]
            for name in ("train", "dev", "test")
        },
        **{
            f"{name}_groups": sections[name]["groups"]
            for name in ("train", "dev", "test")
        },
        "conflict_audit": _split_conflicts(
            examples,
            index_to_split,
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_pairs=pairs,
        ),
    }
    manifest["split_sha256"] = canonical_sha256(manifest)
    return manifest

def make_declared_split_manifest(
    examples: list[dict],
    *,
    split_field: str = "split",
) -> dict:
    """Build a manifest from generator-declared reward/style-disjoint splits."""

    split_names = [str(example.get(split_field) or "") for example in examples]
    invalid = sorted(set(split_names) - {"train", "dev", "test"})
    if invalid or not all(split_names):
        raise ValueError(f"invalid declared split labels: {invalid}")
    sections: dict[str, dict] = {}
    for name in ("train", "dev", "test"):
        indices = [index for index, value in enumerate(split_names) if value == name]
        sections[name] = {
            "indices": indices,
            "groups": sorted(
                {
                    str(examples[index].get("group_id") or examples[index].get("probe_id"))
                    for index in indices
                }
            ),
            "template_families": sorted(
                {template_family(examples[index], index) for index in indices}
            ),
            "reward_configs": sorted(
                {str(examples[index].get("reward_config_id")) for index in indices}
            ),
            "teacher_ids": sorted(
                {
                    str(
                        examples[index].get("teacher_id")
                        or examples[index].get("synthetic_teacher_style_id")
                    )
                    for index in indices
                }
            ),
            "components": sorted(
                {
                    "declared:"
                    + canonical_sha256(
                        [
                            examples[index].get("reward_config_id"),
                            examples[index].get("teacher_id")
                            or examples[index].get("synthetic_teacher_style_id"),
                        ]
                    )[:16]
                    for index in indices
                }
            ),
        }
    normalized_locations: dict[str, set[str]] = defaultdict(set)
    for example, name in zip(examples, split_names):
        normalized = normalize_text(example.get("text"))
        if normalized:
            normalized_locations[normalized].add(name)
    exact_overlap = [
        {"normalized_text": text, "splits": sorted(locations)}
        for text, locations in normalized_locations.items()
        if len(locations) > 1
    ]
    manifest = {
        "version": SPLIT_MANIFEST_VERSION,
        "corpus_sha256": canonical_sha256(examples),
        "grouping": {
            "policy": "generator_declared_context_reward_config_and_teacher",
            "holdout_axis": "context_reward_config_and_teacher",
            "disjoint_fields": ["groups", "reward_configs", "teacher_ids", "components"],
            "split_field": split_field,
            "near_duplicate_grouping_threshold": None,
        },
        "seed": None,
        "dev_fraction": None,
        "test_fraction": None,
        "splits": sections,
        "excluded_indices": [],
        "near_duplicate_exclusions": [],
        **{
            f"{name}_indices": sections[name]["indices"]
            for name in ("train", "dev", "test")
        },
        **{
            f"{name}_groups": sections[name]["groups"]
            for name in ("train", "dev", "test")
        },
        "conflict_audit": {
            "exact_normalized_text_overlap_count": len(exact_overlap),
            "exact_normalized_text_overlaps": exact_overlap[:100],
            "near_duplicate_audit": "not computed for declared multi-reward corpus",
        },
    }
    manifest["split_sha256"] = canonical_sha256(manifest)
    return validate_split_manifest(manifest, examples)

def validate_split_manifest(manifest: dict, examples: list[dict]) -> dict:
    """Validate hashes, coverage, and disjointness before reusing a manifest."""

    version = manifest.get("version")
    if version not in {SPLIT_MANIFEST_VERSION, LEGACY_SPLIT_MANIFEST_VERSION}:
        raise ValueError(
            f"Unsupported split manifest version: {manifest.get('version')!r}"
        )
    expected_corpus_hash = canonical_sha256(examples)
    if manifest.get("corpus_sha256") != expected_corpus_hash:
        raise ValueError(
            "Split manifest corpus hash mismatch: "
            f"expected {expected_corpus_hash}, got {manifest.get('corpus_sha256')}"
        )
    supplied_hash = manifest.get("split_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "split_sha256"}
    expected_split_hash = canonical_sha256(unhashed)
    if supplied_hash != expected_split_hash:
        raise ValueError(
            "Split manifest hash mismatch: "
            f"expected {expected_split_hash}, got {supplied_hash}"
        )
    split_indices = [
        list(manifest.get(f"{name}_indices", []))
        for name in ("train", "dev", "test")
    ]
    flattened = [index for indices in split_indices for index in indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split manifest assigns an example to multiple splits")
    excluded = list(manifest.get("excluded_indices", []))
    if set(flattened) & set(excluded):
        raise ValueError("Excluded examples also occur in a train/dev/test split")
    if sorted([*flattened, *excluded]) != list(range(len(examples))):
        raise ValueError("Split manifest indices and exclusions do not cover the corpus exactly")
    disjoint_fields = manifest.get("grouping", {}).get("disjoint_fields")
    if disjoint_fields is None:
        # v1 manifests were strict joint-component splits.
        disjoint_fields = ("groups", "template_families", "components")
    for field in disjoint_fields:
        values = [
            set(manifest["splits"][name].get(field, []))
            for name in ("train", "dev", "test")
        ]
        if any(values[left] & values[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError(f"Split manifest leaks {field} across splits")
    return manifest


def load_split_manifest(path: str | Path, examples: list[dict]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_split_manifest(manifest, examples)


def write_split_manifest(path: str | Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
