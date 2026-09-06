"""Train the isolated current-ontology direct-fG clean-v3 research candidate.

The only corpus source is the pure typed generator in this package.  Model
selection uses grouped folds inside ``train``.  Temperature scaling uses the
independent ``calibration`` partition.  ``final_eval`` is evaluated only after
the vectorizer, classifier, hyperparameters, and temperature are frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v3 as generator  # noqa: E402


VERSION = "direct-fg-clean-v3-research-candidate-v1"
SEED = 73129
SEQUENCE_THRESHOLD = 0.88
JACCARD_THRESHOLD = 0.70
TARGET_SPLIT_FRACTIONS = {"train": 0.70, "calibration": 0.10, "final_eval": 0.20}
MINIMUM_MODEL_CONFIDENCE = 0.55
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v3"
)

CONFIG_GRID = tuple(
    {
        "min_df": min_df,
        "C": c_value,
        "class_weight": class_weight,
    }
    for min_df in (1, 2)
    for c_value in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    for class_weight in (None, "balanced")
)

NATURAL_PROBES = (
    ("That last tomato pickup was excellent.", "evaluative"),
    ("You handled the pot well on that step.", "evaluative"),
    ("Serving that soup just now was a good choice.", "evaluative"),
    ("Moving aside a moment ago was really helpful.", "evaluative"),
    ("That last onion move was badly timed.", "evaluative"),
    ("Your recent dish pickup was inefficient.", "evaluative"),
    ("The way you cleared the pot access was great.", "evaluative"),
    ("That counter placement was unhelpful.", "evaluative"),
    ("Your last wait action was a poor decision.", "evaluative"),
    ("The recent soup delivery was handled well.", "evaluative"),
    ("Could you grab the onion next?", "imperative"),
    ("Please put that tomato in the pot.", "imperative"),
    ("Move aside by the serving station.", "imperative"),
    ("Would you please pick up a dish?", "imperative"),
    ("Serve the soup at the serving station.", "imperative"),
    ("You should wait while the pot is cooking.", "imperative"),
    ("Place the held onion on an empty accessible counter.", "imperative"),
    ("Why don't you take the ready soup with a dish?", "imperative"),
    ("Clear the access tile in front of the pot.", "imperative"),
    ("Get a tomato from the tomato dispenser.", "imperative"),
    ("The pot currently contains two tomatoes.", "descriptive"),
    ("The ready soup is in the pot.", "descriptive"),
    ("The AI chef is holding a dish.", "descriptive"),
    ("An onion is on an accessible counter.", "descriptive"),
    ("The serving station accepts soup in a dish.", "descriptive"),
    ("The upper corridor is clear for the other chef.", "descriptive"),
    ("The fixed order requires two tomatoes and one onion.", "descriptive"),
    ("The pot is cooking.", "descriptive"),
    ("The AI chef holds soup beside the serving station.", "descriptive"),
    ("The access tile in front of the pot is clear.", "descriptive"),
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _sequence_ratio(left: str, right: str) -> float:
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    # Both quick methods are documented upper bounds.  They may skip expensive
    # exact ratio work but can never suppress a true >= threshold hit.
    if matcher.real_quick_ratio() < SEQUENCE_THRESHOLD:
        return 0.0
    if matcher.quick_ratio() < SEQUENCE_THRESHOLD:
        return 0.0
    return matcher.ratio()


def build_similarity_components(rows: list[dict]) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("cannot split an empty clean-v3 corpus")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["feedback_id"])
    union_find = UnionFind(len(ordered))

    for key in ("surface_family", "scenario_group", "grounding_id"):
        by_value: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(ordered):
            by_value[str(row[key])].append(index)
        for members in by_value.values():
            for index in members[1:]:
                union_find.union(members[0], index)

    normalized = [str(row["normalized_text"]) for row in ordered]
    tokens = [set(text.split()) for text in normalized]
    exact_seen: dict[str, int] = {}
    exact_edges = 0
    sequence_edges = 0
    jaccard_edges = 0
    both_edges = 0
    similarity_edges: list[tuple[int, int, bool, bool, bool]] = []
    for right, text in enumerate(normalized):
        prior = exact_seen.get(text)
        if prior is not None:
            exact_edges += 1
            union_find.union(prior, right)
        else:
            exact_seen[text] = right
        for left in range(right):
            jaccard = _token_jaccard(tokens[left], tokens[right])
            jaccard_hit = jaccard >= JACCARD_THRESHOLD
            sequence = _sequence_ratio(normalized[left], text)
            sequence_hit = sequence >= SEQUENCE_THRESHOLD
            exact_hit = normalized[left] == text
            if sequence_hit:
                sequence_edges += 1
            if jaccard_hit:
                jaccard_edges += 1
            if sequence_hit and jaccard_hit:
                both_edges += 1
            if exact_hit or sequence_hit or jaccard_hit:
                union_find.union(left, right)
                similarity_edges.append(
                    (left, right, exact_hit, sequence_hit, jaccard_hit)
                )

    component_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        component_members[union_find.find(index)].append(index)
    components: list[dict] = []
    for members in component_members.values():
        signature = _canonical_hash(
            sorted(str(ordered[index]["feedback_id"]) for index in members)
        )
        label_counts = Counter(
            str(ordered[index]["expected_feedback_type"]) for index in members
        )
        components.append(
            {
                "component_id": f"clean-v3-component:{signature[:20]}",
                "signature": signature,
                "members": tuple(members),
                "size": len(members),
                "label_counts": dict(sorted(label_counts.items())),
                "scenario_ids": sorted(
                    {str(ordered[index]["scenario_id"]) for index in members}
                ),
            }
        )
    components.sort(key=lambda item: item["signature"])
    audit = {
        "sequence_threshold": SEQUENCE_THRESHOLD,
        "token_jaccard_threshold": JACCARD_THRESHOLD,
        "edge_rule": "sequence_matcher_gte_0.88_OR_token_jaccard_gte_0.70",
        "length_prefilter_used": False,
        "exact_edges": exact_edges,
        "sequence_edges": sequence_edges,
        "jaccard_edges": jaccard_edges,
        "both_edges": both_edges,
        "component_count": len(components),
        "component_sizes_desc": sorted(
            (int(component["size"]) for component in components), reverse=True
        ),
        "largest_component_rows": max(component["size"] for component in components),
        "similarity_edges": similarity_edges,
    }
    return ordered, {"components": components, "audit": audit}


def _select_component_subset(
    components: list[dict], target_rows: int, *, namespace: str
) -> set[str]:
    if not components:
        raise ValueError(f"no components available for {namespace}")
    ordered = sorted(
        components,
        key=lambda component: hashlib.sha256(
            f"{SEED}|{namespace}|{component['signature']}".encode("utf-8")
        ).hexdigest(),
    )
    # Integer subset sum with one deterministic representative per row total.
    states: dict[int, tuple[str, ...]] = {0: ()}
    sizes = {str(component["component_id"]): int(component["size"]) for component in ordered}
    for component in ordered:
        component_id = str(component["component_id"])
        size = int(component["size"])
        additions = {
            total + size: selected + (component_id,)
            for total, selected in list(states.items())
            if total + size not in states
        }
        states.update(additions)
    candidates = [
        (total, selected)
        for total, selected in states.items()
        if selected and len(selected) < len(components)
    ]
    if not candidates:
        raise ValueError(f"cannot reserve a non-empty {namespace} component subset")
    total, selected = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_rows),
            abs(len(item[1]) - max(1, round(len(components) * target_rows / sum(sizes.values())))),
            item[1],
        ),
    )
    if total <= 0:
        raise ValueError(f"invalid {namespace} subset")
    return set(selected)


def split_rows(rows: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    ordered, graph = build_similarity_components(rows)
    components = list(graph["components"])
    if len(components) < 3:
        raise ValueError(
            "strict OR similarity graph has fewer than three components; "
            "cannot create train/calibration/final_eval without relaxing the gate"
        )
    total_rows = len(ordered)
    final_ids = _select_component_subset(
        components,
        round(total_rows * TARGET_SPLIT_FRACTIONS["final_eval"]),
        namespace="final_eval",
    )
    remaining = [
        component for component in components if component["component_id"] not in final_ids
    ]
    if len(remaining) < 2:
        raise ValueError("strict final_eval reservation leaves no independent calibration split")
    calibration_ids = _select_component_subset(
        remaining,
        round(total_rows * TARGET_SPLIT_FRACTIONS["calibration"]),
        namespace="calibration",
    )
    train_ids = {
        str(component["component_id"])
        for component in remaining
        if component["component_id"] not in calibration_ids
    }
    if not train_ids:
        raise ValueError("strict split left an empty train partition")

    index_to_component: dict[int, str] = {}
    for component in components:
        for index in component["members"]:
            index_to_component[int(index)] = str(component["component_id"])
    partitioned = {"train": [], "calibration": [], "final_eval": []}
    row_split: dict[int, str] = {}
    for index, row in enumerate(ordered):
        component_id = index_to_component[index]
        if component_id in final_ids:
            split = "final_eval"
        elif component_id in calibration_ids:
            split = "calibration"
        elif component_id in train_ids:
            split = "train"
        else:
            raise AssertionError(component_id)
        rendered = dict(row)
        rendered["component_id"] = component_id
        rendered["split"] = split
        partitioned[split].append(rendered)
        row_split[index] = split

    required_labels = set(generator.LABELS)
    partition_audit: dict[str, dict] = {}
    for split, split_rows_value in partitioned.items():
        labels = Counter(row["expected_feedback_type"] for row in split_rows_value)
        if set(labels) != required_labels or len(set(labels.values())) != 1:
            raise ValueError(f"{split} is not non-empty and class-balanced: {dict(labels)}")
        partition_audit[split] = {
            "rows": len(split_rows_value),
            "label_counts": dict(sorted(labels.items())),
            "component_count": len({row["component_id"] for row in split_rows_value}),
            "scenario_ids": sorted({row["scenario_id"] for row in split_rows_value}),
            "surface_family_count": len(
                {row["surface_family"] for row in split_rows_value}
            ),
        }

    cross_counts = {"exact": 0, "sequence": 0, "jaccard": 0}
    for left, right, exact_hit, sequence_hit, jaccard_hit in graph["audit"][
        "similarity_edges"
    ]:
        if row_split[left] == row_split[right]:
            continue
        cross_counts["exact"] += int(exact_hit)
        cross_counts["sequence"] += int(sequence_hit)
        cross_counts["jaccard"] += int(jaccard_hit)
    if any(cross_counts.values()):
        raise ValueError(f"strict OR split leaked similar rows: {cross_counts}")

    pairwise_overlap: dict[str, dict] = {}
    for left_name, right_name in (
        ("train", "calibration"),
        ("train", "final_eval"),
        ("calibration", "final_eval"),
    ):
        left_rows = partitioned[left_name]
        right_rows = partitioned[right_name]
        overlap = {}
        for key in (
            "normalized_text",
            "surface_family",
            "scenario_group",
            "grounding_id",
            "component_id",
        ):
            overlap[key] = len(
                {row[key] for row in left_rows} & {row[key] for row in right_rows}
            )
        if any(overlap.values()):
            raise ValueError(f"{left_name}/{right_name} group leakage: {overlap}")
        pairwise_overlap[f"{left_name}__{right_name}"] = overlap

    graph_audit = dict(graph["audit"])
    graph_audit.pop("similarity_edges", None)
    audit = {
        "seed": SEED,
        "target_fractions": dict(TARGET_SPLIT_FRACTIONS),
        "partitions": partition_audit,
        "pairwise_group_overlap": pairwise_overlap,
        "cross_partition_similarity_counts": {
            "normalized_exact": cross_counts["exact"],
            "sequence_matcher_at_least_0_88": cross_counts["sequence"],
            "token_jaccard_at_least_0_70": cross_counts["jaccard"],
        },
        "similarity_quality_gate_passed": (
            cross_counts["sequence"] == 0 and cross_counts["jaccard"] == 0
        ),
        "component_graph": graph_audit,
    }
    return partitioned, audit


def _build_vectorizer(min_df: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def _build_classifier(config: dict):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=float(config["C"]),
        class_weight=config["class_weight"],
        max_iter=3000,
        random_state=SEED,
        solver="lbfgs",
    )


def _basic_metrics(labels: list[str], predictions: Iterable[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    predicted = [str(value) for value in predictions]
    recalls = recall_score(
        labels,
        predicted,
        labels=list(generator.LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=list(generator.LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            label: float(value) for label, value in zip(generator.LABELS, recalls)
        },
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=list(generator.LABELS)
        ).tolist(),
        "label_order": list(generator.LABELS),
    }


def select_hyperparameters_train_only(train_rows: list[dict]) -> tuple[dict, dict]:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    groups = [str(row["component_id"]) for row in train_rows]
    unique_groups = sorted(set(groups))
    n_splits = min(5, len(unique_groups))
    if n_splits < 3:
        raise ValueError("strict train partition needs at least three components for group CV")
    texts = [str(row["text"]) for row in train_rows]
    labels = [str(row["expected_feedback_type"]) for row in train_rows]
    splitter = GroupKFold(n_splits=n_splits)
    fold_indices = list(splitter.split(texts, labels, groups))
    vectorized_folds: dict[tuple[int, int], tuple] = {}
    for min_df in sorted({int(config["min_df"]) for config in CONFIG_GRID}):
        for fold_index, (fit_indices, validation_indices) in enumerate(fold_indices):
            vectorizer = _build_vectorizer(min_df)
            fit_texts = [texts[index] for index in fit_indices]
            validation_texts = [texts[index] for index in validation_indices]
            fit_matrix = vectorizer.fit_transform(fit_texts)
            validation_matrix = vectorizer.transform(validation_texts)
            vectorized_folds[(min_df, fold_index)] = (
                fit_matrix,
                validation_matrix,
                [labels[index] for index in fit_indices],
                [labels[index] for index in validation_indices],
            )

    results: list[dict] = []
    for config in CONFIG_GRID:
        fold_metrics = []
        for fold_index in range(n_splits):
            fit_matrix, validation_matrix, fit_labels, validation_labels = (
                vectorized_folds[(int(config["min_df"]), fold_index)]
            )
            classifier = _build_classifier(config)
            classifier.fit(fit_matrix, fit_labels)
            fold_metrics.append(
                _basic_metrics(validation_labels, classifier.predict(validation_matrix))
            )
        mean_accuracy = float(np.mean([metric["accuracy"] for metric in fold_metrics]))
        mean_macro_f1 = float(np.mean([metric["macro_f1"] for metric in fold_metrics]))
        mean_recalls = {
            label: float(
                np.mean(
                    [metric["per_class_recall"][label] for metric in fold_metrics]
                )
            )
            for label in generator.LABELS
        }
        results.append(
            {
                "config": dict(config),
                "mean_accuracy": mean_accuracy,
                "mean_macro_f1": mean_macro_f1,
                "mean_per_class_recall": mean_recalls,
                "minimum_mean_recall": min(mean_recalls.values()),
                "fold_metrics": fold_metrics,
            }
        )
    selected = max(
        results,
        key=lambda result: (
            result["mean_macro_f1"],
            result["mean_accuracy"],
            result["minimum_mean_recall"],
            -int(result["config"]["min_df"]),
            -float(result["config"]["C"]),
            result["config"]["class_weight"] is None,
        ),
    )
    return dict(selected["config"]), {
        "selection_scope": "train_group_cv_only",
        "n_splits": n_splits,
        "group_count": len(unique_groups),
        "candidate_count": len(CONFIG_GRID),
        "selected": selected,
        "candidates": results,
        "calibration_rows_seen": 0,
        "final_eval_rows_seen": 0,
        "vectorizer_refit_inside_each_fold": True,
    }


def fit_base_model(train_rows: list[dict], config: dict):
    vectorizer = _build_vectorizer(int(config["min_df"]))
    texts = [str(row["text"]) for row in train_rows]
    labels = [str(row["expected_feedback_type"]) for row in train_rows]
    matrix = vectorizer.fit_transform(texts)
    classifier = _build_classifier(config)
    classifier.fit(matrix, labels)
    return vectorizer, classifier


def _temperature_probabilities(raw_probabilities, temperature: float):
    import numpy as np

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    raw = np.asarray(raw_probabilities, dtype=float)
    logits = np.log(np.clip(raw, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _negative_log_likelihood(probabilities, label_indices) -> float:
    import numpy as np

    rows = np.arange(len(label_indices))
    selected = probabilities[rows, np.asarray(label_indices, dtype=int)]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def fit_temperature_calibration_only(classifier, matrix, labels: list[str]) -> tuple[float, dict]:
    import numpy as np

    raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
    class_to_index = {str(label): index for index, label in enumerate(classifier.classes_)}
    indices = [class_to_index[label] for label in labels]
    coarse = np.exp(np.linspace(math.log(0.20), math.log(5.0), 2001))
    losses = [
        _negative_log_likelihood(_temperature_probabilities(raw, float(value)), indices)
        for value in coarse
    ]
    best_index = int(np.argmin(losses))
    lower = coarse[max(0, best_index - 1)]
    upper = coarse[min(len(coarse) - 1, best_index + 1)]
    fine = np.linspace(float(lower), float(upper), 1001)
    fine_losses = [
        _negative_log_likelihood(_temperature_probabilities(raw, float(value)), indices)
        for value in fine
    ]
    fine_index = int(np.argmin(fine_losses))
    temperature = float(fine[fine_index])
    return temperature, {
        "method": "temperature_scaling",
        "version": "clean-v3-independent-calibration-v1",
        "fit_split": "calibration",
        "fit_rows": len(labels),
        "fit_split_used_for_model_selection": False,
        "final_eval_used": False,
        "temperature": temperature,
        "raw_nll": _negative_log_likelihood(raw, indices),
        "calibrated_nll": float(fine_losses[fine_index]),
        "search": "deterministic_log_grid_plus_local_linear_refinement",
    }


def _calibration_metrics(probabilities, labels: list[str], classes: list[str]) -> dict:
    import numpy as np

    values = np.asarray(probabilities, dtype=float)
    class_to_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_to_index[label] for label in labels], dtype=int)
    predictions = values.argmax(axis=1)
    confidence = values.max(axis=1)
    correct = predictions == indices
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        lower, upper = bins[index], bins[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == 9 else confidence < upper
        )
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    one_hot = np.zeros_like(values)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return {
        "negative_log_likelihood": _negative_log_likelihood(values, indices),
        "brier_score_multiclass": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def evaluate_frozen_final(
    vectorizer,
    classifier,
    temperature: float,
    final_rows: list[dict],
    *,
    minimum_confidence: float = MINIMUM_MODEL_CONFIDENCE,
) -> dict:
    import numpy as np

    texts = [str(row["text"]) for row in final_rows]
    labels = [str(row["expected_feedback_type"]) for row in final_rows]
    matrix = vectorizer.transform(texts)
    raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
    calibrated = _temperature_probabilities(raw, temperature)
    classes = [str(label) for label in classifier.classes_]
    predictions = [classes[index] for index in calibrated.argmax(axis=1)]
    metrics = _basic_metrics(labels, predictions)
    metrics["probability_metrics"] = _calibration_metrics(calibrated, labels, classes)
    confidence = calibrated.max(axis=1)
    covered = confidence >= minimum_confidence
    correct = np.asarray(predictions) == np.asarray(labels)
    metrics["fixed_confidence_threshold"] = {
        "threshold": minimum_confidence,
        "selected_before_final_eval": True,
        "coverage": float(covered.mean()),
        "covered_rows": int(covered.sum()),
        "accuracy_when_covered": float(correct[covered].mean()) if covered.any() else None,
        "abstained_rows": int((~covered).sum()),
    }
    metrics["selection_role"] = "none_frozen_final_evaluation_only"
    metrics["model_or_temperature_mutated"] = False
    return metrics


def evaluate_natural_probe(vectorizer, classifier, temperature: float) -> dict:
    import numpy as np

    texts = [text for text, _ in NATURAL_PROBES]
    labels = [label for _, label in NATURAL_PROBES]
    raw = classifier.predict_proba(vectorizer.transform(texts))
    calibrated = _temperature_probabilities(raw, temperature)
    classes = [str(label) for label in classifier.classes_]
    predictions = [classes[index] for index in np.asarray(calibrated).argmax(axis=1)]
    return {
        "provenance": "ai_assisted_targeted_diagnostic_not_human_gold",
        "included_in_pass_gate": False,
        "included_in_selection": False,
        "metrics": _basic_metrics(labels, predictions),
        "errors": [
            {"text": text, "expected": expected, "predicted": predicted}
            for (text, expected), predicted in zip(NATURAL_PROBES, predictions)
            if expected != predicted
        ],
    }


def _model_state_hash(vectorizer, classifier, selected_config: dict, temperature: float) -> str:
    import numpy as np

    payload = hashlib.sha256()
    payload.update(json.dumps(selected_config, sort_keys=True).encode("utf-8"))
    payload.update(np.asarray(classifier.coef_).tobytes())
    payload.update(np.asarray(classifier.intercept_).tobytes())
    payload.update(json.dumps([str(value) for value in classifier.classes_]).encode("utf-8"))
    payload.update(str(float(temperature)).encode("utf-8"))
    payload.update(
        json.dumps(
            [
                (name, len(transformer.vocabulary_))
                for name, transformer in vectorizer.transformer_list
            ],
            sort_keys=True,
        ).encode("utf-8")
    )
    return payload.hexdigest()


def _dependency_manifest() -> dict:
    import joblib
    import numpy
    import sklearn

    local_paths = {
        "generator": Path(generator.__file__).resolve(),
        "trainer": Path(__file__).resolve(),
    }
    external_modules = {"numpy": numpy, "sklearn": sklearn, "joblib": joblib}
    return {
        "local_code": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in local_paths.items()
        },
        "external_dependencies": {
            name: {
                "version": str(getattr(module, "__version__", "unknown")),
                "module_file": str(Path(module.__file__).resolve()),
                "module_file_sha256": _sha256(Path(module.__file__).resolve()),
            }
            for name, module in external_modules.items()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }


def train_candidate_in_memory() -> tuple[dict, dict, dict[str, list[dict]]]:
    rows = generator.generate_rows()
    generation_audit = generator.audit_rows(rows)
    partitions, split_audit = split_rows(rows)
    selected_config, selection_report = select_hyperparameters_train_only(
        partitions["train"]
    )
    vectorizer, classifier = fit_base_model(partitions["train"], selected_config)
    calibration_matrix = vectorizer.transform(
        [row["text"] for row in partitions["calibration"]]
    )
    calibration_labels = [
        row["expected_feedback_type"] for row in partitions["calibration"]
    ]
    temperature, calibration_report = fit_temperature_calibration_only(
        classifier, calibration_matrix, calibration_labels
    )
    frozen_hash_before = _model_state_hash(
        vectorizer, classifier, selected_config, temperature
    )
    final_metrics = evaluate_frozen_final(
        vectorizer, classifier, temperature, partitions["final_eval"]
    )
    frozen_hash_after = _model_state_hash(
        vectorizer, classifier, selected_config, temperature
    )
    if frozen_hash_before != frozen_hash_after:
        raise ValueError("frozen final evaluation mutated the candidate model")
    natural_probe = evaluate_natural_probe(vectorizer, classifier, temperature)

    recalls = final_metrics["per_class_recall"]
    status_checks = {
        "generation_rows_at_least_1440": generation_audit["rows"] >= 1440,
        "generation_balanced": len(set(generation_audit["label_counts"].values())) == 1,
        "forbidden_ontology_hits_zero": generation_audit["forbidden_ontology_hits"] == 0,
        "single_speech_act_failures_zero": generation_audit["single_speech_act_failures"] == 0,
        "surface_grammar_failures_zero": generation_audit["surface_grammar_failures"] == 0,
        "reference_type_not_used": not generation_audit["reference_type_used"],
        "external_input_files_opened_zero": generation_audit["external_input_files_opened"] == 0,
        "strict_sequence_cross_split_zero": split_audit["cross_partition_similarity_counts"]["sequence_matcher_at_least_0_88"] == 0,
        "strict_jaccard_cross_split_zero": split_audit["cross_partition_similarity_counts"]["token_jaccard_at_least_0_70"] == 0,
        "train_only_model_selection": selection_report["calibration_rows_seen"] == 0 and selection_report["final_eval_rows_seen"] == 0,
        "independent_calibration": calibration_report["fit_split"] == "calibration" and not calibration_report["fit_split_used_for_model_selection"] and not calibration_report["final_eval_used"],
        "final_eval_accuracy_at_least_0_87": final_metrics["accuracy"] >= 0.87,
        "final_eval_macro_f1_at_least_0_87": final_metrics["macro_f1"] >= 0.87,
        "each_final_eval_recall_at_least_0_85": min(recalls.values()) >= 0.85,
        "frozen_final_eval_did_not_mutate_model": frozen_hash_before == frozen_hash_after,
    }
    status = (
        "passed_research_candidate"
        if all(status_checks.values())
        else "failed_research_candidate"
    )
    dependencies = _dependency_manifest()
    report = {
        "version": VERSION,
        "status": status,
        "status_checks": status_checks,
        "claim_scope": "synthetic_current_ontology_family_and_scenario_disjoint_final_eval",
        "not_a_real_player_accuracy_claim": True,
        "production_promotion_eligible": False,
        "production_promotion_blocker": "requires independently audited natural human holdout",
        "generation": generation_audit,
        "split": split_audit,
        "selection": selection_report,
        "selected_hyperparameters": selected_config,
        "calibration": calibration_report,
        "final_synthetic_evaluation": final_metrics,
        "natural_probe": natural_probe,
        "frozen_model_state_sha256_before_final": frozen_hash_before,
        "frozen_model_state_sha256_after_final": frozen_hash_after,
        "provenance": {
            "ai_assisted": True,
            "prior_error_informed": True,
            "independent_human_semantic_audit": False,
            "human_gold": False,
            "old_synthetic_rows_read": 0,
            "human_rows_read": 0,
            "dev_test_frozen_membership_files_read": 0,
            "reference_type_used_as_target": False,
            "five_class_reference_classifier_used_as_target": False,
        },
        "dependencies": dependencies,
        "seed": SEED,
    }
    artifact = {
        "artifact_version": VERSION,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "feature_version": "direct-fg-clean-v3-word-char-tfidf-v1",
        "labels": list(generator.LABELS),
        "minimum_model_confidence": MINIMUM_MODEL_CONFIDENCE,
        "selected_hyperparameters": selected_config,
        "probability_calibration": {
            "method": "temperature_scaling",
            "version": calibration_report["version"],
            "temperature": temperature,
            "fit_split": "calibration",
            "fit_split_used_for_model_selection": False,
            "final_eval_used": False,
        },
        "training_scope": {
            "train_rows": len(partitions["train"]),
            "calibration_rows_for_temperature_only": len(partitions["calibration"]),
            "final_eval_rows_seen_during_training_or_calibration": 0,
        },
        "candidate_status": status,
        "production_promotion_eligible": False,
    }
    return artifact, report, partitions


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_candidate_outputs(
    output_dir: Path, artifact: dict, report: dict, partitions: dict[str, list[dict]]
) -> dict:
    from joblib import dump

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.json"
    report_path = output_dir / "evaluation_report.json"
    model_path = output_dir / "model.joblib"
    manifest_path = output_dir / "manifest.json"
    flat_rows = [
        row for split in ("train", "calibration", "final_eval") for row in partitions[split]
    ]
    _write_json(corpus_path, flat_rows)
    _write_json(report_path, report)
    temporary_model = model_path.with_suffix(".joblib.tmp")
    dump(artifact, temporary_model)
    os.replace(temporary_model, model_path)
    manifest = {
        "version": VERSION,
        "status": report["status"],
        "research_candidate_only": True,
        "production_promotion_eligible": False,
        "outputs": {
            "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        },
        "dependencies": report["dependencies"],
        "seed": SEED,
        "temperature": artifact["probability_calibration"]["temperature"],
        "selected_hyperparameters": artifact["selected_hyperparameters"],
        "provenance": report["provenance"],
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-audit-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.split_audit_only:
        rows = generator.generate_rows()
        _, audit = split_rows(rows)
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return

    artifact, report, partitions = train_candidate_in_memory()
    if not args.no_write:
        manifest = write_candidate_outputs(args.output_dir, artifact, report, partitions)
        report = dict(report)
        report["manifest"] = manifest
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
