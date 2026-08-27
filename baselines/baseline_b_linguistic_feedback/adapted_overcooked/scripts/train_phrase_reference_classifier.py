"""Train the paper-style TF-IDF + LogisticRegression reference classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import read_json, write_json  # noqa: E402
from src.evaluation_splits import make_train_dev_test_split, normalize_text  # noqa: E402
from src.phrase_reference_classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    REFERENCE_TYPES,
)
from src.text_analysis import limited_punc_tokenization, preprocess_phrase  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "reference_classifier_feedback.v8.json"
DEFAULT_ORIGINAL_PAPER_LABELS = (
    ROOT.parent
    / "original_rewards_repo"
    / "notebooks"
    / "data"
    / "pilot_chat_messages_labels.csv"
)
DEFAULT_PAPER_STRICT_DATA = ROOT / "data" / "reference_classifier_paper_strict.v1.json"
ROLE_REFERENCE_TYPE = {
    "praise_best": "trajectory",
    "criticize_alt": "action_spatial",
    "command_best": "action_spatial",
    "describe_alt": "feature",
    "describe_behavior_alt": "action_behavioral",
    "other": "other",
}
PAPER_REFERENCE_ACCURACY = 0.87
PAPER_REFERENCE_MACRO_F1 = 0.75
DEFAULT_OVERCOOKED_MIN_DF = 2
OVERCOOKED_INPUT_MODE = "raw_phrase"
OVERCOOKED_FEATURE_VERSION = "raw_word_char_tfidf_v3"
PAPER_STYLE_SOURCES = {"original_paper_human", "human"}
HARD_CONTRASTIVE_ORIGIN = "deterministic_contrastive_v1"


def _reference_type(example: dict) -> str | None:
    label = example.get("reference_type")
    if label in REFERENCE_TYPES:
        return str(label)
    return ROLE_REFERENCE_TYPE.get(str(example.get("role") or ""))


def _safe_preprocess(text: str) -> str:
    try:
        return preprocess_phrase(text)
    except (ImportError, LookupError):
        import re

        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _overcooked_feature_config(min_df: int) -> dict:
    """Describe the deployed raw-text feature extractor for reports/artifacts."""

    return {
        "version": OVERCOOKED_FEATURE_VERSION,
        "input_mode": OVERCOOKED_INPUT_MODE,
        "feature_union": {
            "word": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": int(min_df),
                "sublinear_tf": True,
                "strip_accents": "unicode",
                "stop_words": None,
            },
            "char": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": int(min_df),
                "sublinear_tf": True,
                "weight": 2.0,
            },
        },
    }


def _build_overcooked_vectorizer(min_df: int):
    """Retain discourse/function words and add robust character features."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    sublinear_tf=True,
                    min_df=min_df,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    sublinear_tf=True,
                    min_df=min_df,
                    ngram_range=(3, 5),
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 2.0},
    )


def build_phrase_rows(examples: list[dict]) -> list[dict]:
    rows = []
    for example in examples:
        family = str(
            example.get("paraphrase_family")
            or example.get("intent_id")
            or example.get("feedback_id")
            or ""
        ).split("_llm")[0]
        annotations = example.get("phrase_annotations")
        annotated_rows: list[tuple[str, str]] = []
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                phrase = annotation.get("phrase") or annotation.get("text")
                label = annotation.get("reference_type") or annotation.get("label")
                if isinstance(phrase, str) and label in REFERENCE_TYPES:
                    annotated_rows.append((phrase, str(label)))
        if annotated_rows:
            phrase_labels = annotated_rows
            annotation_source = "phrase_annotations"
        else:
            label = _reference_type(example)
            if label is None:
                continue
            phrase_labels = [
                (phrase, label)
                for phrase in limited_punc_tokenization(example.get("text") or "")
            ]
            annotation_source = "utterance_label"
        for phrase, label in phrase_labels:
            rows.append(
                {
                    "text": phrase,
                    "processed": _safe_preprocess(phrase),
                    "label": label,
                    "group_id": example.get("group_id"),
                    "paraphrase_family": family,
                    "source": example.get("source"),
                    "origin": example.get("classifier_corpus_origin"),
                    "split": example.get("split"),
                    "annotation_source": annotation_source,
                }
            )
    return rows


def _calibrated_probabilities(classifier, matrix, temperature: float):
    import numpy as np

    raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
    logits = np.log(np.clip(raw, 1e-12, 1.0)) / max(float(temperature), 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_temperature(classifier, matrix, labels: list[str]) -> float:
    import numpy as np

    if not labels:
        return 1.0
    class_index = {str(label): index for index, label in enumerate(classifier.classes_)}
    targets = np.asarray([class_index[label] for label in labels], dtype=int)
    candidates = np.geomspace(0.35, 3.0, 81)
    losses = []
    for temperature in candidates:
        probabilities = _calibrated_probabilities(classifier, matrix, float(temperature))
        losses.append(-float(np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()))
    return float(candidates[int(np.argmin(losses))])


def _class_thresholds(
    classifier,
    probabilities,
    labels: list[str],
    *,
    target_precision: float = 0.9,
) -> dict[str, float]:
    import numpy as np

    thresholds: dict[str, float] = {}
    predictions = np.argmax(probabilities, axis=1)
    for index, label in enumerate(classifier.classes_):
        predicted_rows = np.flatnonzero(predictions == index)
        if not len(predicted_rows):
            thresholds[str(label)] = 1.0
            continue
        confidences = probabilities[predicted_rows, index]
        candidates = sorted({float(value) for value in confidences})
        selected = 1.0
        for threshold in candidates:
            accepted = predicted_rows[confidences >= threshold]
            if not len(accepted):
                continue
            precision = sum(labels[row] == label for row in accepted) / len(accepted)
            if precision >= target_precision:
                selected = threshold
                break
        thresholds[str(label)] = float(np.clip(selected, 0.35, 1.0))
    return thresholds


def _metrics(
    classifier,
    matrix,
    labels: list[str],
    *,
    temperature: float = 1.0,
    class_thresholds: dict[str, float] | None = None,
) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    if not labels:
        return {}
    probabilities = _calibrated_probabilities(classifier, matrix, temperature)
    best_indices = probabilities.argmax(axis=1)
    predictions = np.asarray(classifier.classes_)[best_indices]
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    confidences = probabilities[np.arange(len(labels)), best_indices]
    correct = predictions == np.asarray(labels)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    label_index = {str(label): index for index, label in enumerate(classifier.classes_)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(labels):
        one_hot[row, label_index[label]] = 1.0
    thresholds = class_thresholds or {}
    abstained = np.asarray(
        [
            confidence < thresholds.get(str(label), 0.0) or margin < 0.05
            for label, confidence, margin in zip(predictions, confidences, margins)
        ]
    )
    ordered = list(REFERENCE_TYPES)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": classification_report(
            labels, predictions, labels=ordered, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=ordered).tolist(),
        "labels": ordered,
        "calibration": {
            "temperature": float(temperature),
            "ece_10_bin": float(ece),
            "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "mean_binary_brier": float(np.mean([
                brier_score_loss(one_hot[:, index], probabilities[:, index])
                for index in range(probabilities.shape[1])
            ])),
            "mean_top2_margin": float(margins.mean()),
            "abstention_rate": float(abstained.mean()),
            "selective_accuracy": (
                float(correct[~abstained].mean()) if (~abstained).any() else None
            ),
        },
    }


def paper_comparable_benchmark(
    examples: list[dict],
    *,
    min_df: int = 5,
) -> dict:
    """Reproduce the paper notebook's random 85/15 classifier protocol.

    This number is intentionally reported separately from the stricter
    scenario/template-disjoint test used by the deployed classifier.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rows = build_phrase_rows(examples)
    processed = [row["processed"] for row in rows]
    labels = [row["label"] for row in rows]
    train_text, test_text, train_labels, test_labels = train_test_split(
        processed,
        labels,
        test_size=0.15,
        random_state=1,
    )
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=min_df,
        ngram_range=(1, 2),
        stop_words="english",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    classifier = LogisticRegression(max_iter=2000, random_state=1)
    classifier.fit(train_matrix, train_labels)
    metrics = _metrics(classifier, test_matrix, test_labels)
    return {
        "protocol": "overcooked_random_85_15_diagnostic",
        "train_rows": len(train_text),
        "test_rows": len(test_text),
        "original_paper": {
            "accuracy": PAPER_REFERENCE_ACCURACY,
            "macro_f1": PAPER_REFERENCE_MACRO_F1,
            "test_rows": 148,
        },
        "ours": metrics,
        "accuracy_gap": float(metrics["accuracy"] - PAPER_REFERENCE_ACCURACY),
        "macro_f1_gap": float(metrics["macro_f1"] - PAPER_REFERENCE_MACRO_F1),
    }


def reproduce_original_paper_benchmark(path: str | Path = DEFAULT_ORIGINAL_PAPER_LABELS) -> dict:
    """Reproduce the paper notebook protocol, including its random row split.

    This is a compatibility result, not an external/generalization benchmark.
    """

    import csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    label_map = {
        "trajectory": "trajectory",
        "features": "feature",
        "object_spatial": "action_spatial",
        "object_behavior": "action_behavioral",
        "other": "other",
    }
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        {
            "text": str(row.get("phrase") or row.get("chat") or ""),
            "processed": _safe_preprocess(row.get("phrase") or row.get("chat") or ""),
            "label": label_map[row["reference_type"]],
            "task_uuid": str(row.get("task_uuid") or ""),
        }
        for row in source_rows
        if row.get("reference_type") in label_map
    ]
    train_rows, test_rows = train_test_split(rows, test_size=0.15, random_state=1)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        ngram_range=(1, 2),
        stop_words="english",
    )
    train_matrix = vectorizer.fit_transform([row["processed"] for row in train_rows])
    test_matrix = vectorizer.transform([row["processed"] for row in test_rows])
    classifier = LogisticRegression(max_iter=2000, random_state=1)
    classifier.fit(train_matrix, [row["label"] for row in train_rows])
    metrics = _metrics(classifier, test_matrix, [row["label"] for row in test_rows])
    train_texts = {
        normalize_text(row["text"])
        for row in train_rows
        if normalize_text(row["text"])
    }
    overlapping_test_rows = sum(
        bool(normalize_text(row["text"]))
        and normalize_text(row["text"]) in train_texts
        for row in test_rows
    )
    train_tasks = {row["task_uuid"] for row in train_rows if row["task_uuid"]}
    test_tasks = {row["task_uuid"] for row in test_rows if row["task_uuid"]}
    return {
        "protocol": "original_notebook_random_85_15_random_state_1",
        "benchmark_role": "paper_protocol_reproduction_not_external",
        "source": str(path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "reported_paper_accuracy": PAPER_REFERENCE_ACCURACY,
        "reported_paper_macro_f1": PAPER_REFERENCE_MACRO_F1,
        "normalized_text_overlap_test_rows": overlapping_test_rows,
        "task_uuid_overlap_count": len(train_tasks & test_tasks),
        "reproduced": metrics,
    }


def active_model_paper_strict_benchmark(
    examples: list[dict],
    vectorizer,
    classifier,
    *,
    temperature: float,
    class_thresholds: dict[str, float],
    path: str | Path = DEFAULT_PAPER_STRICT_DATA,
    strict_rows: list[dict] | None = None,
) -> dict:
    """Evaluate the active model on the fixed task/text-disjoint paper split.

    Every exposure check is computed from the actual text/task metadata. Any
    overlap fails closed instead of being reported as a hard-coded zero.  This
    split is leakage-free, but it has already been inspected during migration,
    so it is a regression benchmark rather than a sealed external test.
    """

    paper_rows = strict_rows if strict_rows is not None else read_json(Path(path))
    if not isinstance(paper_rows, list):
        raise ValueError("paper strict benchmark must be a JSON row list")
    for split in ("train", "dev", "test"):
        if not any(row.get("split") == split for row in paper_rows):
            raise ValueError(f"paper strict benchmark is missing split={split}")

    strict_groups = {
        split: {
            str(row.get("paper_task_uuid") or "")
            for row in paper_rows
            if row.get("split") == split and row.get("paper_task_uuid")
        }
        for split in ("train", "dev", "test")
    }
    strict_texts = {
        split: {
            normalize_text(row.get("text"))
            for row in paper_rows
            if row.get("split") == split and normalize_text(row.get("text"))
        }
        for split in ("train", "dev", "test")
    }
    internal_group_overlap = sum(
        len(strict_groups[left] & strict_groups[right])
        for index, left in enumerate(("train", "dev", "test"))
        for right in ("train", "dev", "test")[index + 1 :]
    )
    internal_text_overlap = sum(
        len(strict_texts[left] & strict_texts[right])
        for index, left in enumerate(("train", "dev", "test"))
        for right in ("train", "dev", "test")[index + 1 :]
    )
    if internal_group_overlap or internal_text_overlap:
        raise ValueError(
            "paper strict benchmark leaks internally: "
            f"groups={internal_group_overlap}, texts={internal_text_overlap}"
        )

    paper_test = [row for row in paper_rows if row.get("split") == "test"]
    test_ids = {str(row.get("feedback_id") or "") for row in paper_test}
    active_ids = {str(example.get("feedback_id") or "") for example in examples}
    id_overlap = sorted(test_ids & active_ids)
    active_texts = {
        normalize_text(example.get("text"))
        for example in examples
        if normalize_text(example.get("text"))
    }
    active_paper_tasks: set[str] = set()
    for example in examples:
        task_uuid = str(example.get("paper_task_uuid") or "").strip()
        group_id = str(example.get("group_id") or "")
        if not task_uuid and group_id.startswith("paper_task_uuid:"):
            task_uuid = group_id.split(":", 1)[1].strip()
        if task_uuid:
            active_paper_tasks.add(task_uuid)
    text_overlap = sorted(strict_texts["test"] & active_texts)
    task_overlap = sorted(strict_groups["test"] & active_paper_tasks)
    if id_overlap or text_overlap or task_overlap:
        raise ValueError(
            "active classifier data leaks into paper strict regression split: "
            f"ids={len(id_overlap)}, texts={len(text_overlap)}, "
            f"tasks={len(task_overlap)}"
        )

    texts = [str(row.get("text") or "") for row in paper_test]
    labels = [str(row["reference_type"]) for row in paper_test]
    metrics = _metrics(
        classifier,
        vectorizer.transform(texts),
        labels,
        temperature=temperature,
        class_thresholds=class_thresholds,
    )
    return {
        "protocol": "paper-task-text-disjoint-v1",
        "benchmark_role": "paper_strict_exposed_regression_not_for_selection",
        "source": str(path),
        "test_rows": len(paper_test),
        "test_task_uuids": len(strict_groups["test"]),
        "active_id_overlap_count": len(id_overlap),
        "active_normalized_text_overlap_count": len(text_overlap),
        "active_task_uuid_overlap_count": len(task_overlap),
        "internal_group_overlap_count": internal_group_overlap,
        "internal_normalized_text_overlap_count": internal_text_overlap,
        "metrics": metrics,
    }


def train_classifier(
    examples: list[dict],
    *,
    seed: int = 1,
    min_df: int = DEFAULT_OVERCOOKED_MIN_DF,
    tune_on_dev: bool = True,
) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression

    rows = build_phrase_rows(examples)
    if not rows:
        raise ValueError("no reference-type labelled phrases found")
    declared_splits = {
        str(row["split"]) for row in rows if row.get("split") in {"train", "dev", "test"}
    }
    if declared_splits:
        group_locations: dict[str, set[str]] = {}
        text_locations: dict[str, set[str]] = {}
        assigned: list[str] = []
        for row in rows:
            split_name = (
                str(row["split"])
                if row.get("split") in {"train", "dev", "test"}
                else "train"
            )
            assigned.append(split_name)
            if row.get("group_id") is not None:
                group_locations.setdefault(str(row["group_id"]), set()).add(split_name)
            normalized = normalize_text(row["text"])
            if normalized:
                text_locations.setdefault(normalized, set()).add(split_name)
        leaking_groups = [key for key, value in group_locations.items() if len(value) > 1]
        leaking_texts = [key for key, value in text_locations.items() if len(value) > 1]
        if leaking_groups or leaking_texts:
            raise ValueError(
                "declared classifier split leaks groups/text: "
                f"groups={len(leaking_groups)}, texts={len(leaking_texts)}"
            )
        split = {
            f"{name}_indices": [
                index for index, split_name in enumerate(assigned) if split_name == name
            ]
            for name in ("train", "dev", "test")
        }
        split.update(
            {
                f"{name}_groups": sorted(
                    {
                        str(rows[index]["group_id"])
                        for index in split[f"{name}_indices"]
                        if rows[index].get("group_id") is not None
                    }
                )
                for name in ("train", "dev", "test")
            }
        )
        split["policy"] = "generator_declared_split"
    else:
        split = make_train_dev_test_split(
            [
                None
                if row["group_id"] is None
                else str(row["group_id"])
                for row in rows
            ],
            seed=seed,
            dev_fraction=0.15,
            test_fraction=0.15,
        )
        split["policy"] = "scenario_grouped_legacy"
    train_rows = [rows[index] for index in split["train_indices"]]
    dev_rows = [rows[index] for index in split["dev_indices"]]
    test_rows = [rows[index] for index in split["test_indices"]]
    train_labels = [row["label"] for row in train_rows]
    if len(set(train_labels)) < 2:
        raise ValueError("reference classifier needs at least two train classes")

    feature_config = _overcooked_feature_config(min_df)
    vectorizer = _build_overcooked_vectorizer(min_df)
    train_matrix = vectorizer.fit_transform([row["text"] for row in train_rows])
    dev_matrix = vectorizer.transform([row["text"] for row in dev_rows])
    dev_labels = [row["label"] for row in dev_rows]

    def evaluate(classifier, rows_for_split: list[dict]) -> dict:
        if not rows_for_split:
            return {}
        matrix = vectorizer.transform([row["text"] for row in rows_for_split])
        return _metrics(
            classifier,
            matrix,
            [row["label"] for row in rows_for_split],
            temperature=temperature,
            class_thresholds=thresholds,
        )

    has_paper_style_train = any(
        row.get("source") in PAPER_STYLE_SOURCES for row in train_rows
    )
    has_hard_contrastive_train = any(
        row.get("origin") == HARD_CONTRASTIVE_ORIGIN for row in train_rows
    )
    candidates = (
        [
            {
                "C": c_value,
                "class_weight": "balanced",
                "paper_style_source_weight": paper_weight,
                "hard_contrastive_source_weight": hard_weight,
            }
            for c_value in (0.125, 0.25, 0.5)
            for paper_weight in ((8.0, 12.0, 16.0) if has_paper_style_train else (1.0,))
            for hard_weight in ((1.0, 2.0, 4.0) if has_hard_contrastive_train else (1.0,))
        ]
        if tune_on_dev
        else [
            {
                "C": 1.0,
                "class_weight": None,
                "paper_style_source_weight": 1.0,
                "hard_contrastive_source_weight": 1.0,
            }
        ]
    )

    def candidate_metrics(candidate, selected_rows: list[dict], selected_indices: list[int]) -> dict:
        if not selected_rows:
            return {}
        return _metrics(
            candidate,
            dev_matrix[selected_indices],
            [row["label"] for row in selected_rows],
        )

    paper_dev_indices = [
        index
        for index, row in enumerate(dev_rows)
        if row.get("source") in PAPER_STYLE_SOURCES
    ]
    hard_dev_indices = [
        index
        for index, row in enumerate(dev_rows)
        if row.get("origin") == HARD_CONTRASTIVE_ORIGIN
    ]
    legacy_synthetic_dev_indices = [
        index
        for index, row in enumerate(dev_rows)
        if row.get("source") not in PAPER_STYLE_SOURCES
        and row.get("origin") != HARD_CONTRASTIVE_ORIGIN
    ]
    fitted = []
    for config in candidates:
        candidate = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=2000,
            random_state=seed,
        )
        sample_weight = [
            (
                float(config["paper_style_source_weight"])
                if row.get("source") in PAPER_STYLE_SOURCES
                else (
                    float(config["hard_contrastive_source_weight"])
                    if row.get("origin") == HARD_CONTRASTIVE_ORIGIN
                    else 1.0
                )
            )
            for row in train_rows
        ]
        candidate.fit(train_matrix, train_labels, sample_weight=sample_weight)
        dev_metrics = (
            _metrics(candidate, dev_matrix, dev_labels) if dev_labels else {}
        )
        domain_metrics = {
            "paper_style": candidate_metrics(
                candidate,
                [dev_rows[index] for index in paper_dev_indices],
                paper_dev_indices,
            ),
            "hard_contrastive": candidate_metrics(
                candidate,
                [dev_rows[index] for index in hard_dev_indices],
                hard_dev_indices,
            ),
            "legacy_synthetic": candidate_metrics(
                candidate,
                [dev_rows[index] for index in legacy_synthetic_dev_indices],
                legacy_synthetic_dev_indices,
            ),
        }
        fitted.append((config, candidate, dev_metrics, domain_metrics))
    selected_config, classifier, selected_dev = max(
        fitted,
        key=lambda item: (
            item[2].get("macro_f1", float("-inf")),
            item[2].get("balanced_accuracy", float("-inf")),
            item[2].get("accuracy", float("-inf")),
            item[3]["paper_style"].get("macro_f1", float("-inf")),
        ),
    )[:3]
    temperature = _fit_temperature(classifier, dev_matrix, dev_labels)
    dev_probabilities = _calibrated_probabilities(classifier, dev_matrix, temperature)
    thresholds = _class_thresholds(classifier, dev_probabilities, dev_labels)
    selected_dev = _metrics(
        classifier,
        dev_matrix,
        dev_labels,
        temperature=temperature,
        class_thresholds=thresholds,
    ) if dev_labels else {}

    artifact = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "input_mode": OVERCOOKED_INPUT_MODE,
        "feature_config": feature_config,
        "reference_types": list(REFERENCE_TYPES),
        "temperature": temperature,
        "class_thresholds": thresholds,
        "top2_margin_threshold": 0.05,
        "target_selective_precision": 0.9,
    }
    report = {
        "benchmark_contract": {
            "synthetic_scope_cue_regression": "development regression; not external",
            "paper_protocol_reproduction": (
                "exact original random-row protocol; not external"
            ),
            "paper_strict_external_test": (
                "fixed leakage-free but already exposed regression; never used for selection"
            ),
        },
        "selection_protocol": {
            "partition": "declared_dev_only",
            "primary_metric": "overall_dev_macro_f1",
            "tie_breakers": [
                "overall_dev_balanced_accuracy",
                "overall_dev_accuracy",
                "paper_style_dev_macro_f1",
            ],
            "test_examples_evaluated_during_selection": 0,
            "paper_style_sample_weight_is_dev_selected": True,
        },
        "feature_config": feature_config,
        "split_policy": split["policy"],
        "split_groups": {
            name: split.get(f"{name}_groups", [])
            for name in ("train", "dev", "test")
        },
        "split_sizes": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
        },
        "train_classes": sorted(set(train_labels)),
        "annotation_sources": {
            source: sum(row["annotation_source"] == source for row in rows)
            for source in sorted({row["annotation_source"] for row in rows})
        },
        "class_thresholds": thresholds,
        "target_selective_precision": 0.9,
        "selected_hyperparameters": selected_config,
        "dev_candidates": [
            {"config": config, "metrics": metrics, "development_by_domain": domains}
            for config, _candidate, metrics, domains in fitted
        ],
        "dev": selected_dev,
        "development_by_domain": {
            "legacy_synthetic": evaluate(
                classifier,
                [
                    row
                    for row in dev_rows
                    if row.get("source") not in PAPER_STYLE_SOURCES
                    and row.get("origin") != HARD_CONTRASTIVE_ORIGIN
                ],
            ),
            "hard_contrastive": evaluate(
                classifier,
                [row for row in dev_rows if row.get("origin") == HARD_CONTRASTIVE_ORIGIN],
            ),
            "paper_style": evaluate(
                classifier,
                [row for row in dev_rows if row.get("source") in PAPER_STYLE_SOURCES],
            ),
        },
        "synthetic_scope_cue_regression": {
            "benchmark_role": "development_regression_not_external",
            "rows": sum(
                row.get("source") not in PAPER_STYLE_SOURCES
                and row.get("origin") != HARD_CONTRASTIVE_ORIGIN
                for row in test_rows
            ),
            "metrics": evaluate(
                classifier,
                [
                    row
                    for row in test_rows
                    if row.get("source") not in PAPER_STYLE_SOURCES
                    and row.get("origin") != HARD_CONTRASTIVE_ORIGIN
                ],
            ),
        },
        "hard_contrastive_regression": {
            "benchmark_role": "deterministic_challenge_regression_not_external",
            "rows": sum(row.get("origin") == HARD_CONTRASTIVE_ORIGIN for row in test_rows),
            "metrics": evaluate(
                classifier,
                [row for row in test_rows if row.get("origin") == HARD_CONTRASTIVE_ORIGIN],
            ),
        },
        "paper_strict_external_test": (
            active_model_paper_strict_benchmark(
                examples,
                vectorizer,
                classifier,
                temperature=temperature,
                class_thresholds=thresholds,
            )
        ),
        "paper_protocol_reproduction": reproduce_original_paper_benchmark(),
    }
    return artifact, report


def main() -> int:
    from joblib import dump

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--augment",
        type=Path,
        action="append",
        default=[],
        help="Additional labelled corpora used for classifier training only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_MODEL_PATH.with_suffix(".report.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-df", type=int, default=DEFAULT_OVERCOOKED_MIN_DF)
    parser.add_argument(
        "--no-dev-tuning",
        action="store_true",
        help="Use the paper default LogisticRegression without dev tuning.",
    )
    parser.add_argument(
        "--minimum-synthetic-regression-accuracy",
        "--minimum-test-accuracy",
        dest="minimum_synthetic_regression_accuracy",
        type=float,
        default=0.87,
        help=(
            "Regression gate for the already-used synthetic scope-cue set; "
            "this is not an external-test claim."
        ),
    )
    parser.add_argument(
        "--minimum-selective-accuracy",
        type=float,
        default=0.84,
        help=(
            "Fail if accepted predictions on the already-used legacy synthetic "
            "regression set fall below this accuracy."
        ),
    )
    args = parser.parse_args()

    raw = args.input.read_bytes()
    examples = json.loads(raw.decode("utf-8"))
    for augment_path in args.augment:
        examples.extend(read_json(augment_path))
    deduplicated = {}
    for example in examples:
        key = (
            str(example.get("group_id")),
            str(example.get("text") or "").strip().lower(),
            str(_reference_type(example)),
        )
        deduplicated.setdefault(key, example)
    examples = list(deduplicated.values())
    artifact, report = train_classifier(
        examples,
        seed=args.seed,
        min_df=args.min_df,
        tune_on_dev=not args.no_dev_tuning,
    )
    artifact["manifest"] = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "augment": [str(path) for path in args.augment],
        "augment_inputs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.augment
        ],
        "seed": args.seed,
        "min_df": args.min_df,
        "preprocessing": OVERCOOKED_INPUT_MODE,
        "feature_config": report["feature_config"],
        "selected_hyperparameters": report["selected_hyperparameters"],
        "paper_strict_data": str(DEFAULT_PAPER_STRICT_DATA),
        "paper_strict_data_sha256": hashlib.sha256(
            DEFAULT_PAPER_STRICT_DATA.read_bytes()
        ).hexdigest(),
    }
    report["artifact_manifest"] = artifact["manifest"]
    reproduced_accuracy = report["paper_protocol_reproduction"]["reproduced"][
        "accuracy"
    ]
    synthetic_metrics = report["synthetic_scope_cue_regression"]["metrics"]
    synthetic_accuracy = synthetic_metrics["accuracy"]
    strict_accuracy = report["paper_strict_external_test"]["metrics"]["accuracy"]
    selective_accuracy = synthetic_metrics["calibration"]["selective_accuracy"]
    if reproduced_accuracy < PAPER_REFERENCE_ACCURACY:
        raise RuntimeError(
            "failed to reproduce the paper classifier accuracy: "
            f"{reproduced_accuracy:.3f} < {PAPER_REFERENCE_ACCURACY:.3f}"
        )
    if synthetic_accuracy < args.minimum_synthetic_regression_accuracy:
        raise RuntimeError(
            "synthetic scope-cue regression accuracy is too low: "
            f"{synthetic_accuracy:.3f} < "
            f"{args.minimum_synthetic_regression_accuracy:.3f}"
        )
    if selective_accuracy is None or selective_accuracy < args.minimum_selective_accuracy:
        raise RuntimeError(
            "legacy-synthetic selective classifier accuracy is too low: "
            f"{selective_accuracy} < {args.minimum_selective_accuracy:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, args.output)
    write_json(args.report, report)
    print(f"Saved reference classifier: {args.output}")
    print(f"Synthetic scope-cue regression: {synthetic_metrics}")
    print(
        "Original-paper reproduction accuracy: "
        f"{reproduced_accuracy:.3f} (paper: {PAPER_REFERENCE_ACCURACY:.2f})"
    )
    print(
        "Paper strict exposed-regression accuracy (reported, never selected on): "
        f"{strict_accuracy:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
