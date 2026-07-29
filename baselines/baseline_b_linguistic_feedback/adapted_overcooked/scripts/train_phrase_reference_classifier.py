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


DEFAULT_INPUT = ROOT / "data" / "synthetic_feedback.validated.json"
ROLE_REFERENCE_TYPE = {
    "praise_best": "trajectory",
    "criticize_alt": "action_spatial",
    "command_best": "action_spatial",
    "describe_alt": "feature",
    "describe_behavior_alt": "action_behavioral",
    "other": "other",
}
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


def train_classifier(
    examples: list[dict],
    *,
    seed: int = 1,
    min_df: int = 5,
    tune_on_dev: bool = True,
) -> tuple[dict, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
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

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=min_df,
        ngram_range=(1, 2),
        stop_words="english",
    )
    train_matrix = vectorizer.fit_transform([row["processed"] for row in train_rows])
    dev_matrix = vectorizer.transform([row["processed"] for row in dev_rows])
    dev_labels = [row["label"] for row in dev_rows]

    def evaluate(classifier, rows_for_split: list[dict]) -> dict:
        if not rows_for_split:
            return {}
        matrix = vectorizer.transform([row["processed"] for row in rows_for_split])
        return _metrics(
            classifier,
            matrix,
            [row["label"] for row in rows_for_split],
            temperature=temperature,
            class_thresholds=thresholds,
        )

    candidates = (
        [
            {"C": c_value, "class_weight": class_weight}
            for c_value in (0.5, 1.0, 2.0, 4.0)
            for class_weight in (None, "balanced")
        ]
        if tune_on_dev
        else [{"C": 1.0, "class_weight": None}]
    )
    fitted = []
    for config in candidates:
        candidate = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=2000,
            random_state=seed,
        )
        candidate.fit(train_matrix, train_labels)
        dev_metrics = (
            _metrics(candidate, dev_matrix, dev_labels) if dev_labels else {}
        )
        fitted.append((config, candidate, dev_metrics))
    selected_config, classifier, selected_dev = max(
        fitted,
        key=lambda item: (
            item[2].get("macro_f1", float("-inf")),
            item[2].get("balanced_accuracy", float("-inf")),
            item[2].get("accuracy", float("-inf")),
        ),
    )
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
        "reference_types": list(REFERENCE_TYPES),
        "temperature": temperature,
        "class_thresholds": thresholds,
        "top2_margin_threshold": 0.05,
        "target_selective_precision": 0.9,
    }
    report = {
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
            {"config": config, "metrics": metrics}
            for config, _candidate, metrics in fitted
        ],
        "dev": selected_dev,
        "untouched_test": evaluate(classifier, test_rows),
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
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument(
        "--no-dev-tuning",
        action="store_true",
        help="Use the paper default LogisticRegression without dev tuning.",
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
        "seed": args.seed,
        "min_df": args.min_df,
        "preprocessing": "paper_preprocess_phrase",
        "selected_hyperparameters": report["selected_hyperparameters"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, args.output)
    write_json(args.report, report)
    print(f"Saved reference classifier: {args.output}")
    print(f"Untouched test: {report['untouched_test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
