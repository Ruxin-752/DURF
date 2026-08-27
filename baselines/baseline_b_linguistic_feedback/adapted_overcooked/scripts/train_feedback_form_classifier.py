"""Train the paper feedback-strategy classifier without touching test data.

For synthetic data, the three-way target is derived only from the paper-style
reference label: trajectory -> evaluative, feature -> descriptive, and
action_spatial -> imperative. ``action_behavioral`` and ``other`` are excluded,
matching the original notebook's three-way analysis. The generator's
cross-product ``expected_feedback_type`` field is not a target. Human rows keep
their explicit three-class annotation. Role strings and model predictions are
never converted into labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    FEEDBACK_TYPES,
)


DEFAULT_SYNTHETIC_INPUT = ROOT / "data" / "reference_classifier_feedback.v8.json"
DEFAULT_HUMAN_TRAIN = ROOT / "data" / "human_feedback_form_train.json"
DEFAULT_HUMAN_DEV = ROOT / "data" / "human_feedback_form_dev.json"
DEFAULT_HUMAN_AUGMENTATION = (
    ROOT / "data" / "human_feedback_form_train_augmentation.json"
)
# Keep the paper reference-collapse model on its original mapping corpus.  The
# v2 surface-speech hard set is evaluated separately for the UI classifier and
# must not silently change this paper-aligned target.
DEFAULT_HARD_TRAIN = ROOT / "data" / "feedback_form_hard_train.v1.json"
DEFAULT_PAPER_HUMAN_TRAIN = ROOT / "data" / "paper_feedback_form_human_train.v1.json"
DEFAULT_PAPER_HUMAN_DEV = ROOT / "data" / "paper_feedback_form_human_dev.v1.json"
DEFAULT_PAPER_BENCHMARK_MANIFEST = (
    ROOT
    / "outputs"
    / "feedback_form_classifier"
    / "paper_human_benchmark.v1.manifest.json"
)
FEATURE_VERSION = "feedback-form-raw-word-char-tfidf-v3"
PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE = {
    "trajectory": "evaluative",
    # The paper appendix says the main experiment treated object behavior as
    # Evaluative, although the released strict three-way notebook excluded it.
    "action_behavioral": "evaluative",
    "feature": "descriptive",
    "action_spatial": "imperative",
}
# Synthetic behavior examples were generated as commands under a different
# feedback-form convention.  They remain excluded rather than being relabeled
# against their generation contract.  The paper-main-experiment behavioral
# mapping is learned from the original-paper human rows below.
SYNTHETIC_REFERENCE_TYPE_TO_FEEDBACK_TYPE = {
    key: value
    for key, value in PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE.items()
    if key != "action_behavioral"
}
EXCLUDED_SYNTHETIC_REFERENCE_TYPES = ("action_behavioral", "other")
SYNTHETIC_ORIGIN_TO_SOURCE = {
    "deepseek_reference_v7": "deepseek_reference_v7",
    "deterministic_contrastive_v1": "deterministic_contrastive",
}
COMPLETE_DEV_SOURCES = (
    "deepseek_reference_v7",
    "deterministic_contrastive",
    "original_paper_human",
)
SOURCE_WEIGHT_GRID = {
    "deterministic_contrastive": (8.0, 16.0),
    "original_paper_human": (4.0, 8.0, 16.0, 32.0),
}
FIXED_SOURCE_WEIGHTS = {
    "deepseek_reference_v7": 1.0,
    "feedback_form_hard_train": 8.0,
    "human": 1.0,
    "human_augmentation": 0.5,
}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _read_json_list(path: Path, *, missing_ok: bool = False) -> list[dict]:
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _text_digest(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _make_row(
    raw: dict,
    *,
    split: str,
    source: str,
    index: int,
    label: str,
) -> dict:
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source} {split} row {index}: missing non-empty text")
    if label not in FEEDBACK_TYPES:
        raise ValueError(
            f"{source} {split} row {index}: target must be one of {list(FEEDBACK_TYPES)}"
        )
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError(f"{source} {split} row {index}: text is empty after normalization")
    if source == "human":
        expected_canonical = CANONICAL_FEEDBACK_LABELS[str(label)]
        if raw.get("classification_label") != expected_canonical:
            raise ValueError(
                f"human {split} row {index}: canonical classification_label must be "
                f"{expected_canonical!r}"
            )
        if raw.get("label_source") != "human_explicit":
            raise ValueError(
                f"human {split} row {index}: label_source must be 'human_explicit'"
            )
    elif source == "human_augmentation":
        expected_canonical = CANONICAL_FEEDBACK_LABELS[str(label)]
        if split != "train":
            raise ValueError("human augmentation is training-only")
        if raw.get("classification_label") != expected_canonical:
            raise ValueError(
                f"human augmentation row {index}: canonical label mismatch"
            )
        if raw.get("source") != "human_train_augmentation":
            raise ValueError(
                f"human augmentation row {index}: invalid source provenance"
            )
        if raw.get("label_source") != "deterministic_label_preserving_transform":
            raise ValueError(
                f"human augmentation row {index}: invalid label provenance"
            )
        if not isinstance(raw.get("parent_feedback_id"), str) or not raw[
            "parent_feedback_id"
        ].strip():
            raise ValueError(
                f"human augmentation row {index}: missing parent_feedback_id"
            )
    elif source == "feedback_form_hard_train":
        expected_canonical = CANONICAL_FEEDBACK_LABELS[str(label)]
        if split != "train":
            raise ValueError("feedback-form hard expansion is training-only")
        if raw.get("classification_label") != expected_canonical:
            raise ValueError(f"hard expansion row {index}: canonical label mismatch")
        if raw.get("source") != "feedback_form_hard_train":
            raise ValueError(f"hard expansion row {index}: invalid source provenance")
        if raw.get("label_source") != "paper_mapping_contrastive_train_only":
            raise ValueError(
                f"hard expansion row {index}: invalid label provenance"
            )
    elif source == "original_paper_human":
        expected_canonical = CANONICAL_FEEDBACK_LABELS[str(label)]
        if raw.get("classification_label") != expected_canonical:
            raise ValueError(
                f"original-paper human {split} row {index}: canonical label mismatch"
            )
        reference_type = str(raw.get("reference_type") or "")
        expected_label = PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE.get(reference_type)
        if expected_label != label:
            raise ValueError(
                f"original-paper human {split} row {index}: reference mapping mismatch"
            )
        if raw.get("label_source") != (
            "original_paper_human_reference_main_experiment_mapping"
        ):
            raise ValueError(
                f"original-paper human {split} row {index}: invalid label provenance"
            )
    group_id = (
        raw.get("group_id")
        or raw.get("paraphrase_family")
        or raw.get("feedback_id")
        or f"{source}:{_text_digest(normalized)}"
    )
    return {
        "text": text.strip(),
        "normalized_text": normalized,
        "label": label,
        "split": split,
        "source": source,
        "group_id": f"{source}:{group_id}",
        "origin": raw.get("classifier_corpus_origin") or source,
        "template_family": raw.get("template_family"),
        "paper_task_uuid": raw.get("paper_task_uuid"),
    }


def load_training_rows(
    synthetic_path: Path,
    human_train_path: Path,
    human_dev_path: Path,
    human_augmentation_path: Path | None = None,
    hard_train_path: Path | None = None,
    paper_human_train_path: Path | None = None,
    paper_human_dev_path: Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Load explicit train/dev labels; test rows are skipped before label access."""

    synthetic_raw = _read_json_list(synthetic_path)
    rows_by_split: dict[str, list[dict]] = {"train": [], "dev": []}
    synthetic_test_rows_skipped = 0
    embedded_paper_rows_skipped = 0
    synthetic_excluded_reference_counts = {
        split: Counter({label: 0 for label in EXCLUDED_SYNTHETIC_REFERENCE_TYPES})
        for split in ("train", "dev")
    }
    synthetic_reference_counts = {"train": Counter(), "dev": Counter()}
    for index, raw in enumerate(synthetic_raw, start=1):
        split = raw.get("split")
        if split not in {"train", "dev"}:
            if split == "test":
                synthetic_test_rows_skipped += 1
            continue
        if (
            paper_human_train_path is not None
            and raw.get("classifier_corpus_origin")
            == "original_paper_strict_development"
        ):
            # A separately pinned source is loaded below.  Keeping the embedded
            # copy would make it harder to prove the exact human split/hash.
            embedded_paper_rows_skipped += 1
            continue
        reference_type = raw.get("reference_type")
        if reference_type in EXCLUDED_SYNTHETIC_REFERENCE_TYPES:
            synthetic_excluded_reference_counts[str(split)][str(reference_type)] += 1
            continue
        if reference_type not in SYNTHETIC_REFERENCE_TYPE_TO_FEEDBACK_TYPE:
            raise ValueError(
                f"synthetic {split} row {index}: reference_type must be one of "
                f"{sorted((*SYNTHETIC_REFERENCE_TYPE_TO_FEEDBACK_TYPE, *EXCLUDED_SYNTHETIC_REFERENCE_TYPES))}"
            )
        synthetic_reference_counts[str(split)][str(reference_type)] += 1
        origin = raw.get("classifier_corpus_origin")
        source = SYNTHETIC_ORIGIN_TO_SOURCE.get(str(origin), "synthetic_other")
        rows_by_split[str(split)].append(
            _make_row(
                raw,
                split=str(split),
                source=source,
                index=index,
                label=SYNTHETIC_REFERENCE_TYPE_TO_FEEDBACK_TYPE[str(reference_type)],
            )
        )

    for split, path in (("train", human_train_path), ("dev", human_dev_path)):
        for index, raw in enumerate(_read_json_list(path, missing_ok=True), start=1):
            if raw.get("split") != split:
                raise ValueError(
                    f"{path}: row {index} declares split {raw.get('split')!r}, expected {split!r}"
                )
            explicit_label = raw.get("expected_feedback_type")
            rows_by_split[split].append(
                _make_row(
                    raw,
                    split=split,
                    source="human",
                    index=index,
                    label=str(explicit_label),
                )
            )

    if human_augmentation_path is not None:
        for index, raw in enumerate(
            _read_json_list(human_augmentation_path, missing_ok=True), start=1
        ):
            if raw.get("split") != "train":
                raise ValueError(
                    f"{human_augmentation_path}: row {index} must declare split='train'"
                )
            rows_by_split["train"].append(
                _make_row(
                    raw,
                    split="train",
                    source="human_augmentation",
                    index=index,
                    label=str(raw.get("expected_feedback_type")),
                )
            )

    if hard_train_path is not None:
        for index, raw in enumerate(
            _read_json_list(hard_train_path, missing_ok=True), start=1
        ):
            if raw.get("split") != "train":
                raise ValueError(
                    f"{hard_train_path}: row {index} must declare split='train'"
                )
            rows_by_split["train"].append(
                _make_row(
                    raw,
                    split="train",
                    source="feedback_form_hard_train",
                    index=index,
                    label=str(raw.get("expected_feedback_type")),
                )
            )

    paper_paths = {
        "train": paper_human_train_path,
        "dev": paper_human_dev_path,
    }
    if any(path is not None for path in paper_paths.values()) and not all(
        path is not None for path in paper_paths.values()
    ):
        raise ValueError("original-paper human train and dev paths must be supplied together")
    paper_human_counts = Counter()
    for split, path in paper_paths.items():
        if path is None:
            continue
        for index, raw in enumerate(_read_json_list(path), start=1):
            if raw.get("split") != split:
                raise ValueError(
                    f"{path}: row {index} declares split {raw.get('split')!r}, expected {split!r}"
                )
            reference_type = str(raw.get("reference_type") or "")
            mapped_label = PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE.get(reference_type)
            if mapped_label is None:
                raise ValueError(
                    f"{path}: row {index} has unmapped reference_type {reference_type!r}"
                )
            if raw.get("expected_feedback_type") != mapped_label:
                raise ValueError(f"{path}: row {index} has inconsistent collapsed label")
            rows_by_split[split].append(
                _make_row(
                    raw,
                    split=split,
                    source="original_paper_human",
                    index=index,
                    label=mapped_label,
                )
            )
            paper_human_counts[(split, reference_type)] += 1

    duplicate_counts: dict[str, int] = {}
    synthetic_conflict_text_counts: dict[str, int] = {}
    synthetic_conflict_row_counts: dict[str, int] = {}
    human_overrides_synthetic_counts: dict[str, int] = {}
    for split in ("train", "dev"):
        # Explicit human labels take precedence over an exact synthetic copy.
        # A human-human conflict remains fatal.  Synthetic-only exact conflicts
        # are discarded as ambiguous instead of silently choosing a target.
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows_by_split[split]:
            grouped[row["normalized_text"]].append(row)
        unique: dict[str, dict] = {}
        dropped = 0
        synthetic_conflict_texts = 0
        synthetic_conflict_rows = 0
        human_overrides = 0
        for key, same_text_rows in grouped.items():
            human_rows = [
                row
                for row in same_text_rows
                if row["source"] in {"human", "original_paper_human"}
            ]
            if human_rows:
                human_labels = {row["label"] for row in human_rows}
                if len(human_labels) != 1:
                    raise ValueError(
                        f"{split}: normalized human text {key!r} has conflicting labels "
                        f"{sorted(human_labels)}"
                    )
                unique[key] = human_rows[0]
                non_human_rows = [
                    row
                    for row in same_text_rows
                    if row["source"] not in {"human", "original_paper_human"}
                ]
                human_overrides += len(non_human_rows)
                dropped += len(same_text_rows) - 1
                continue
            labels = {row["label"] for row in same_text_rows}
            if len(labels) > 1:
                synthetic_conflict_texts += 1
                synthetic_conflict_rows += len(same_text_rows)
                dropped += len(same_text_rows)
                continue
            unique[key] = same_text_rows[0]
            dropped += len(same_text_rows) - 1
        rows_by_split[split] = list(unique.values())
        duplicate_counts[split] = dropped
        synthetic_conflict_text_counts[split] = synthetic_conflict_texts
        synthetic_conflict_row_counts[split] = synthetic_conflict_rows
        human_overrides_synthetic_counts[split] = human_overrides

    train_texts = {row["normalized_text"] for row in rows_by_split["train"]}
    dev_texts = {row["normalized_text"] for row in rows_by_split["dev"]}
    text_overlap = train_texts & dev_texts
    if text_overlap:
        raise ValueError(f"train/dev normalized-text leakage: {len(text_overlap)} rows")
    train_groups = {row["group_id"] for row in rows_by_split["train"]}
    dev_groups = {row["group_id"] for row in rows_by_split["dev"]}
    group_overlap = train_groups & dev_groups
    if group_overlap:
        raise ValueError(f"train/dev group leakage: {len(group_overlap)} groups")

    audit = {
        "synthetic_test_rows_skipped_without_label_use": synthetic_test_rows_skipped,
        "test_examples_evaluated_during_selection": 0,
        "duplicate_same_label_rows_dropped": duplicate_counts,
        "synthetic_cross_label_texts_dropped": synthetic_conflict_text_counts,
        "synthetic_cross_label_rows_dropped": synthetic_conflict_row_counts,
        "exact_synthetic_rows_overridden_by_explicit_human": human_overrides_synthetic_counts,
        "train_dev_normalized_text_overlap": 0,
        "train_dev_group_overlap": 0,
        "synthetic_target_source": "reference_type_fixed_mapping",
        "synthetic_reference_type_mapping": dict(
            SYNTHETIC_REFERENCE_TYPE_TO_FEEDBACK_TYPE
        ),
        "paper_human_reference_type_mapping": dict(
            PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE
        ),
        "paper_human_reference_counts": {
            split: {
                reference_type: paper_human_counts[(split, reference_type)]
                for reference_type in PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE
            }
            for split in ("train", "dev")
        },
        "embedded_paper_rows_skipped_in_favor_of_pinned_source": (
            embedded_paper_rows_skipped
        ),
        "synthetic_excluded_reference_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in synthetic_excluded_reference_counts.items()
        },
        "synthetic_reference_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in synthetic_reference_counts.items()
        },
        "synthetic_expected_feedback_type_used_as_target": False,
        "human_target_source": "explicit_classification_label",
        "human_augmentation_target_source": (
            "deterministic_label_preserving_transform_from_train_only"
        ),
        "human_augmentation_dev_or_test_rows": 0,
        "hard_contrast_target_source": "paper_mapping_contrastive_train_only",
        "hard_contrast_dev_or_test_rows": 0,
        "labels_inferred_from_role_or_predictions": False,
    }
    return rows_by_split["train"], rows_by_split["dev"], audit


def bind_frozen_benchmark_without_loading_test(
    manifest_path: Path,
    paper_train_path: Path,
    paper_dev_path: Path,
    train_rows: list[dict],
    dev_rows: list[dict],
) -> dict:
    """Verify train/dev and test-membership hashes without opening test data."""

    manifest = _read_json_object(manifest_path)
    outputs = manifest.get("outputs")
    membership = manifest.get("frozen_test_membership")
    if not isinstance(outputs, dict) or not isinstance(membership, dict):
        raise ValueError("paper benchmark manifest is missing outputs or membership")
    for name, path in (("train", paper_train_path), ("dev", paper_dev_path)):
        entry = outputs.get(name)
        if not isinstance(entry, dict) or entry.get("sha256") != _sha256(path):
            raise ValueError(f"paper benchmark {name} hash mismatch")
    frozen_entry = outputs.get("frozen_test")
    if not isinstance(frozen_entry, dict) or not frozen_entry.get("sha256"):
        raise ValueError("paper benchmark manifest has no frozen-test hash")
    frozen_text_hashes = set(membership.get("normalized_text_sha256") or [])
    frozen_group_hashes = set(membership.get("paper_task_uuid_sha256") or [])
    if not frozen_text_hashes or not frozen_group_hashes:
        raise ValueError("paper benchmark manifest has empty frozen-test membership")
    development = [*train_rows, *dev_rows]
    text_overlap = {
        _text_digest(row["normalized_text"])
        for row in development
        if _text_digest(row["normalized_text"]) in frozen_text_hashes
    }
    group_overlap = {
        _text_digest(str(row["paper_task_uuid"]))
        for row in development
        if row.get("paper_task_uuid")
        and _text_digest(str(row["paper_task_uuid"])) in frozen_group_hashes
    }
    if text_overlap or group_overlap:
        raise ValueError(
            "frozen paper benchmark leaked into development: "
            f"texts={len(text_overlap)}, groups={len(group_overlap)}"
        )
    return {
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": _sha256(manifest_path),
        "benchmark_version": manifest.get("benchmark_version"),
        "mapping_version": manifest.get("mapping_version"),
        "paper_train_sha256_verified": _sha256(paper_train_path),
        "paper_dev_sha256_verified": _sha256(paper_dev_path),
        "frozen_test_path_bound_not_opened": frozen_entry.get("path"),
        "frozen_test_sha256_bound_not_opened": frozen_entry.get("sha256"),
        "frozen_test_normalized_text_overlap": 0,
        "frozen_test_task_overlap": 0,
        "frozen_test_loaded": False,
    }


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


def classification_metrics(classifier, matrix, labels: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    if not labels:
        return {}
    predictions = classifier.predict(matrix)
    report = classification_report(
        labels,
        predictions,
        labels=list(FEEDBACK_TYPES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(FEEDBACK_TYPES),
                average="macro",
                zero_division=0,
            )
        ),
        "label_support": dict(sorted(Counter(labels).items())),
        "all_labels_present": set(labels) == set(FEEDBACK_TYPES),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in FEEDBACK_TYPES
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(FEEDBACK_TYPES)
        ).tolist(),
        "label_order": list(FEEDBACK_TYPES),
    }


def temperature_scaled_probabilities(classifier, matrix, temperature: float):
    """Temperature-scale deployed probabilities without changing their argmax."""

    import numpy as np

    if not temperature > 0:
        raise ValueError("temperature must be positive")
    raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(classifier.classes_):
        raise ValueError("temperature scaling requires multiclass probabilities")
    # Scaling log probabilities is identical to logit temperature scaling for
    # multinomial LR and remains internally consistent if sklearn's probability
    # implementation changes.
    scaled = np.log(np.clip(raw, 1e-12, 1.0)) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def probability_metrics(probabilities, labels: list[str], classes) -> dict:
    """Measure multiclass calibration using a fixed ten-bin ECE."""

    import numpy as np

    values = np.asarray(probabilities, dtype=float)
    class_names = [str(value) for value in classes]
    class_to_index = {label: index for index, label in enumerate(class_names)}
    targets = np.asarray([class_to_index[label] for label in labels], dtype=int)
    one_hot = np.eye(len(class_names), dtype=float)[targets]
    clipped = np.clip(values, 1e-12, 1.0)
    predictions = values.argmax(axis=1)
    confidence = values.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "rows": len(labels),
        "negative_log_likelihood": float(
            -np.log(clipped[np.arange(len(targets)), targets]).mean()
        ),
        "multiclass_brier": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "ece_10_bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "accuracy_from_probabilities": float(correct.mean()),
    }


def fit_temperature(classifier, matrix, labels: list[str]) -> tuple[float, dict]:
    """Fit one scalar on dev only; the frozen test remains unopened."""

    import numpy as np

    raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
    raw_metrics = probability_metrics(raw, labels, classifier.classes_)
    candidates = np.geomspace(0.05, 20.0, 601)
    best_temperature = 1.0
    best_probabilities = temperature_scaled_probabilities(classifier, matrix, 1.0)
    best_metrics = probability_metrics(best_probabilities, labels, classifier.classes_)
    for candidate in candidates:
        probabilities = temperature_scaled_probabilities(
            classifier, matrix, float(candidate)
        )
        metrics = probability_metrics(probabilities, labels, classifier.classes_)
        if (
            metrics["negative_log_likelihood"],
            abs(float(candidate) - 1.0),
        ) < (
            best_metrics["negative_log_likelihood"],
            abs(best_temperature - 1.0),
        ):
            best_temperature = float(candidate)
            best_probabilities = probabilities
            best_metrics = metrics
    return best_temperature, {
        "method": "temperature_scaling",
        "version": "feedback-form-temperature-dev-v1",
        "fit_split": "dev_only",
        "candidate_temperature_count": len(candidates),
        "raw_predict_proba": raw_metrics,
        "calibrated": best_metrics,
    }


def train_feedback_form_classifier(
    train_rows: list[dict],
    dev_rows: list[dict],
    *,
    seed: int = 1,
    min_df: int = 2,
) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression

    if not train_rows or not dev_rows:
        raise ValueError("feedback-form training requires non-empty train and dev rows")
    train_labels = [row["label"] for row in train_rows]
    dev_labels = [row["label"] for row in dev_rows]
    missing_train = sorted(set(FEEDBACK_TYPES) - set(train_labels))
    missing_dev = sorted(set(FEEDBACK_TYPES) - set(dev_labels))
    if missing_train or missing_dev:
        raise ValueError(
            f"all three labels are required; missing_train={missing_train}, missing_dev={missing_dev}"
        )

    vectorizer = _build_vectorizer(min_df)
    train_matrix = vectorizer.fit_transform([row["text"] for row in train_rows])
    dev_matrix = vectorizer.transform([row["text"] for row in dev_rows])
    dev_sources = sorted({row["source"] for row in dev_rows})
    dev_rows_by_source = {
        source: [row for row in dev_rows if row["source"] == source]
        for source in dev_sources
    }
    dev_matrices_by_source = {
        source: vectorizer.transform([row["text"] for row in rows])
        for source, rows in dev_rows_by_source.items()
    }

    def metrics_by_source(classifier) -> dict[str, dict]:
        return {
            source: classification_metrics(
                classifier,
                dev_matrices_by_source[source],
                [row["label"] for row in rows],
            )
            for source, rows in dev_rows_by_source.items()
        }

    def selection_summary(overall: dict, by_source: dict[str, dict]) -> dict:
        complete = [
            by_source[source]["macro_f1"]
            for source in COMPLETE_DEV_SOURCES
            if source in by_source and by_source[source]["all_labels_present"]
        ]
        if not complete:
            complete = [overall["macro_f1"]]
        harmonic = len(complete) / sum(1.0 / max(value, 1e-12) for value in complete)
        return {
            "complete_dev_sources": [
                source
                for source in COMPLETE_DEV_SOURCES
                if source in by_source and by_source[source]["all_labels_present"]
            ],
            "minimum_complete_source_macro_f1": min(complete),
            "harmonic_mean_complete_source_macro_f1": harmonic,
        }

    candidates: list[tuple[dict, object, dict, dict, dict]] = []
    for c_value in (0.05, 0.1, 0.25, 0.5, 1.0):
        for class_weight in (None, "balanced"):
            for contrast_weight in SOURCE_WEIGHT_GRID["deterministic_contrastive"]:
                for paper_weight in SOURCE_WEIGHT_GRID["original_paper_human"]:
                    source_weights = {
                        **FIXED_SOURCE_WEIGHTS,
                        "deterministic_contrastive": contrast_weight,
                        "original_paper_human": paper_weight,
                    }
                    sample_weights = [
                        float(source_weights.get(row["source"], 1.0))
                        for row in train_rows
                    ]
                    classifier = LogisticRegression(
                        C=c_value,
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=seed,
                    )
                    classifier.fit(
                        train_matrix, train_labels, sample_weight=sample_weights
                    )
                    metrics = classification_metrics(
                        classifier, dev_matrix, dev_labels
                    )
                    by_source = metrics_by_source(classifier)
                    selection = selection_summary(metrics, by_source)
                    config = {
                        "C": c_value,
                        "class_weight": class_weight,
                        "source_sample_weights": source_weights,
                    }
                    candidates.append(
                        (config, classifier, metrics, by_source, selection)
                    )

    selected = max(
        candidates,
        key=lambda item: (
            item[3].get("original_paper_human", {}).get("macro_f1", -1.0),
            item[3]
            .get("original_paper_human", {})
            .get("balanced_accuracy", -1.0),
            item[3].get("original_paper_human", {}).get("accuracy", -1.0),
            item[4]["minimum_complete_source_macro_f1"],
            item[4]["harmonic_mean_complete_source_macro_f1"],
            item[2]["macro_f1"],
            item[2]["balanced_accuracy"],
            item[2]["accuracy"],
            -float(item[0]["C"]),
            item[0]["class_weight"] is None,
        ),
    )
    (
        selected_config,
        selected_classifier,
        selected_metrics,
        selected_metrics_by_source,
        selected_selection_summary,
    ) = selected
    calibration_source = (
        "original_paper_human"
        if "original_paper_human" in dev_rows_by_source
        else "all_dev_fallback"
    )
    calibration_rows = (
        dev_rows_by_source["original_paper_human"]
        if calibration_source == "original_paper_human"
        else dev_rows
    )
    calibration_matrix = (
        dev_matrices_by_source["original_paper_human"]
        if calibration_source == "original_paper_human"
        else dev_matrix
    )
    temperature, calibration_report = fit_temperature(
        selected_classifier,
        calibration_matrix,
        [row["label"] for row in calibration_rows],
    )
    calibration_report["fit_source"] = calibration_source
    selected_metrics["calibration"] = calibration_report

    membership = {
        "train_normalized_text_sha256": sorted(
            _text_digest(row["normalized_text"]) for row in train_rows
        ),
        "dev_normalized_text_sha256": sorted(
            _text_digest(row["normalized_text"]) for row in dev_rows
        ),
        "train_group_sha256": sorted(
            _text_digest(str(row["group_id"])) for row in train_rows
        ),
        "dev_group_sha256": sorted(
            _text_digest(str(row["group_id"])) for row in dev_rows
        ),
        "human_train_template_family_sha256": sorted(
            {
                _text_digest(str(row["template_family"]))
                for row in train_rows
                if row["source"] in {"human", "human_augmentation"}
                and row.get("template_family")
            }
        ),
        "human_dev_template_family_sha256": sorted(
            {
                _text_digest(str(row["template_family"]))
                for row in dev_rows
                if row["source"] == "human" and row.get("template_family")
            }
        ),
    }
    artifact = {
        "model_type": "feedback_form_tfidf_logistic_regression",
        "model_version": 3,
        "input_mode": "raw_text",
        "labels": list(FEEDBACK_TYPES),
        "canonical_labels": dict(CANONICAL_FEEDBACK_LABELS),
        "vectorizer": vectorizer,
        "classifier": selected_classifier,
        "selected_hyperparameters": selected_config,
        "minimum_model_confidence": DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
        "probability_calibration": {
            "method": "temperature_scaling",
            "temperature": temperature,
            "version": "feedback-form-temperature-dev-v1",
            "fit_split": (
                "original_paper_human_dev_only"
                if calibration_source == "original_paper_human"
                else "all_dev_fallback"
            ),
            "fit_split_used_for_model_selection": True,
            "independently_validated": False,
        },
        "reference_type_mapping": dict(PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE),
        "mapping_version": (
            "paper-main-experiment-reference-collapse-v1"
        ),
        "data_membership": membership,
    }
    report = {
        "target": "paper_feedback_strategy/main_experiment_reference_collapsed",
        "paper_main_experiment_reference_type_mapping": dict(
            PAPER_REFERENCE_TYPE_TO_FEEDBACK_TYPE
        ),
        "mapping_claim": {
            "strict_original_notebook_excludes": ["action_behavioral", "other"],
            "paper_main_experiment_mapping": {
                "action_behavioral": "evaluative"
            },
            "paper_notebook_exact": False,
            "paper_main_experiment_exact": True,
        },
        "canonical_labels": list(CANONICAL_FEEDBACK_LABELS.values()),
        "internal_labels": list(FEEDBACK_TYPES),
        "synthetic_target_is_reference_collapsed": True,
        "generator_cross_product_expected_feedback_type_used_as_target": False,
        "selection_protocol": {
            "partition": "train_and_dev_only",
            "primary_metric": "original_paper_human_dev_macro_f1",
            "tie_breakers": [
                "original_paper_human_dev_balanced_accuracy",
                "original_paper_human_dev_accuracy",
                "minimum_complete_source_dev_macro_f1",
                "harmonic_mean_complete_source_dev_macro_f1",
                "overall_dev_macro_f1",
                "overall_dev_balanced_accuracy",
                "overall_dev_accuracy",
            ],
            "human_dev_used_for_selection": True,
            "human_dev_selection_role": (
                "included only through late overall-dev tie-breakers; not primary"
            ),
            "original_paper_human_dev_used_for_selection": True,
            "human_dev_limit": (
                "the current small family-disjoint human dev does not contain all labels"
            ),
            "test_examples_evaluated_during_selection": 0,
            "final_human_test_used_for_selection": False,
        },
        "feature_config": {
            "version": FEATURE_VERSION,
            "word_ngram_range": [1, 3],
            "char_wb_ngram_range": [2, 5],
            "min_df": min_df,
        },
        "split_sizes": {"train": len(train_rows), "dev": len(dev_rows)},
        "split_label_counts": {
            "train": dict(sorted(Counter(train_labels).items())),
            "dev": dict(sorted(Counter(dev_labels).items())),
        },
        "source_counts": {
            split: dict(sorted(Counter(row["source"] for row in rows).items()))
            for split, rows in (("train", train_rows), ("dev", dev_rows))
        },
        "selected_hyperparameters": selected_config,
        "selected_source_balanced_dev": selected_selection_summary,
        "runtime_low_confidence_fallback": {
            "threshold": DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
            "deployment_behavior": "model_label_preserved_with_abstained_true",
            "fallback_rule_role": "audit_only",
            "threshold_selected_on_calibration_data": False,
            "model_probabilities_and_confidence_preserved": True,
        },
        "dev": selected_metrics,
        "development_by_source": selected_metrics_by_source,
        "dev_candidates": [
            {
                "config": config,
                "metrics": metrics,
                "development_by_source": by_source,
                "source_balanced_selection": selection,
            }
            for config, _classifier, metrics, by_source, selection in candidates
        ],
        "test_metrics": None,
    }
    return artifact, report


def main() -> int:
    from joblib import dump

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-input", type=Path, default=DEFAULT_SYNTHETIC_INPUT)
    parser.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN)
    parser.add_argument("--human-dev", type=Path, default=DEFAULT_HUMAN_DEV)
    parser.add_argument(
        "--human-augmentation",
        type=Path,
        default=DEFAULT_HUMAN_AUGMENTATION,
    )
    parser.add_argument("--hard-train", type=Path, default=DEFAULT_HARD_TRAIN)
    parser.add_argument(
        "--paper-human-train", type=Path, default=DEFAULT_PAPER_HUMAN_TRAIN
    )
    parser.add_argument(
        "--paper-human-dev", type=Path, default=DEFAULT_PAPER_HUMAN_DEV
    )
    parser.add_argument(
        "--paper-benchmark-manifest",
        type=Path,
        default=DEFAULT_PAPER_BENCHMARK_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_MODEL_PATH.with_suffix(".report.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    train_rows, dev_rows, audit = load_training_rows(
        args.synthetic_input,
        args.human_train,
        args.human_dev,
        args.human_augmentation,
        args.hard_train,
        args.paper_human_train,
        args.paper_human_dev,
    )
    frozen_binding = bind_frozen_benchmark_without_loading_test(
        args.paper_benchmark_manifest,
        args.paper_human_train,
        args.paper_human_dev,
        train_rows,
        dev_rows,
    )
    artifact, report = train_feedback_form_classifier(
        train_rows, dev_rows, seed=args.seed, min_df=args.min_df
    )
    manifest = {
        "synthetic_input": str(args.synthetic_input),
        "synthetic_input_sha256": _sha256(args.synthetic_input),
        "human_train": str(args.human_train),
        "human_train_sha256": _sha256(args.human_train),
        "human_dev": str(args.human_dev),
        "human_dev_sha256": _sha256(args.human_dev),
        "human_augmentation": str(args.human_augmentation),
        "human_augmentation_sha256": _sha256(args.human_augmentation),
        "hard_train": str(args.hard_train),
        "hard_train_sha256": _sha256(args.hard_train),
        "paper_human_train": str(args.paper_human_train),
        "paper_human_train_sha256": _sha256(args.paper_human_train),
        "paper_human_dev": str(args.paper_human_dev),
        "paper_human_dev_sha256": _sha256(args.paper_human_dev),
        "paper_benchmark_manifest": str(args.paper_benchmark_manifest),
        "paper_benchmark_manifest_sha256": _sha256(args.paper_benchmark_manifest),
        "frozen_test_sha256_bound_not_loaded": frozen_binding[
            "frozen_test_sha256_bound_not_opened"
        ],
        "frozen_test_loaded": False,
        "human_test_loaded": False,
        "seed": args.seed,
        "min_df": args.min_df,
    }
    artifact["manifest"] = manifest
    report["artifact_manifest"] = manifest
    artifact["frozen_benchmark_binding"] = frozen_binding
    report["frozen_benchmark_binding"] = frozen_binding
    audit.update(
        {
            "frozen_test_loaded": False,
            "frozen_test_normalized_text_overlap": 0,
            "frozen_test_task_overlap": 0,
        }
    )
    report["data_audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, args.output)
    report["artifact_model_sha256"] = _sha256(args.output)
    write_json(args.report, report)
    print(f"Saved feedback-form classifier: {args.output}")
    print(f"Dev metrics: {report['dev']}")
    print("Human test rows used during training/selection: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
