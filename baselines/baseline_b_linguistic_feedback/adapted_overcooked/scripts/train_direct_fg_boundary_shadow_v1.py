"""Train the locked direct-fG boundary shadow candidate.

This trainer has one explicit data input: the reviewed 240-row train JSON with
the locked canonical SHA-256 below.  It never searches the data directory and
never opens the known-regression, mixed-speech-act, human, dev, test, frozen,
or private diagnostic files.  Model features are raw ``text`` strings only.

All reported model metrics are nested group out-of-fold diagnostics on the
synthetic training corpus.  They are not independent, player, production, or
paper-comparable accuracy.  The resulting artifact is shadow-only and cannot
be promoted by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Sequence
import unicodedata
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSON = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1"
    / "train.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1_model"
)
GENERATOR_PATH = ROOT / "scripts" / "generate_direct_fg_boundary_shadow_v1.py"
PREDICTOR_PATH = ROOT / "scripts" / "predict_direct_fg_boundary_shadow_v1.py"

VERSION = "direct-fg-boundary-shadow-model-v1"
SEED = 752
EXPECTED_TRAIN_CANONICAL_SHA256 = (
    "5a7181dbd67d8aeb9dc1dfe941daf07c454140cc803f93a493ccb7bdf2609db5"
)
EXPECTED_GENERATOR_SHA256 = (
    "e247cf010e6f330c56622215fe4d7986271adaa8feadd519fbbb6f5cd22c1c87"
)
CLASS_ORDER = ("descriptive", "evaluative", "imperative")
OUTER_SPLITS = 5
INNER_SPLITS = 5
FINAL_SELECTION_SPLITS = 5

CONFIG_GRID: tuple[dict, ...] = tuple(
    {
        "config_id": f"C={c:g};min_df={min_df}",
        "C": float(c),
        "min_df": int(min_df),
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5],
        "char_weight": 0.7,
    }
    for c in (0.25, 1.0, 4.0)
    for min_df in (1, 2)
)


def canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def extract_model_input(value: str) -> str:
    """Return the sole model feature; reject rows/dicts fail-closed."""
    if type(value) is not str:
        raise TypeError("model input must be a raw text string")
    if not value.strip():
        raise ValueError("model input text must be non-empty")
    return value


def _validate_locked_rows(rows: object) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != 240:
        raise ValueError("locked train corpus must contain exactly 240 rows")
    if canonical_hash(rows) != EXPECTED_TRAIN_CANONICAL_SHA256:
        raise ValueError("locked train canonical SHA-256 mismatch")
    ordered = list(rows)
    if any(not isinstance(row, dict) for row in ordered):
        raise ValueError("every train row must be a JSON object")
    labels = Counter(row.get("expected_feedback_type") for row in ordered)
    if labels != Counter({label: 80 for label in CLASS_ORDER}):
        raise ValueError("locked train labels must be exactly 80 per direct-fG class")
    ids = [row.get("feedback_id") for row in ordered]
    if len(set(ids)) != 240 or any(not value for value in ids):
        raise ValueError("locked train feedback_id values must be unique and non-empty")
    groups = Counter(row.get("surface_bundle_id") for row in ordered)
    if len(groups) != 20 or set(groups.values()) != {12} or None in groups:
        raise ValueError("locked train must have twenty 12-row surface bundles")
    group_labels = Counter(
        (row.get("surface_bundle_id"), row.get("expected_feedback_type"))
        for row in ordered
    )
    if len(group_labels) != 60 or set(group_labels.values()) != {4}:
        raise ValueError("each surface bundle must contain four rows per class")
    for row in ordered:
        text = extract_model_input(row.get("text"))
        if row.get("split") != "train":
            raise ValueError("trainer accepts train rows only")
        if row.get("normalized_text") != normalize_text(text):
            raise ValueError("normalized_text does not match raw text")
        if "reference_type" in row or "classification_label" not in row:
            raise ValueError("trainer accepts the direct three-class schema only")
        expected_canonical = str(row["expected_feedback_type"]).capitalize()
        if row.get("classification_label") != expected_canonical:
            raise ValueError("classification_label disagrees with direct-fG target")
    return ordered


def load_locked_train_rows(train_json: Path) -> list[dict]:
    """Read exactly one explicit JSON file and enforce its canonical hash."""
    resolved = Path(train_json).resolve()
    rows = json.loads(resolved.read_text(encoding="utf-8"))
    return _validate_locked_rows(rows)


def _texts(rows: Sequence[dict]) -> list[str]:
    return [extract_model_input(row["text"]) for row in rows]


def _labels(rows: Sequence[dict]) -> list[str]:
    return [str(row["expected_feedback_type"]) for row in rows]


def _groups(rows: Sequence[dict]) -> list[str]:
    return [str(row["surface_bundle_id"]) for row in rows]


def build_vectorizer(config: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    min_df = int(config["min_df"])
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    ngram_range=tuple(config["word_ngram_range"]),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=tuple(config["char_ngram_range"]),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": float(config["char_weight"])},
    )


def build_classifier(config: dict):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=float(config["C"]),
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
        class_weight=None,
    )


def _align_logits(classifier, matrix):
    import numpy as np

    observed = [str(value) for value in classifier.classes_]
    if set(observed) != set(CLASS_ORDER):
        raise ValueError(f"classifier classes changed: {observed}")
    scores = np.asarray(classifier.decision_function(matrix), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(observed):
        raise ValueError("expected a three-class decision matrix")
    return scores[:, [observed.index(label) for label in CLASS_ORDER]]


def fit_model(train_rows: Sequence[dict], config: dict):
    vectorizer = build_vectorizer(config)
    matrix = vectorizer.fit_transform(_texts(train_rows))
    classifier = build_classifier(config)
    classifier.fit(matrix, _labels(train_rows))
    return vectorizer, classifier


def _predict_logits(
    train_rows: Sequence[dict], validation_rows: Sequence[dict], config: dict
):
    vectorizer, classifier = fit_model(train_rows, config)
    validation_matrix = vectorizer.transform(_texts(validation_rows))
    return _align_logits(classifier, validation_matrix)


def softmax_temperature(logits, temperature: float):
    import numpy as np

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("non-finite calibrated probability")
    return probabilities


def _label_indices(labels: Sequence[str]):
    import numpy as np

    lookup = {label: index for index, label in enumerate(CLASS_ORDER)}
    if any(label not in lookup for label in labels):
        raise ValueError("unknown direct-fG label")
    return np.asarray([lookup[label] for label in labels], dtype=int)


def _negative_log_likelihood(probabilities, labels: Sequence[str]) -> float:
    import numpy as np

    indices = _label_indices(labels)
    values = np.asarray(probabilities, dtype=float)[np.arange(len(indices)), indices]
    return float(-np.mean(np.log(np.clip(values, 1e-15, 1.0))))


def fit_oof_temperature(logits, labels: Sequence[str]) -> tuple[float, dict]:
    """Fit one scalar using only nested out-of-fold train predictions."""
    low = math.log(0.05)
    high = math.log(20.0)
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(log_temperature: float) -> float:
        return _negative_log_likelihood(
            softmax_temperature(logits, math.exp(log_temperature)), labels
        )

    left = high - golden * (high - low)
    right = low + golden * (high - low)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(100):
        if left_value <= right_value:
            high, right, right_value = right, left, left_value
            left = high - golden * (high - low)
            left_value = objective(left)
        else:
            low, left, left_value = left, right, right_value
            right = low + golden * (high - low)
            right_value = objective(right)
    candidates = (1.0, math.exp((low + high) / 2.0))
    temperature = min(candidates, key=lambda value: objective(math.log(value)))
    raw = softmax_temperature(logits, 1.0)
    calibrated = softmax_temperature(logits, temperature)
    raw_argmax = raw.argmax(axis=1)
    calibrated_argmax = calibrated.argmax(axis=1)
    if not (raw_argmax == calibrated_argmax).all():
        raise ValueError("temperature scaling changed argmax predictions")
    return float(temperature), {
        "method": "scalar_temperature_on_nested_group_oof_logits",
        "fit_scope": "synthetic_train_nested_group_oof_only",
        "temperature": float(temperature),
        "nll_before": _negative_log_likelihood(raw, labels),
        "nll_after": _negative_log_likelihood(calibrated, labels),
        "argmax_unchanged": True,
        "independently_validated": False,
    }


def probability_metrics(labels: Sequence[str], probabilities) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    array = np.asarray(probabilities, dtype=float)
    predictions = [CLASS_ORDER[index] for index in array.argmax(axis=1)]
    recalls = recall_score(
        labels, predictions, labels=list(CLASS_ORDER), average=None, zero_division=0
    )
    one_hot = np.eye(len(CLASS_ORDER), dtype=float)[_label_indices(labels)]
    confidence = array.max(axis=1)
    correct = np.asarray([left == right for left, right in zip(labels, predictions)])
    ece = 0.0
    for start in [value / 10.0 for value in range(10)]:
        end = start + 0.1
        mask = (confidence >= start) & (
            (confidence <= end) if end >= 1.0 else (confidence < end)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "per_class_recall": {
            label: float(value) for label, value in zip(CLASS_ORDER, recalls)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(CLASS_ORDER)
        ).tolist(),
        "class_order": list(CLASS_ORDER),
        "negative_log_likelihood": _negative_log_likelihood(array, labels),
        "multiclass_brier": float(((array - one_hot) ** 2).sum(axis=1).mean()),
        "expected_calibration_error_10_bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "row_count": len(labels),
    }


def _group_folds(rows: Sequence[dict], n_splits: int) -> list[tuple[list[int], list[int]]]:
    from sklearn.model_selection import GroupKFold

    groups = _groups(rows)
    if len(set(groups)) < n_splits:
        raise ValueError("not enough surface bundles for grouped cross-validation")
    splitter = GroupKFold(n_splits=n_splits)
    indices = list(range(len(rows)))
    folds = []
    for train_index, validation_index in splitter.split(indices, _labels(rows), groups):
        train_values = [int(value) for value in train_index]
        validation_values = [int(value) for value in validation_index]
        train_groups = {groups[index] for index in train_values}
        validation_groups = {groups[index] for index in validation_values}
        if train_groups & validation_groups:
            raise ValueError("surface bundle leaked across a grouped fold")
        folds.append((train_values, validation_values))
    return folds


def _subset(rows: Sequence[dict], indices: Iterable[int]) -> list[dict]:
    return [rows[index] for index in indices]


def select_hyperparameters(
    rows: Sequence[dict], *, n_splits: int
) -> tuple[dict, dict]:
    import numpy as np

    fold_indices = _group_folds(rows, n_splits)
    candidates = []
    for config in CONFIG_GRID:
        oof_logits = np.zeros((len(rows), len(CLASS_ORDER)), dtype=float)
        seen = np.zeros(len(rows), dtype=bool)
        fold_metrics = []
        for fold_index, (train_index, validation_index) in enumerate(fold_indices):
            train_rows = _subset(rows, train_index)
            validation_rows = _subset(rows, validation_index)
            logits = _predict_logits(train_rows, validation_rows, config)
            oof_logits[validation_index, :] = logits
            seen[validation_index] = True
            fold_probabilities = softmax_temperature(logits, 1.0)
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "train_groups": len(set(_groups(train_rows))),
                    "validation_groups": len(set(_groups(validation_rows))),
                    "metrics": probability_metrics(
                        _labels(validation_rows), fold_probabilities
                    ),
                }
            )
        if not seen.all():
            raise ValueError("group CV did not produce exactly one prediction per row")
        metrics = probability_metrics(
            _labels(rows), softmax_temperature(oof_logits, 1.0)
        )
        candidates.append(
            {
                "config": dict(config),
                "metrics": metrics,
                "folds": fold_metrics,
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["metrics"]["macro_f1"],
            -item["metrics"]["accuracy"],
            item["config"]["config_id"],
        ),
    )
    return dict(ranked[0]["config"]), {
        "selection_scope": "synthetic_train_group_cv_only",
        "selection_metric": "macro_f1_then_accuracy_then_config_id",
        "n_splits": n_splits,
        "group_key": "surface_bundle_id",
        "group_count": len(set(_groups(rows))),
        "candidate_count": len(CONFIG_GRID),
        "selected_config_id": ranked[0]["config"]["config_id"],
        "candidates": candidates,
    }


def nested_group_oof(rows: Sequence[dict]) -> tuple[object, dict]:
    import numpy as np

    outer_folds = _group_folds(rows, OUTER_SPLITS)
    oof_logits = np.zeros((len(rows), len(CLASS_ORDER)), dtype=float)
    seen = np.zeros(len(rows), dtype=bool)
    fold_reports = []
    for fold_index, (train_index, validation_index) in enumerate(outer_folds):
        outer_train = _subset(rows, train_index)
        outer_validation = _subset(rows, validation_index)
        selected, inner_report = select_hyperparameters(
            outer_train, n_splits=INNER_SPLITS
        )
        logits = _predict_logits(outer_train, outer_validation, selected)
        oof_logits[validation_index, :] = logits
        seen[validation_index] = True
        fold_reports.append(
            {
                "outer_fold": fold_index,
                "selected_config": selected,
                "outer_train_rows": len(outer_train),
                "outer_validation_rows": len(outer_validation),
                "outer_train_groups": sorted(set(_groups(outer_train))),
                "outer_validation_groups": sorted(set(_groups(outer_validation))),
                "inner_selection": inner_report,
                "uncalibrated_outer_metrics": probability_metrics(
                    _labels(outer_validation), softmax_temperature(logits, 1.0)
                ),
            }
        )
    if not seen.all() or int(seen.sum()) != len(rows):
        raise ValueError("nested outer CV did not cover every train row exactly once")
    return oof_logits, {
        "protocol": "nested_group_cv",
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "group_key": "surface_bundle_id",
        "each_row_predicted_once_out_of_fold": True,
        "folds": fold_reports,
    }


def train_locked_rows(rows: list[dict]) -> tuple[dict, dict, dict]:
    locked = _validate_locked_rows(rows)
    nested_logits, nested_report = nested_group_oof(locked)
    labels = _labels(locked)
    temperature, temperature_report = fit_oof_temperature(nested_logits, labels)
    raw_oof = probability_metrics(labels, softmax_temperature(nested_logits, 1.0))
    calibrated_oof = probability_metrics(
        labels, softmax_temperature(nested_logits, temperature)
    )
    final_config, final_selection = select_hyperparameters(
        locked, n_splits=FINAL_SELECTION_SPLITS
    )
    vectorizer, classifier = fit_model(locked, final_config)
    artifact = {
        "artifact_version": VERSION,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": list(CLASS_ORDER),
        "temperature": temperature,
        "selected_hyperparameters": final_config,
        "model_input_contract": {
            "input_type": "raw_string_only",
            "field": "text",
            "metadata_features": [],
        },
        "status": "shadow_diagnostic_only",
        "promotion_eligible": False,
        "production_artifact": False,
        "confidence_scope": "temperature_fit_on_nested_train_oof_not_independently_validated",
    }
    frozen_config = {
        "version": VERSION,
        "seed": SEED,
        "expected_train_canonical_sha256": EXPECTED_TRAIN_CANONICAL_SHA256,
        "class_order": list(CLASS_ORDER),
        "feature_contract": artifact["model_input_contract"],
        "selected_hyperparameters": final_config,
        "temperature": temperature,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "final_selection_splits": FINAL_SELECTION_SPLITS,
        "group_key": "surface_bundle_id",
        "promotion_eligible": False,
    }
    report = {
        "version": VERSION,
        "status": "trained_shadow_diagnostic_only",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "production_promotion_eligible": False,
        "confidence_ready_for_player_ui": False,
        "production_model_replaced": False,
        "browser_model_replaced": False,
        "pygame_model_replaced": False,
        "metric_claim": (
            "nested group out-of-fold diagnostics on targeted synthetic train data; "
            "not independent, player, production, or paper-comparable accuracy"
        ),
        "train_input": {
            "rows": len(locked),
            "canonical_sha256": canonical_hash(locked),
            "label_counts": dict(sorted(Counter(labels).items())),
            "surface_bundle_count": len(set(_groups(locked))),
            "explicit_data_files_opened_by_trainer": 1,
            "input_feature": "raw text only",
        },
        "excluded_from_trainer": {
            "known_regression": {
                "opened": False,
                "fit": False,
                "selection": False,
                "temperature": False,
                "accuracy": False,
            },
            "mixed_speech_act": {
                "opened": False,
                "fit": False,
                "selection": False,
                "temperature": False,
                "accuracy": False,
            },
            "private_diagnostic": {"opened": False, "used": False},
            "human_dev_test_frozen": {"opened": False, "used": False},
        },
        "nested_group_oof": nested_report,
        "nested_oof_uncalibrated_metrics": raw_oof,
        "nested_oof_temperature_scaled_metrics": calibrated_oof,
        "temperature_calibration": temperature_report,
        "final_train_only_selection": final_selection,
        "selected_hyperparameters": final_config,
        "limitations": [
            "targeted AI-assisted synthetic corpus with limited phrasing",
            "twenty of the current typed scenarios are represented",
            "negative commands are valid only for fG intent, not action grounding",
            "no independent human or real-player accuracy is available yet",
            "confidence calibration has no independent validation",
        ],
    }
    return artifact, report, frozen_config


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_outputs(
    output_dir: Path,
    train_rows: list[dict],
    artifact: dict,
    report: dict,
    frozen_config: dict,
    *,
    train_input_path: Path,
) -> dict:
    import joblib
    import numpy
    import sklearn

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.locked.json"
    report_path = output / "training_report.json"
    config_path = output / "frozen_config.json"
    model_path = output / "model.joblib"
    manifest_path = output / "manifest.json"
    _write_json(train_path, train_rows)
    _write_json(report_path, report)
    _write_json(config_path, frozen_config)
    temporary_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(artifact, temporary_model)
    os.replace(temporary_model, model_path)
    if canonical_hash(json.loads(train_path.read_text(encoding="utf-8"))) != EXPECTED_TRAIN_CANONICAL_SHA256:
        raise ValueError("written locked train corpus hash mismatch")
    if file_sha256(GENERATOR_PATH) != EXPECTED_GENERATOR_SHA256:
        raise ValueError("reviewed generator script changed before artifact freeze")
    if not PREDICTOR_PATH.is_file():
        raise ValueError("frozen predictor interface is missing")
    manifest = {
        "version": VERSION,
        "status": "shadow_diagnostic_only",
        "promotion_eligible": False,
        "model_input": "raw text string only",
        "train_input": {
            "path": str(Path(train_input_path).resolve()),
            "canonical_sha256": EXPECTED_TRAIN_CANONICAL_SHA256,
            "data_files_opened": [str(Path(train_input_path).resolve())],
            "directory_glob_used": False,
        },
        "artifacts": {
            "model": {"path": str(model_path), "sha256": file_sha256(model_path)},
            "frozen_config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
                "canonical_sha256": canonical_hash(frozen_config),
            },
            "predictor": {
                "path": str(PREDICTOR_PATH.resolve()),
                "sha256": file_sha256(PREDICTOR_PATH),
            },
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "generator": {
                "path": str(GENERATOR_PATH.resolve()),
                "sha256": file_sha256(GENERATOR_PATH),
            },
            "locked_train_copy": {
                "path": str(train_path),
                "sha256": file_sha256(train_path),
                "canonical_sha256": EXPECTED_TRAIN_CANONICAL_SHA256,
            },
            "training_report": {
                "path": str(report_path),
                "sha256": file_sha256(report_path),
            },
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "seed": SEED,
        "temperature": artifact["temperature"],
        "selected_hyperparameters": artifact["selected_hyperparameters"],
        "private_diagnostic_opened": False,
        "known_or_mixed_opened_by_trainer": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    rows = load_locked_train_rows(args.train_json)
    artifact, report, frozen_config = train_locked_rows(rows)
    result = dict(report)
    if not args.no_write:
        result["manifest"] = write_outputs(
            args.output_dir,
            rows,
            artifact,
            report,
            frozen_config,
            train_input_path=args.train_json,
        )
    summary = {
        "version": result["version"],
        "status": result["status"],
        "promotion_eligible": result["promotion_eligible"],
        "metric_claim": result["metric_claim"],
        "train_canonical_sha256": result["train_input"]["canonical_sha256"],
        "nested_oof_metrics": result["nested_oof_temperature_scaled_metrics"],
        "temperature_calibration": result["temperature_calibration"],
        "selected_hyperparameters": result["selected_hyperparameters"],
    }
    if "manifest" in result:
        summary["manifest"] = result["manifest"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
