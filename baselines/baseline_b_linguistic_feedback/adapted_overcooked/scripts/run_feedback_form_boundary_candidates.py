"""Train isolated low-weight boundary-corpus candidates using development data only.

The current production artifact is a fixed baseline.  New boundary rows are
added under a separate source so the existing hard corpus keeps its audited
weight.  Candidate selection never loads a human test or frozen-paper test.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_feedback_form_classifier as trainer  # noqa: E402
from scripts.generate_feedback_form_boundary_train import (  # noqa: E402
    DEFAULT_HOLDOUT_SPECS,
    DEFAULT_PAPER_MANIFEST,
    _load_holdout_gate,
)
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import artifact_probabilities  # noqa: E402


VERSION = "feedback-form-boundary-candidate-sweep-v1"
BOUNDARY_SOURCE = "feedback_form_boundary_train"
DEFAULT_BOUNDARY = ROOT / "data" / "feedback_form_boundary_train.v1.json"
DEFAULT_BOUNDARY_REPORT = (
    ROOT / "outputs" / "synth" / "feedback_form_boundary_train.v1.report.json"
)
DEFAULT_BASELINE_MODEL = ROOT / "outputs" / "feedback_form_classifier" / "model.joblib"
DEFAULT_BASELINE_REPORT = (
    ROOT / "outputs" / "feedback_form_classifier" / "model.report.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "feedback_form_classifier_candidates" / "boundary_v1"
)
DEFAULT_WEIGHTS = (0.1, 0.25, 0.5, 1.0)


# This small diagnostic suite is manually labeled but is not part of the
# candidate selection key.  Exact text overlap with train/dev/known holdouts is
# rejected before any model is evaluated.
NATURAL_PROBES = (
    ("That move was very good.", "evaluative"),
    ("That last handoff was excellent work.", "evaluative"),
    ("Your earlier detour really slowed the order.", "evaluative"),
    ("I appreciated the way you cleared that counter.", "evaluative"),
    ("The move you just made helped us a lot.", "evaluative"),
    ("You ought to have delivered that sooner.", "evaluative"),
    ("What you did there was not useful.", "evaluative"),
    ("Please fetch the next clean dish.", "imperative"),
    ("Could you set the onion beside the chopping board?", "imperative"),
    ("Stop waiting and help with the soup.", "imperative"),
    ("I want you to keep the right passage free.", "imperative"),
    ("Would you mind taking this tomato to the prep area?", "imperative"),
    ("Make sure the finished meal reaches the service counter.", "imperative"),
    ("One chopped tomato is sitting near the pot.", "descriptive"),
    ("The right-hand passage is currently free.", "descriptive"),
    ("Carrying an extra dish reduces available counter space.", "descriptive"),
    ("This recipe requires three vegetables.", "descriptive"),
    ("Only the upper chopping board is occupied.", "descriptive"),
    ("The serving counter is closer by the lower route.", "descriptive"),
)


def resolve_min_df(baseline_report: dict, requested: int | None) -> tuple[int, int]:
    """Keep candidate vectorization comparable with the production baseline."""
    baseline_min_df = int(baseline_report["feature_config"]["min_df"])
    resolved = baseline_min_df if requested is None else int(requested)
    if resolved < 1:
        raise ValueError("min_df must be at least 1")
    return baseline_min_df, resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extension_rows(
    boundary_path: Path, occupied_texts: set[str], extension_mode: str
) -> list[dict]:
    raw_rows = _read_json(boundary_path)
    if not isinstance(raw_rows, list):
        raise ValueError("boundary corpus must be a JSON list")
    extension = []
    for index, raw in enumerate(raw_rows, start=1):
        family = str(raw.get("template_family") or "")
        if extension_mode == "short" and ":short_" not in family:
            continue
        if extension_mode not in {"full", "short"}:
            raise ValueError(f"unsupported extension mode: {extension_mode}")
        text = raw.get("text")
        label = raw.get("expected_feedback_type")
        normalized = normalize_text(text)
        if normalized in occupied_texts:
            continue
        if label not in trainer.FEEDBACK_TYPES:
            raise ValueError(f"boundary row {index}: invalid label")
        if raw.get("split") != "train":
            raise ValueError(f"boundary row {index}: not train-only")
        if raw.get("source") != "feedback_form_hard_train":
            raise ValueError(f"boundary row {index}: invalid source provenance")
        if raw.get("label_source") != "direct_speech_act_contrastive_templates_train_only":
            raise ValueError(f"boundary row {index}: invalid label provenance")
        extension.append(
            {
                "text": str(text).strip(),
                "normalized_text": normalized,
                "label": str(label),
                "split": "train",
                "source": BOUNDARY_SOURCE,
                "group_id": f"{BOUNDARY_SOURCE}:{raw['group_id']}",
                "origin": "feedback-form-boundary-train-v1",
                "template_family": family,
                "paper_task_uuid": None,
            }
        )
    if len({row["normalized_text"] for row in extension}) != len(extension):
        raise ValueError("boundary extension contains duplicate normalized text")
    if Counter(row["label"] for row in extension).values() and len(
        set(Counter(row["label"] for row in extension).values())
    ) != 1:
        raise ValueError("boundary extension must remain class balanced")
    return extension


def _probe_report(artifact: dict) -> dict:
    import numpy as np

    texts = [text for text, _label in NATURAL_PROBES]
    labels = [label for _text, label in NATURAL_PROBES]
    matrix = artifact["vectorizer"].transform(texts)
    classification = trainer.classification_metrics(
        artifact["classifier"], matrix, labels
    )
    raw = np.asarray(artifact["classifier"].predict_proba(matrix), dtype=float)
    temperature = float(artifact["probability_calibration"]["temperature"])
    calibrated = trainer.temperature_scaled_probabilities(
        artifact["classifier"], matrix, temperature
    )
    return {
        "selection_uses_this_probe": False,
        "rows": len(texts),
        "classification": classification,
        "raw_probability_metrics": trainer.probability_metrics(
            raw, labels, artifact["classifier"].classes_
        ),
        "calibrated_probability_metrics": trainer.probability_metrics(
            calibrated, labels, artifact["classifier"].classes_
        ),
    }


def _phrase_probabilities(artifact: dict, phrase: str) -> dict:
    classifier = artifact["classifier"]
    matrix = artifact["vectorizer"].transform([phrase])
    raw_values = classifier.predict_proba(matrix)[0]
    raw = {
        str(label): float(value)
        for label, value in zip(classifier.classes_, raw_values)
    }
    calibrated, metadata = artifact_probabilities(artifact, matrix)
    return {
        "text": phrase,
        "raw_probabilities": raw,
        "raw_prediction": max(raw, key=raw.get),
        "raw_confidence": max(raw.values()),
        "calibrated_probabilities": calibrated,
        "calibrated_prediction": max(calibrated, key=calibrated.get),
        "calibrated_confidence": max(calibrated.values()),
        "calibration": metadata,
    }


def _summary_metrics(name: str, artifact: dict, report: dict) -> dict:
    human = report["development_by_source"]["original_paper_human"]
    overall = report["dev"]
    calibrated = overall["calibration"]["calibrated"]
    return {
        "name": name,
        "overall_dev": {
            "rows": overall["rows"],
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "balanced_accuracy": overall["balanced_accuracy"],
            "per_class_recall": {
                label: overall["per_class"][label]["recall"]
                for label in trainer.FEEDBACK_TYPES
            },
        },
        "paper_reference_proxy_selection_dev": {
            "source": "original_paper_human",
            "rows": human["rows"],
            "accuracy": human["accuracy"],
            "correct_rows": round(human["accuracy"] * human["rows"]),
            "macro_f1": human["macro_f1"],
            "balanced_accuracy": human["balanced_accuracy"],
            "per_class_recall": {
                label: human["per_class"][label]["recall"]
                for label in trainer.FEEDBACK_TYPES
            },
            "minimum_class_recall": min(
                human["per_class"][label]["recall"]
                for label in trainer.FEEDBACK_TYPES
            ),
            "calibrated_ece_10_bin": calibrated["ece_10_bin"],
            "calibration_warning": "temperature fitted and measured on selection dev",
        },
        "natural_probe": _probe_report(artifact),
        "requested_phrase": _phrase_probabilities(
            artifact, "That move was very good."
        ),
    }


def run_candidates(
    boundary_path: Path,
    boundary_report_path: Path,
    baseline_model_path: Path,
    baseline_report_path: Path,
    output_dir: Path,
    weights: tuple[float, ...],
    extension_mode: str,
    min_df: int | None = None,
) -> dict:
    from joblib import dump, load

    if any(weight <= 0 for weight in weights):
        raise ValueError("candidate source weights must be positive")
    boundary_report = _read_json(boundary_report_path)
    if boundary_report.get("output_sha256") != _sha256(boundary_path):
        raise ValueError("boundary data/report hash mismatch")
    if boundary_report.get("holdout_leakage_gate", {}).get("exact_overlap_count") != 0:
        raise ValueError("boundary corpus failed its holdout leakage gate")

    base_train, dev_rows, base_audit = trainer.load_training_rows(
        trainer.DEFAULT_SYNTHETIC_INPUT,
        trainer.DEFAULT_HUMAN_TRAIN,
        trainer.DEFAULT_HUMAN_DEV,
        trainer.DEFAULT_HUMAN_AUGMENTATION,
        trainer.DEFAULT_HARD_TRAIN,
        trainer.DEFAULT_PAPER_HUMAN_TRAIN,
        trainer.DEFAULT_PAPER_HUMAN_DEV,
    )
    occupied = {row["normalized_text"] for row in [*base_train, *dev_rows]}
    extension = _extension_rows(boundary_path, occupied, extension_mode)
    if not extension:
        raise ValueError("boundary extension is empty after removing the base corpus")
    extension_texts = {row["normalized_text"] for row in extension}
    if extension_texts & occupied:
        raise AssertionError("boundary extension overlaps train/dev")

    holdout_hashes, _groups, _texts, _gate = _load_holdout_gate(
        DEFAULT_HOLDOUT_SPECS, DEFAULT_PAPER_MANIFEST
    )
    probe_hashes = {
        hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        for text, _label in NATURAL_PROBES
    }
    development_hashes = {
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in occupied
    }
    if probe_hashes & (holdout_hashes | development_hashes):
        raise ValueError("natural probe has exact train/dev/holdout overlap")
    if extension_texts & {normalize_text(text) for text, _label in NATURAL_PROBES}:
        raise ValueError("natural probe has exact boundary-train overlap")

    augmented_train = [*base_train, *extension]
    frozen_binding = trainer.bind_frozen_benchmark_without_loading_test(
        trainer.DEFAULT_PAPER_BENCHMARK_MANIFEST,
        trainer.DEFAULT_PAPER_HUMAN_TRAIN,
        trainer.DEFAULT_PAPER_HUMAN_DEV,
        augmented_train,
        dev_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_artifact = load(baseline_model_path)
    baseline_report = _read_json(baseline_report_path)
    baseline_min_df, resolved_min_df = resolve_min_df(baseline_report, min_df)
    summaries = [
        _summary_metrics("production_baseline", baseline_artifact, baseline_report)
    ]
    candidate_records = []
    saved_weights = dict(trainer.FIXED_SOURCE_WEIGHTS)
    try:
        for weight in weights:
            trainer.FIXED_SOURCE_WEIGHTS.clear()
            trainer.FIXED_SOURCE_WEIGHTS.update(
                {**saved_weights, BOUNDARY_SOURCE: float(weight)}
            )
            artifact, report = trainer.train_feedback_form_classifier(
                augmented_train, dev_rows, seed=1, min_df=resolved_min_df
            )
            slug = str(weight).replace(".", "p")
            candidate_dir = output_dir / f"weight_{slug}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model_path = candidate_dir / "model.joblib"
            report_path = candidate_dir / "model.report.json"
            manifest = {
                "version": VERSION,
                "synthetic_input": str(trainer.DEFAULT_SYNTHETIC_INPUT),
                "synthetic_input_sha256": _sha256(trainer.DEFAULT_SYNTHETIC_INPUT),
                "base_hard_train": str(trainer.DEFAULT_HARD_TRAIN),
                "base_hard_train_sha256": _sha256(trainer.DEFAULT_HARD_TRAIN),
                "boundary_train": str(boundary_path),
                "boundary_train_sha256": _sha256(boundary_path),
                "boundary_report_sha256": _sha256(boundary_report_path),
                "boundary_extension_rows": len(extension),
                "boundary_extension_mode": extension_mode,
                "boundary_source_weight": float(weight),
                "human_test_loaded": False,
                "frozen_test_loaded": False,
                "seed": 1,
                "min_df": resolved_min_df,
                "baseline_min_df": baseline_min_df,
            }
            artifact["manifest"] = manifest
            artifact["frozen_benchmark_binding"] = frozen_binding
            report["artifact_manifest"] = manifest
            report["frozen_benchmark_binding"] = frozen_binding
            report["data_audit"] = {
                **base_audit,
                "boundary_extension_rows": len(extension),
                "boundary_train_dev_normalized_text_overlap": 0,
                "boundary_known_holdout_normalized_text_overlap": 0,
                "frozen_test_loaded": False,
                "test_examples_evaluated_during_selection": 0,
            }
            dump(artifact, model_path)
            report["artifact_model_sha256"] = _sha256(model_path)
            write_json(report_path, report)
            name = f"boundary_weight_{weight:g}"
            summary = _summary_metrics(name, artifact, report)
            summary["model_path"] = str(model_path)
            summary["model_sha256"] = report["artifact_model_sha256"]
            summary["report_path"] = str(report_path)
            summaries.append(summary)
            candidate_records.append(
                {"name": name, "weight": weight, "report": report_path}
            )
    finally:
        trainer.FIXED_SOURCE_WEIGHTS.clear()
        trainer.FIXED_SOURCE_WEIGHTS.update(saved_weights)

    baseline = summaries[0]
    baseline_overall = baseline["overall_dev"]
    baseline_human = baseline["paper_reference_proxy_selection_dev"]
    for summary in summaries:
        overall = summary["overall_dev"]
        human = summary["paper_reference_proxy_selection_dev"]
        summary["preserves_baseline_guard"] = all(
            (
                overall["accuracy"] >= baseline_overall["accuracy"],
                overall["macro_f1"] >= baseline_overall["macro_f1"],
                human["accuracy"] >= baseline_human["accuracy"],
                human["macro_f1"] >= baseline_human["macro_f1"],
            )
        )
    eligible = [row for row in summaries if row["preserves_baseline_guard"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["paper_reference_proxy_selection_dev"]["macro_f1"],
            row["paper_reference_proxy_selection_dev"]["minimum_class_recall"],
            -row["paper_reference_proxy_selection_dev"]["calibrated_ece_10_bin"],
            row["overall_dev"]["macro_f1"],
        ),
    )
    return {
        "version": VERSION,
        "selection_protocol": {
            "model_selection_data": "train_and_dev_only",
            "primary": "paper_reference_proxy_selection_dev_macro_f1",
            "guard": "do_not_reduce baseline overall accuracy/macro-F1 or human-proxy accuracy/macro-F1",
            "tie_breakers": [
                "human_proxy_minimum_class_recall",
                "human_proxy_calibrated_ece_10_bin_lower_is_better",
                "overall_dev_macro_f1",
            ],
            "natural_probe_used_for_selection": False,
            "test_examples_evaluated_during_selection": 0,
            "human_test_loaded": False,
            "frozen_test_loaded": False,
            "candidate_min_df": resolved_min_df,
            "baseline_min_df": baseline_min_df,
        },
        "data": {
            "boundary_extension_mode": extension_mode,
            "base_train_rows": len(base_train),
            "boundary_extension_rows": len(extension),
            "candidate_train_rows": len(augmented_train),
            "dev_rows": len(dev_rows),
            "boundary_label_counts": dict(
                sorted(Counter(row["label"] for row in extension).items())
            ),
            "train_dev_exact_overlap": 0,
            "known_holdout_exact_overlap": 0,
            "frozen_test_bound_by_manifest_not_opened": True,
        },
        "candidates": summaries,
        "selected": selected["name"],
        "production_replaced": False,
        "promotion_requires_separate_approval_and_frozen_evaluation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument(
        "--boundary-report", type=Path, default=DEFAULT_BOUNDARY_REPORT
    )
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--extension-mode", choices=("full", "short"), default="full"
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=None,
        help="Feature minimum document frequency (default: inherit baseline artifact)",
    )
    args = parser.parse_args()
    report = run_candidates(
        args.boundary,
        args.boundary_report,
        args.baseline_model,
        args.baseline_report,
        args.output_dir,
        tuple(args.weights),
        args.extension_mode,
        args.min_df,
    )
    output = args.output_dir / "candidate_selection.report.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved candidate selection report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
