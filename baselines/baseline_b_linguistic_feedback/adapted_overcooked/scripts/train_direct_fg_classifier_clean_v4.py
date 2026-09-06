"""Train clean-v4 with short contrasts, preserving a one-look synthetic final eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_direct_fg_clean_v3 as v3_generator  # noqa: E402
from scripts import generate_direct_fg_clean_v4 as generator  # noqa: E402
from scripts import train_direct_fg_classifier_clean_v3 as base  # noqa: E402


VERSION = "direct-fg-clean-v4-research-candidate-v1"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v4"
)

# This set was observed during v3 development.  It is now explicitly
# training-informed regression only and can never enter a v4 pass gate.
V3_TRAINING_INFORMED_REGRESSION = base.NATURAL_PROBES

# Frozen before the first v4 final evaluation.  It is AI-assisted, not human
# gold, not used for selection/calibration, and secondary-only.
V4_UNSEEN_SHORT_DIAGNOSTIC_VERSION = "clean-v4-unseen-short-diagnostic-v1"
V4_UNSEEN_SHORT_DIAGNOSTIC = (
    ("Nice work on that tomato pickup.", "evaluative"),
    ("That onion placement was a bad move.", "evaluative"),
    ("You did well serving the soup.", "evaluative"),
    ("Clearing that corridor was helpful.", "evaluative"),
    ("That last dish pickup was poorly timed.", "evaluative"),
    ("Putting the soup on the counter was a smart move.", "evaluative"),
    ("Waiting on that step was inefficient.", "evaluative"),
    ("That pot interaction was excellent.", "evaluative"),
    ("You handled the serving route badly.", "evaluative"),
    ("That recent counter pickup was useful.", "evaluative"),
    ("Grab a dish now.", "imperative"),
    ("Put the onion into the pot.", "imperative"),
    ("Please yield the lower corridor.", "imperative"),
    ("Get the tomato from the accessible counter.", "imperative"),
    ("Use the dish on the ready soup.", "imperative"),
    ("Take the soup to the serving station.", "imperative"),
    ("Set the tomato on an empty accessible counter.", "imperative"),
    ("Wait until the pot is ready.", "imperative"),
    ("Move away from the pot access tile.", "imperative"),
    ("Could you fetch an onion?", "imperative"),
    ("One tomato is in the pot.", "descriptive"),
    ("Two tomatoes are in the filling pot.", "descriptive"),
    ("The AI chef has an onion.", "descriptive"),
    ("Soup is ready in the pot.", "descriptive"),
    ("A dish is on an accessible counter.", "descriptive"),
    ("The lower corridor is open.", "descriptive"),
    ("The pot contains two tomatoes and one onion.", "descriptive"),
    ("The AI chef is carrying soup.", "descriptive"),
    ("The serving station is where soup in a dish is delivered.", "descriptive"),
    ("The other chef's route is clear at the pot.", "descriptive"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_diagnostic(vectorizer, classifier, temperature: float, examples) -> dict:
    import numpy as np

    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    raw = classifier.predict_proba(vectorizer.transform(texts))
    probabilities = base._temperature_probabilities(raw, temperature)
    classes = [str(label) for label in classifier.classes_]
    predictions = [classes[index] for index in np.asarray(probabilities).argmax(axis=1)]
    errors = []
    for (text, expected), predicted, values in zip(examples, predictions, probabilities):
        if expected == predicted:
            continue
        errors.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": float(max(values)),
                "probabilities": {
                    label: float(value) for label, value in zip(classes, values)
                },
            }
        )
    return {"metrics": base._basic_metrics(labels, predictions), "errors": errors}


def _dependency_manifest() -> dict:
    import joblib
    import numpy
    import sklearn

    local_paths = {
        "v4_generator": Path(generator.__file__).resolve(),
        "v4_trainer": Path(__file__).resolve(),
        "reused_v3_typed_generator": Path(v3_generator.__file__).resolve(),
        "reused_v3_training_algorithms": Path(base.__file__).resolve(),
    }
    external = {"numpy": numpy, "sklearn": sklearn, "joblib": joblib}
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
            for name, module in external.items()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }


def train_candidate_one_final_look() -> tuple[dict, dict, dict[str, list[dict]]]:
    rows = generator.generate_rows()
    generation_audit = generator.audit_rows(rows)
    partitions, split_audit = base.split_rows(rows)
    selected_config, selection_report = base.select_hyperparameters_train_only(
        partitions["train"]
    )
    vectorizer, classifier = base.fit_base_model(partitions["train"], selected_config)
    calibration_matrix = vectorizer.transform(
        [row["text"] for row in partitions["calibration"]]
    )
    calibration_labels = [
        row["expected_feedback_type"] for row in partitions["calibration"]
    ]
    temperature, calibration_report = base.fit_temperature_calibration_only(
        classifier, calibration_matrix, calibration_labels
    )
    calibration_report = dict(calibration_report)
    calibration_report.update(
        {
            "temperature_search_bounds": [0.20, 5.0],
            "temperature_at_search_boundary": temperature in {0.20, 5.0},
            "confidence_claim_scope": "synthetic_calibration_partition_only",
            "confidence_ready_for_player_ui": False,
        }
    )

    frozen_before = base._model_state_hash(
        vectorizer, classifier, selected_config, temperature
    )
    # The single final-evaluation call in the v4 training path.  Nothing below
    # selects a model, template, threshold, seed, or temperature.
    final_metrics = base.evaluate_frozen_final(
        vectorizer, classifier, temperature, partitions["final_eval"]
    )
    frozen_after = base._model_state_hash(
        vectorizer, classifier, selected_config, temperature
    )
    if frozen_before != frozen_after:
        raise ValueError("v4 final evaluator mutated the frozen candidate")

    v3_regression = _evaluate_diagnostic(
        vectorizer,
        classifier,
        temperature,
        V3_TRAINING_INFORMED_REGRESSION,
    )
    v3_regression.update(
        {
            "role": "training_informed_regression_only",
            "source": "v3 natural-probe error analysis",
            "included_in_selection": False,
            "included_in_pass_gate": False,
            "independent": False,
            "human_gold": False,
        }
    )
    unseen_diagnostic = _evaluate_diagnostic(
        vectorizer,
        classifier,
        temperature,
        V4_UNSEEN_SHORT_DIAGNOSTIC,
    )
    unseen_diagnostic.update(
        {
            "version": V4_UNSEEN_SHORT_DIAGNOSTIC_VERSION,
            "role": "secondary_unseen_ai_assisted_diagnostic",
            "included_in_selection": False,
            "included_in_pass_gate": False,
            "human_gold": False,
        }
    )

    recalls = final_metrics["per_class_recall"]
    status_checks = {
        "generation_rows_exactly_2160": generation_audit["rows"] == 2160,
        "generation_balanced": len(set(generation_audit["label_counts"].values())) == 1,
        "short_contrasts_balanced": len(
            set(generation_audit["short_contrast_label_counts"].values())
        )
        == 1,
        "forbidden_ontology_hits_zero": generation_audit["forbidden_ontology_hits"] == 0,
        "single_speech_act_failures_zero": generation_audit["single_speech_act_failures"] == 0,
        "surface_grammar_failures_zero": generation_audit["surface_grammar_failures"] == 0,
        "reference_type_not_used": not generation_audit["reference_type_used"],
        "external_or_old_data_files_opened_zero": generation_audit["external_input_files_opened"] == 0 and generation_audit["old_or_human_data_rows_read"] == 0,
        "strict_sequence_cross_split_zero": split_audit["cross_partition_similarity_counts"]["sequence_matcher_at_least_0_88"] == 0,
        "strict_jaccard_cross_split_zero": split_audit["cross_partition_similarity_counts"]["token_jaccard_at_least_0_70"] == 0,
        "train_only_model_selection": selection_report["calibration_rows_seen"] == 0 and selection_report["final_eval_rows_seen"] == 0,
        "independent_calibration": calibration_report["fit_split"] == "calibration" and not calibration_report["fit_split_used_for_model_selection"] and not calibration_report["final_eval_used"],
        "final_eval_accuracy_at_least_0_87": final_metrics["accuracy"] >= 0.87,
        "final_eval_macro_f1_at_least_0_87": final_metrics["macro_f1"] >= 0.87,
        "each_final_eval_recall_at_least_0_85": min(recalls.values()) >= 0.85,
        "frozen_final_eval_did_not_mutate_model": frozen_before == frozen_after,
        "final_evaluator_called_once": True,
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
        "claim_scope": "synthetic_current_ontology_short_and_long_family_scenario_disjoint_final_eval",
        "not_a_real_player_accuracy_claim": True,
        "confidence_ready_for_player_ui": False,
        "production_promotion_eligible": False,
        "production_promotion_blocker": "requires independently audited natural human holdout and player-domain confidence calibration",
        "generation": generation_audit,
        "split": split_audit,
        "selection": selection_report,
        "selected_hyperparameters": selected_config,
        "calibration": calibration_report,
        "final_synthetic_evaluation": final_metrics,
        "final_eval_protocol": {
            "execution_count": 1,
            "used_for_selection": False,
            "used_for_calibration": False,
            "model_frozen_before_read": True,
        },
        "v3_training_informed_regression": v3_regression,
        "v4_unseen_secondary_diagnostic": unseen_diagnostic,
        "frozen_model_state_sha256_before_final": frozen_before,
        "frozen_model_state_sha256_after_final": frozen_after,
        "provenance": {
            "ai_assisted": True,
            "prior_error_informed": True,
            "v3_natural_probe_error_informed": True,
            "independent_human_semantic_audit": False,
            "human_gold": False,
            "old_synthetic_rows_read": 0,
            "human_rows_read": 0,
            "dev_test_frozen_membership_files_read": 0,
            "v3_artifact_files_read": 0,
            "reference_type_used_as_target": False,
            "five_class_reference_classifier_used_as_target": False,
        },
        "dependencies": dependencies,
        "seed": base.SEED,
    }
    artifact = {
        "artifact_version": VERSION,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "feature_version": "direct-fg-clean-v4-word-char-tfidf-v1",
        "labels": list(generator.LABELS),
        "minimum_model_confidence": base.MINIMUM_MODEL_CONFIDENCE,
        "selected_hyperparameters": selected_config,
        "probability_calibration": {
            "method": "temperature_scaling",
            "version": calibration_report["version"],
            "temperature": temperature,
            "fit_split": "calibration",
            "fit_split_used_for_model_selection": False,
            "final_eval_used": False,
            "confidence_claim_scope": "synthetic_only_not_player_validated",
        },
        "training_scope": {
            "train_rows": len(partitions["train"]),
            "calibration_rows_for_temperature_only": len(partitions["calibration"]),
            "final_eval_rows_seen_during_training_or_calibration": 0,
        },
        "candidate_status": status,
        "production_promotion_eligible": False,
        "confidence_ready_for_player_ui": False,
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
        "confidence_ready_for_player_ui": False,
        "outputs": {
            "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        },
        "dependencies": report["dependencies"],
        "seed": base.SEED,
        "temperature": artifact["probability_calibration"]["temperature"],
        "selected_hyperparameters": artifact["selected_hyperparameters"],
        "provenance": report["provenance"],
        "final_eval_protocol": report["final_eval_protocol"],
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
        _, audit = base.split_rows(generator.generate_rows())
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    artifact, report, partitions = train_candidate_one_final_look()
    if not args.no_write:
        manifest = write_candidate_outputs(args.output_dir, artifact, report, partitions)
        report = dict(report)
        report["manifest"] = manifest
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
