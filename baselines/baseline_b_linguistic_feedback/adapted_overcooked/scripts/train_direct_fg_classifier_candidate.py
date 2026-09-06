"""Build and train an isolated direct-fG feedback-form candidate.

The target is the surface speech act fG (Evaluative, Imperative, or
Descriptive).  Training and primary development labels come only from an
explicit ``expected_feedback_type`` or a human ``classification_label``.
``reference_type`` is retained as provenance when present and is never mapped
into the target.  Paper reference-collapse development data is evaluated only
as a secondary, non-selection proxy.  No frozen/test file is opened.

The produced artifact is a research candidate.  It never overwrites the
runtime model under ``outputs/feedback_form_classifier``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_feedback_form_classifier as shared  # noqa: E402
from src.evaluation_splits import normalize_text  # noqa: E402
from src.feature_schema import write_json  # noqa: E402
from src.feedback_form_classifier import (  # noqa: E402
    CANONICAL_FEEDBACK_LABELS,
    DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
    FEEDBACK_TYPES,
)


VERSION = "direct-fg-classifier-candidate-v1"
DIRECT_BOUNDARY_LABEL_SOURCE = (
    "direct_speech_act_contrastive_templates_train_only"
)
HUMAN_LABEL_SOURCE = "human_explicit"

DEFAULT_BOUNDARY = ROOT / "data" / "feedback_form_boundary_train.v1.json"
DEFAULT_HUMAN_TRAIN = ROOT / "data" / "human_feedback_form_train.json"
DEFAULT_HUMAN_DEV = ROOT / "data" / "human_feedback_form_dev.json"
DEFAULT_PAPER_PROXY = ROOT / "data" / "paper_feedback_form_human_dev.v1.json"
DEFAULT_BASELINE_MODEL = ROOT / "outputs" / "feedback_form_classifier" / "model.joblib"
DEFAULT_CORPUS_OUTPUT = ROOT / "data" / "direct_fg_classifier_corpus.v1.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "feedback_form_classifier_candidates" / "direct_fg_v1"
)

DEV_FAMILY_FRACTION = 0.25
SEED = 1
MIN_DF = 1
C_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CLASS_WEIGHT_GRID = (None, "balanced")

# Manually labeled diagnostics.  They are never used to select a candidate.
NATURAL_PROBES = {
    "evaluative": (
        "That move was very good.",
        "You did a great job just then.",
        "That was the wrong thing to do.",
        "Thanks, that really helped.",
        "Your previous move slowed us down.",
        "You should have brought the plate sooner.",
        "Why did you leave the soup there?",
        "That handoff was excellent.",
        "I did not like that decision.",
        "The last delivery was a mistake.",
    ),
    "imperative": (
        "Could you grab an onion?",
        "Why don't you chop the onion?",
        "Please move the plate to the counter.",
        "Would you mind watching the pot?",
        "Get a clean dish next.",
        "Do not block the middle lane.",
        "I need you to serve the soup.",
        "How about taking the lower route?",
        "You should fetch another tomato.",
        "Let's clear this counter.",
    ),
    "descriptive": (
        "The left path is blocked.",
        "The soup is ready to serve.",
        "There is one onion beside the pot.",
        "The next order needs three tomatoes.",
        "The upper chopping board is free.",
        "A clean plate is on the table.",
        "The side corridor is the quickest route.",
        "We have ten seconds remaining.",
        "Orders leave through the service hatch.",
        "Carrying a plate uses one hand.",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _explicit_target(raw: dict, *, source: str, index: int) -> tuple[str, str]:
    """Resolve a direct fG label without consulting ``reference_type``."""

    expected = raw.get("expected_feedback_type")
    canonical = raw.get("classification_label")
    canonical_target = None
    if isinstance(canonical, str):
        canonical_target = {
            value: key for key, value in CANONICAL_FEEDBACK_LABELS.items()
        }.get(canonical)
    if expected in FEEDBACK_TYPES:
        if canonical is not None and canonical_target != expected:
            raise ValueError(
                f"{source} row {index}: explicit expected/canonical labels disagree"
            )
        return str(expected), "expected_feedback_type"
    if raw.get("label_source") == HUMAN_LABEL_SOURCE and canonical_target:
        return canonical_target, "classification_label"
    raise ValueError(
        f"{source} row {index}: no admissible explicit direct-fG label"
    )


_FAMILY_LABEL = re.compile(
    r"^(.*?):(?:evaluative|imperative|descriptive):(.*)$"
)


def shared_family_id(template_family: str) -> str:
    """Remove the class token so minimal contrasts share one split family."""

    match = _FAMILY_LABEL.fullmatch(template_family)
    if not match:
        raise ValueError(
            "direct boundary template_family must contain an explicit class token: "
            f"{template_family!r}"
        )
    return f"{match.group(1)}:{match.group(2)}"


def _convert_boundary_rows(path: Path) -> tuple[list[dict], dict]:
    raw_rows = _read_json_list(path)
    accepted = []
    excluded = Counter()
    for index, raw in enumerate(raw_rows, start=1):
        if raw.get("label_source") != DIRECT_BOUNDARY_LABEL_SOURCE:
            excluded[str(raw.get("label_source") or "missing")] += 1
            continue
        if raw.get("split") != "train":
            raise ValueError(f"boundary row {index}: source must be train-only")
        target, target_field = _explicit_target(
            raw, source="direct boundary", index=index
        )
        text = raw.get("text")
        template_family = raw.get("template_family")
        group_id = raw.get("group_id")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"boundary row {index}: missing text")
        if not isinstance(template_family, str) or not template_family:
            raise ValueError(f"boundary row {index}: missing template_family")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"boundary row {index}: missing group_id")
        accepted.append(
            {
                "text": text.strip(),
                "normalized_text": normalize_text(text),
                "label": target,
                "classification_label": CANONICAL_FEEDBACK_LABELS[target],
                "split": None,
                "source": "direct_speech_act_synthetic",
                "label_source": DIRECT_BOUNDARY_LABEL_SOURCE,
                "target_field": target_field,
                "group_id": f"direct_speech_act_synthetic:{group_id}",
                "template_family": template_family,
                # The source is already encoded in the family name.  Hash the
                # shared (label-stripped) value itself so the checked-in split
                # remains stable across loader/provenance refactors.
                "split_family": shared_family_id(template_family),
                # Five-way reference information is provenance only.
                "reference_type": raw.get("reference_type"),
            }
        )
    if not accepted:
        raise ValueError("no direct-speech-act boundary rows passed provenance gate")
    return accepted, {
        "input_rows": len(raw_rows),
        "accepted_rows": len(accepted),
        "excluded_rows": len(raw_rows) - len(accepted),
        "excluded_label_source_counts": dict(sorted(excluded.items())),
    }


def _convert_human_train(path: Path) -> list[dict]:
    rows = []
    for index, raw in enumerate(_read_json_list(path), start=1):
        if raw.get("split") != "train" or raw.get("label_source") != HUMAN_LABEL_SOURCE:
            raise ValueError(f"human train row {index}: invalid split/provenance")
        target, target_field = _explicit_target(raw, source="human train", index=index)
        text = raw.get("text")
        family = raw.get("template_family")
        feedback_id = raw.get("feedback_id")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"human train row {index}: missing text")
        if not isinstance(family, str) or not family:
            raise ValueError(f"human train row {index}: missing template_family")
        if not isinstance(feedback_id, str) or not feedback_id:
            raise ValueError(f"human train row {index}: missing feedback_id")
        rows.append(
            {
                "text": text.strip(),
                "normalized_text": normalize_text(text),
                "label": target,
                "classification_label": CANONICAL_FEEDBACK_LABELS[target],
                "split": "train",
                "source": "human_explicit",
                "label_source": HUMAN_LABEL_SOURCE,
                "target_field": target_field,
                "group_id": f"human_explicit:{feedback_id}",
                "template_family": family,
                "split_family": f"human_explicit:{family}",
                "reference_type": raw.get("reference_type"),
            }
        )
    return rows


def _split_boundary_families(rows: list[dict]) -> tuple[set[str], set[str]]:
    families = sorted(
        {row["split_family"] for row in rows},
        key=lambda value: (_digest(value), value),
    )
    if len(families) < 12:
        raise ValueError("at least 12 direct synthetic families are required")
    dev_count = max(3, round(len(families) * DEV_FAMILY_FRACTION))
    dev_families = set(families[:dev_count])
    train_families = set(families[dev_count:])
    if not train_families:
        raise ValueError("family split left no training families")
    return train_families, dev_families


def _assert_no_collisions(rows: list[dict]) -> dict:
    by_text: dict[str, list[dict]] = defaultdict(list)
    by_group_split: dict[str, set[str]] = defaultdict(set)
    by_family_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_text[row["normalized_text"]].append(row)
        by_group_split[row["group_id"]].add(row["split"])
        by_family_split[row["split_family"]].add(row["split"])
    duplicate_text_rows = sum(len(values) - 1 for values in by_text.values())
    conflicting_texts = sum(
        len({row["label"] for row in values}) > 1 for values in by_text.values()
    )
    cross_split_texts = sum(
        len({row["split"] for row in values}) > 1 for values in by_text.values()
    )
    cross_split_groups = sum(len(splits) > 1 for splits in by_group_split.values())
    cross_split_families = sum(len(splits) > 1 for splits in by_family_split.values())
    if any(
        (
            duplicate_text_rows,
            conflicting_texts,
            cross_split_texts,
            cross_split_groups,
            cross_split_families,
        )
    ):
        raise ValueError(
            "direct-fG split audit failed: "
            f"duplicates={duplicate_text_rows}, conflicts={conflicting_texts}, "
            f"text_overlap={cross_split_texts}, group_overlap={cross_split_groups}, "
            f"family_overlap={cross_split_families}"
        )
    return {
        "normalized_duplicate_rows": 0,
        "cross_label_text_conflicts": 0,
        "train_dev_text_overlap": 0,
        "train_dev_group_overlap": 0,
        "train_dev_family_overlap": 0,
    }


def build_direct_fg_corpus(
    boundary_path: Path = DEFAULT_BOUNDARY,
    human_train_path: Path = DEFAULT_HUMAN_TRAIN,
) -> tuple[list[dict], dict]:
    """Create a deterministic family/group-disjoint direct-fG split."""

    boundary_rows, boundary_audit = _convert_boundary_rows(boundary_path)
    human_rows = _convert_human_train(human_train_path)
    train_families, dev_families = _split_boundary_families(boundary_rows)
    for row in boundary_rows:
        row["split"] = "dev" if row["split_family"] in dev_families else "train"
    if {row["split_family"] for row in boundary_rows if row["split"] == "train"} != train_families:
        raise AssertionError("resolved train family membership changed unexpectedly")
    rows = [*boundary_rows, *human_rows]
    overlap_audit = _assert_no_collisions(rows)

    counts = {
        split: Counter(row["label"] for row in rows if row["split"] == split)
        for split in ("train", "dev")
    }
    if any(set(counts[split]) != set(FEEDBACK_TYPES) for split in counts):
        raise ValueError("all direct-fG classes must exist in train and dev")
    if min(counts["train"].values()) < 100 or min(counts["dev"].values()) < 30:
        raise ValueError(f"insufficient explicit direct-fG coverage: {counts}")
    if any(
        row["label_source"]
        not in {DIRECT_BOUNDARY_LABEL_SOURCE, HUMAN_LABEL_SOURCE}
        for row in rows
    ):
        raise AssertionError("non-direct label provenance passed the fail-closed gate")

    report = {
        "version": VERSION,
        "target": "direct_fG_surface_speech_act",
        "rows": len(rows),
        "split_rows": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "split_label_counts": {
            split: dict(sorted(count.items())) for split, count in counts.items()
        },
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "label_source_counts": dict(
            sorted(Counter(row["label_source"] for row in rows).items())
        ),
        "target_field_counts": dict(
            sorted(Counter(row["target_field"] for row in rows).items())
        ),
        "boundary_audit": boundary_audit,
        "family_split": {
            "method": "sha256_ranked_shared_template_family",
            "dev_fraction": DEV_FAMILY_FRACTION,
            "train_families": len(train_families),
            "dev_families": len(dev_families),
            "train_family_sha256": sorted(_digest(value) for value in train_families),
            "dev_family_sha256": sorted(_digest(value) for value in dev_families),
        },
        "split_integrity": overlap_audit,
        "reference_type_policy": {
            "used_as_target": False,
            "mapping_applied": False,
            "retained_as_provenance_only": True,
            "non_null_rows": sum(row["reference_type"] is not None for row in rows),
        },
        "quality_gate": {
            "status": "passed",
            "fail_closed": True,
            "minimum_train_rows_per_class": 100,
            "minimum_dev_rows_per_class": 30,
            "allowed_label_sources": [
                DIRECT_BOUNDARY_LABEL_SOURCE,
                HUMAN_LABEL_SOURCE,
            ],
            "llm_rule_teacher_rows_used": 0,
            "paper_reference_collapsed_rows_used_for_training_or_primary_dev": 0,
        },
    }
    return rows, report


def _metric_bundle(classifier, matrix, labels: list[str]) -> dict:
    raw = classifier.predict_proba(matrix)
    return {
        "classification": shared.classification_metrics(classifier, matrix, labels),
        "raw_probability": shared.probability_metrics(
            raw, labels, classifier.classes_
        ),
    }


def _natural_probe_rows() -> list[dict]:
    return [
        {"text": text, "label": label}
        for label in FEEDBACK_TYPES
        for text in NATURAL_PROBES[label]
    ]


def _load_secondary_rows(path: Path, *, required_split: str) -> list[dict]:
    rows = []
    for index, raw in enumerate(_read_json_list(path), start=1):
        if raw.get("split") != required_split:
            raise ValueError(f"{path.name} row {index}: expected split={required_split}")
        target, target_field = _explicit_target(
            raw, source=path.name, index=index
        )
        text = raw.get("text")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"{path.name} row {index}: missing text")
        rows.append(
            {
                "text": text.strip(),
                "label": target,
                "target_field": target_field,
                "reference_type": raw.get("reference_type"),
            }
        )
    return rows


def _evaluate_rows(artifact: dict, rows: list[dict]) -> dict:
    matrix = artifact["vectorizer"].transform([row["text"] for row in rows])
    labels = [row["label"] for row in rows]
    result = _metric_bundle(artifact["classifier"], matrix, labels)
    calibrated = shared.temperature_scaled_probabilities(
        artifact["classifier"],
        matrix,
        float(artifact["probability_calibration"]["temperature"]),
    )
    result["calibrated_probability"] = shared.probability_metrics(
        calibrated, labels, artifact["classifier"].classes_
    )
    return result


def _baseline_probe_report(path: Path, probes: list[dict]) -> dict:
    if not path.exists():
        return {"available": False, "path": str(path)}
    from joblib import load

    artifact = load(path)
    matrix = artifact["vectorizer"].transform([row["text"] for row in probes])
    return {
        "available": True,
        "path": str(path),
        "sha256": _sha256(path),
        "classification": shared.classification_metrics(
            artifact["classifier"], matrix, [row["label"] for row in probes]
        ),
    }


def _requested_phrase_report(artifact: dict, text: str) -> dict:
    classifier = artifact["classifier"]
    matrix = artifact["vectorizer"].transform([text])
    raw = classifier.predict_proba(matrix)[0]
    calibrated = shared.temperature_scaled_probabilities(
        classifier,
        matrix,
        float(artifact["probability_calibration"]["temperature"]),
    )[0]
    raw_values = {
        str(label): float(value) for label, value in zip(classifier.classes_, raw)
    }
    calibrated_values = {
        str(label): float(value)
        for label, value in zip(classifier.classes_, calibrated)
    }
    return {
        "text": text,
        "raw_probabilities": raw_values,
        "raw_prediction": max(raw_values, key=raw_values.get),
        "calibrated_probabilities": calibrated_values,
        "calibrated_prediction": max(calibrated_values, key=calibrated_values.get),
        "confidence_interpretation": "model score, not measured correctness",
    }


def train_direct_fg_candidate(
    rows: list[dict],
    *,
    seed: int = SEED,
    min_df: int = MIN_DF,
) -> tuple[dict, dict]:
    """Train with direct dev selection; secondary proxies cannot affect selection."""

    from sklearn.linear_model import LogisticRegression

    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("direct-fG training requires non-empty train and dev")

    vectorizer = shared._build_vectorizer(min_df)
    train_matrix = vectorizer.fit_transform([row["text"] for row in train_rows])
    dev_matrix = vectorizer.transform([row["text"] for row in dev_rows])
    train_labels = [row["label"] for row in train_rows]
    dev_labels = [row["label"] for row in dev_rows]

    candidates = []
    for c_value in C_GRID:
        for class_weight in CLASS_WEIGHT_GRID:
            classifier = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=3000,
                random_state=seed,
            )
            classifier.fit(train_matrix, train_labels)
            metrics = _metric_bundle(classifier, dev_matrix, dev_labels)
            candidates.append(
                {
                    "config": {"C": c_value, "class_weight": class_weight},
                    "classifier": classifier,
                    "metrics": metrics,
                }
            )

    def selection_key(candidate: dict) -> tuple:
        classification = candidate["metrics"]["classification"]
        minimum_recall = min(
            classification["per_class"][label]["recall"]
            for label in FEEDBACK_TYPES
        )
        return (
            classification["macro_f1"],
            minimum_recall,
            classification["balanced_accuracy"],
            classification["accuracy"],
            -candidate["metrics"]["raw_probability"]["negative_log_likelihood"],
            -float(candidate["config"]["C"]),
            candidate["config"]["class_weight"] is None,
        )

    selected = max(candidates, key=selection_key)
    classifier = selected["classifier"]
    temperature, calibration_report = shared.fit_temperature(
        classifier, dev_matrix, dev_labels
    )
    artifact = {
        "model_type": "feedback_form_tfidf_logistic_regression",
        "model_version": 4,
        "candidate_version": VERSION,
        "candidate_only": True,
        "promotion_status": "not_promoted",
        "target": "direct_fG_surface_speech_act",
        "input_mode": "raw_text",
        "labels": list(FEEDBACK_TYPES),
        "canonical_labels": dict(CANONICAL_FEEDBACK_LABELS),
        "vectorizer": vectorizer,
        "classifier": classifier,
        "selected_hyperparameters": selected["config"],
        "minimum_model_confidence": DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
        "probability_calibration": {
            "method": "temperature_scaling",
            "temperature": temperature,
            "version": "direct-fg-temperature-selection-dev-v1",
            "fit_split": "direct_fg_family_disjoint_selection_dev",
            "fit_split_used_for_model_selection": True,
            "independently_validated": False,
        },
        "reference_type_mapping": None,
        "reference_type_used_as_target": False,
        "data_membership": {
            "train_normalized_text_sha256": sorted(
                _digest(row["normalized_text"]) for row in train_rows
            ),
            "dev_normalized_text_sha256": sorted(
                _digest(row["normalized_text"]) for row in dev_rows
            ),
            "train_group_sha256": sorted(
                {_digest(row["group_id"]) for row in train_rows}
            ),
            "dev_group_sha256": sorted(
                {_digest(row["group_id"]) for row in dev_rows}
            ),
            "train_family_sha256": sorted(
                {_digest(row["split_family"]) for row in train_rows}
            ),
            "dev_family_sha256": sorted(
                {_digest(row["split_family"]) for row in dev_rows}
            ),
        },
    }
    report = {
        "version": VERSION,
        "target": "direct_fG_surface_speech_act",
        "candidate_only": True,
        "production_artifact_overwritten": False,
        "selection_protocol": {
            "primary": "direct_fG_family_and_group_disjoint_dev_macro_f1",
            "tie_breakers": [
                "minimum_per_class_recall",
                "balanced_accuracy",
                "accuracy",
                "raw_negative_log_likelihood",
                "smaller_C",
            ],
            "natural_probe_used_for_selection": False,
            "human_direct_dev_used_for_selection": False,
            "paper_reference_proxy_used_for_selection": False,
            "frozen_or_test_rows_used": 0,
        },
        "feature_config": {
            "version": shared.FEATURE_VERSION,
            "word_ngram_range": [1, 3],
            "char_wb_ngram_range": [2, 5],
            "min_df": min_df,
        },
        "split_sizes": {"train": len(train_rows), "dev": len(dev_rows)},
        "selected_hyperparameters": selected["config"],
        "direct_fg_family_group_disjoint_dev": {
            **selected["metrics"],
            "calibration": calibration_report,
        },
        "candidate_grid": [
            {"config": item["config"], "metrics": item["metrics"]}
            for item in candidates
        ],
        "reference_type_policy": {
            "mapping": None,
            "used_as_target": False,
            "five_class_role": "provenance_or_downstream_routing_only",
        },
        "confidence_limit": (
            "temperature is fitted and measured on selection dev; probabilities are "
            "model scores, not independently measured correctness"
        ),
        "frozen_test": {
            "path_opened": False,
            "labels_read": False,
            "metrics_computed": False,
        },
        "test_metrics": None,
    }
    return artifact, report


def run_pipeline(
    boundary_path: Path,
    human_train_path: Path,
    human_dev_path: Path,
    paper_proxy_path: Path,
    baseline_model_path: Path,
    corpus_output: Path,
    output_dir: Path,
) -> dict:
    from joblib import dump

    rows, corpus_report = build_direct_fg_corpus(boundary_path, human_train_path)
    write_json(corpus_output, rows)
    corpus_report["output"] = str(corpus_output)
    corpus_report["output_sha256"] = _sha256(corpus_output)

    artifact, report = train_direct_fg_candidate(rows)
    probes = _natural_probe_rows()
    occupied = {row["normalized_text"] for row in rows}
    probe_normalized = [normalize_text(row["text"]) for row in probes]
    if len(set(probe_normalized)) != len(probe_normalized):
        raise ValueError("natural probe contains normalized duplicates")
    if occupied & set(probe_normalized):
        raise ValueError("natural probe has exact corpus overlap")

    human_dev = _load_secondary_rows(human_dev_path, required_split="dev")
    paper_proxy = _load_secondary_rows(paper_proxy_path, required_split="dev")
    report["corpus_audit"] = corpus_report
    report["secondary_evaluations"] = {
        "human_explicit_dev": {
            "selection_role": "none",
            "all_three_labels_present": (
                {row["label"] for row in human_dev} == set(FEEDBACK_TYPES)
            ),
            "metrics": _evaluate_rows(artifact, human_dev),
        },
        "paper_reference_collapse_proxy": {
            "selection_role": "none",
            "target_semantics_match_direct_fG": False,
            "interpretation": (
                "secondary compatibility diagnostic only; low performance does not "
                "invalidate a direct-speech-act target"
            ),
            "reference_type_counts": dict(
                sorted(Counter(row["reference_type"] for row in paper_proxy).items())
            ),
            "metrics": _evaluate_rows(artifact, paper_proxy),
        },
        "natural_probe": {
            "selection_role": "none",
            "manually_labeled": True,
            "exact_corpus_overlap": 0,
            "candidate": _evaluate_rows(artifact, probes),
            "production_baseline": _baseline_probe_report(
                baseline_model_path, probes
            ),
        },
    }
    report["requested_phrase"] = _requested_phrase_report(
        artifact, "That move was very good."
    )

    primary = report["direct_fg_family_group_disjoint_dev"]["classification"]
    natural = report["secondary_evaluations"]["natural_probe"]["candidate"][
        "classification"
    ]
    report["candidate_quality_gate"] = {
        "direct_dev_accuracy_at_least_0_87": primary["accuracy"] >= 0.87,
        "direct_dev_macro_f1_at_least_0_87": primary["macro_f1"] >= 0.87,
        "minimum_direct_dev_class_recall_at_least_0_70": min(
            primary["per_class"][label]["recall"] for label in FEEDBACK_TYPES
        )
        >= 0.70,
        "natural_probe_accuracy_at_least_0_80": natural["accuracy"] >= 0.80,
        "split_integrity_passed": all(
            value == 0 for value in corpus_report["split_integrity"].values()
        ),
        "reference_type_used_as_target": False,
        "status": "passed_research_candidate_not_promoted",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    report_path = output_dir / "model.report.json"
    manifest_path = output_dir / "manifest.json"
    dump(artifact, model_path)
    write_json(report_path, report)
    manifest = {
        "version": VERSION,
        "candidate_only": True,
        "promotion_status": "not_promoted",
        "production_artifact_overwritten": False,
        "target": "direct_fG_surface_speech_act",
        "inputs": {
            "boundary": {"path": str(boundary_path), "sha256": _sha256(boundary_path)},
            "human_train": {
                "path": str(human_train_path),
                "sha256": _sha256(human_train_path),
            },
            "human_dev_secondary": {
                "path": str(human_dev_path),
                "sha256": _sha256(human_dev_path),
            },
            "paper_proxy_secondary": {
                "path": str(paper_proxy_path),
                "sha256": _sha256(paper_proxy_path),
            },
        },
        "outputs": {
            "corpus": {"path": str(corpus_output), "sha256": _sha256(corpus_output)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        },
        "selection": report["selection_protocol"],
        "reference_type_policy": report["reference_type_policy"],
        "quality_gate": report["candidate_quality_gate"],
        "frozen_test": report["frozen_test"],
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN)
    parser.add_argument("--human-dev", type=Path, default=DEFAULT_HUMAN_DEV)
    parser.add_argument("--paper-proxy", type=Path, default=DEFAULT_PAPER_PROXY)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_pipeline(
        args.boundary,
        args.human_train,
        args.human_dev,
        args.paper_proxy,
        args.baseline_model,
        args.corpus_output,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
