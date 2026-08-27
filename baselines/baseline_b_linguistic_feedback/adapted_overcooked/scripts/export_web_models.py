"""Export the deployed feedback models to deterministic browser-readable JSON.

The desktop runtime keeps sklearn and PyTorch artifacts as the source of truth.
This exporter only changes their container format; it does not retrain, prune, or
alter model parameters.  The generated manifest binds every browser artifact to
the exact source SHA-256 used by the audited Python runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
ADAPTED_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER = (
    ADAPTED_ROOT / "outputs" / "feedback_form_classifier" / "model.joblib"
)
DEFAULT_CLASSIFIER_REPORT = (
    ADAPTED_ROOT / "outputs" / "feedback_form_classifier" / "model.report.json"
)
DEFAULT_SHADOW_CLASSIFIER = (
    ADAPTED_ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1_model"
    / "model.joblib"
)
DEFAULT_SHADOW_CLASSIFIER_REPORT = (
    ADAPTED_ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1_model"
    / "training_report.json"
)
DEFAULT_SHADOW_FROZEN_CONFIG = DEFAULT_SHADOW_CLASSIFIER.parent / "frozen_config.json"
DEFAULT_SHADOW_CANDIDATE_MANIFEST = DEFAULT_SHADOW_CLASSIFIER.parent / "manifest.json"
DEFAULT_SHADOW_PREDICTOR = (
    ADAPTED_ROOT / "scripts" / "predict_direct_fg_boundary_shadow_v1.py"
)
SHADOW_MODEL_SHA256 = "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55"
SHADOW_REPORT_SHA256 = "4b23b8ca052e11c530929d640f1a6c15bbf24db7ae8e20c78b13b99e66ef9019"
SHADOW_CONFIG_SHA256 = "cb0e4ccf00c96030167428ee057dbdcf134f3f075d6af77c2616445483191327"
SHADOW_CANDIDATE_MANIFEST_SHA256 = "60258ce84ed959aa75f91c6bb43cd02975c3377300b8b0426fd2aeb223db5bee"
SHADOW_PREDICTOR_SHA256 = "d76fe50fe2e7d079c2709d37c020c88227c5708d087595347f483d6b63c8c507"
SHADOW_TEMPERATURE = 0.0823618560384671
SHADOW_BROWSER_SHA256 = "580ab91c21d988d4ea7e146b70e810b68d0e454372b5cf3cb0ea32abb7f81dc3"
SHADOW_BROWSER_FILENAME = f"feedback-form-boundary-shadow-v1-{SHADOW_BROWSER_SHA256}.json"
SHADOW_MANIFEST_FILENAME = "manifest-boundary-shadow-v1.json"
ROUTE2_BROWSER_SHA256 = "fa9ddb2aff1a451713e62417fcd3bca56a9e06987fbc6d7dbec2cf1651b80177"
SHADOW_HYPERPARAMETERS = {
    "config_id": "C=4;min_df=2",
    "C": 4.0,
    "min_df": 2,
    "word_ngram_range": [1, 2],
    "char_ngram_range": [3, 5],
    "char_weight": 0.7,
}
DEFAULT_ROUTE2 = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_v5_seed137_selected"
    / "ensemble_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "web" / "public" / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    return hashlib.sha256(encoded + b"\n").hexdigest()


def finite_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("model contains a non-finite numeric parameter")
    return [float(value) for value in array]


def export_classifier(
    model_path: Path,
    report_path: Path,
    output_path: Path,
    *,
    release: str = "production",
) -> str:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("feedback classifier artifact must be a dictionary")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    transformer_weights = dict(getattr(vectorizer, "transformer_weights", {}) or {})
    transformers = []
    feature_offset = 0
    for name, fitted in vectorizer.transformer_list:
        vocabulary = {
            str(term): int(index)
            for term, index in fitted.vocabulary_.items()
        }
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError(f"{name} vectorizer vocabulary is not contiguous")
        transformer = {
            "name": str(name),
            "analyzer": str(fitted.analyzer),
            "ngram_range": [int(value) for value in fitted.ngram_range],
            "lowercase": bool(fitted.lowercase),
            "strip_accents": fitted.strip_accents,
            "sublinear_tf": bool(fitted.sublinear_tf),
            "norm": fitted.norm,
            "token_pattern": fitted.token_pattern,
            "feature_offset": feature_offset,
            "vocabulary": vocabulary,
            "idf": finite_list(fitted.idf_),
        }
        weight = float(transformer_weights.get(name, 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name} vectorizer weight must be finite and positive")
        # Omit the default to keep the existing production export byte-stable.
        if weight != 1.0:
            transformer["weight"] = weight
        transformers.append(transformer)
        feature_offset += len(vocabulary)

    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    if coefficients.shape[1] != feature_offset:
        raise ValueError(
            "classifier/vectorizer feature mismatch: "
            f"{coefficients.shape[1]} != {feature_offset}"
        )
    if release == "production":
        if artifact.get("artifact_version") == "direct-fg-boundary-shadow-model-v1":
            raise ValueError("shadow classifier requires --classifier-release shadow-preview")
        calibration = dict(artifact.get("probability_calibration") or {})
        model_type = artifact.get("model_type")
        model_version = artifact.get("model_version")
        canonical_labels = artifact.get("canonical_labels")
        minimum_confidence = float(artifact["minimum_model_confidence"])
        claim_scope = {
            "paper_reference_proxy_accuracy": 0.9166666666666666,
            "paper_reference_proxy_examples": 96,
            "paper_reference_proxy_is_previously_exposed": True,
            "independent_current_player_accuracy": None,
            "calibration_independently_validated": False,
        }
    elif release == "shadow-preview":
        expected_hashes = {
            "model": (model_path, SHADOW_MODEL_SHA256),
            "report": (report_path, SHADOW_REPORT_SHA256),
            "frozen config": (DEFAULT_SHADOW_FROZEN_CONFIG, SHADOW_CONFIG_SHA256),
            "candidate manifest": (
                DEFAULT_SHADOW_CANDIDATE_MANIFEST,
                SHADOW_CANDIDATE_MANIFEST_SHA256,
            ),
            "predictor": (DEFAULT_SHADOW_PREDICTOR, SHADOW_PREDICTOR_SHA256),
        }
        for label, (path, expected_hash) in expected_hashes.items():
            if sha256(path) != expected_hash:
                raise ValueError(f"shadow {label} SHA-256 changed")
        if artifact.get("artifact_version") != "direct-fg-boundary-shadow-model-v1":
            raise ValueError("shadow-preview requires the boundary shadow v1 artifact")
        if artifact.get("status") != "shadow_diagnostic_only":
            raise ValueError("shadow classifier status changed")
        if artifact.get("promotion_eligible") is not False:
            raise ValueError("shadow classifier must remain non-promotable")
        if artifact.get("production_artifact") is not False:
            raise ValueError("shadow classifier must not claim production status")
        if artifact.get("model_input_contract") != {
            "input_type": "raw_string_only",
            "field": "text",
            "metadata_features": [],
        }:
            raise ValueError("shadow classifier input contract changed")
        classes = [str(value) for value in artifact.get("classes", [])]
        if classes != ["descriptive", "evaluative", "imperative"]:
            raise ValueError("shadow classifier class order changed")
        if [str(value) for value in classifier.classes_] != classes:
            raise ValueError("shadow classifier fitted class order changed")
        if artifact.get("selected_hyperparameters") != SHADOW_HYPERPARAMETERS:
            raise ValueError("shadow classifier hyperparameters changed")
        temperature = float(artifact["temperature"])
        if not np.isfinite(temperature) or temperature != SHADOW_TEMPERATURE:
            raise ValueError("shadow classifier temperature changed")
        calibration = {
            "method": "temperature_scaling",
            "temperature": temperature,
            "version": "direct-fg-boundary-shadow-train-oof-v1",
            "fit_split": "synthetic_train_nested_group_oof_only",
            "independently_validated": False,
            "scope": "train_oof",
        }
        model_type = "direct_fg_tfidf_logistic_regression_shadow"
        model_version = artifact["artifact_version"]
        canonical_labels = {
            "descriptive": "Descriptive",
            "evaluative": "Evaluative",
            "imperative": "Imperative",
        }
        # Keep the existing Web preview abstention policy. This threshold was
        # not estimated on an independent player set and is not an accuracy claim.
        minimum_confidence = 0.55
        private_metrics = report.get("private_diagnostic_v2") or {}
        claim_scope = {
            "diagnostic_shadow_preview": True,
            "synthetic_training_rows": report.get("train_input", {}).get("rows"),
            "private_diagnostic_accuracy": private_metrics.get("accuracy"),
            "independent_current_player_accuracy": None,
            "calibration_independently_validated": False,
            "confidence_ready_for_player_ui": False,
            "minimum_confidence_source": "existing_web_preview_policy",
        }
    else:
        raise ValueError(f"unsupported classifier release: {release}")

    source = {
        "model_sha256": sha256(model_path),
        "report_sha256": sha256(report_path),
    }
    if release == "shadow-preview":
        source.update(
            {
                "frozen_config_sha256": SHADOW_CONFIG_SHA256,
                "predictor_sha256": SHADOW_PREDICTOR_SHA256,
                "candidate_manifest_sha256": SHADOW_CANDIDATE_MANIFEST_SHA256,
            }
        )
    payload = {
        "schema_version": "durf-feedback-form-browser-v1",
        "source": source,
        "model_type": model_type,
        "model_version": model_version,
        "classes": [str(value) for value in classifier.classes_],
        "canonical_labels": canonical_labels,
        "minimum_confidence": minimum_confidence,
        "calibration": calibration,
        "transformers": transformers,
        "classifier": {
            "kind": "multinomial_logistic_regression",
            "coefficients": [finite_list(row) for row in coefficients],
            "intercept": finite_list(classifier.intercept_),
        },
        "claim_scope": claim_scope,
    }
    return write_json(output_path, payload)


def encode_float32(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Route2 checkpoint contains a non-finite parameter")
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": "float32-le",
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def export_route2(manifest_path: Path, output_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "route2-ensemble-v1":
        raise ValueError("unsupported Route2 ensemble manifest")
    features = [str(value) for value in manifest["features"]]
    members = []
    for entry in manifest["members"]:
        checkpoint_path = manifest_path.parent / entry["checkpoint"]
        checkpoint_hash = sha256(checkpoint_path)
        if checkpoint_hash != entry["checkpoint_sha256"]:
            raise ValueError(f"Route2 checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if [str(value) for value in checkpoint["features"]] != features:
            raise ValueError(f"Route2 feature order mismatch: {checkpoint_path}")
        config = dict(checkpoint["config"])
        state = checkpoint["state_dict"]
        members.append(
            {
                "fold": int(entry["fold"]),
                "checkpoint_sha256": checkpoint_hash,
                "vocab": {
                    str(token): int(index)
                    for token, index in checkpoint["vocab"].items()
                },
                "config": {
                    "vocab_size": int(config["vocab_size"]),
                    "n_features": int(config["n_features"]),
                    "embedding_dim": int(state["embedding.weight"].shape[1]),
                    "hidden_dim": int(state["fc1.bias"].shape[0]),
                    "use_feature_counts": bool(config.get("use_feature_counts", False)),
                },
                "parameters": {
                    name: encode_float32(state[name])
                    for name in (
                        "embedding.weight",
                        "fc1.weight",
                        "fc1.bias",
                        "fc2.weight",
                        "fc2.bias",
                    )
                },
            }
        )
    if len(members) != 10:
        raise ValueError(f"paper Route2 export requires 10 members, got {len(members)}")
    payload = {
        "schema_version": "durf-route2-browser-v1",
        "source": {
            "ensemble_manifest_sha256": sha256(manifest_path),
            "ensemble_identity": manifest.get("manifest_sha256"),
        },
        "features": features,
        "members": members,
        "update": {
            "kind": "paper_independent_gaussian",
            "prior_variance": 25.0,
            "observation_precision": 2.0,
        },
        "claim_scope": {
            "paper_core_architecture_aligned": True,
            "original_paper_reward_dimensions": 9,
            "overcooked_reward_dimensions": len(features),
            "trained_on_synthetic_overcooked_feedback": True,
            "independent_current_player_accuracy": None,
        },
    }
    return write_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-release",
        choices=("production", "shadow-preview"),
        default="production",
    )
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--classifier-report", type=Path)
    parser.add_argument("--classifier-output-name")
    parser.add_argument("--route2-manifest", type=Path, default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.classifier_release == "shadow-preview":
        classifier_source = args.classifier or DEFAULT_SHADOW_CLASSIFIER
        classifier_report = args.classifier_report or DEFAULT_SHADOW_CLASSIFIER_REPORT
        classifier_output_name = args.classifier_output_name or SHADOW_BROWSER_FILENAME
    else:
        classifier_source = args.classifier or DEFAULT_CLASSIFIER
        classifier_report = args.classifier_report or DEFAULT_CLASSIFIER_REPORT
        classifier_output_name = (
            args.classifier_output_name or "feedback-form-v3.json"
        )
    if (
        Path(classifier_output_name).name != classifier_output_name
        or not classifier_output_name.endswith(".json")
    ):
        raise ValueError("classifier output name must be one JSON filename")
    if (
        args.classifier_release == "shadow-preview"
        and classifier_output_name == "feedback-form-v3.json"
    ):
        raise ValueError("shadow preview cannot overwrite feedback-form-v3.json")
    classifier_path = args.output_dir / classifier_output_name
    route2_path = args.output_dir / "route2-v5.json"
    classifier_hash = export_classifier(
        classifier_source.resolve(),
        classifier_report.resolve(),
        classifier_path,
        release=args.classifier_release,
    )
    if (
        args.classifier_release == "shadow-preview"
        and (
            classifier_output_name != SHADOW_BROWSER_FILENAME
            or classifier_hash != SHADOW_BROWSER_SHA256
        )
    ):
        raise ValueError("shadow browser artifact content identity changed")
    if args.classifier_release == "shadow-preview":
        route2_hash = sha256(route2_path)
        if route2_hash != ROUTE2_BROWSER_SHA256:
            raise ValueError("existing Route2 browser artifact identity changed")
    else:
        route2_hash = export_route2(args.route2_manifest.resolve(), route2_path)
    manifest = {
        "schema_version": "durf-browser-model-manifest-v1",
        "models": {
            "feedback_form": {
                "path": f"/models/{classifier_output_name}",
                "sha256": classifier_hash,
            },
            "route2": {
                "path": "/models/route2-v5.json",
                "sha256": route2_hash,
            },
        },
    }
    manifest_name = (
        SHADOW_MANIFEST_FILENAME
        if args.classifier_release == "shadow-preview"
        else "manifest.json"
    )
    manifest_hash = write_json(args.output_dir / manifest_name, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "classifier_sha256": classifier_hash,
                "route2_sha256": route2_hash,
                "manifest_sha256": manifest_hash,
                "manifest_path": str((args.output_dir / manifest_name).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
