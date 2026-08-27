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


def export_classifier(model_path: Path, report_path: Path, output_path: Path) -> str:
    artifact = joblib.load(model_path)
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    transformers = []
    feature_offset = 0
    for name, fitted in vectorizer.transformer_list:
        vocabulary = {
            str(term): int(index)
            for term, index in fitted.vocabulary_.items()
        }
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError(f"{name} vectorizer vocabulary is not contiguous")
        transformers.append(
            {
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
        )
        feature_offset += len(vocabulary)

    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    if coefficients.shape[1] != feature_offset:
        raise ValueError(
            "classifier/vectorizer feature mismatch: "
            f"{coefficients.shape[1]} != {feature_offset}"
        )
    calibration = dict(artifact.get("probability_calibration") or {})
    payload = {
        "schema_version": "durf-feedback-form-browser-v1",
        "source": {
            "model_sha256": sha256(model_path),
            "report_sha256": sha256(report_path),
        },
        "model_type": artifact.get("model_type"),
        "model_version": artifact.get("model_version"),
        "classes": [str(value) for value in classifier.classes_],
        "canonical_labels": artifact.get("canonical_labels"),
        "minimum_confidence": float(artifact["minimum_model_confidence"]),
        "calibration": calibration,
        "transformers": transformers,
        "classifier": {
            "kind": "multinomial_logistic_regression",
            "coefficients": [finite_list(row) for row in coefficients],
            "intercept": finite_list(classifier.intercept_),
        },
        "claim_scope": {
            "paper_reference_proxy_accuracy": 0.9166666666666666,
            "paper_reference_proxy_examples": 96,
            "paper_reference_proxy_is_previously_exposed": True,
            "independent_current_player_accuracy": None,
            "calibration_independently_validated": False,
        },
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
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument(
        "--classifier-report", type=Path, default=DEFAULT_CLASSIFIER_REPORT
    )
    parser.add_argument("--route2-manifest", type=Path, default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier_path = args.output_dir / "feedback-form-v3.json"
    route2_path = args.output_dir / "route2-v5.json"
    classifier_hash = export_classifier(
        args.classifier.resolve(),
        args.classifier_report.resolve(),
        classifier_path,
    )
    route2_hash = export_route2(args.route2_manifest.resolve(), route2_path)
    manifest = {
        "schema_version": "durf-browser-model-manifest-v1",
        "models": {
            "feedback_form": {
                "path": "/models/feedback-form-v3.json",
                "sha256": classifier_hash,
            },
            "route2": {
                "path": "/models/route2-v5.json",
                "sha256": route2_hash,
            },
        },
    }
    manifest_hash = write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "classifier_sha256": classifier_hash,
                "route2_sha256": route2_hash,
                "manifest_sha256": manifest_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
