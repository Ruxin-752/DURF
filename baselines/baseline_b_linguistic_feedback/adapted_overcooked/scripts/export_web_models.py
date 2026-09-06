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
import io
import inspect
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
DEFAULT_CLASSIFIER_RELEASE_CONTRACT = (
    ADAPTED_ROOT
    / "outputs"
    / "feedback_form_classifier"
    / "release.production.json"
)
PRODUCTION_RELEASE_CONTRACT_CANONICAL_SHA256 = (
    "f1bace4cfb0b3abad28518299ea6b6607f470e9a59d49698682114b8d7cd0ac5"
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
DEFAULT_SHADOW_DIAGNOSTIC_RECEIPT = (
    DEFAULT_SHADOW_CLASSIFIER.parent / "private_diagnostic_v2_one_shot_receipt.json"
)
DEFAULT_SHADOW_PREDICTOR = (
    ADAPTED_ROOT / "scripts" / "predict_direct_fg_boundary_shadow_v1.py"
)
SHADOW_MODEL_SHA256 = "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55"
SHADOW_REPORT_SHA256 = "4b23b8ca052e11c530929d640f1a6c15bbf24db7ae8e20c78b13b99e66ef9019"
SHADOW_CONFIG_SHA256 = "cb0e4ccf00c96030167428ee057dbdcf134f3f075d6af77c2616445483191327"
SHADOW_CANDIDATE_MANIFEST_SHA256 = "60258ce84ed959aa75f91c6bb43cd02975c3377300b8b0426fd2aeb223db5bee"
SHADOW_PREDICTOR_SHA256 = "d76fe50fe2e7d079c2709d37c020c88227c5708d087595347f483d6b63c8c507"
SHADOW_DIAGNOSTIC_RECEIPT_SHA256 = (
    "5cef25eb4ddba21f182ec4ba9bdc8f1a135ad741b3ef49e9593ab055454ce5d2"
)
# The train-OOF fitted temperature (0.0823618560384671) is intentionally not
# exported to the player-facing trial: it made uncalibrated scores saturate at
# 100%.  T=1 preserves the classifier's raw multinomial softmax output.
SHADOW_DISPLAY_TEMPERATURE = 1.0
SHADOW_BROWSER_SHA256 = "2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef"
SHADOW_BROWSER_FILENAME = f"feedback-form-boundary-shadow-raw-v2-{SHADOW_BROWSER_SHA256}.json"
SHADOW_MANIFEST_FILENAME = "manifest-boundary-shadow-raw-v2.json"
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
DEFAULT_ROUTE2_MANIFEST_SHA256 = (
    "b863aea76336a7c200dee950a6874032793505ea0b54a898750bb57ac7fd168a"
)
DEFAULT_ROUTE2_ENSEMBLE_IDENTITY = (
    "256c9551a738c53c723e00d09364ea1d15b6ab52daf3e515c98e772f9a29c180"
)
DEFAULT_OUTPUT = ROOT / "web" / "public" / "models"
TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = (
    "weights_only" in inspect.signature(torch.load).parameters
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bytes_with_sha256(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def read_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = read_bytes_with_sha256(path)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return decoded, digest


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for field in dotted_path.split("."):
        if not isinstance(value, dict) or field not in value:
            raise ValueError(f"missing required report field: {dotted_path}")
        value = value[field]
    return value


def validate_source_model_fields(
    report: dict[str, Any],
    fields: list[str],
    expected_model_sha256: str,
    *,
    report_id: str,
) -> None:
    if not fields:
        raise ValueError(f"release report has no source-model binding: {report_id}")
    for field in fields:
        if nested_value(report, field) != expected_model_sha256:
            raise ValueError(
                f"release report source model SHA-256 mismatch: {report_id}.{field}"
            )


def release_report_path(contract_path: Path, filename: Any) -> Path:
    relative = Path(str(filename))
    if relative.name != str(relative) or relative.suffix != ".json":
        raise ValueError("release report must be one JSON filename")
    return (contract_path.parent / relative).resolve()


def validate_production_release(
    model_sha256: str,
    report_path: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    contract_path = contract_path.resolve()
    contract, contract_sha256 = read_json_with_sha256(contract_path)
    if (
        contract_path == DEFAULT_CLASSIFIER_RELEASE_CONTRACT.resolve()
        and canonical_json_sha256(contract)
        != PRODUCTION_RELEASE_CONTRACT_CANONICAL_SHA256
    ):
        raise ValueError("production release contract canonical SHA-256 changed")
    if contract.get("schema_version") != "durf-feedback-form-release-contract-v1":
        raise ValueError("unsupported production release contract")
    if contract.get("model_sha256") != model_sha256:
        raise ValueError("production model SHA-256 does not match release contract")
    release_metadata = contract.get("release")
    if not isinstance(release_metadata, dict) or release_metadata != {
        "channel": "production",
        "status": "active_legacy_default",
        "default_eligible": True,
        "promotion_eligible": False,
        "promotion_status": "legacy_active_not_requalified",
        "promotion_reason": "No independent current-player evaluation is bound to this model.",
    }:
        raise ValueError("production release status or promotion contract changed")
    training_metadata = contract.get("training")
    if not isinstance(training_metadata, dict) or training_metadata != {
        "trained": True,
        "frozen": True,
        "report_id": "training",
    }:
        raise ValueError("production training status changed")
    claims = contract.get("claims")
    if not isinstance(claims, dict) or claims != {
        "independent_current_player_accuracy": None,
        "calibration_independently_validated": False,
    }:
        raise ValueError("production claim scope changed")

    report_specs = contract.get("reports")
    if not isinstance(report_specs, list) or not report_specs:
        raise ValueError("production release must reference bound reports")
    reports: dict[str, dict[str, Any]] = {}
    report_bindings: list[dict[str, Any]] = []
    for spec in report_specs:
        if not isinstance(spec, dict):
            raise ValueError("production release report specification is invalid")
        report_id = str(spec.get("id", ""))
        if not report_id or report_id in reports:
            raise ValueError("production release report id is missing or duplicated")
        source_path = release_report_path(contract_path, spec.get("file"))
        payload, actual_sha256 = read_json_with_sha256(source_path)
        if actual_sha256 != spec.get("sha256"):
            raise ValueError(f"production release report SHA-256 changed: {report_id}")
        source_fields = spec.get("source_model_sha256_fields")
        if not isinstance(source_fields, list) or not all(
            isinstance(value, str) for value in source_fields
        ):
            raise ValueError(f"production release report binding is invalid: {report_id}")
        validate_source_model_fields(
            payload,
            source_fields,
            model_sha256,
            report_id=report_id,
        )
        reports[report_id] = {"payload": payload, "path": source_path}
        report_bindings.append(
            {
                "id": report_id,
                "role": str(spec.get("role", "unspecified")),
                "report_sha256": actual_sha256,
                "source_model_sha256": model_sha256,
                "binding_fields": list(source_fields),
            }
        )

    training_id = str(training_metadata["report_id"])
    if training_id not in reports or reports[training_id]["path"] != report_path.resolve():
        raise ValueError("selected production training report is not release-bound")
    display_id = str(contract.get("display_evidence_id", ""))
    if display_id not in reports or display_id == training_id:
        raise ValueError("production display evidence is not release-bound")
    training_report = reports[training_id]["payload"]
    evidence_report = reports[display_id]["payload"]
    training_sha256 = next(
        item["report_sha256"] for item in report_bindings if item["id"] == training_id
    )
    if evidence_report.get("training_report_sha256") != training_sha256:
        raise ValueError("production evidence report references another training report")
    if (
        evidence_report.get("model_sha256_unchanged") is not True
        or evidence_report.get("current_overcooked_player_accuracy_claim") is not False
        or evidence_report.get("independent_human_speech_form_gold") is not False
        or evidence_report.get("independent_sealed_test") is not False
    ):
        raise ValueError("production evidence claim scope changed")
    metrics = evidence_report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("production evidence metrics are missing")
    evidence = {
        "id": display_id,
        "report_sha256": next(
            item["report_sha256"] for item in report_bindings if item["id"] == display_id
        ),
        "source_model_sha256": model_sha256,
        "role": str(evidence_report.get("benchmark_role")),
        "scope": str(evidence_report.get("claim_scope")),
        "rows": int(metrics["rows"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "human_or_player_gold": False,
        "independent_current_player": False,
        "previously_exposed": bool(evidence_report.get("previously_exposed_source_test")),
        "requested_accuracy_target": float(
            evidence_report["diagnostic_accuracy_threshold"]
        ),
        "target_passed": bool(evidence_report.get("diagnostic_threshold_met")),
    }
    model_card = {
        "schema_version": "durf-feedback-form-model-card-v1",
        "source_model_sha256": model_sha256,
        "release": dict(release_metadata),
        "training": {
            "trained": True,
            "frozen": True,
            "status": "trained",
            "report_sha256": training_sha256,
            "source_model_sha256": model_sha256,
        },
        "reports": report_bindings,
        "display_evidence_id": display_id,
        "evidence": [evidence],
        "claims": dict(claims),
    }
    return training_report, model_card, evidence, contract_sha256


def validate_shadow_release(
    model_sha256: str,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if model_sha256 != SHADOW_MODEL_SHA256:
        raise ValueError("shadow model SHA-256 changed")
    report, report_sha256 = read_json_with_sha256(report_path)
    _, config_sha256 = read_json_with_sha256(DEFAULT_SHADOW_FROZEN_CONFIG)
    manifest, manifest_sha256 = read_json_with_sha256(
        DEFAULT_SHADOW_CANDIDATE_MANIFEST
    )
    receipt, receipt_sha256 = read_json_with_sha256(DEFAULT_SHADOW_DIAGNOSTIC_RECEIPT)
    _, predictor_sha256 = read_bytes_with_sha256(DEFAULT_SHADOW_PREDICTOR)
    actual_hashes = {
        "report": (report_sha256, SHADOW_REPORT_SHA256),
        "frozen config": (config_sha256, SHADOW_CONFIG_SHA256),
        "candidate manifest": (
            manifest_sha256,
            SHADOW_CANDIDATE_MANIFEST_SHA256,
        ),
        "predictor": (predictor_sha256, SHADOW_PREDICTOR_SHA256),
        "diagnostic receipt": (
            receipt_sha256,
            SHADOW_DIAGNOSTIC_RECEIPT_SHA256,
        ),
    }
    for label, (actual_hash, expected_hash) in actual_hashes.items():
        if actual_hash != expected_hash:
            raise ValueError(f"shadow {label} SHA-256 changed")
    if (
        manifest.get("status") != "shadow_diagnostic_only"
        or manifest.get("promotion_eligible") is not False
        or nested_value(manifest, "artifacts.model.sha256") != model_sha256
        or nested_value(manifest, "artifacts.training_report.sha256") != report_sha256
    ):
        raise ValueError("shadow manifest status, promotion, or report binding changed")
    if (
        report.get("status") != "trained_shadow_diagnostic_only"
        or report.get("promotion_eligible") is not False
        or report.get("production_promotion_eligible") is not False
    ):
        raise ValueError("shadow training report status or promotion changed")
    validate_source_model_fields(
        receipt,
        ["artifact_binding.model_sha256"],
        model_sha256,
        report_id="private_diagnostic_v2_one_shot_receipt",
    )
    if (
        nested_value(receipt, "artifact_binding.frozen_config_file_sha256")
        != SHADOW_CONFIG_SHA256
        or nested_value(receipt, "artifact_binding.predictor_sha256")
        != SHADOW_PREDICTOR_SHA256
        or nested_value(receipt, "interpretation.promotion_eligible") is not False
        or nested_value(receipt, "interpretation.failed_target") is not True
    ):
        raise ValueError("shadow diagnostic receipt contract changed")
    scored = nested_value(receipt, "results.scored")
    if not isinstance(scored, dict):
        raise ValueError("shadow diagnostic metrics are missing")
    evidence = {
        "id": "private_diagnostic_v2_one_shot",
        "report_sha256": receipt_sha256,
        "source_model_sha256": model_sha256,
        "role": str(nested_value(receipt, "diagnostic.role")),
        "scope": str(nested_value(receipt, "interpretation.limitations")[0]),
        "rows": int(scored["rows"]),
        "accuracy": float(scored["accuracy"]),
        "macro_f1": float(scored["macro_f1"]),
        "human_or_player_gold": bool(
            nested_value(receipt, "diagnostic.human_or_player_gold")
        ),
        "independent_current_player": False,
        "previously_exposed": False,
        "requested_accuracy_target": float(
            nested_value(receipt, "interpretation.requested_accuracy_target")
        ),
        "target_passed": not bool(nested_value(receipt, "interpretation.failed_target")),
    }
    release_metadata = {
        "channel": "shadow-preview",
        "status": str(manifest["status"]),
        "default_eligible": False,
        "promotion_eligible": False,
        "promotion_status": str(nested_value(receipt, "interpretation.status")),
        "promotion_reason": str(nested_value(receipt, "interpretation.limitations")[1]),
    }
    model_card = {
        "schema_version": "durf-feedback-form-model-card-v1",
        "source_model_sha256": model_sha256,
        "release": release_metadata,
        "training": {
            "trained": True,
            "frozen": True,
            "status": str(report["status"]),
            "report_sha256": report_sha256,
            "source_model_sha256": model_sha256,
        },
        "reports": [
            {
                "id": "training",
                "role": "training",
                "report_sha256": report_sha256,
                "source_model_sha256": model_sha256,
                "binding_fields": [
                    "candidate_manifest.artifacts.model.sha256",
                    "candidate_manifest.artifacts.training_report.sha256",
                ],
            },
            {
                "id": evidence["id"],
                "role": "diagnostic",
                "report_sha256": receipt_sha256,
                "source_model_sha256": model_sha256,
                "binding_fields": ["artifact_binding.model_sha256"],
            },
        ],
        "display_evidence_id": evidence["id"],
        "evidence": [evidence],
        "claims": {
            "independent_current_player_accuracy": None,
            "calibration_independently_validated": False,
        },
    }
    return report, model_card, evidence


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
    release_contract_path: Path | None = None,
) -> str:
    model_bytes, model_sha256 = read_bytes_with_sha256(model_path)
    if release == "production":
        report, model_card, evidence, release_contract_sha256 = (
            validate_production_release(
                model_sha256,
                report_path,
                release_contract_path or DEFAULT_CLASSIFIER_RELEASE_CONTRACT,
            )
        )
    elif release == "shadow-preview":
        report, model_card, evidence = validate_shadow_release(
            model_sha256,
            report_path,
        )
        release_contract_sha256 = None
    else:
        raise ValueError(f"unsupported classifier release: {release}")

    # joblib/pickle can execute constructors. Deserialize only the exact bytes
    # whose SHA and release evidence were validated above, closing both the
    # untrusted-before-check and path time-of-check/time-of-use gaps.
    artifact = joblib.load(io.BytesIO(model_bytes))
    if not isinstance(artifact, dict):
        raise ValueError("feedback classifier artifact must be a dictionary")
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
            "paper_reference_proxy_accuracy": evidence["accuracy"],
            "paper_reference_proxy_examples": evidence["rows"],
            "paper_reference_proxy_is_previously_exposed": evidence[
                "previously_exposed"
            ],
            "independent_current_player_accuracy": model_card["claims"][
                "independent_current_player_accuracy"
            ],
            "calibration_independently_validated": model_card["claims"][
                "calibration_independently_validated"
            ],
        }
    elif release == "shadow-preview":
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
        fitted_temperature = float(artifact["temperature"])
        if not np.isfinite(fitted_temperature) or fitted_temperature != 0.0823618560384671:
            raise ValueError("shadow classifier fitted temperature changed")
        calibration = {
            "method": "none",
            "scope": "raw_model_output",
            "independently_validated": False,
        }
        score_policy = {
            "kind": "raw_softmax",
            "version": "boundary-shadow-raw-softmax-v2",
            "temperature": SHADOW_DISPLAY_TEMPERATURE,
            "probability_of_correctness": False,
            "independently_calibrated": False,
            "routing_threshold": 0.55,
            "threshold_policy": "existing_web_preview_policy_not_validated_in_raw_score_space",
        }
        model_type = "direct_fg_tfidf_logistic_regression_shadow"
        model_version = artifact["artifact_version"]
        canonical_labels = {
            "descriptive": "Descriptive",
            "evaluative": "Evaluative",
            "imperative": "Imperative",
        }
        # This is a trial routing policy, not a calibrated probability or an
        # accuracy claim. It is applied to the raw T=1 softmax top score.
        minimum_confidence = 0.55
        claim_scope = {
            "diagnostic_shadow_preview": True,
            "synthetic_training_rows": report.get("train_input", {}).get("rows"),
            "private_diagnostic_accuracy": evidence["accuracy"],
            "private_diagnostic_examples": evidence["rows"],
            "independent_current_player_accuracy": model_card["claims"][
                "independent_current_player_accuracy"
            ],
            "calibration_independently_validated": model_card["claims"][
                "calibration_independently_validated"
            ],
            "confidence_ready_for_player_ui": False,
            "displayed_score_kind": "raw_model_softmax_score",
            "displayed_score_is_probability_of_correctness": False,
            "routing_policy_threshold": minimum_confidence,
            "minimum_confidence_source": "existing_web_preview_policy_not_validated_in_raw_score_space",
        }
    source = {
        "model_sha256": model_sha256,
        "report_sha256": model_card["training"]["report_sha256"],
    }
    if release_contract_sha256 is not None:
        source["release_contract_sha256"] = release_contract_sha256
    if release == "shadow-preview":
        source.update(
            {
                "frozen_config_sha256": SHADOW_CONFIG_SHA256,
                "predictor_sha256": SHADOW_PREDICTOR_SHA256,
                "candidate_manifest_sha256": SHADOW_CANDIDATE_MANIFEST_SHA256,
                "diagnostic_report_sha256": SHADOW_DIAGNOSTIC_RECEIPT_SHA256,
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
        "model_card": model_card,
    }
    if release == "shadow-preview":
        payload["score_policy"] = score_policy
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


def load_route2_checkpoint(checkpoint_bytes: bytes) -> dict[str, Any]:
    options: dict[str, Any] = {"map_location": "cpu"}
    if TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        options["weights_only"] = True
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), **options)
    if not isinstance(checkpoint, dict):
        raise ValueError("Route2 checkpoint must be a dictionary")
    return checkpoint


def export_route2(manifest_path: Path, output_path: Path) -> str:
    manifest_path = manifest_path.resolve()
    manifest, manifest_sha256 = read_json_with_sha256(manifest_path)
    if manifest.get("schema_version") != "route2-ensemble-v1":
        raise ValueError("unsupported Route2 ensemble manifest")
    if manifest_path == DEFAULT_ROUTE2.resolve() and (
        manifest_sha256 != DEFAULT_ROUTE2_MANIFEST_SHA256
        or manifest.get("manifest_sha256") != DEFAULT_ROUTE2_ENSEMBLE_IDENTITY
    ):
        raise ValueError("default production Route2 ensemble identity changed")
    features = [str(value) for value in manifest["features"]]
    members = []
    for entry in manifest["members"]:
        checkpoint_path = manifest_path.parent / entry["checkpoint"]
        checkpoint_bytes, checkpoint_hash = read_bytes_with_sha256(checkpoint_path)
        if checkpoint_hash != entry["checkpoint_sha256"]:
            raise ValueError(f"Route2 checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = load_route2_checkpoint(checkpoint_bytes)
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
            "ensemble_manifest_sha256": manifest_sha256,
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


def browser_release_metadata(
    classifier_path: Path,
    classifier_sha256: str,
) -> dict[str, Any]:
    artifact, actual_sha256 = read_json_with_sha256(classifier_path)
    if actual_sha256 != classifier_sha256:
        raise ValueError("exported feedback classifier SHA-256 changed before manifest write")
    model_card = artifact.get("model_card")
    if not isinstance(model_card, dict):
        raise ValueError("exported feedback classifier model card is missing")
    release = model_card.get("release")
    if not isinstance(release, dict):
        raise ValueError("exported feedback classifier release metadata is missing")
    if model_card.get("source_model_sha256") != artifact.get("source", {}).get(
        "model_sha256"
    ):
        raise ValueError("exported feedback classifier model card binding changed")
    return {
        **release,
        "source_model_sha256": model_card["source_model_sha256"],
        "model_card_schema_version": model_card.get("schema_version"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-release",
        choices=("production", "shadow-preview"),
        default="production",
    )
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--classifier-report", type=Path)
    parser.add_argument("--classifier-release-contract", type=Path)
    parser.add_argument("--classifier-output-name")
    parser.add_argument("--route2-manifest", type=Path, default=DEFAULT_ROUTE2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.classifier_release == "shadow-preview"
        and args.classifier_release_contract is not None
    ):
        raise ValueError("shadow preview does not accept a production release contract")
    if args.classifier_release == "shadow-preview":
        classifier_source = args.classifier or DEFAULT_SHADOW_CLASSIFIER
        classifier_report = args.classifier_report or DEFAULT_SHADOW_CLASSIFIER_REPORT
        classifier_output_name = args.classifier_output_name or SHADOW_BROWSER_FILENAME
    else:
        classifier_source = args.classifier or DEFAULT_CLASSIFIER
        classifier_report = args.classifier_report or DEFAULT_CLASSIFIER_REPORT
        classifier_release_contract = (
            args.classifier_release_contract or DEFAULT_CLASSIFIER_RELEASE_CONTRACT
        )
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
        release_contract_path=(
            classifier_release_contract.resolve()
            if args.classifier_release == "production"
            else None
        ),
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
        "schema_version": "durf-browser-model-manifest-v2",
        "release": browser_release_metadata(classifier_path, classifier_hash),
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
