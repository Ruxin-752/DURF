"""Raw-text-only prediction interface for the boundary shadow candidate.

This interface is intentionally separate from the production/Pygame/Web model.
Its confidence is train-OOF temperature scaled and not independently validated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_VERSION = "direct-fg-boundary-shadow-model-v1"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_boundary_shadow_v1_model"
    / "model.joblib"
)


def extract_model_input(value: str) -> str:
    if type(value) is not str:
        raise TypeError("shadow predictor accepts one raw text string only")
    if not value.strip():
        raise ValueError("shadow predictor text must be non-empty")
    return value


def load_artifact(model_path: Path):
    import joblib

    artifact = joblib.load(Path(model_path).resolve())
    if not isinstance(artifact, dict):
        raise ValueError("shadow artifact must be a dictionary")
    if artifact.get("artifact_version") != EXPECTED_VERSION:
        raise ValueError("unexpected shadow artifact version")
    contract = artifact.get("model_input_contract")
    if contract != {
        "input_type": "raw_string_only",
        "field": "text",
        "metadata_features": [],
    }:
        raise ValueError("shadow artifact does not enforce raw-text-only features")
    if artifact.get("promotion_eligible") is not False:
        raise ValueError("shadow artifact must remain non-promotable")
    if artifact.get("production_artifact") is not False:
        raise ValueError("production artifacts are not accepted by this interface")
    classes = artifact.get("classes")
    if classes != ["descriptive", "evaluative", "imperative"]:
        raise ValueError("unexpected direct-fG class order")
    temperature = float(artifact.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("shadow temperature must be finite and positive")
    return artifact


def _probabilities(artifact: dict, texts: list[str]):
    import numpy as np

    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    matrix = vectorizer.transform(texts)
    observed = [str(value) for value in classifier.classes_]
    target = list(artifact["classes"])
    if set(observed) != set(target):
        raise ValueError("classifier and artifact class sets differ")
    logits = np.asarray(classifier.decision_function(matrix), dtype=float)
    logits = logits[:, [observed.index(label) for label in target]]
    logits /= float(artifact["temperature"])
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("non-finite shadow prediction")
    return probabilities


def predict_text(artifact: dict, text: str) -> dict:
    raw_text = extract_model_input(text)
    probabilities = _probabilities(artifact, [raw_text])[0]
    classes = list(artifact["classes"])
    winner = int(probabilities.argmax())
    return {
        "label": classes[winner],
        "confidence": float(probabilities[winner]),
        "probabilities": {
            label: float(value) for label, value in zip(classes, probabilities)
        },
        "confidence_scope": (
            "train_oof_temperature_scaled_not_independently_validated"
        ),
        "artifact_status": "shadow_diagnostic_only",
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    artifact = load_artifact(args.model)
    print(json.dumps(predict_text(artifact, args.text), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
