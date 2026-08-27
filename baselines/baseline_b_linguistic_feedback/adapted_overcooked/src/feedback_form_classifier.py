"""Three-class feedback-form classifier with a deterministic missing-model fallback.

This target is a paper-aligned feedback strategy obtained by collapsing
reference classes: trajectory -> Evaluative, feature -> Descriptive, and
action_spatial -> Imperative. The deployed v3 artifact also uses the paper main
experiment's action_behavioral -> Evaluative mapping; the released notebook's
strict analysis excluded behavioral-action and other references. It is
predicted by a separate three-class model at runtime; the
five-class reference classifier remains intact. A trained TF-IDF +
LogisticRegression artifact is used when present. Transparent rules keep live
play functional only when that artifact has not been trained; low confidence is
reported as uncertainty and never replaces a learned prediction with a rule.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "outputs" / "feedback_form_classifier" / "model.joblib"
DEFAULT_MODEL_CONFIDENCE_THRESHOLD = 0.55
FEEDBACK_TYPES = ("evaluative", "imperative", "descriptive")
CANONICAL_FEEDBACK_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}


IMPERATIVE_MARKERS = (
    "please ",
    "could you ",
    "should",
    "need to",
    "go ",
    "take ",
    "get ",
    "put ",
    "serve",
    "move",
    "clear",
    "grab ",
    "fetch ",
    "pick up ",
    "bring ",
    "stop ",
    "don't ",
    "do not ",
    "avoid ",
)

EVALUATIVE_MARKERS = (
    "good",
    "bad",
    "nice",
    "great",
    "wrong",
    "correct",
    "thanks",
)

DESCRIPTIVE_MARKERS = (
    "block",
    "blocking",
    "stuck",
    "in my way",
    "duplicate",
    "repeat",
    "shortest path",
    "crowd",
)


def fallback_prediction(text: str | None, *, scalar_value: int | None = None) -> dict:
    """Return the legacy rule result with explicit provenance."""

    if scalar_value is not None and not text:
        label = "evaluative"
        return {
            "feedback_type": label,
            "classification_label": CANONICAL_FEEDBACK_LABELS[label],
            "confidence": 1.0,
            "probabilities": {label: 1.0},
            "classifier": "fallback_rules",
            "calibrated": False,
        }

    lowered = (text or "").strip().lower()
    if not lowered:
        return {
            "feedback_type": "unknown",
            "classification_label": "Unknown",
            "confidence": 1.0,
            "probabilities": {"unknown": 1.0},
            "classifier": "fallback_rules",
            "calibrated": False,
        }

    imperative_score = sum(marker in lowered for marker in IMPERATIVE_MARKERS)
    descriptive_score = sum(marker in lowered for marker in DESCRIPTIVE_MARKERS)
    evaluative_score = sum(marker in lowered for marker in EVALUATIVE_MARKERS)

    if imperative_score:
        label, confidence = "imperative", 0.9
    elif descriptive_score:
        label, confidence = "descriptive", 0.85
    elif evaluative_score:
        label, confidence = "evaluative", 0.85
    else:
        label, confidence = "descriptive", 0.55
    remainder = (1.0 - confidence) / (len(FEEDBACK_TYPES) - 1)
    probabilities = {
        candidate: confidence if candidate == label else remainder
        for candidate in FEEDBACK_TYPES
    }
    return {
        "feedback_type": label,
        "classification_label": CANONICAL_FEEDBACK_LABELS[label],
        "confidence": confidence,
        "probabilities": probabilities,
        "classifier": "fallback_rules",
        "calibrated": False,
    }


@lru_cache(maxsize=8)
def _load_artifact(path: str):
    from joblib import load

    return load(path)


def artifact_probabilities(artifact: dict, matrix) -> tuple[dict[str, float], dict]:
    """Return the artifact's deployed probabilities and calibration metadata."""

    classifier = artifact["classifier"]
    calibration = artifact.get("probability_calibration")
    calibrated = False
    calibration_version = None
    if isinstance(calibration, dict) and calibration.get("method") == "temperature_scaling":
        import numpy as np

        temperature = float(calibration.get("temperature", 0.0))
        if not temperature > 0:
            raise ValueError("feedback-form artifact has invalid calibration temperature")
        raw = np.asarray(classifier.predict_proba(matrix), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(classifier.classes_):
            raise ValueError("feedback-form calibration requires multiclass probabilities")
        logits = np.log(np.clip(raw, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities_raw = (exponentiated / exponentiated.sum(axis=1, keepdims=True))[0]
        calibrated = True
        calibration_version = calibration.get("version")
    else:
        probabilities_raw = classifier.predict_proba(matrix)[0]
    probabilities = {
        str(label): float(value)
        for label, value in zip(classifier.classes_, probabilities_raw)
    }
    return probabilities, {
        "calibrated": calibrated,
        "calibration_version": calibration_version,
        "calibration_status": (
            "temperature_scaled_on_selection_dev_not_independently_validated"
            if calibrated
            else "uncalibrated"
        ),
    }


def predict_feedback_form(
    text: str | None,
    *,
    scalar_value: int | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Predict one feedback form while preserving lowercase internal labels."""

    if not (text or "").strip():
        return fallback_prediction(text, scalar_value=scalar_value)
    path = Path(model_path)
    if not path.exists():
        return fallback_prediction(text, scalar_value=scalar_value)
    artifact = _load_artifact(str(path.resolve()))
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    matrix = vectorizer.transform([str(text)])
    probabilities, calibration_metadata = artifact_probabilities(artifact, matrix)
    unknown_labels = sorted(set(probabilities) - set(FEEDBACK_TYPES))
    if unknown_labels:
        raise ValueError(f"feedback-form artifact has unknown labels: {unknown_labels}")
    label = max(probabilities, key=probabilities.get)
    model_confidence = probabilities[label]
    threshold = float(
        artifact.get("minimum_model_confidence", DEFAULT_MODEL_CONFIDENCE_THRESHOLD)
    )
    if model_confidence < threshold:
        fallback = fallback_prediction(text, scalar_value=scalar_value)
        return {
            "feedback_type": label,
            "classification_label": CANONICAL_FEEDBACK_LABELS[label],
            "confidence": model_confidence,
            "probabilities": probabilities,
            "model_feedback_type": label,
            "model_confidence": model_confidence,
            "model_probabilities": probabilities,
            # The rule result is retained only as an audit/ablation field. It
            # must not silently overwrite a trained model at deployment.
            "fallback_audit_feedback_type": fallback["feedback_type"],
            "fallback_audit_confidence": fallback["confidence"],
            "confidence_threshold": threshold,
            "classifier": "tfidf_logistic_regression_low_confidence",
            "abstained": True,
            "model_path": str(path),
            **calibration_metadata,
        }
    return {
        "feedback_type": label,
        "classification_label": CANONICAL_FEEDBACK_LABELS[label],
        "confidence": model_confidence,
        "probabilities": probabilities,
        "confidence_threshold": threshold,
        "classifier": "tfidf_logistic_regression",
        "abstained": False,
        "model_path": str(path),
        **calibration_metadata,
    }


def classify_feedback(
    text: str | None,
    *,
    scalar_value: int | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> str:
    """Return the legacy lowercase string API used throughout Route 1."""

    return str(
        predict_feedback_form(
            text, scalar_value=scalar_value, model_path=model_path
        )["feedback_type"]
    )
