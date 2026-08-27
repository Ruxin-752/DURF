"""Paper-style phrase reference-type classifier.

The paper's TF-IDF + LogisticRegression model predicts what a phrase refers
to; it does *not* predict the evaluative/imperative/descriptive speech act.
This module loads an Overcooked-domain artifact when available and otherwise
uses a conservative deterministic fallback so live play remains functional.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from .text_analysis import limited_punc_tokenization, preprocess_phrase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "outputs" / "reference_classifier" / "model.joblib"
REFERENCE_TYPES = (
    "trajectory",
    "feature",
    "action_spatial",
    "action_behavioral",
    "other",
)

_SPATIAL = ("left", "right", "top", "bottom", "corner", "near", "next to", "station")
_BEHAVIORAL = ("keep ", "always ", "again", "before", "after", "every time", "usually")
_COMMAND = (
    "please ",
    "go ",
    "get ",
    "grab ",
    "take ",
    "put ",
    "serve ",
    "move ",
    "stop ",
    "don't ",
    "do not ",
    "avoid ",
    "should ",
)
_FEATURE = (
    "onion",
    "tomato",
    "dish",
    "plate",
    "soup",
    "pot",
    "serve",
    "block",
    "path",
    "way",
    "crowd",
    "duplicate",
    "wait",
    "idle",
)
_TRAJECTORY = ("good", "great", "nice", "perfect", "bad", "wrong", "thanks", "well done")


def _safe_preprocess(text: str) -> str:
    try:
        return preprocess_phrase(text)
    except (ImportError, LookupError):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@lru_cache(maxsize=4)
def _load_artifact(path: str):
    from joblib import load

    return load(path)


def fallback_prediction(phrase: str) -> dict:
    """Return a transparent fallback prediction with a confidence estimate."""

    lowered = phrase.strip().lower()
    if not lowered:
        return {"reference_type": "other", "confidence": 1.0, "probabilities": {"other": 1.0}}
    if any(marker in lowered for marker in _SPATIAL):
        label, confidence = "action_spatial", 0.82
    elif any(marker in lowered for marker in _BEHAVIORAL):
        label, confidence = "action_behavioral", 0.76
    elif any(lowered.startswith(marker) for marker in _COMMAND) and any(
        marker in lowered for marker in _FEATURE
    ):
        label, confidence = "action_spatial", 0.72
    elif any(marker in lowered for marker in _FEATURE):
        label, confidence = "feature", 0.68
    elif any(marker in lowered for marker in _TRAJECTORY):
        label, confidence = "trajectory", 0.72
    else:
        label, confidence = "other", 0.55
    remainder = (1.0 - confidence) / (len(REFERENCE_TYPES) - 1)
    probabilities = {
        candidate: confidence if candidate == label else remainder
        for candidate in REFERENCE_TYPES
    }
    return {
        "reference_type": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "classifier": "fallback_rules",
        "top2_margin": confidence - remainder,
        "threshold": 0.55,
        "abstained": confidence < 0.55,
    }


def predict_reference_type(
    phrase: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Predict one paper reference type and calibrated class probabilities."""

    path = Path(model_path)
    if not path.exists():
        return fallback_prediction(phrase)
    artifact = _load_artifact(str(path.resolve()))
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    manifest = artifact.get("manifest") or {}
    input_mode = (
        artifact.get("input_mode")
        or manifest.get("preprocessing")
        or "paper_preprocess_phrase"
    )
    model_text = str(phrase or "") if input_mode == "raw_phrase" else _safe_preprocess(phrase)
    matrix = vectorizer.transform([model_text])
    probabilities_raw = np.asarray(classifier.predict_proba(matrix)[0], dtype=float)
    temperature = max(float(artifact.get("temperature", 1.0)), 1e-3)
    logits = np.log(np.clip(probabilities_raw, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    probabilities_raw = np.exp(logits)
    probabilities_raw /= probabilities_raw.sum()
    probabilities = {
        str(label): float(value)
        for label, value in zip(classifier.classes_, probabilities_raw)
    }
    label = max(probabilities, key=probabilities.get)
    ordered = sorted(probabilities.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    # Legacy artifacts predate dev-set thresholds; retain their former permissive
    # behavior while every newly trained artifact carries calibrated thresholds.
    threshold = float((artifact.get("class_thresholds") or {}).get(label, 0.45))
    margin_threshold = float(artifact.get("top2_margin_threshold", 0.05))
    abstained = probabilities[label] < threshold or margin < margin_threshold
    return {
        "reference_type": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "top2_margin": float(margin),
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "abstained": bool(abstained),
        "classifier": "tfidf_logistic_regression",
        "input_mode": input_mode,
        "model_path": str(path),
    }


def classify_utterance(
    text: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict]:
    """Split an utterance using the paper rule and classify each phrase."""

    return [
        {"phrase": phrase, **predict_reference_type(phrase, model_path=model_path)}
        for phrase in limited_punc_tokenization(text)
    ]
