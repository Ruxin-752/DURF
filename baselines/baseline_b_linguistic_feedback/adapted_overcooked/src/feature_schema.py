"""Feature schema helpers for the adapted Overcooked Baseline B pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_FEATURES_PATH = DATA_DIR / "overcooked_features.json"
DEFAULT_WEIGHTS_PATH = DATA_DIR / "initial_weights.json"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_features(path: str | Path = DEFAULT_FEATURES_PATH) -> list[str]:
    data = read_json(path)
    features = data["features"] if isinstance(data, dict) else data
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError(f"Invalid feature schema: {path}")
    return list(dict.fromkeys(features))


def empty_weights(features: list[str] | None = None) -> dict[str, float]:
    return {feature: 0.0 for feature in (features or load_features())}


def load_weights(
    path: str | Path = DEFAULT_WEIGHTS_PATH,
    *,
    features: list[str] | None = None,
) -> dict[str, float]:
    weights = empty_weights(features)
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Weights must be a JSON object: {path}")
    for feature, value in raw.items():
        weights[str(feature)] = float(value)
    return weights


def validate_feature_vector(
    feature_vector: dict[str, int | float],
    *,
    known_features: set[str] | None = None,
) -> list[str]:
    known = known_features or set(load_features())
    return sorted(feature for feature in feature_vector if feature not in known)


def collect_action_feature_library(probe_states: list[dict]) -> dict[str, dict[str, float]]:
    """Collect action_id -> feature vector from authored probe states."""

    library: dict[str, dict[str, float]] = {}
    for probe in probe_states:
        for action in probe.get("available_actions", []):
            action_id = action.get("action_id")
            features = action.get("features", {})
            if action_id and isinstance(features, dict):
                library[str(action_id)] = {
                    str(feature): float(value)
                    for feature, value in features.items()
                }
    return library
