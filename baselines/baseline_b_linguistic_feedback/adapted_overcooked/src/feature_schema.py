"""Feature schema helpers for the adapted Overcooked Baseline B pipeline."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_FEATURES_PATH = DATA_DIR / "overcooked_features.json"
DEFAULT_WEIGHTS_PATH = DATA_DIR / "initial_weights.json"
VALID_FEEDBACK_TYPES = frozenset({"evaluative", "imperative", "descriptive"})
GROUNDING_FIELDS = ("target_features", "trajectory_features", "target_action")


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


def validate_feedback_examples(
    feedback_examples: list[dict],
    *,
    probe_states: list[dict],
    known_features: set[str] | None = None,
) -> None:
    """Validate authored or converted feedback before it can update weights."""

    if not isinstance(feedback_examples, list):
        raise ValueError("Feedback examples must be a JSON list")

    known = known_features or set(load_features())
    action_library = collect_action_feature_library(probe_states)
    probe_ids = {
        str(probe["probe_id"])
        for probe in probe_states
        if isinstance(probe, dict) and probe.get("probe_id")
    }
    seen_ids: set[str] = set()
    errors: list[str] = []

    for index, example in enumerate(feedback_examples):
        if not isinstance(example, dict):
            errors.append(f"item[{index}]: expected an object")
            continue

        feedback_id = example.get("feedback_id")
        label = str(feedback_id) if feedback_id else f"item[{index}]"
        if not isinstance(feedback_id, str) or not feedback_id.strip():
            errors.append(f"{label}: feedback_id must be a non-empty string")
        elif feedback_id in seen_ids:
            errors.append(f"{label}: duplicate feedback_id")
        else:
            seen_ids.add(feedback_id)

        text = example.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}: text must be a non-empty string")

        feedback_type = example.get("expected_feedback_type")
        if feedback_type not in VALID_FEEDBACK_TYPES:
            errors.append(
                f"{label}: expected_feedback_type must be one of "
                f"{sorted(VALID_FEEDBACK_TYPES)}"
            )

        has_target_features = isinstance(example.get("target_features"), dict) and bool(
            example["target_features"]
        )
        has_trajectory_features = isinstance(
            example.get("trajectory_features"), dict
        ) and bool(example["trajectory_features"])
        has_target_action = isinstance(example.get("target_action"), str) and bool(
            example["target_action"].strip()
        )
        # target_action may accompany explicit target_features as a semantic
        # label. The explicit vector remains the sole effective grounding
        # source, matching ground_feedback's precedence.
        active_grounding = (
            ["target_features"]
            if has_target_features
            else ["trajectory_features"]
            if has_trajectory_features
            else ["target_action"]
            if has_target_action
            else []
        )
        if not active_grounding or (has_target_features and has_trajectory_features):
            errors.append(
                f"{label}: expected exactly one effective grounding source from "
                f"{list(GROUNDING_FIELDS)}, found {active_grounding}"
            )

        for field in ("target_features", "trajectory_features"):
            if field not in example:
                continue
            vector = example[field]
            if not isinstance(vector, dict):
                errors.append(f"{label}: {field} must be an object")
                continue
            unknown = validate_feature_vector(vector, known_features=known)
            if unknown:
                errors.append(f"{label}: {field} contains unknown features {unknown}")
            for feature, value in vector.items():
                if isinstance(value, bool) or not isinstance(value, Real):
                    errors.append(f"{label}: {field}.{feature} must be numeric")
                elif not math.isfinite(float(value)):
                    errors.append(f"{label}: {field}.{feature} must be finite")

        target_action = example.get("target_action")
        if target_action is not None:
            if not isinstance(target_action, str) or target_action not in action_library:
                errors.append(f"{label}: unknown target_action {target_action!r}")
            if feedback_type != "imperative":
                errors.append(f"{label}: target_action requires imperative feedback")

        probe_id = example.get("probe_id")
        if probe_id is not None and str(probe_id) not in probe_ids:
            errors.append(f"{label}: unknown probe_id {probe_id!r}")

    if errors:
        raise ValueError("Invalid feedback examples:\n- " + "\n- ".join(errors))


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
