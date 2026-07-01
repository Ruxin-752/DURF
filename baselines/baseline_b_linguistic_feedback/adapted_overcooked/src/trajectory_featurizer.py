"""First-pass trajectory featurizer for saved Overcooked session logs.

Milestone 1-2 use hand-authored trajectory feature examples. This module is a
small bridge for Milestone 4: it can count simple action and state facts from
`trajectory.jsonl` records produced by `durf.feedback_attribution`.
"""

from __future__ import annotations

from pathlib import Path

from .feature_schema import read_json


ACTION_FEATURES = {
    "interact": "interact",
    "stay": "time_cost",
}


def held_object_name(held_object) -> str | None:
    if isinstance(held_object, dict):
        return held_object.get("name")
    return None


def featurize_trajectory_steps(trajectory_steps: list[dict]) -> dict[str, float]:
    counts: dict[str, float] = {}

    def add(feature: str, value: float = 1.0) -> None:
        counts[feature] = counts.get(feature, 0.0) + value

    for step in trajectory_steps:
        ai_action_name = step.get("ai_action_name")
        if ai_action_name == "stay":
            add("time_cost")

        state_facts = step.get("state_facts") or {}
        ai_held = held_object_name(state_facts.get("ai_held_object"))
        if ai_held == "tomato":
            add("ingredient_tomato")
        elif ai_held == "onion":
            add("ingredient_onion")
        elif ai_held == "dish":
            add("pick_dish")
        elif ai_held == "soup":
            add("pick_ready_soup")

        pot_states = state_facts.get("pot_states") or {}
        if isinstance(pot_states, dict):
            if pot_states.get("empty"):
                add("pot_empty")
            if pot_states.get("cooking"):
                add("pot_cooking")
            if pot_states.get("ready"):
                add("soup_ready")

    return counts


def featurize_trajectory_jsonl(path: str | Path) -> dict[str, float]:
    records = read_jsonl(path)
    return featurize_trajectory_steps(records)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(read_json_string(line))
    return records


def read_json_string(value: str) -> dict:
    # Kept local to avoid adding a second JSON helper dependency for this bridge.
    import json

    return json.loads(value)
