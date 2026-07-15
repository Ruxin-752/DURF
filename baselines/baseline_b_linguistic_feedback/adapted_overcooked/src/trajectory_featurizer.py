"""Featurize attributed windows from normalized Overcooked trajectories."""

from __future__ import annotations

from pathlib import Path

from .feature_schema import read_json


ACTION_FEATURES = {
    "stay": "time_cost",
}
MOVE_DELTAS = {
    "north": (0, -1),
    "up": (0, -1),
    "south": (0, 1),
    "down": (0, 1),
    "east": (1, 0),
    "right": (1, 0),
    "west": (-1, 0),
    "left": (-1, 0),
}


def held_object_name(held_object) -> str | None:
    if isinstance(held_object, dict):
        return held_object.get("name")
    return None


def _position(value) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _pot_state_active(pot_states, name: str) -> bool:
    if not isinstance(pot_states, dict):
        return False
    value = pot_states.get(name)
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def featurize_trajectory_steps(trajectory_steps: list[dict]) -> dict[str, float]:
    counts: dict[str, float] = {}
    indicators: set[str] = set()

    def add(feature: str, value: float = 1.0) -> None:
        counts[feature] = counts.get(feature, 0.0) + value

    def indicate(feature: str) -> None:
        if feature not in indicators:
            indicators.add(feature)
            add(feature)

    for step in trajectory_steps:
        ai_action_name = str(step.get("ai_action_name") or "").lower()
        action_feature = ACTION_FEATURES.get(ai_action_name)
        if action_feature:
            add(action_feature)

        after = step.get("state_facts") or {}
        before = (step.get("extra") or {}).get("state_before") or {}
        held_before = held_object_name(before.get("ai_held_object"))
        held_after = held_object_name(after.get("ai_held_object"))

        if held_after != held_before:
            pickup_features = {
                "tomato": ("ingredient_tomato", "pick_tomato"),
                "onion": ("ingredient_onion", "pick_onion"),
                "dish": ("pick_dish",),
                "soup": ("pick_ready_soup",),
            }
            if held_after in pickup_features:
                for feature in pickup_features[held_after]:
                    add(feature)

            if ai_action_name == "interact" and held_after is None:
                if held_before == "tomato":
                    add("adds_needed_tomato")
                elif held_before == "onion":
                    add("adds_needed_onion")
                elif held_before == "soup" and float(step.get("environment_reward") or 0) > 0:
                    add("serve_ready_soup")

        pot_states = after.get("pot_states") or {}
        if _pot_state_active(pot_states, "empty"):
            indicate("pot_empty")
        if _pot_state_active(pot_states, "cooking"):
            indicate("pot_cooking")
        if _pot_state_active(pot_states, "ready"):
            indicate("soup_ready")
            if held_after != "dish":
                indicate("dish_needed_for_ready_soup")

        human_before = _position(before.get("human_pos"))
        human_after = _position(after.get("human_pos"))
        ai_before = _position(before.get("ai_pos"))
        ai_after = _position(after.get("ai_pos"))
        human_action = str(step.get("human_action_name") or "").lower()
        delta = MOVE_DELTAS.get(human_action)
        if human_before and delta:
            human_target = (
                human_before[0] + delta[0],
                human_before[1] + delta[1],
            )
            if human_after == human_before and ai_before == human_target:
                add("blocks_human_path")
                add("human_wait_cost")
            if ai_after == human_target and ai_before != human_target:
                add("cuts_in_front_of_human")
                add("collision_risk")

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
