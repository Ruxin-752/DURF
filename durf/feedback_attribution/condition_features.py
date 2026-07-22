"""Normalize trajectory facts into Hu condition features.

The attribution/provenance records may keep rich evidence, but Hu training uses
a fixed condition vector so missing fields, false fields, and irrelevant fields
do not get mixed together.
"""

from __future__ import annotations

from typing import Any


ACTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}

CONDITION_KEYS = (
    "pot_empty",
    "pot_partially_filled",
    "pot_cooking_or_ready",
    "human_has_dish",
    "ai_has_dish",
    "human_has_tomato",
    "human_has_onion",
    "ai_has_tomato",
    "ai_has_onion",
    "ai_empty_handed",
    "recipe_needs_tomato",
    "recipe_needs_onion",
    "human_trying_to_pass",
    "narrow_corridor",
    "ai_on_human_path",
    "useful_object_adjacent",
    "human_waiting_near_pot",
)

RING_TOMATO_ONION_RECIPE = ("tomato", "tomato", "onion")


def null_condition_features() -> dict[str, bool | None]:
    return {key: None for key in CONDITION_KEYS}


def object_name(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("name")
    return getattr(obj, "name", None)


def pos_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def add_pos(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return pos[0] + delta[0], pos[1] + delta[1]


def terrain_at(terrain: list[str], pos: tuple[int, int]) -> str | None:
    x, y = pos
    if y < 0 or y >= len(terrain):
        return None
    if x < 0 or x >= len(terrain[y]):
        return None
    return terrain[y][x]


def is_walkable(terrain: list[str], pos: tuple[int, int]) -> bool:
    return terrain_at(terrain, pos) == " "


def walkable_neighbor_count(terrain: list[str], pos: tuple[int, int]) -> int | None:
    if not terrain:
        return None
    return sum(1 for delta in ACTION_DELTAS.values() if is_walkable(terrain, add_pos(pos, delta)))


def adjacent_positions(pos: tuple[int, int]) -> set[tuple[int, int]]:
    return {add_pos(pos, delta) for delta in ACTION_DELTAS.values()}


def pot_state_has(pot_states: Any, *keys: str) -> bool | None:
    if not isinstance(pot_states, dict):
        return None
    return any(bool(pot_states.get(key)) for key in keys)


def soup_ingredients_from_objects(facts: dict[str, Any]) -> list[str]:
    ingredients: list[str] = []
    for obj in facts.get("objects") or []:
        if not isinstance(obj, dict) or obj.get("name") != "soup":
            continue
        for ingredient in obj.get("ingredients") or []:
            ingredients.append(str(ingredient))
    return ingredients


def recipe_needs_from_facts(facts: dict[str, Any], ingredient: str) -> bool | None:
    layout_name = (facts.get("layout_features") or {}).get("layout_name") or ""
    if "tomato_onion" not in layout_name and "ring_tomato_onion" not in layout_name:
        return None
    target_count = RING_TOMATO_ONION_RECIPE.count(ingredient)
    current_count = soup_ingredients_from_objects(facts).count(ingredient)
    return current_count < target_count


def object_adjacent_to_ai(facts: dict[str, Any]) -> bool | None:
    ai_pos = pos_tuple(facts.get("ai_pos"))
    if ai_pos is None:
        return None
    adjacent = adjacent_positions(ai_pos)
    for obj in facts.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        obj_pos = pos_tuple(obj.get("position"))
        if obj_pos in adjacent and object_name(obj) in {"tomato", "onion", "dish", "soup"}:
            return True
    terrain = (facts.get("layout_features") or {}).get("terrain") or []
    return any(terrain_at(terrain, pos) in {"T", "O", "D", "P"} for pos in adjacent)


def extract_condition_features(step: dict[str, Any] | None) -> dict[str, bool | None]:
    features = null_condition_features()
    if not step:
        return features
    recorded = step.get("ai_condition_features")
    if isinstance(recorded, dict) and recorded:
        features.update({key: recorded.get(key) for key in features})
        return features

    facts = step.get("state_facts") or {}
    before = (step.get("extra") or {}).get("state_before") or {}
    pot_states = facts.get("pot_states")
    ai_held = object_name(facts.get("ai_held_object"))
    human_held = object_name(facts.get("human_held_object"))

    features["pot_empty"] = pot_state_has(pot_states, "empty")
    features["pot_partially_filled"] = pot_state_has(pot_states, "partially_full")
    features["pot_cooking_or_ready"] = pot_state_has(pot_states, "cooking", "ready")
    features["human_has_dish"] = human_held == "dish" if human_held is not None else False
    features["ai_has_dish"] = ai_held == "dish" if ai_held is not None else False
    features["human_has_tomato"] = human_held == "tomato" if human_held is not None else False
    features["human_has_onion"] = human_held == "onion" if human_held is not None else False
    features["ai_has_tomato"] = ai_held == "tomato" if ai_held is not None else False
    features["ai_has_onion"] = ai_held == "onion" if ai_held is not None else False
    features["ai_empty_handed"] = ai_held is None
    features["recipe_needs_tomato"] = recipe_needs_from_facts(facts, "tomato")
    features["recipe_needs_onion"] = recipe_needs_from_facts(facts, "onion")
    features["useful_object_adjacent"] = object_adjacent_to_ai(facts)
    features["human_waiting_near_pot"] = bool(
        features["pot_cooking_or_ready"] and features["human_has_dish"]
    )

    human_action_name = step.get("human_action_name")
    delta = ACTION_DELTAS.get(str(human_action_name))
    human_before = pos_tuple(before.get("human_pos"))
    human_after = pos_tuple(facts.get("human_pos"))
    ai_before = pos_tuple(before.get("ai_pos"))
    ai_after = pos_tuple(facts.get("ai_pos"))
    if delta and human_before is not None and human_after is not None:
        attempted_target = add_pos(human_before, delta)
        features["human_trying_to_pass"] = human_before == human_after
        features["ai_on_human_path"] = attempted_target in {ai_before, ai_after}
        terrain = (before.get("layout_features") or {}).get("terrain") or []
        corridor_width = walkable_neighbor_count(terrain, human_before)
        features["narrow_corridor"] = (
            corridor_width is not None and corridor_width <= 2
        )
    else:
        features["human_trying_to_pass"] = False
        features["ai_on_human_path"] = False
        features["narrow_corridor"] = False

    return features


def latest_step_at_or_before(
    trajectory: list[dict[str, Any]],
    total_step: int | None,
) -> dict[str, Any] | None:
    if not trajectory:
        return None
    if total_step is None:
        return trajectory[-1]
    eligible = [
        step for step in trajectory if int(step.get("total_step") or -1) <= int(total_step)
    ]
    return eligible[-1] if eligible else trajectory[0]
