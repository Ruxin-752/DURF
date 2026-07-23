"""Normalize trajectory facts into Hu condition features.

The attribution/provenance records may keep rich evidence, but Hu training uses
a fixed condition vector so missing fields, false fields, and irrelevant fields
do not get mixed together.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any


ACTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}

MODEL_CONDITION_KEYS = (
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
    "human_holding_last_needed_ingredient",
    "human_closer_to_dish",
    "human_closer_to_pot",
    "ai_closer_to_ingredient",
    "useful_counter_object_available",
    "useful_counter_object_closer_than_dispenser",
    "useful_counter_object_closer_to_pot_than_dispenser",
    "useful_counter_object_lower_task_cost_than_dispenser",
)

# Kept as the public model feature list for compatibility with Hu-v0.
CONDITION_KEYS = MODEL_CONDITION_KEYS

# These values are retained for attribution provenance and human review. Hu-v0
# consumes the fixed boolean MODEL_CONDITION_KEYS above, not raw strings or
# unscaled distances.
CONDITION_CONTEXT_KEYS = (
    "needed_ingredient",
    "human_inferred_subgoal",
    "human_distance_to_dish",
    "ai_distance_to_dish",
    "human_distance_to_pot",
    "ai_distance_to_pot",
    "human_distance_to_needed_ingredient",
    "ai_distance_to_needed_ingredient",
    "useful_counter_object_type",
    "useful_counter_object_position",
    "useful_counter_object_distance",
    "useful_counter_object_pot_distance",
    "useful_counter_total_task_distance",
    "matching_dispenser_distance",
    "matching_dispenser_pot_distance",
    "matching_dispenser_total_task_distance",
)

RING_TOMATO_ONION_RECIPE = ("tomato", "tomato", "onion")


def null_condition_features() -> dict[str, Any]:
    return {
        key: None
        for key in (*MODEL_CONDITION_KEYS, *CONDITION_CONTEXT_KEYS)
    }


def object_name(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        nested = obj.get("object")
        if isinstance(nested, dict):
            return nested.get("name")
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


def object_position(obj: Any) -> tuple[int, int] | None:
    if not isinstance(obj, dict):
        return pos_tuple(getattr(obj, "position", None))
    nested = obj.get("object")
    if isinstance(nested, dict):
        return pos_tuple(obj.get("position") or nested.get("position"))
    return pos_tuple(obj.get("position"))


def object_payload(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    nested = obj.get("object")
    return nested if isinstance(nested, dict) else obj


def feature_positions(terrain: list[str], symbol: str) -> list[tuple[int, int]]:
    positions = []
    for y, row in enumerate(terrain):
        for x, cell in enumerate(row):
            if cell == symbol:
                positions.append((x, y))
    return positions


def shortest_walk_distance(
    terrain: list[str],
    start: tuple[int, int] | None,
    target_features: list[tuple[int, int]],
) -> int | None:
    """Shortest movement distance to a tile adjacent to a feature."""

    if not terrain or start is None or not target_features:
        return None
    targets = {
        neighbor
        for feature in target_features
        for neighbor in adjacent_positions(feature)
        if is_walkable(terrain, neighbor)
    }
    if not targets:
        return None
    if start in targets:
        return 0

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, distance = queue.popleft()
        for delta in ACTION_DELTAS.values():
            nxt = add_pos(current, delta)
            if nxt in visited or not is_walkable(terrain, nxt):
                continue
            if nxt in targets:
                return distance + 1
            visited.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def shortest_feature_distance(
    terrain: list[str],
    source_feature: tuple[int, int],
    target_features: list[tuple[int, int]],
) -> int | None:
    starts = [
        position
        for position in adjacent_positions(source_feature)
        if is_walkable(terrain, position)
    ]
    distances = [
        shortest_walk_distance(terrain, start, target_features)
        for start in starts
    ]
    return min(
        [distance for distance in distances if distance is not None],
        default=None,
    )


def closer(first: int | None, second: int | None) -> bool | None:
    if first is None or second is None:
        return None
    return first < second


def pot_state_has(pot_states: Any, *keys: str) -> bool | None:
    if not isinstance(pot_states, dict):
        return None
    return any(bool(pot_states.get(key)) for key in keys)


def pot_ingredients_from_facts(facts: dict[str, Any]) -> list[str]:
    terrain = (facts.get("layout_features") or {}).get("terrain") or []
    pot_locations = set(feature_positions(terrain, "P"))
    soups = []
    for obj in facts.get("objects") or []:
        payload = object_payload(obj)
        if object_name(obj) != "soup" or object_position(obj) not in pot_locations:
            continue
        if payload.get("is_cooking") or payload.get("is_ready"):
            return list(RING_TOMATO_ONION_RECIPE)
        soups.append([str(item) for item in payload.get("ingredients") or []])
    return max(soups, key=len, default=[])


def missing_ingredients_from_facts(facts: dict[str, Any]) -> list[str]:
    pot_states = facts.get("pot_states")
    if pot_state_has(pot_states, "cooking", "ready"):
        return []
    missing = Counter(RING_TOMATO_ONION_RECIPE)
    missing.subtract(Counter(pot_ingredients_from_facts(facts)))
    ordered = []
    for ingredient in RING_TOMATO_ONION_RECIPE:
        if missing[ingredient] > 0:
            ordered.append(ingredient)
            missing[ingredient] -= 1
    return ordered


def recipe_needs_from_facts(facts: dict[str, Any], ingredient: str) -> bool | None:
    layout_name = (facts.get("layout_features") or {}).get("layout_name") or ""
    if "tomato_onion" not in layout_name and "ring_tomato_onion" not in layout_name:
        return None
    return ingredient in missing_ingredients_from_facts(facts)


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


def loose_object_positions(facts: dict[str, Any], name: str) -> list[tuple[int, int]]:
    positions = []
    for obj in facts.get("objects") or []:
        if object_name(obj) != name:
            continue
        position = object_position(obj)
        if position is not None:
            positions.append(position)
    return positions


def dispenser_positions(terrain: list[str], name: str) -> list[tuple[int, int]]:
    symbol = {"tomato": "T", "onion": "O", "dish": "D"}.get(name)
    return feature_positions(terrain, symbol) if symbol else []


def pickup_positions(
    facts: dict[str, Any],
    terrain: list[str],
    name: str,
) -> list[tuple[int, int]]:
    return list(
        dict.fromkeys(
            [
                *loose_object_positions(facts, name),
                *dispenser_positions(terrain, name),
            ]
        )
    )


def useful_counter_object(
    facts: dict[str, Any],
    *,
    ai_pos: tuple[int, int] | None,
    terrain: list[str],
    missing_ingredients: list[str],
    pot_cooking_or_ready: bool | None,
    pot_positions: list[tuple[int, int]],
) -> dict[str, Any] | None:
    useful_names = set(missing_ingredients)
    if pot_cooking_or_ready:
        useful_names.add("dish")

    candidates = []
    for obj in facts.get("objects") or []:
        name = object_name(obj)
        position = object_position(obj)
        if name not in useful_names or position is None:
            continue
        if terrain_at(terrain, position) != "X":
            continue
        distance = shortest_walk_distance(terrain, ai_pos, [position])
        if distance is None:
            continue
        pot_distance = shortest_feature_distance(terrain, position, pot_positions)
        total_task_distance = (
            distance + pot_distance
            if pot_distance is not None
            else distance
        )
        candidates.append(
            {
                "name": name,
                "position": position,
                "distance": distance,
                "pot_distance": pot_distance,
                "total_task_distance": total_task_distance,
            }
        )
    return min(
        candidates,
        key=lambda item: (item["total_task_distance"], item["distance"]),
        default=None,
    )


def best_dispenser_route(
    *,
    terrain: list[str],
    ai_pos: tuple[int, int] | None,
    object_type: str,
    pot_positions: list[tuple[int, int]],
) -> dict[str, int | None] | None:
    candidates = []
    for position in dispenser_positions(terrain, object_type):
        pickup_distance = shortest_walk_distance(terrain, ai_pos, [position])
        if pickup_distance is None:
            continue
        pot_distance = shortest_feature_distance(terrain, position, pot_positions)
        total_task_distance = (
            pickup_distance + pot_distance
            if pot_distance is not None
            else pickup_distance
        )
        candidates.append(
            {
                "distance": pickup_distance,
                "pot_distance": pot_distance,
                "total_task_distance": total_task_distance,
            }
        )
    return min(
        candidates,
        key=lambda item: (int(item["total_task_distance"]), int(item["distance"])),
        default=None,
    )


def infer_human_subgoal(
    *,
    human_held: str | None,
    pot_cooking_or_ready: bool | None,
) -> str | None:
    if human_held == "tomato":
        return "PUT_TOMATO_IN_POT"
    if human_held == "onion":
        return "PUT_ONION_IN_POT"
    if human_held == "soup":
        return "SERVE_SOUP"
    if human_held == "dish" and pot_cooking_or_ready:
        return "PICKUP_SOUP"
    return None


def extract_condition_features(step: dict[str, Any] | None) -> dict[str, Any]:
    features = null_condition_features()
    if not step:
        return features
    recorded = step.get("ai_condition_features")

    facts = step.get("state_facts") or {}
    before = (step.get("extra") or {}).get("state_before") or {}
    pot_states = facts.get("pot_states")
    terrain = (facts.get("layout_features") or {}).get("terrain") or []
    ai_held = object_name(facts.get("ai_held_object"))
    human_held = object_name(facts.get("human_held_object"))
    ai_pos = pos_tuple(facts.get("ai_pos"))
    human_pos = pos_tuple(facts.get("human_pos"))
    missing_ingredients = missing_ingredients_from_facts(facts)
    needed_ingredient = missing_ingredients[0] if missing_ingredients else None

    features["pot_empty"] = pot_state_has(pot_states, "empty")
    features["pot_partially_filled"] = pot_state_has(
        pot_states,
        "1_items",
        "2_items",
        "partially_full",
    )
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
    features["needed_ingredient"] = needed_ingredient
    features["human_holding_last_needed_ingredient"] = bool(
        len(missing_ingredients) == 1 and human_held == missing_ingredients[0]
    )
    features["human_inferred_subgoal"] = infer_human_subgoal(
        human_held=human_held,
        pot_cooking_or_ready=features["pot_cooking_or_ready"],
    )

    dish_positions = pickup_positions(facts, terrain, "dish")
    pot_positions = feature_positions(terrain, "P")
    needed_positions = (
        pickup_positions(facts, terrain, needed_ingredient)
        if needed_ingredient
        else []
    )
    human_dish_distance = shortest_walk_distance(terrain, human_pos, dish_positions)
    ai_dish_distance = shortest_walk_distance(terrain, ai_pos, dish_positions)
    human_pot_distance = shortest_walk_distance(terrain, human_pos, pot_positions)
    ai_pot_distance = shortest_walk_distance(terrain, ai_pos, pot_positions)
    human_ingredient_distance = shortest_walk_distance(
        terrain,
        human_pos,
        needed_positions,
    )
    ai_ingredient_distance = shortest_walk_distance(terrain, ai_pos, needed_positions)
    features["human_distance_to_dish"] = human_dish_distance
    features["ai_distance_to_dish"] = ai_dish_distance
    features["human_distance_to_pot"] = human_pot_distance
    features["ai_distance_to_pot"] = ai_pot_distance
    features["human_waiting_near_pot"] = bool(
        features["pot_cooking_or_ready"]
        and features["human_has_dish"]
        and human_pot_distance is not None
        and human_pot_distance <= 1
    )
    features["human_distance_to_needed_ingredient"] = human_ingredient_distance
    features["ai_distance_to_needed_ingredient"] = ai_ingredient_distance
    features["human_closer_to_dish"] = closer(
        human_dish_distance,
        ai_dish_distance,
    )
    features["human_closer_to_pot"] = closer(
        human_pot_distance,
        ai_pot_distance,
    )
    features["ai_closer_to_ingredient"] = closer(
        ai_ingredient_distance,
        human_ingredient_distance,
    )

    counter_object = useful_counter_object(
        facts,
        ai_pos=ai_pos,
        terrain=terrain,
        missing_ingredients=missing_ingredients,
        pot_cooking_or_ready=features["pot_cooking_or_ready"],
        pot_positions=pot_positions,
    )
    features["useful_counter_object_available"] = counter_object is not None
    if counter_object:
        object_type = str(counter_object["name"])
        counter_distance = int(counter_object["distance"])
        counter_pot_distance = counter_object.get("pot_distance")
        counter_total_distance = counter_object.get("total_task_distance")
        dispenser_route = best_dispenser_route(
            terrain=terrain,
            ai_pos=ai_pos,
            object_type=object_type,
            pot_positions=pot_positions,
        )
        dispenser_distance = (
            dispenser_route.get("distance") if dispenser_route else None
        )
        dispenser_pot_distance = (
            dispenser_route.get("pot_distance") if dispenser_route else None
        )
        dispenser_total_distance = (
            dispenser_route.get("total_task_distance") if dispenser_route else None
        )
        features["useful_counter_object_type"] = object_type
        features["useful_counter_object_position"] = list(counter_object["position"])
        features["useful_counter_object_distance"] = counter_distance
        features["useful_counter_object_pot_distance"] = counter_pot_distance
        features["useful_counter_total_task_distance"] = counter_total_distance
        features["matching_dispenser_distance"] = dispenser_distance
        features["matching_dispenser_pot_distance"] = dispenser_pot_distance
        features["matching_dispenser_total_task_distance"] = (
            dispenser_total_distance
        )
        features["useful_counter_object_closer_than_dispenser"] = closer(
            counter_distance,
            dispenser_distance,
        )
        features["useful_counter_object_closer_to_pot_than_dispenser"] = closer(
            counter_pot_distance,
            dispenser_pot_distance,
        )
        features["useful_counter_object_lower_task_cost_than_dispenser"] = closer(
            counter_total_distance,
            dispenser_total_distance,
        )
    else:
        features["useful_counter_object_closer_than_dispenser"] = False
        features["useful_counter_object_closer_to_pot_than_dispenser"] = False
        features["useful_counter_object_lower_task_cost_than_dispenser"] = False

    human_action_name = step.get("human_action_name")
    delta = ACTION_DELTAS.get(str(human_action_name))
    human_before = pos_tuple(before.get("human_pos"))
    human_after = pos_tuple(facts.get("human_pos"))
    ai_before = pos_tuple(before.get("ai_pos"))
    ai_after = pos_tuple(facts.get("ai_pos"))
    if delta and human_before is not None and human_after is not None:
        attempted_target = add_pos(human_before, delta)
        target_was_ai = attempted_target in {ai_before, ai_after}
        features["human_trying_to_pass"] = target_was_ai
        features["ai_on_human_path"] = target_was_ai
        terrain = (before.get("layout_features") or {}).get("terrain") or []
        corridor_width = walkable_neighbor_count(terrain, human_before)
        features["narrow_corridor"] = (
            corridor_width is not None and corridor_width <= 2
        )
    else:
        features["human_trying_to_pass"] = False
        features["ai_on_human_path"] = False
        features["narrow_corridor"] = False

    if isinstance(recorded, dict):
        for key, value in recorded.items():
            if key in features and features[key] is None:
                features[key] = value

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
