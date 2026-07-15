"""Runtime-only H0 rule teacher and MotionPlanner helpers."""

from __future__ import annotations

from collections import Counter, deque

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner


SUBGOALS = (
    "GET_TOMATO",
    "PUT_TOMATO_IN_POT",
    "GET_ONION",
    "PUT_ONION_IN_POT",
    "GET_DISH",
    "PICKUP_SOUP",
    "SERVE_SOUP",
    "WAIT",
)
SUBGOAL_TO_INDEX = {name: index for index, name in enumerate(SUBGOALS)}


def target_recipe(mdp) -> list[str]:
    if not mdp.start_all_orders:
        return ["onion", "onion", "onion"]
    order = mdp.start_all_orders[0]
    if isinstance(order, dict):
        return list(order.get("ingredients", []))
    return list(getattr(order, "ingredients", order))


def pot_ingredients(state, pot_pos: tuple[int, int]) -> list[str]:
    if not state.has_object(pot_pos):
        return []
    obj = state.get_object(pot_pos)
    if getattr(obj, "name", None) != "soup":
        return []
    return [
        getattr(ingredient, "name", ingredient)
        for ingredient in getattr(obj, "ingredients", [])
    ]


def missing_ingredients(recipe: list[str], current: list[str]) -> list[str]:
    missing = Counter(recipe)
    missing.subtract(Counter(current))
    ordered = []
    for ingredient in recipe:
        if missing[ingredient] > 0:
            ordered.append(ingredient)
            missing[ingredient] -= 1
    return ordered


def pots_needing_ingredient(state, mdp, ingredient: str) -> list[tuple[int, int]]:
    recipe = target_recipe(mdp)
    targets = []
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            if getattr(obj, "name", None) != "soup":
                continue
            if getattr(obj, "is_ready", False) or getattr(obj, "is_cooking", False):
                continue
        if ingredient in missing_ingredients(recipe, pot_ingredients(state, pot_pos)):
            targets.append(pot_pos)
    return targets


def next_needed_ingredient(state, mdp) -> str | None:
    recipe = target_recipe(mdp)
    best_missing: list[str] = []
    best_filled = -1
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            if getattr(obj, "name", None) != "soup":
                continue
            if getattr(obj, "is_ready", False) or getattr(obj, "is_cooking", False):
                continue
        current = pot_ingredients(state, pot_pos)
        missing = missing_ingredients(recipe, current)
        if missing and len(current) > best_filled:
            best_missing = missing
            best_filled = len(current)
    return best_missing[0] if best_missing else None


def ingredient_dispenser_locations(mdp, ingredient: str) -> list[tuple[int, int]]:
    if ingredient == "tomato":
        return mdp.get_tomato_dispenser_locations()
    if ingredient == "onion":
        return mdp.get_onion_dispenser_locations()
    return []


def object_positions(state, name: str) -> list[tuple[int, int]]:
    return [
        position
        for position, obj in state.objects.items()
        if getattr(obj, "name", None) == name
    ]


def ingredient_pickup_locations(state, mdp, ingredient: str) -> list[tuple[int, int]]:
    locations = [
        *object_positions(state, ingredient),
        *ingredient_dispenser_locations(mdp, ingredient),
    ]
    return list(dict.fromkeys(locations))


def adjacent_position(player) -> tuple[int, int]:
    return Action.move_in_direction(player.position, player.orientation)


def adjacent_feature_action(
    player,
    feature_positions: list[tuple[int, int]],
) -> int | None:
    feature_set = set(feature_positions)
    for direction in Action.MOTION_ACTIONS:
        if direction == Action.STAY:
            continue
        if Action.move_in_direction(player.position, direction) in feature_set:
            if player.orientation == direction:
                return int(Action.ACTION_TO_INDEX[Action.INTERACT])
            return int(Action.ACTION_TO_INDEX[direction])
    return None


def motion_action_moves_to_open_tile(mdp, player, action) -> bool:
    if action not in Action.MOTION_ACTIONS or action == Action.STAY:
        return True
    return Action.move_in_direction(
        player.position,
        action,
    ) in mdp.get_valid_player_positions()


def action_plan_hits_blocked_position(
    start_position: tuple[int, int],
    action_plan,
    blocked_positions: set[tuple[int, int]],
) -> bool:
    position = start_position
    for action in action_plan:
        if action in Action.MOTION_ACTIONS and action != Action.STAY:
            position = Action.move_in_direction(position, action)
        if position in blocked_positions:
            return True
    return False


def bfs_first_action_to_feature(
    mdp,
    player,
    feature_positions: list[tuple[int, int]],
    blocked_positions: set[tuple[int, int]] | None = None,
) -> int | None:
    blocked_positions = blocked_positions or set()
    valid_positions = set(mdp.get_valid_player_positions()) - blocked_positions
    start = player.position
    if start not in valid_positions:
        valid_positions.add(start)

    targets: dict[tuple[int, int], tuple[int, int]] = {}
    for feature_pos in feature_positions:
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            candidate = Action.move_in_direction(feature_pos, direction)
            if candidate in valid_positions:
                targets[candidate] = feature_pos
    if not targets:
        return None

    if start in targets:
        feature_pos = targets[start]
        desired_orientation = (
            feature_pos[0] - start[0],
            feature_pos[1] - start[1],
        )
        if player.orientation == desired_orientation:
            return int(Action.ACTION_TO_INDEX[Action.INTERACT])
        return int(Action.ACTION_TO_INDEX[desired_orientation])

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None] = {
        start: None
    }
    while queue:
        position = queue.popleft()
        if position in targets:
            first = position
            while parent[first] is not None and parent[first][0] != start:
                first = parent[first][0]
            if parent[first] is None:
                return None
            return int(Action.ACTION_TO_INDEX[parent[first][1]])
        for action in Action.MOTION_ACTIONS:
            if action == Action.STAY:
                continue
            nxt = Action.move_in_direction(position, action)
            if nxt not in valid_positions or nxt in parent:
                continue
            parent[nxt] = (position, action)
            queue.append(nxt)
    return None


def first_action_to_feature(
    motion_planner: MotionPlanner,
    player,
    feature_positions: list[tuple[int, int]],
    blocked_positions: set[tuple[int, int]] | None = None,
) -> int | None:
    blocked_positions = blocked_positions or set()
    mdp = motion_planner.mdp
    if not feature_positions:
        return None
    adjacent_action = adjacent_feature_action(player, feature_positions)
    if adjacent_action is not None:
        return adjacent_action

    best_plan = None
    best_cost = float("inf")
    for feature_pos in feature_positions:
        for goal in motion_planner.motion_goals_for_pos.get(feature_pos, []):
            start = player.pos_and_or
            if not motion_planner.is_valid_motion_start_goal_pair(start, goal):
                continue
            action_plan, _, cost = motion_planner.get_plan(start, goal)
            if action_plan:
                first_action = action_plan[0]
                if not motion_action_moves_to_open_tile(mdp, player, first_action):
                    continue
                if action_plan_hits_blocked_position(
                    player.position,
                    action_plan,
                    blocked_positions,
                ):
                    continue
                if (
                    first_action in Action.MOTION_ACTIONS
                    and Action.move_in_direction(player.position, first_action)
                    in blocked_positions
                ):
                    continue
            if cost < best_cost:
                best_plan = action_plan
                best_cost = cost
    if best_plan:
        return int(Action.ACTION_TO_INDEX[best_plan[0]])
    if best_plan == [] and adjacent_position(player) in feature_positions:
        return int(Action.ACTION_TO_INDEX[Action.INTERACT])
    return bfs_first_action_to_feature(
        mdp,
        player,
        feature_positions,
        blocked_positions,
    )


def rule_teacher_decision(
    state,
    motion_planner: MotionPlanner,
    player_index: int = 0,
) -> tuple[str, int]:
    mdp = motion_planner.mdp
    player = state.players[player_index]
    blocked_positions = {
        other.position
        for index, other in enumerate(state.players)
        if index != player_index
    }
    held_name = getattr(player.held_object, "name", None)
    pot_states = mdp.get_pot_states(state)

    if held_name in ("tomato", "onion"):
        targets = pots_needing_ingredient(state, mdp, held_name)
        if not targets:
            return "WAIT", int(Action.ACTION_TO_INDEX[Action.STAY])
        subgoal = (
            "PUT_TOMATO_IN_POT"
            if held_name == "tomato"
            else "PUT_ONION_IN_POT"
        )
        action = first_action_to_feature(
            motion_planner,
            player,
            targets,
            blocked_positions,
        )
        return subgoal, (
            action
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )
    if held_name == "dish":
        ready_pots = mdp.get_ready_pots(pot_states)
        if not ready_pots:
            return "WAIT", int(Action.ACTION_TO_INDEX[Action.STAY])
        action = first_action_to_feature(
            motion_planner,
            player,
            ready_pots,
            blocked_positions,
        )
        return "PICKUP_SOUP", (
            action
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )
    if held_name == "soup":
        action = first_action_to_feature(
            motion_planner,
            player,
            mdp.get_serving_locations(),
            blocked_positions,
        )
        return "SERVE_SOUP", (
            action
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )

    ready_pots = mdp.get_ready_pots(pot_states)
    if ready_pots:
        action = first_action_to_feature(
            motion_planner,
            player,
            mdp.get_dish_dispenser_locations(),
            blocked_positions,
        )
        return "GET_DISH", (
            action
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )

    needed = next_needed_ingredient(state, mdp)
    if needed is not None:
        subgoal = "GET_TOMATO" if needed == "tomato" else "GET_ONION"
        action = first_action_to_feature(
            motion_planner,
            player,
            ingredient_pickup_locations(state, mdp, needed),
            blocked_positions,
        )
        return subgoal, (
            action
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )
    return "WAIT", int(Action.ACTION_TO_INDEX[Action.STAY])


def make_motion_planner(layout: str, seed: int, horizon: int) -> MotionPlanner:
    from durf.baseline.runtime import make_direct_multi_env

    env = make_direct_multi_env(layout, seed=seed, horizon=horizon)
    try:
        mdp = env.base_env.mdp
        counter_goals = []
        if mdp.start_state is not None:
            counter_goals = [
                position
                for position, _ in mdp.start_state.objects.items()
                if mdp.get_terrain_type_at_pos(position) == "X"
            ]
        return MotionPlanner.from_pickle_or_compute(
            mdp,
            counter_goals=counter_goals,
            info=False,
        )
    finally:
        env.close()


def plan_h0_subgoal(state, mdp, player_index: int = 0) -> str:
    """Choose the high-level task without constructing a MotionPlanner."""

    player = state.players[player_index]
    held = player.held_object
    held_name = getattr(held, "name", None)
    pot_states = mdp.get_pot_states(state)

    if held_name in ("tomato", "onion"):
        if not pots_needing_ingredient(state, mdp, held_name):
            return "WAIT"
        return (
            "PUT_TOMATO_IN_POT"
            if held_name == "tomato"
            else "PUT_ONION_IN_POT"
        )
    if held_name == "dish":
        return "PICKUP_SOUP" if mdp.get_ready_pots(pot_states) else "WAIT"
    if held_name == "soup":
        return "SERVE_SOUP"

    if mdp.get_ready_pots(pot_states):
        return "GET_DISH"
    needed = next_needed_ingredient(state, mdp)
    if needed == "tomato":
        return "GET_TOMATO"
    if needed == "onion":
        return "GET_ONION"
    return "WAIT"
