"""Runtime-only H0 rule teacher and MotionPlanner helpers."""

from __future__ import annotations

from collections import Counter, deque

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner


H0_MODEL_SUBGOALS = (
    "GET_TOMATO",
    "PUT_TOMATO_IN_POT",
    "GET_ONION",
    "PUT_ONION_IN_POT",
    "GET_DISH",
    "PICKUP_SOUP",
    "SERVE_SOUP",
    "WAIT",
)
# The bundled H0 network keeps its original 8-way subgoal input contract.
# STASH and YIELD_PATH are runtime feasibility actions executed by this
# planner, not new neural output dimensions.  They are still ranked by the
# learned reward once live geometry says they are executable.
SUBGOALS = H0_MODEL_SUBGOALS[:-1] + (
    "STASH_HELD_OBJECT",
    "YIELD_PATH",
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


def dish_pickup_locations(state, mdp) -> list[tuple[int, int]]:
    """Prefer reachable placed dishes while retaining the dispenser fallback."""

    locations = [
        *object_positions(state, "dish"),
        *mdp.get_dish_dispenser_locations(),
    ]
    return list(dict.fromkeys(locations))


def empty_non_feature_counter_locations(state, mdp) -> list[tuple[int, int]]:
    """Empty counters on which a held object can safely be stashed.

    Counter occupancy is state-dependent, while dispensers, pots and serving
    windows are never valid stash targets.  The latter exclusion is retained
    explicitly so this helper also behaves correctly with lightweight test
    MDPs whose terrain APIs are less strict than OvercookedGridworld's.
    """

    get_counters = getattr(mdp, "get_counter_locations", None)
    counters = list(get_counters()) if get_counters is not None else []
    feature_positions: set[tuple[int, int]] = set()
    for getter_name in (
        "get_pot_locations",
        "get_serving_locations",
        "get_dish_dispenser_locations",
        "get_tomato_dispenser_locations",
        "get_onion_dispenser_locations",
    ):
        getter = getattr(mdp, getter_name, None)
        if getter is not None:
            feature_positions.update(tuple(position) for position in getter())
    occupied = {tuple(position) for position in getattr(state, "objects", {})}
    return [
        tuple(position)
        for position in counters
        if tuple(position) not in feature_positions
        and tuple(position) not in occupied
    ]


def useful_stash_counter_locations(state, mdp) -> list[tuple[int, int]]:
    """Empty counters closest to a pot, with all-empty fallback.

    STASH is preparation for a later pot/serving step, so placing an object on
    a task-adjacent counter is a state-based utility criterion.  It does not
    depend on feedback wording or on the learned reward, and the reward still
    decides whether STASH is selected at all.
    """

    counters = empty_non_feature_counter_locations(state, mdp)
    pots = [tuple(position) for position in mdp.get_pot_locations()]
    if not counters or not pots:
        return counters
    distance = {
        counter: min(
            abs(counter[0] - pot[0]) + abs(counter[1] - pot[1]) for pot in pots
        )
        for counter in counters
    }
    best = min(distance.values())
    return [counter for counter in counters if distance[counter] == best]

def adjacent_position(player) -> tuple[int, int]:
    return Action.move_in_direction(player.position, player.orientation)


def yield_path_action(state, mdp, player_index: int = 0) -> int | None:
    """Return one legal step that vacates the human's explicit next tile.

    This is deliberately a geometry predicate, not a language rule.  A yield
    exists only when the teammate is adjacent, is currently facing the AI,
    and the AI can move to another unoccupied floor tile in one step.  Side
    exits are preferred over remaining on the teammate's forward ray; a
    corridor-forward step is retained as the generic fallback.
    """

    players = list(getattr(state, "players", ()))
    if len(players) < 2 or not 0 <= player_index < len(players):
        return None
    human_index = (
        1 - player_index
        if len(players) == 2
        else (player_index + 1) % len(players)
    )
    ai = players[player_index]
    human = players[human_index]
    human_direction = tuple(getattr(human, "orientation", ()))
    if human_direction not in Action.MOTION_ACTIONS or human_direction == Action.STAY:
        return None
    if Action.move_in_direction(human.position, human_direction) != ai.position:
        return None

    valid = set(mdp.get_valid_player_positions())
    occupied = {
        tuple(player.position)
        for index, player in enumerate(players)
        if index != player_index
    }
    hx, hy = human.position
    dx, dy = human_direction
    options: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for order, direction in enumerate(Action.MOTION_ACTIONS):
        if direction == Action.STAY:
            continue
        destination = Action.move_in_direction(ai.position, direction)
        if destination not in valid or destination in occupied:
            continue
        offset = (destination[0] - hx, destination[1] - hy)
        # A zero cross product and positive dot product means the destination
        # remains directly ahead of the human. Prefer a side exit when one is
        # available, then prefer a tile with more onward room.
        on_forward_ray = int(
            offset[0] * dy - offset[1] * dx == 0
            and offset[0] * dx + offset[1] * dy > 0
        )
        onward_degree = sum(
            Action.move_in_direction(destination, nxt) in valid
            and Action.move_in_direction(destination, nxt) not in occupied
            for nxt in Action.MOTION_ACTIONS
            if nxt != Action.STAY
        )
        options.append(((on_forward_ray, -onward_degree, order), direction))
    if not options:
        return None
    _, chosen = min(options, key=lambda item: item[0])
    return int(Action.ACTION_TO_INDEX[chosen])


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


def bfs_first_action_to_positions(
    mdp,
    player,
    target_positions: set[tuple[int, int]],
    blocked_positions: set[tuple[int, int]] | None = None,
) -> int | None:
    """Move toward an open floor target without treating it as interactable."""

    blocked_positions = blocked_positions or set()
    valid_positions = set(mdp.get_valid_player_positions()) - blocked_positions
    targets = set(target_positions) & valid_positions
    start = player.position
    if start in targets or not targets:
        return None
    valid_positions.add(start)
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None] = {
        start: None
    }
    while queue:
        position = queue.popleft()
        for action in Action.MOTION_ACTIONS:
            if action == Action.STAY:
                continue
            nxt = Action.move_in_direction(position, action)
            if nxt not in valid_positions or nxt in parent:
                continue
            parent[nxt] = (position, action)
            if nxt in targets:
                first = nxt
                while parent[first] is not None and parent[first][0] != start:
                    first = parent[first][0]
                return int(Action.ACTION_TO_INDEX[parent[first][1]])
            queue.append(nxt)
    return None


def staging_action_for_blocked_feature(
    mdp,
    player,
    feature_positions: list[tuple[int, int]],
    blocked_positions: set[tuple[int, int]],
) -> int | None:
    """Approach a human-blocked interaction tile instead of freezing far away."""

    valid_positions = set(mdp.get_valid_player_positions())
    blocked_access: set[tuple[int, int]] = set()
    for feature_pos in feature_positions:
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            access = Action.move_in_direction(feature_pos, direction)
            if access in valid_positions and access in blocked_positions:
                blocked_access.add(access)
    staging: set[tuple[int, int]] = set()
    for access in blocked_access:
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            candidate = Action.move_in_direction(access, direction)
            if candidate in valid_positions and candidate not in blocked_positions:
                staging.add(candidate)
    return bfs_first_action_to_positions(mdp, player, staging, blocked_positions)

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
    bfs_action = bfs_first_action_to_feature(
        mdp,
        player,
        feature_positions,
        blocked_positions,
    )
    if bfs_action is not None:
        return bfs_action
    return staging_action_for_blocked_feature(
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
            action = first_action_to_feature(
                motion_planner,
                player,
                useful_stash_counter_locations(state, mdp),
                blocked_positions,
            )
            return "STASH_HELD_OBJECT", (
                action
                if action is not None
                else int(Action.ACTION_TO_INDEX[Action.STAY])
            )
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
            action = first_action_to_feature(
                motion_planner,
                player,
                useful_stash_counter_locations(state, mdp),
                blocked_positions,
            )
            return "STASH_HELD_OBJECT", (
                action
                if action is not None
                else int(Action.ACTION_TO_INDEX[Action.STAY])
            )
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
            dish_pickup_locations(state, mdp),
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


def subgoal_target_positions(
    state, mdp, subgoal: str, player_index: int = 0
) -> list[tuple[int, int]]:
    """Feature positions a given subgoal wants to reach.

    Mirrors the per-subgoal target logic inside ``rule_teacher_decision`` but
    keyed by an *externally chosen* subgoal, so a comfort reranker can decide
    WHICH subgoal to pursue while H0 keeps owning execution.
    """

    if subgoal == "STASH_HELD_OBJECT":
        return useful_stash_counter_locations(state, mdp)
    pot_states = mdp.get_pot_states(state)
    if subgoal == "GET_TOMATO":
        # A counter tomato is already a completed staging result while every
        # pot is closed (cooking/ready).  Treating it as the source of a new
        # prefetch task makes the agent immediately undo STASH by picking the
        # same object back up.  Once a pot can accept tomato again, staged
        # objects regain priority over the dispenser.
        if pots_needing_ingredient(state, mdp, "tomato"):
            return ingredient_pickup_locations(state, mdp, "tomato")
        return ingredient_dispenser_locations(mdp, "tomato")
    if subgoal == "GET_ONION":
        if pots_needing_ingredient(state, mdp, "onion"):
            return ingredient_pickup_locations(state, mdp, "onion")
        return ingredient_dispenser_locations(mdp, "onion")
    if subgoal == "PUT_TOMATO_IN_POT":
        return pots_needing_ingredient(state, mdp, "tomato")
    if subgoal == "PUT_ONION_IN_POT":
        return pots_needing_ingredient(state, mdp, "onion")
    if subgoal == "GET_DISH":
        # Likewise, a staged dish should be consumed when soup is ready, but
        # must not be picked up and stashed repeatedly during prefetch.
        if mdp.get_ready_pots(pot_states):
            return dish_pickup_locations(state, mdp)
        return list(mdp.get_dish_dispenser_locations())
    if subgoal == "PICKUP_SOUP":
        return mdp.get_ready_pots(pot_states)
    if subgoal == "SERVE_SOUP":
        return mdp.get_serving_locations()
    return []  # YIELD_PATH, WAIT, or unknown


def execute_subgoal(
    state, motion_planner: MotionPlanner, subgoal: str, player_index: int = 0
) -> int:
    """Return the next action that advances ``subgoal`` (H0-owned execution).

    ``WAIT`` (and any subgoal with no reachable target) resolves to ``STAY``.
    """

    if subgoal == "WAIT":
        return int(Action.ACTION_TO_INDEX[Action.STAY])
    mdp = motion_planner.mdp
    if subgoal == "YIELD_PATH":
        action = yield_path_action(state, mdp, player_index=player_index)
        return (
            int(action)
            if action is not None
            else int(Action.ACTION_TO_INDEX[Action.STAY])
        )
    player = state.players[player_index]
    blocked_positions = {
        other.position
        for index, other in enumerate(state.players)
        if index != player_index
    }
    targets = subgoal_target_positions(state, mdp, subgoal, player_index)
    action = first_action_to_feature(motion_planner, player, targets, blocked_positions)
    if action is None:
        return int(Action.ACTION_TO_INDEX[Action.STAY])
    return int(action)


def make_motion_planner(layout: str, seed: int, horizon: int) -> MotionPlanner:
    from durf.baseline.runtime import make_direct_multi_env

    env = make_direct_multi_env(layout, seed=seed, horizon=horizon)
    try:
        mdp = env.base_env.mdp
        # Empty-counter availability changes during play, so every counter
        # must be a possible motion goal. The executor filters occupied ones
        # against the live state before selecting a stash target.
        counter_goals = list(mdp.get_counter_locations())
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
            return "STASH_HELD_OBJECT"
        return (
            "PUT_TOMATO_IN_POT"
            if held_name == "tomato"
            else "PUT_ONION_IN_POT"
        )
    if held_name == "dish":
        return (
            "PICKUP_SOUP"
            if mdp.get_ready_pots(pot_states)
            else "STASH_HELD_OBJECT"
        )
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
