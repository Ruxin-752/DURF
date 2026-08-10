"""Collect clean task demonstrations from a recipe-aware rule teacher.

This is the first-stage backbone dataset: the rule teacher actively rolls out
the full task from the reset state instead of labeling states visited by a weak
PPO policy.  The resulting dataset is meant to teach task mechanics, not human
preference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import dill
import numpy as np

from durf.baseline.evaluate_baseline import count_event_value
from durf.baseline.runtime import make_direct_multi_env, resolve_agent_dir
from durf.baseline.task_logic import next_unstaged_ingredient
from human_aware_rl.rllib.rllib import load_trainer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "rule_teacher_datasets"
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


@dataclass
class CandidateSubgoal:
    subgoal: str
    task_score: float
    action: int
    reason: str
    feasible: bool = True
    metadata: dict = field(default_factory=dict)
    hu_score: float = 0.0
    final_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-agent", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-id", default="ppo_0")
    parser.add_argument("--teacher-player-index", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--partner-mode",
        choices=("stay", "safe_corner", "safe_left", "safe_recycle", "dynamic_avoid"),
        default="stay",
        help=(
            "First-stage clean rollout partner behavior. 'safe_corner' moves "
            "the partner out of the top corridor before staying still."
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument(
        "--start-jitter-steps",
        type=int,
        default=0,
        help=(
            "Before recording a rollout, execute this many random motion-only "
            "teacher actions. This creates recovery starts away from the fixed spawn."
        ),
    )
    parser.add_argument(
        "--stop-after-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop each rollout as soon as cumulative reward reaches the success threshold.",
    )
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Also keep examples from failed rollouts for debugging.",
    )
    return parser.parse_args()


def load_params_for_agent(agent_dir: Path) -> dict:
    config_path = agent_dir.parent / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("rb") as handle:
        return dill.load(handle)


def featurize_for_policy(featurize_fn, state, player_index: int) -> np.ndarray:
    try:
        obs_pair = featurize_fn(state, debug=False)
    except TypeError:
        obs_pair = featurize_fn(state)
    return np.asarray(obs_pair[player_index], dtype=np.float32)


def object_name(obj) -> str | None:
    if obj is None:
        return None
    return getattr(obj, "name", str(obj))


def state_summary(state) -> dict:
    players = []
    for player in state.players:
        players.append(
            {
                "position": list(player.position),
                "orientation": list(player.orientation),
                "held_object": object_name(player.held_object),
            }
        )
    objects = []
    for position, obj in sorted(state.objects.items()):
        objects.append(
            {
                "position": list(position),
                "name": object_name(obj),
                "ingredients": list(getattr(obj, "ingredients", [])),
                "is_ready": bool(getattr(obj, "is_ready", False)),
                "is_cooking": bool(getattr(obj, "is_cooking", False)),
            }
        )
    return {"timestep": int(state.timestep), "players": players, "objects": objects}


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
    return list(getattr(obj, "ingredients", []))


def missing_ingredients(recipe: list[str], current: list[str]) -> list[str]:
    missing = Counter(recipe)
    missing.subtract(Counter(current))
    ordered = []
    for ingredient in recipe:
        if missing[ingredient] > 0:
            ordered.append(ingredient)
            missing[ingredient] -= 1
    return ordered


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


def staged_ingredients_for_next_cycle(state) -> list[str]:
    """Count loose and teammate-held ingredients already prepared for later."""
    staged = [
        getattr(obj, "name", None)
        for obj in state.objects.values()
        if getattr(obj, "name", None) in {"tomato", "onion"}
    ]
    staged.extend(
        getattr(getattr(player, "held_object", None), "name", None)
        for player in state.players
        if getattr(getattr(player, "held_object", None), "name", None)
        in {"tomato", "onion"}
    )
    return [ingredient for ingredient in staged if ingredient is not None]


def ingredient_pickup_locations(state, mdp, ingredient: str) -> list[tuple[int, int]]:
    """Return all available sources for an ingredient.

    Loose ingredients placed on counters are first-class resources, not a
    special case.  This lets the baseline use tomatoes/onions staged by the
    human instead of always walking back to the corner dispenser.
    """
    locations = [
        *object_positions(state, ingredient),
        *ingredient_dispenser_locations(mdp, ingredient),
    ]
    return list(dict.fromkeys(locations))


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


def teammate_holding(state, player_index: int, object_name_: str) -> bool:
    for index, other in enumerate(state.players):
        if index == player_index:
            continue
        held = getattr(other, "held_object", None)
        if getattr(held, "name", None) == object_name_:
            return True
    return False


def adjacent_position(player) -> tuple[int, int]:
    return Action.move_in_direction(player.position, player.orientation)


def adjacent_feature_action(player, feature_positions: list[tuple[int, int]]) -> int | None:
    feature_set = set(feature_positions)
    for direction in Action.MOTION_ACTIONS:
        if direction == Action.STAY:
            continue
        if Action.move_in_direction(player.position, direction) in feature_set:
            if player.orientation == direction:
                return int(Action.ACTION_TO_INDEX[Action.INTERACT])
            return int(Action.ACTION_TO_INDEX[direction])
    return None


def is_open_player_position(mdp, position: tuple[int, int]) -> bool:
    return position in mdp.get_valid_player_positions()


def motion_action_moves_to_open_tile(mdp, player, action) -> bool:
    if action not in Action.MOTION_ACTIONS or action == Action.STAY:
        return True
    return is_open_player_position(mdp, Action.move_in_direction(player.position, action))


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


def stay_candidate(reason: str, task_score: float = 0.0) -> CandidateSubgoal:
    return CandidateSubgoal(
        subgoal="WAIT",
        task_score=task_score,
        action=int(Action.ACTION_TO_INDEX[Action.STAY]),
        reason=reason,
        feasible=True,
    )


def feature_candidate(
    *,
    subgoal: str,
    task_score: float,
    reason: str,
    motion_planner: MotionPlanner,
    player,
    feature_positions: list[tuple[int, int]],
    blocked_positions: set[tuple[int, int]],
    metadata: dict | None = None,
) -> CandidateSubgoal | None:
    action = first_action_to_feature(
        motion_planner,
        player,
        feature_positions,
        blocked_positions,
    )
    if action is None:
        return None
    candidate_metadata = dict(metadata or {})
    candidate_metadata.setdefault(
        "target_positions",
        [list(position) for position in feature_positions],
    )
    return CandidateSubgoal(
        subgoal=subgoal,
        task_score=task_score,
        action=int(action),
        reason=reason,
        feasible=True,
        metadata=candidate_metadata,
    )


def put_down_candidates(
    state,
    mdp,
    motion_planner: MotionPlanner,
    player,
    blocked_positions: set[tuple[int, int]],
    held_name: str,
) -> list[CandidateSubgoal]:
    """Candidate pool for holding an ingredient no pot needs.

    Dropping the unneeded ingredient on a free counter is the productive
    choice, so PUT_DOWN_OBJECT must be in the pool.  Without it, probes that
    expect PUT_DOWN_OBJECT can never evaluate against the real task pool and
    always report preferred_available=False.
    """
    counters = (
        list(mdp.get_counter_locations())
        if hasattr(mdp, "get_counter_locations")
        else []
    )
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
            feature_positions.update(getter())
    occupied = set(getattr(state, "objects", {}).keys())
    player_pos = tuple(player.position)
    free = sorted(
        (
            tuple(position)
            for position in counters
            if tuple(position) not in feature_positions
            and tuple(position) not in occupied
        ),
        key=lambda position: (
            abs(position[0] - player_pos[0]) + abs(position[1] - player_pos[1])
        ),
    )
    candidate = feature_candidate(
        subgoal="PUT_DOWN_OBJECT",
        task_score=60.0,
        reason="holding_unneeded_ingredient_drop_on_free_counter",
        motion_planner=motion_planner,
        player=player,
        feature_positions=free,
        blocked_positions=blocked_positions,
        metadata={"held_object": held_name},
    )
    if candidate is not None:
        return [candidate, stay_candidate("holding_unneeded_ingredient")]
    return [stay_candidate("holding_unneeded_ingredient")]


def generate_candidate_subgoals(
    state,
    motion_planner: MotionPlanner,
    player_index: int,
) -> list[CandidateSubgoal]:
    mdp = motion_planner.mdp
    player = state.players[player_index]
    blocked_positions = {
        other.position
        for index, other in enumerate(state.players)
        if index != player_index
    }
    held = player.held_object
    pot_states = mdp.get_pot_states(state)
    candidates: list[CandidateSubgoal] = []

    if held is not None:
        held_name = getattr(held, "name", None)
        if held_name in ("tomato", "onion"):
            ingredient_targets = pots_needing_ingredient(state, mdp, held_name)
            if not ingredient_targets:
                return put_down_candidates(
                    state,
                    mdp,
                    motion_planner,
                    player,
                    blocked_positions,
                    held_name,
                )
            subgoal = (
                "PUT_TOMATO_IN_POT"
                if held_name == "tomato"
                else "PUT_ONION_IN_POT"
            )
            candidate = feature_candidate(
                subgoal=subgoal,
                task_score=80.0,
                reason="held_ingredient_needed_by_pot",
                motion_planner=motion_planner,
                player=player,
                feature_positions=ingredient_targets,
                blocked_positions=blocked_positions,
                metadata={"held_object": held_name},
            )
            return [candidate] if candidate else [stay_candidate("no_path_to_needed_pot")]
        if held_name == "dish":
            ready_pots = mdp.get_ready_pots(pot_states)
            if not ready_pots:
                return [stay_candidate("holding_dish_waiting_for_soup", task_score=5.0)]
            candidate = feature_candidate(
                subgoal="PICKUP_SOUP",
                task_score=95.0,
                reason="held_dish_and_soup_ready",
                motion_planner=motion_planner,
                player=player,
                feature_positions=ready_pots,
                blocked_positions=blocked_positions,
            )
            return [candidate] if candidate else [stay_candidate("no_path_to_ready_pot")]
        if held_name == "soup":
            candidate = feature_candidate(
                subgoal="SERVE_SOUP",
                task_score=100.0,
                reason="held_soup_deliver_immediately",
                motion_planner=motion_planner,
                player=player,
                feature_positions=mdp.get_serving_locations(),
                blocked_positions=blocked_positions,
            )
            return [candidate] if candidate else [stay_candidate("no_path_to_serving")]

    ready_pots = mdp.get_ready_pots(pot_states)
    if ready_pots:
        candidate = feature_candidate(
            subgoal="GET_DISH",
            task_score=90.0,
            reason="soup_ready_get_dish",
            motion_planner=motion_planner,
            player=player,
            feature_positions=mdp.get_dish_dispenser_locations(),
            blocked_positions=blocked_positions,
        )
        if candidate:
            candidates.append(candidate)

    cooking_pots = mdp.get_cooking_pots(pot_states)
    if cooking_pots:
        if not teammate_holding(state, player_index, "dish"):
            candidate = feature_candidate(
                subgoal="GET_DISH",
                task_score=60.0,
                reason="soup_cooking_prepare_dish",
                motion_planner=motion_planner,
                player=player,
                feature_positions=mdp.get_dish_dispenser_locations(),
                blocked_positions=blocked_positions,
            )
            if candidate:
                candidates.append(candidate)

        recipe = target_recipe(mdp)
        prep_ingredient = next_unstaged_ingredient(
            recipe,
            staged_ingredients_for_next_cycle(state),
        )
        if prep_ingredient in ("tomato", "onion"):
            subgoal = "GET_TOMATO" if prep_ingredient == "tomato" else "GET_ONION"
            candidate = feature_candidate(
                subgoal=subgoal,
                task_score=50.0,
                reason="soup_cooking_prepare_unstaged_next_cycle_ingredient",
                motion_planner=motion_planner,
                player=player,
                # Loose ingredients are already counted as staged. Fetch a new
                # unit from the dispenser instead of re-picking the object that
                # the guard just placed on a counter.
                feature_positions=ingredient_dispenser_locations(
                    mdp,
                    prep_ingredient,
                ),
                blocked_positions=blocked_positions,
                metadata={"ingredient": prep_ingredient},
            )
            if candidate:
                candidates.append(candidate)

    needed = next_needed_ingredient(state, mdp)
    if needed is not None:
        subgoal = "GET_TOMATO" if needed == "tomato" else "GET_ONION"
        candidate = feature_candidate(
            subgoal=subgoal,
            task_score=70.0,
            reason="pot_needs_ingredient",
            motion_planner=motion_planner,
            player=player,
            feature_positions=ingredient_pickup_locations(state, mdp, needed),
            blocked_positions=blocked_positions,
            metadata={"ingredient": needed},
        )
        if candidate:
            candidates.append(candidate)

    candidates.append(stay_candidate("fallback_wait", task_score=0.0))
    return candidates


def choose_task_candidate(
    candidates: list[CandidateSubgoal],
    *,
    hu_lambda: float = 0.0,
) -> CandidateSubgoal:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return stay_candidate("no_feasible_candidate")
    for candidate in feasible:
        candidate.final_score = candidate.task_score + hu_lambda * candidate.hu_score
    return max(
        feasible,
        key=lambda candidate: (
            candidate.final_score if candidate.final_score is not None else candidate.task_score,
            candidate.task_score,
        ),
    )


def rule_teacher_candidates(
    state,
    motion_planner: MotionPlanner,
    player_index: int,
) -> list[CandidateSubgoal]:
    return generate_candidate_subgoals(state, motion_planner, player_index)


def rule_teacher_decision(
    state,
    motion_planner: MotionPlanner,
    player_index: int,
) -> tuple[str, int]:
    chosen = choose_task_candidate(
        rule_teacher_candidates(state, motion_planner, player_index)
    )
    return chosen.subgoal, chosen.action


def rule_teacher_action(state, motion_planner: MotionPlanner, player_index: int) -> int:
    return rule_teacher_decision(state, motion_planner, player_index)[1]


def action_name(action_index: int):
    return Action.INDEX_TO_ACTION[int(action_index)]


def partner_action(partner_mode: str, partner_index: int, step: int) -> int:
    if partner_mode in ("safe_corner", "safe_left", "safe_recycle"):
        if partner_index == 1:
            # From H0 player 1 start (6, 1), move to the lower-right corridor
            # so player 0 can pass through the top corridor to serve.
            script = [
                Action.ACTION_TO_INDEX[(1, 0)],
                Action.ACTION_TO_INDEX[(1, 0)],
                Action.ACTION_TO_INDEX[(0, 1)],
                Action.ACTION_TO_INDEX[(0, 1)],
                Action.ACTION_TO_INDEX[(0, 1)],
            ]
            if partner_mode == "safe_left":
                # Then clear the service exit and lower corridor before the
                # first delivery finishes.  The left-top holding tile keeps p1
                # out of p0's repeated tomato/onion/service route.
                script.extend(
                    [
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                    ]
                )
            elif partner_mode == "safe_recycle":
                # For repeated deliveries, stay in the lower-right corner
                # during the first cooking cycle, then move to (2, 1).  That
                # leaves the service exit (8, 4), tomato access (1, 4), and
                # dish access (1, 1) open for p0's next cycle.
                script.extend([Action.ACTION_TO_INDEX[Action.STAY]] * 65)
                script.extend(
                    [
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(-1, 0)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                        Action.ACTION_TO_INDEX[(0, -1)],
                        Action.ACTION_TO_INDEX[(1, 0)],
                    ]
                )
        else:
            # From H0 player 0 start (3, 1), move to a left corridor tile that
            # leaves both the tomato approach and the lower onion route open.
            script = [
                Action.ACTION_TO_INDEX[(-1, 0)],
                Action.ACTION_TO_INDEX[(-1, 0)],
                Action.ACTION_TO_INDEX[(0, 1)],
                Action.ACTION_TO_INDEX[(0, 1)],
            ]
        if step < len(script):
            return int(script[step])
    return int(Action.ACTION_TO_INDEX[Action.STAY])


def dynamic_partner_action(
    state,
    motion_planner: MotionPlanner,
    teacher_player_index: int,
    teacher_action: int,
) -> int:
    mdp = motion_planner.mdp
    partner_index = 1 - teacher_player_index
    teacher = state.players[teacher_player_index]
    partner = state.players[partner_index]
    action = Action.INDEX_TO_ACTION[int(teacher_action)]
    if action not in Action.MOTION_ACTIONS or action == Action.STAY:
        return int(Action.ACTION_TO_INDEX[Action.STAY])

    teacher_next = Action.move_in_direction(teacher.position, action)
    if partner.position != teacher_next:
        return int(Action.ACTION_TO_INDEX[Action.STAY])

    valid_positions = set(mdp.get_valid_player_positions())
    blocked = {teacher.position, teacher_next}
    best_action = None
    best_score = None
    for candidate_action in Action.MOTION_ACTIONS:
        candidate_pos = partner.position
        if candidate_action != Action.STAY:
            candidate_pos = Action.move_in_direction(partner.position, candidate_action)
        if candidate_pos not in valid_positions or candidate_pos in blocked:
            continue
        # Prefer moves that clear narrow service/tomato corridors and keep some
        # distance from the teacher's immediate path.
        score = (
            abs(candidate_pos[0] - teacher.position[0])
            + abs(candidate_pos[1] - teacher.position[1])
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = candidate_action
    if best_action is None:
        return int(Action.ACTION_TO_INDEX[Action.STAY])
    return int(Action.ACTION_TO_INDEX[best_action])


def partner_action_for_state(
    partner_mode: str,
    state,
    motion_planner: MotionPlanner,
    teacher_player_index: int,
    step: int,
    teacher_action: int,
) -> int:
    if partner_mode == "dynamic_avoid":
        return dynamic_partner_action(
            state,
            motion_planner,
            teacher_player_index,
            teacher_action,
        )
    partner_index = 1 - teacher_player_index
    return partner_action(partner_mode, partner_index, step)


def make_motion_planner(layout: str, seed: int, horizon: int) -> MotionPlanner:
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


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.start_jitter_steps < 0:
        raise ValueError("--start-jitter-steps must be non-negative")

    reference_dir = resolve_agent_dir(args.reference_agent)
    trainer = load_trainer(str(reference_dir))
    output_dir = args.output_dir or (
        OUTPUT_ROOT
        / f"{args.layout}_rule_teacher_p{args.teacher_player_index}_seed{args.seed}_n{args.episodes}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    observations: list[np.ndarray] = []
    labels: list[int] = []
    subgoals: list[int] = []
    jsonl_records: list[dict] = []
    episode_summaries: list[dict] = []
    event_counts_total: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()

    try:
        dummy_env = trainer.env_creator(trainer.config["env_config"])
        featurize_fn = dummy_env._get_featurize_fn(args.policy_id)
        motion_planner = make_motion_planner(args.layout, args.seed, args.horizon)
        for episode_index in range(args.episodes):
            if len(labels) >= args.max_examples:
                break
            episode_seed = args.seed + episode_index
            episode_rng = np.random.default_rng(episode_seed)
            env = make_direct_multi_env(args.layout, seed=episode_seed, horizon=args.horizon)
            env.multi_reset()
            done = False
            reward_sum = 0.0
            info = {}
            episode_observations: list[np.ndarray] = []
            episode_labels: list[int] = []
            episode_subgoals: list[int] = []
            episode_records: list[dict] = []

            try:
                for jitter_step in range(args.start_jitter_steps):
                    teacher_move = int(
                        Action.ACTION_TO_INDEX[
                            Action.MOTION_ACTIONS[
                                episode_rng.integers(0, len(Action.MOTION_ACTIONS))
                            ]
                        ]
                    )
                    partner_move = partner_action_for_state(
                        args.partner_mode,
                        env.base_env.state,
                        motion_planner,
                        args.teacher_player_index,
                        jitter_step,
                        teacher_move,
                    )
                    joint_action = [partner_move, partner_move]
                    joint_action[args.teacher_player_index] = teacher_move
                    _, reward, done, info = env.multi_step(
                        int(joint_action[0]),
                        int(joint_action[1]),
                    )
                    reward_sum += float(reward[0])
                    if done:
                        break
                while not done:
                    state = env.base_env.state
                    step = len(episode_records)
                    subgoal_name, teacher_action = rule_teacher_decision(
                        state,
                        motion_planner,
                        args.teacher_player_index,
                    )
                    partner_move = partner_action_for_state(
                        args.partner_mode,
                        state,
                        motion_planner,
                        args.teacher_player_index,
                        step,
                        teacher_action,
                    )
                    joint_action = [partner_move, partner_move]
                    joint_action[args.teacher_player_index] = teacher_action

                    episode_observations.append(
                        featurize_for_policy(
                            featurize_fn,
                            state,
                            args.teacher_player_index,
                        )
                    )
                    episode_labels.append(int(teacher_action))
                    episode_subgoals.append(int(SUBGOAL_TO_INDEX[subgoal_name]))
                    episode_records.append(
                        {
                            "episode_index": episode_index,
                            "seed": episode_seed,
                            "step": step,
                            "teacher_player_index": args.teacher_player_index,
                            "subgoal": subgoal_name,
                            "subgoal_id": int(SUBGOAL_TO_INDEX[subgoal_name]),
                            "teacher_action": action_name(teacher_action),
                            "partner_action": action_name(partner_move),
                            "state": state_summary(state),
                        }
                    )

                    _, reward, done, info = env.multi_step(
                        int(joint_action[0]),
                        int(joint_action[1]),
                    )
                    reward_sum += float(reward[0])
                    if args.stop_after_success and reward_sum >= args.success_threshold:
                        done = True
            finally:
                env.close()

            episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
            episode_stats = episode_info.get("ep_game_stats", {})
            event_counts = Counter()
            for event_name, value in episode_stats.items():
                event_counts[event_name] += count_event_value(value)
            event_counts_total.update(event_counts)

            success = reward_sum >= args.success_threshold
            if success or args.keep_failed:
                remaining = args.max_examples - len(labels)
                observations.extend(episode_observations[:remaining])
                labels.extend(episode_labels[:remaining])
                subgoals.extend(episode_subgoals[:remaining])
                jsonl_records.extend(episode_records[:remaining])
                label_counts.update(episode_labels[:remaining])

            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "reward": reward_sum,
                    "success": success,
                    "steps": len(episode_labels),
                    "kept": bool(success or args.keep_failed),
                    "event_counts": dict(event_counts),
                }
            )
            print(
                f"Episode {episode_index + 1:02d}/{args.episodes}: "
                f"seed={episode_seed}, reward={reward_sum:.1f}, "
                f"success={success}, kept_examples={len(labels)}"
            )

        if not labels:
            raise RuntimeError("Rule teacher produced no kept examples.")

        observation_array = np.stack(observations).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        subgoal_array = np.asarray(subgoals, dtype=np.int32)
        np.savez_compressed(
            output_dir / "dataset.npz",
            observations=observation_array,
            labels=label_array,
            subgoals=subgoal_array,
        )
        with (output_dir / "dataset.jsonl").open("w", encoding="utf-8") as handle:
            for record in jsonl_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metadata = {
            "reference_agent": str(reference_dir),
            "layout": args.layout,
            "policy_id": args.policy_id,
            "teacher_player_index": args.teacher_player_index,
            "partner_mode": args.partner_mode,
            "episodes": args.episodes,
            "seed": args.seed,
            "horizon": args.horizon,
            "success_threshold": args.success_threshold,
            "start_jitter_steps": args.start_jitter_steps,
            "stop_after_success": args.stop_after_success,
            "max_examples": args.max_examples,
            "examples": int(label_array.shape[0]),
            "observation_shape": list(observation_array.shape),
            "subgoals": list(SUBGOALS),
            "subgoal_counts": {
                name: int((subgoal_array == index).sum())
                for index, name in enumerate(SUBGOALS)
            },
            "label_counts": {
                str(Action.INDEX_TO_ACTION[index]): int(label_counts[index])
                for index in range(Action.NUM_ACTIONS)
            },
            "event_counts": dict(event_counts_total),
            "episodes_summary": episode_summaries,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"Saved dataset: {output_dir / 'dataset.npz'}")
    finally:
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
