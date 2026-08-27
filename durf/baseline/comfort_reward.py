"""Bridge the learned comfort reward into the live Overcooked env.

The Route 2 stage exports a frozen comfort weight vector ``w`` over the shared
53-dim feature schema (``outputs/route2/learned_comfort_weights.json``). This
module turns ``w`` into a per-step scalar shaping term for PPO:

    r_comfort(state) = coeff * ( w . phi_comfort(context, ai_subgoal) )

where the AI's committed subgoal is what H0 would pick, ``phi`` is the same
featurizer used in learning, and only the coordination/comfort feature
partition is scored (task progress is already rewarded by the env). This keeps
training and runtime on ONE feature schema.

Runtime dependency note: only numpy-level modules from the offline Baseline B
package are imported (featurizer / reranker / schema / score_action). No torch
or nltk is needed here, so it is safe to import inside the RLlib/tf training
process.
"""

from __future__ import annotations

from collections import Counter, deque
import json
import sys
from pathlib import Path

from overcooked_ai_py.mdp.actions import Action

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTED_ROOT = (
    REPO_ROOT
    / "baselines"
    / "baseline_b_linguistic_feedback"
    / "adapted_overcooked"
)
if str(ADAPTED_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTED_ROOT))

from src.belief_model import score_action  # noqa: E402
from src.human_intent import infer_human_intent  # noqa: E402
from src.subgoal_featurizer import SubgoalContext, featurize_subgoal  # noqa: E402
from src.subgoal_reranker import COMFORT_FEATURES  # noqa: E402

from durf.baseline.h0_planner import (  # noqa: E402
    SUBGOALS,
    ingredient_pickup_locations,
    missing_ingredients,
    pot_ingredients,
    pots_needing_ingredient,
    plan_h0_subgoal,
    subgoal_target_positions,
    target_recipe,
    yield_path_action,
)


DEFAULT_WEIGHTS_PATH = ADAPTED_ROOT / "outputs" / "route2" / "learned_comfort_weights.json"


def _held_name(player) -> str | None:
    held = getattr(player, "held_object", None)
    return getattr(held, "name", None) if held is not None else None


def _facing_target_resource(state, mdp, player) -> str | None:
    """Conservative straight-line intent for an empty-handed human.

    The first interactable feature visible along the current orientation wins.
    A turn, branch, or non-resource counter remains ambiguous and returns None.
    """

    if _held_name(player) is not None:
        return None
    resource_locations = {
        "tomato": set(mdp.get_tomato_dispenser_locations()),
        "onion": set(mdp.get_onion_dispenser_locations()),
        "dish": set(mdp.get_dish_dispenser_locations()),
        "serving": set(mdp.get_serving_locations()),
    }
    ready_pots = set(mdp.get_ready_pots(mdp.get_pot_states(state)))
    valid = set(mdp.get_valid_player_positions())
    position = player.position
    # A layout dimension is a safe upper bound for a straight corridor ray.
    for _ in range(len(valid) + 1):
        position = Action.move_in_direction(position, player.orientation)
        for resource, locations in resource_locations.items():
            if position in locations:
                return resource
        if position in ready_pots:
            return "soup"
        if position not in valid:
            return None
    return None


def _all_pot_snapshots(state, mdp) -> list[dict]:
    """Return every physical pot without merging independent recipes."""

    pot_states = mdp.get_pot_states(state)
    ready = set(mdp.get_ready_pots(pot_states))
    cooking = set(mdp.get_cooking_pots(pot_states))

    snapshots: list[dict] = []
    for pot_pos in mdp.get_pot_locations():
        ingredients = pot_ingredients(state, pot_pos)
        if pot_pos in ready:
            status = "ready"
        elif pot_pos in cooking:
            status = "cooking"
        elif ingredients:
            status = "partial"
        else:
            status = "empty"
        snapshots.append(
            {
                "position": pot_pos,
                "ingredients": list(ingredients),
                "status": status,
            }
        )
    return snapshots


def _staged_inventory(state, mdp) -> dict[str, int]:
    """Count objects on ordinary counters, excluding fixed map features."""

    get_counters = getattr(mdp, "get_counter_locations", None)
    if get_counters is None:
        return {}
    counters = {tuple(position) for position in get_counters()}
    for getter_name in (
        "get_pot_locations",
        "get_serving_locations",
        "get_dish_dispenser_locations",
        "get_tomato_dispenser_locations",
        "get_onion_dispenser_locations",
    ):
        getter = getattr(mdp, getter_name, None)
        if getter is not None:
            counters.difference_update(tuple(position) for position in getter())

    counts: Counter[str] = Counter()
    for position, obj in getattr(state, "objects", {}).items():
        if tuple(position) not in counters:
            continue
        name = getattr(obj, "name", None)
        if name:
            counts[str(name)] += 1
    return dict(counts)


def _focus_pot_snapshot(
    snapshots: list[dict],
    recipe: list[str],
    preferred_ingredient: str | None = None,
) -> tuple[list[str], str]:
    """Choose one legacy focus pot while plural consumers retain all pots."""

    # If the AI is already carrying an ingredient, represent the best pot that
    # can actually accept it. Otherwise a ready pot could hide a partial second
    # pot and make the feasible set incorrectly collapse to WAIT.
    if preferred_ingredient in ("tomato", "onion"):
        accepting = [
            snapshot
            for snapshot in snapshots
            if snapshot["status"] in {"empty", "partial"}
            and preferred_ingredient
            in missing_ingredients(recipe, snapshot["ingredients"])
        ]
        if accepting:
            selected = max(
                accepting, key=lambda snapshot: len(snapshot["ingredients"])
            )
            return list(selected["ingredients"]), str(selected["status"])

    best_ingredients: list[str] = []
    best_status = "empty"
    best_rank = (-1, -1)
    status_rank = {"empty": 0, "partial": 1, "cooking": 2, "ready": 3}
    for snapshot in snapshots:
        ingredients = snapshot["ingredients"]
        status = snapshot["status"]
        rank = (status_rank[status], len(ingredients))
        if rank > best_rank:
            best_rank = rank
            best_status = status
            best_ingredients = ingredients
    return best_ingredients, best_status


def _pot_snapshot(
    state, mdp, preferred_ingredient: str | None = None
) -> tuple[list[str], str]:
    """Return the backwards-compatible focus-pot snapshot."""

    return _focus_pot_snapshot(
        _all_pot_snapshots(state, mdp),
        target_recipe(mdp),
        preferred_ingredient=preferred_ingredient,
    )


def _interaction_access_positions(mdp, features) -> set[tuple[int, int]]:
    valid = set(mdp.get_valid_player_positions())
    access: set[tuple[int, int]] = set()
    for feature in features:
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            position = Action.move_in_direction(feature, direction)
            if position in valid:
                access.add(position)
    return access


def _shortest_distance(
    valid: set[tuple[int, int]],
    start: tuple[int, int],
    targets: set[tuple[int, int]],
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> int | None:
    blocked = set(blocked or ())
    blocked.discard(start)
    allowed = set(valid) - blocked
    if start in targets:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            nxt = Action.move_in_direction(position, direction)
            if nxt not in allowed or nxt in seen:
                continue
            if nxt in targets:
                return distance + 1
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def _first_step_toward(
    mdp,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Grid first step toward an interaction access cell, or ``start``."""

    valid = set(mdp.get_valid_player_positions())
    blocked = set(blocked or ())
    blocked.discard(start)
    allowed = valid - blocked
    if not targets or start in targets:
        return start
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found: tuple[int, int] | None = None
    while queue and found is None:
        position = queue.popleft()
        for direction in Action.MOTION_ACTIONS:
            if direction == Action.STAY:
                continue
            nxt = Action.move_in_direction(position, direction)
            if nxt not in allowed or nxt in parent:
                continue
            parent[nxt] = position
            if nxt in targets:
                found = nxt
                break
            queue.append(nxt)
    if found is None:
        return start
    while parent[found] is not None and parent[found] != start:
        found = parent[found]
    return found


def _human_task_targets(state, mdp, human_index: int, resource: str | None):
    human = state.players[human_index]
    holding = _held_name(human)
    pot_states = mdp.get_pot_states(state)
    if holding in ("tomato", "onion"):
        return pots_needing_ingredient(state, mdp, holding)
    if holding == "dish":
        return mdp.get_ready_pots(pot_states)
    if holding == "soup":
        return mdp.get_serving_locations()
    if resource in ("tomato", "onion"):
        return ingredient_pickup_locations(state, mdp, resource)
    if resource == "dish":
        return mdp.get_dish_dispenser_locations()
    if resource == "soup":
        return mdp.get_ready_pots(pot_states)
    if resource == "serving":
        return mdp.get_serving_locations()
    return []


def _candidate_geometry(
    state,
    mdp,
    *,
    ai_index: int,
    human_index: int,
    human_target_resource: str | None,
) -> tuple[dict[str, str], dict[str, bool]]:
    """Immediate occupancy effect of each subgoal on the human's live route."""

    yield_action = yield_path_action(state, mdp, player_index=ai_index)
    if human_target_resource is None:
        # Orientation is enough to establish one conservative short-term path
        # fact even when the human's eventual task target is unknown.
        if yield_action is None:
            return {}, {}
        return (
            {"YIELD_PATH": "clears", "WAIT": "blocks"},
            {"YIELD_PATH": False, "WAIT": False},
        )
    ai = state.players[ai_index]
    human = state.players[human_index]
    human_features = set(
        _human_task_targets(state, mdp, human_index, human_target_resource)
    )
    human_access = _interaction_access_positions(mdp, human_features)
    if not human_access:
        if yield_action is None:
            return {}, {}
        return (
            {"YIELD_PATH": "clears", "WAIT": "blocks"},
            {"YIELD_PATH": False, "WAIT": False},
        )
    valid = set(mdp.get_valid_player_positions())
    baseline = _shortest_distance(valid, human.position, human_access)

    def blocks(position: tuple[int, int]) -> bool:
        if baseline is None or position == human.position:
            return False
        obstructed = _shortest_distance(
            valid, human.position, human_access, blocked={position}
        )
        return obstructed is None or obstructed > baseline

    current_blocks = blocks(ai.position)
    effects: dict[str, str] = {}
    overlaps: dict[str, bool] = {}
    for subgoal in SUBGOALS:
        candidate_features = set(
            subgoal_target_positions(state, mdp, subgoal, player_index=ai_index)
        )
        candidate_access = _interaction_access_positions(mdp, candidate_features)
        overlaps[subgoal] = bool(
            candidate_features & human_features or candidate_access & human_access
        )
        next_position = (
            ai.position
            if subgoal == "WAIT"
            else _first_step_toward(
                mdp,
                ai.position,
                candidate_access,
                blocked={human.position},
            )
        )
        next_blocks = blocks(next_position)
        if current_blocks and not next_blocks:
            effects[subgoal] = "clears"
        elif not current_blocks and next_blocks:
            effects[subgoal] = "enters"
        elif next_blocks:
            effects[subgoal] = "blocks"
        else:
            effects[subgoal] = "neutral"
    # A teammate directly facing the AI is a stronger immediate occupancy
    # observation than a longer-horizon resource route approximation.
    if yield_action is not None:
        effects["YIELD_PATH"] = "clears"
        effects["WAIT"] = "blocks"
        overlaps.setdefault("YIELD_PATH", False)
        overlaps.setdefault("WAIT", False)
    return effects, overlaps


def context_from_state(state, mdp, ai_index: int = 0) -> SubgoalContext:
    """Build a :class:`SubgoalContext` from a live Overcooked state."""

    human_index = 1 - ai_index if len(state.players) == 2 else (ai_index + 1) % len(state.players)
    ai_player = state.players[ai_index]
    human_player = state.players[human_index]

    pot_snapshots = _all_pot_snapshots(state, mdp)
    ingredients, status = _focus_pot_snapshot(
        pot_snapshots,
        target_recipe(mdp),
        preferred_ingredient=_held_name(ai_player),
    )
    human_holding = _held_name(human_player)
    facing_resource = _facing_target_resource(state, mdp, human_player)
    human_intent = infer_human_intent(
        human_holding=human_holding,
        target_resource=facing_resource,
    )
    human_target_resource = {
        "tomato": "tomato",
        "onion": "onion",
        "dish": "dish",
        "soup": "serving",
    }.get(human_holding) or facing_resource
    path_effects, target_overlaps = _candidate_geometry(
        state,
        mdp,
        ai_index=ai_index,
        human_index=human_index,
        human_target_resource=human_target_resource,
    )

    return SubgoalContext(
        recipe=list(target_recipe(mdp)),
        pot_ingredients=list(ingredients),
        pot_status=status,
        agent_holding=_held_name(ai_player),
        human_holding=human_holding,
        human_intent=human_intent,
        human_committed_units=1 if human_target_resource is not None else 0,
        candidate_path_effects=path_effects,
        candidate_target_overlaps_human=target_overlaps,
        pot_snapshots=pot_snapshots,
        staged_inventory=_staged_inventory(state, mdp),
    )


class ComfortReward:
    """Frozen learned comfort reward, evaluated per step for PPO shaping."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        coeff: float = 1.0,
    ) -> None:
        if weights is None:
            weights = self.load_weights(weights_path)
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.comfort_weights = {
            feature: value
            for feature, value in self.weights.items()
            if feature in COMFORT_FEATURES
        }
        self.coeff = float(coeff)

    @staticmethod
    def load_weights(path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Comfort weights must be a JSON object: {path}")
        return {str(k): float(v) for k, v in raw.items()}

    def comfort_score(self, state, mdp, ai_index: int = 0) -> float:
        """Raw ``w . phi_comfort`` for the AI's committed subgoal (no coeff)."""

        context = context_from_state(state, mdp, ai_index=ai_index)
        subgoal = plan_h0_subgoal(state, mdp, player_index=ai_index)
        phi = featurize_subgoal(context, subgoal)
        comfort_phi = {f: v for f, v in phi.items() if f in COMFORT_FEATURES}
        return float(score_action(self.comfort_weights, comfort_phi))

    def shaping(self, state, mdp, ai_index: int = 0) -> float:
        """Comfort shaping term ``coeff * comfort_score`` for one step."""

        return self.coeff * self.comfort_score(state, mdp, ai_index=ai_index)
