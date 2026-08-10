"""Candidate event detectors for saved trajectory logs.

These detectors are intentionally conservative. They turn factual trajectory
records into candidate events for later semantic attribution, not final labels.
"""

from __future__ import annotations

from .condition_features import extract_condition_features
from .schemas import candidate_event


ACTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


DEFAULT_LOOKBACK_STEPS = 25


VALENCE_POSITIVE_PROGRESS = "positive_progress"
VALENCE_NEGATIVE_PROBLEM = "negative_problem"
VALENCE_MISSED_OPPORTUNITY = "missed_opportunity"
VALENCE_NEUTRAL_CONTEXT = "neutral_context"

ACTOR_AI = "ai"
ACTOR_HUMAN = "human"
ACTOR_TEAM = "team"
ACTOR_UNKNOWN = "unknown"

# Sliding-window parameters for hold-unneeded detection: real trajectories
# interrupt holding with short pick/drop actions, so a contiguous threshold
# never fires on the typical "grab a few frames, put down, grab again" pattern.
HELD_UNNEEDED_WINDOW_STEPS = 15
HELD_UNNEEDED_MIN_FRAMES = 5


def detect_coordination_decision_events(window: list[dict]) -> list[dict]:
    """Convert explicit runtime coordination decisions into neutral events."""
    grouped: dict[str, list[dict]] = {}
    for step in window:
        decision = step.get("coordination_decision") or {}
        decision_id = str(decision.get("decision_id") or "")
        if decision_id:
            grouped.setdefault(decision_id, []).append(step)

    events = []
    for decision_id, steps in grouped.items():
        ordered = sorted(
            steps,
            key=lambda step: int(step.get("total_step") or -1),
        )
        first = ordered[0]
        last = ordered[-1]
        decision = first.get("coordination_decision") or {}
        selected = decision.get("selected")
        if selected == "YIELD":
            event_type = "AI_successfully_yielded"
        elif selected == "CONTINUE_CURRENT_SUBGOAL":
            event_type = "AI_maintained_current_subgoal_during_conflict"
        else:
            continue
        candidates = [
            str(option)
            for option in decision.get("candidate_set") or []
            if option and option != selected
        ]
        events.append(
            candidate_event(
                event_type=event_type,
                start_timestep=int(first.get("total_step") or 0),
                end_timestep=int(last.get("total_step") or 0),
                evidence={
                    "decision_id": decision_id,
                    "decision_level": "coordination",
                    "conflict_type": decision.get("conflict_type"),
                    "task_subgoal": decision.get("task_subgoal"),
                    "selected_option": selected,
                    "candidate_set": decision.get("candidate_set") or [],
                    "runtime_candidates": decision.get("candidates") or [],
                    "termination_visible_until": int(
                        last.get("total_step") or 0
                    ),
                },
                severity=0.2,
                confidence=0.95,
                actor=ACTOR_AI,
                event_valence=VALENCE_NEUTRAL_CONTEXT,
                related_subgoal=selected,
                alternative_subgoals=candidates,
                condition_features=(
                    decision.get("condition_at_decision") or {}
                ),
            )
        )
    return events


def recent_window(
    trajectory: list[dict],
    *,
    feedback_total_step: int | None,
    lookback_steps: int = DEFAULT_LOOKBACK_STEPS,
) -> list[dict]:
    if feedback_total_step is None:
        return trajectory[-lookback_steps:]
    start = max(0, feedback_total_step - lookback_steps)
    return [
        step
        for step in trajectory
        if start <= int(step.get("total_step") or -1) <= feedback_total_step
    ]


def pos_tuple(value) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def add_pos(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return pos[0] + delta[0], pos[1] + delta[1]


def manhattan(a: tuple[int, int] | None, b: tuple[int, int] | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
    return sum(
        1
        for delta in ACTION_DELTAS.values()
        if is_walkable(terrain, add_pos(pos, delta))
    )


def held_name(held_object) -> str | None:
    if not isinstance(held_object, dict):
        return None
    return held_object.get("name")


def object_ingredients(obj) -> list[str]:
    if not isinstance(obj, dict):
        return []
    return [str(item) for item in obj.get("ingredients") or []]


def soup_ingredients(facts: dict) -> list[str]:
    ingredients: list[str] = []
    for obj in facts.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        nested = obj.get("object") if isinstance(obj.get("object"), dict) else obj
        if nested.get("name") == "soup":
            ingredients.extend(object_ingredients(nested))
    return ingredients


def recipe_needs(step: dict, ingredient: str) -> bool | None:
    conditions = extract_condition_features(step)
    key = f"recipe_needs_{ingredient}"
    value = conditions.get(key)
    return value if isinstance(value, bool) else None


def object_positions(facts: dict, name: str) -> list[tuple[int, int]]:
    positions = []
    for obj in facts.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        nested = obj.get("object") if isinstance(obj.get("object"), dict) else obj
        if nested.get("name") != name:
            continue
        pos = pos_tuple(obj.get("position") or nested.get("position"))
        if pos is not None:
            positions.append(pos)
    return positions


def pot_all_positions(facts: dict) -> list[tuple[int, int]]:
    pot_states = facts.get("pot_states") or {}
    return pot_positions(
        pot_states,
        ("empty", "1_items", "2_items", "3_items", "partially_full", "cooking", "ready"),
    )


def pot_positions(pot_states: dict, keys: tuple[str, ...]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if not isinstance(pot_states, dict):
        return positions
    for key in keys:
        for raw_pos in pot_states.get(key, []) or []:
            pos = pos_tuple(raw_pos)
            if pos is not None:
                positions.append(pos)
    return positions


def step_state_before(step: dict) -> dict:
    return step.get("extra", {}).get("state_before") or {}


def step_state_after(step: dict) -> dict:
    return step.get("state_facts") or {}


def step_ai_subgoal(step: dict) -> str | None:
    return step.get("ai_subgoal") or step.get("extra", {}).get("ai_subgoal")


def event_context(step: dict) -> dict:
    return {
        "related_subgoal": step_ai_subgoal(step),
        "condition_features": extract_condition_features(step),
    }


def missing_facts(step: dict, names: list[str], *, before: bool = False) -> list[str]:
    facts = step_state_before(step) if before else step_state_after(step)
    return [name for name in names if facts.get(name) is None]


def detect_ai_blocked_human_path(window: list[dict]) -> list[dict]:
    """Detect cases where human tried to move but remained in place near AI."""

    events: list[dict] = []
    for step in window:
        required = ["ai_pos", "human_pos", "layout_features"]
        missing = missing_facts(step, required, before=True) + missing_facts(
            step,
            ["ai_pos", "human_pos"],
        )
        if missing:
            continue

        human_action_name = step.get("human_action_name")
        delta = ACTION_DELTAS.get(str(human_action_name))
        if delta is None:
            continue

        before = step_state_before(step)
        after = step_state_after(step)
        human_before = pos_tuple(before.get("human_pos"))
        human_after = pos_tuple(after.get("human_pos"))
        ai_before = pos_tuple(before.get("ai_pos"))
        ai_after = pos_tuple(after.get("ai_pos"))
        if human_before is None or human_after is None:
            continue

        attempted_target = add_pos(human_before, delta)
        human_failed_to_move = human_before == human_after
        target_was_ai = attempted_target in {ai_before, ai_after}
        terrain = before.get("layout_features", {}).get("terrain") or []
        corridor_width = walkable_neighbor_count(terrain, human_before)
        narrow_corridor = corridor_width is not None and corridor_width <= 2

        if human_failed_to_move and target_was_ai:
            events.append(
                candidate_event(
                    event_type="AI_blocked_human_path",
                    start_timestep=int(step["total_step"]),
                    end_timestep=int(step["total_step"]),
                    evidence={
                        "human_action": human_action_name,
                        "human_before": list(human_before),
                        "human_after": list(human_after),
                        "human_attempted_target": list(attempted_target),
                        "ai_before": list(ai_before) if ai_before else None,
                        "ai_after": list(ai_after) if ai_after else None,
                        "target_was_ai": target_was_ai,
                        "narrow_corridor": narrow_corridor,
                        "walkable_neighbor_count": corridor_width,
                    },
                    severity=0.8,
                    confidence=0.8,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_NEGATIVE_PROBLEM,
                    **event_context(step),
                )
            )

    if events:
        return events

    if window:
        missing = sorted(
            set(
                missing_facts(window[-1], ["ai_pos", "human_pos", "layout_features"])
            )
        )
        if missing:
            return [
                candidate_event(
                    event_type="AI_blocked_human_path",
                    start_timestep=int(window[0]["total_step"]),
                    end_timestep=int(window[-1]["total_step"]),
                    evidence={
                        "reason": "Missing facts needed for path-blocking detection.",
                        "recent_ai_actions": [s.get("ai_action_name") for s in window],
                        "recent_human_actions": [s.get("human_action_name") for s in window],
                    },
                    severity=None,
                    confidence=0.05,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_NEUTRAL_CONTEXT,
                    **event_context(window[-1]),
                    missing_required_facts=missing,
                )
            ]

    return []


def detect_ai_failed_to_yield_or_clear_path(window: list[dict]) -> list[dict]:
    """Detect detour-inducing AI standing: the human's destination is free,
    but the AI occupies the tile just beyond it along the human's motion
    direction, so the human must route around the AI.

    ``AI_blocked_human_path`` only fires when the human's destination equals
    the AI's tile.  Real sessions rarely produce that instant geometry, so the
    "AI standing so the human must detour" case (the main coordination data
    bottleneck) is detected here instead: the human successfully moves to the
    free destination, then has to turn because the next tile belongs to the AI.
    """

    events: list[dict] = []
    for step in window:
        required = ["ai_pos", "human_pos", "layout_features"]
        missing = missing_facts(step, required, before=True) + missing_facts(
            step,
            ["ai_pos", "human_pos"],
        )
        if missing:
            continue

        human_action_name = step.get("human_action_name")
        delta = ACTION_DELTAS.get(str(human_action_name))
        if delta is None:
            continue

        before = step_state_before(step)
        after = step_state_after(step)
        human_before = pos_tuple(before.get("human_pos"))
        human_after = pos_tuple(after.get("human_pos"))
        ai_before = pos_tuple(before.get("ai_pos"))
        ai_after = pos_tuple(after.get("ai_pos"))
        if human_before is None or human_after is None:
            continue

        attempted_target = add_pos(human_before, delta)
        human_moved = human_before != human_after
        target_was_ai = attempted_target in {ai_before, ai_after}
        step_after_target = add_pos(attempted_target, delta)
        ai_blocks_next_tile = step_after_target in {ai_before, ai_after}
        if not human_moved or target_was_ai or not ai_blocks_next_tile:
            continue
        terrain = before.get("layout_features", {}).get("terrain") or []
        corridor_width = walkable_neighbor_count(terrain, human_before)
        narrow_corridor = corridor_width is not None and corridor_width <= 2
        events.append(
            candidate_event(
                event_type="AI_failed_to_yield_or_clear_path",
                start_timestep=int(step["total_step"]),
                end_timestep=int(step["total_step"]),
                evidence={
                    "human_action": human_action_name,
                    "human_before": list(human_before),
                    "human_after": list(human_after),
                    "human_attempted_target": list(attempted_target),
                    "ai_before": list(ai_before) if ai_before else None,
                    "ai_after": list(ai_after) if ai_after else None,
                    "target_was_ai": target_was_ai,
                    "ai_blocks_next_tile": ai_blocks_next_tile,
                    "narrow_corridor": narrow_corridor,
                    "walkable_neighbor_count": corridor_width,
                },
                severity=0.7,
                confidence=0.7,
                actor=ACTOR_AI,
                event_valence=VALENCE_NEGATIVE_PROBLEM,
                **event_context(step),
            )
        )

    if events:
        return events

    if window:
        missing = sorted(
            set(
                missing_facts(window[-1], ["ai_pos", "human_pos", "layout_features"])
            )
        )
        if missing:
            return [
                candidate_event(
                    event_type="AI_failed_to_yield_or_clear_path",
                    start_timestep=int(window[0]["total_step"]),
                    end_timestep=int(window[-1]["total_step"]),
                    evidence={
                        "reason": "Missing facts needed for detour-path detection.",
                        "recent_ai_actions": [s.get("ai_action_name") for s in window],
                        "recent_human_actions": [s.get("human_action_name") for s in window],
                    },
                    severity=None,
                    confidence=0.05,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_NEUTRAL_CONTEXT,
                    **event_context(window[-1]),
                    missing_required_facts=missing,
                )
            ]

    return []


def detect_ai_ignored_ready_or_nearly_ready_pot(window: list[dict]) -> list[dict]:
    """Detect ready/cooking-pot windows where the AI does not interact with the pot.

    Cooking pots are included: once a pot starts cooking, plate work becomes
    the productive use of the same waiting time, so ignoring the pot window
    is an opportunity regardless of whether the pot is ready yet.
    """

    events: list[dict] = []
    streak: list[dict] = []

    for step in window:
        after = step_state_after(step)
        missing = missing_facts(step, ["ai_pos", "pot_states"])
        if missing:
            streak = []
            continue

        ready_positions = pot_positions(after.get("pot_states"), ("cooking", "ready"))
        if not ready_positions:
            streak = []
            continue

        ai_pos = pos_tuple(after.get("ai_pos"))
        ai_holding = held_name(after.get("ai_held_object"))
        distances = [manhattan(ai_pos, pot_pos) for pot_pos in ready_positions]
        nearest_ready_pot_dist = min(
            [distance for distance in distances if distance is not None],
            default=None,
        )
        ai_action_name = str(step.get("ai_action_name"))
        ai_can_pickup_soup = ai_holding in {None, "dish"}
        ai_handled_pot = ai_action_name == "interact" and nearest_ready_pot_dist in {0, 1}

        if ai_can_pickup_soup and not ai_handled_pot:
            streak.append(step)
        else:
            if len(streak) >= 3:
                events.append(ready_pot_event(streak))
            streak = []

    if len(streak) >= 3:
        events.append(ready_pot_event(streak))

    if events:
        return events

    if window:
        missing = sorted(set(missing_facts(window[-1], ["pot_states"])))
        if missing:
            return [
                candidate_event(
                    event_type="AI_ignored_ready_or_nearly_ready_pot",
                    start_timestep=int(window[0]["total_step"]),
                    end_timestep=int(window[-1]["total_step"]),
                    evidence={
                        "reason": "Missing pot states needed for ready-pot detection.",
                        "recent_ai_actions": [s.get("ai_action_name") for s in window],
                    },
                    severity=None,
                    confidence=0.05,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_NEUTRAL_CONTEXT,
                    **event_context(window[-1]),
                    missing_required_facts=missing,
                )
            ]

    return []


def detect_ai_failed_to_prepare_ingredient_while_waiting(window: list[dict]) -> list[dict]:
    """Detect cooking-wait periods where AI has idle capacity but does not prep.

    This catches feedback such as "prepare food while I wait for the soup to
    cook". It is deliberately a candidate event: it says the opportunity
    existed, not that this is always the optimal strategy.
    """

    events: list[dict] = []
    streak: list[dict] = []
    ingredient_pickups = {"onion", "tomato"}

    for step in window:
        after = step_state_after(step)
        missing = missing_facts(step, ["ai_pos", "human_pos", "pot_states"])
        if missing:
            if len(streak) >= 4:
                events.append(prep_wait_event(streak))
            streak = []
            continue

        pot_states = after.get("pot_states") or {}
        pot_is_cooking = bool(pot_positions(pot_states, ("cooking",)))
        if not pot_is_cooking:
            if len(streak) >= 4:
                events.append(prep_wait_event(streak))
            streak = []
            continue

        ai_holding = held_name(after.get("ai_held_object"))
        human_holding = held_name(after.get("human_held_object"))
        ai_action_name = str(step.get("ai_action_name"))
        ai_picked_up_ingredient = (
            ai_action_name == "interact" and ai_holding in ingredient_pickups
        )

        ai_has_free_hand = ai_holding is None
        human_waiting_with_dish = human_holding == "dish"
        ai_not_prepping = ai_has_free_hand and not ai_picked_up_ingredient

        if ai_not_prepping and human_waiting_with_dish:
            streak.append(step)
        else:
            if len(streak) >= 4:
                events.append(prep_wait_event(streak))
            streak = []

    if len(streak) >= 4:
        events.append(prep_wait_event(streak))

    return events


def detect_successfully_delivered_soup(window: list[dict]) -> list[dict]:
    """Detect positive delivery moments and attribute the acting player.

    Environment reward alone only tells us the team delivered soup.  We infer
    the actor from who held soup before the step and performed interact.
    """

    events = []
    for step in window:
        try:
            reward = float(step.get("environment_reward") or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        if reward <= 0.0:
            continue
        before = step_state_before(step)
        after = step_state_after(step)
        ai_delivered = (
            held_name(before.get("ai_held_object")) == "soup"
            and step.get("ai_action_name") == "interact"
        )
        human_delivered = (
            held_name(before.get("human_held_object")) == "soup"
            and step.get("human_action_name") == "interact"
        )
        if ai_delivered:
            event_type = "AI_successfully_delivered_soup"
            actor = ACTOR_AI
            related_subgoal = step_ai_subgoal(step) or "SERVE_SOUP"
            confidence = 0.9
        elif human_delivered:
            event_type = "Human_successfully_delivered_soup"
            actor = ACTOR_HUMAN
            related_subgoal = None
            confidence = 0.85
        else:
            event_type = "Team_successfully_delivered_soup"
            actor = ACTOR_TEAM
            related_subgoal = step_ai_subgoal(step)
            confidence = 0.65
        events.append(
            candidate_event(
                event_type=event_type,
                start_timestep=int(step["total_step"]),
                end_timestep=int(step["total_step"]),
                evidence={
                    "reason": (
                        "Environment reward increased. Actor is inferred from "
                        "who held soup before the step and used interact."
                    ),
                    "environment_reward": reward,
                    "ai_action": step.get("ai_action_name"),
                    "human_action": step.get("human_action_name"),
                    "ai_held_before": before.get("ai_held_object"),
                    "human_held_before": before.get("human_held_object"),
                    "ai_pos": after.get("ai_pos"),
                    "human_pos": after.get("human_pos"),
                },
                severity=0.2,
                confidence=confidence,
                actor=actor,
                event_valence=VALENCE_POSITIVE_PROGRESS,
                related_subgoal=related_subgoal,
                condition_features=extract_condition_features(step),
            )
        )
    return events


def detect_ai_successfully_picked_up_soup(window: list[dict]) -> list[dict]:
    """Detect when AI turns a dish into held soup."""

    events = []
    for step in window:
        before = step_state_before(step)
        after = step_state_after(step)
        before_held = held_name(before.get("ai_held_object"))
        after_held = held_name(after.get("ai_held_object"))
        if before_held == "dish" and after_held == "soup":
            events.append(
                candidate_event(
                    event_type="AI_successfully_picked_up_soup",
                    start_timestep=int(step["total_step"]),
                    end_timestep=int(step["total_step"]),
                    evidence={
                        "reason": "AI held a dish before the step and held soup after the step.",
                        "ai_action": step.get("ai_action_name"),
                        "ai_before": before.get("ai_pos"),
                        "ai_after": after.get("ai_pos"),
                    },
                    severity=0.2,
                    confidence=0.85,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_POSITIVE_PROGRESS,
                    related_subgoal=step_ai_subgoal(step) or "PICKUP_SOUP",
                    condition_features=extract_condition_features(step),
                )
            )
    return events


def detect_ai_successfully_put_ingredient_into_pot(window: list[dict]) -> list[dict]:
    """Detect when AI's held ingredient appears in a pot soup.

    The soup ingredient count is read from the after-state only, so a same-step
    human interact could also explain the count increase.  Require the AI to be
    adjacent to a pot and downgrade confidence when the human also interacted.
    """

    events = []
    for step in window:
        before = step_state_before(step)
        after = step_state_after(step)
        before_held = held_name(before.get("ai_held_object"))
        after_held = held_name(after.get("ai_held_object"))
        if before_held not in {"tomato", "onion"} or after_held is not None:
            continue
        before_count = soup_ingredients(before).count(before_held)
        after_count = soup_ingredients(after).count(before_held)
        if after_count <= before_count:
            continue
        ai_after = pos_tuple(after.get("ai_pos"))
        nearest = min(
            (
                manhattan(ai_after, pot_pos)
                for pot_pos in pot_all_positions(after)
                if manhattan(ai_after, pot_pos) is not None
            ),
            default=None,
        )
        if nearest is None or nearest > 1:
            continue
        human_acted_same_step = (
            step.get("human_action_name") == "interact"
            and held_name(before.get("human_held_object"))
            != held_name(after.get("human_held_object"))
        )
        subgoal = (
            "PUT_TOMATO_IN_POT" if before_held == "tomato" else "PUT_ONION_IN_POT"
        )
        confidence = 0.7 if human_acted_same_step else 0.85
        events.append(
            candidate_event(
                event_type="AI_successfully_put_ingredient_into_pot",
                start_timestep=int(step["total_step"]),
                end_timestep=int(step["total_step"]),
                evidence={
                    "reason": "AI's held ingredient count increased in soup after interaction.",
                    "ingredient": before_held,
                    "ai_action": step.get("ai_action_name"),
                    "ai_after": after.get("ai_pos"),
                    "nearest_pot_distance": nearest,
                    "human_acted_same_step": human_acted_same_step,
                    "soup_ingredients_before": soup_ingredients(before),
                    "soup_ingredients_after": soup_ingredients(after),
                },
                severity=0.2,
                confidence=confidence,
                actor=ACTOR_AI,
                event_valence=VALENCE_POSITIVE_PROGRESS,
                related_subgoal=step_ai_subgoal(step) or subgoal,
                condition_features=extract_condition_features(step),
            )
        )
    return events


def detect_ai_pick_drop_loop(window: list[dict]) -> list[dict]:
    """Detect idle pick/drop loops: the same object repeatedly grabbed and
    dropped on the same tile without task progress.

    The conservative criterion is deliberately narrow: at least three pick/drop
    actions on the same object at the same tile.  Moves between tiles are not
    loops (the AI may be relocating an object); different objects in sequence
    are not loops (the AI may be sorting or staging).
    """

    events = []
    interaction_steps = []
    for step in window:
        before = step_state_before(step)
        after = step_state_after(step)
        before_held = held_name(before.get("ai_held_object"))
        after_held = held_name(after.get("ai_held_object"))
        if (
            step.get("ai_action_name") == "interact"
            and before_held != after_held
            and {before_held, after_held} & {"tomato", "onion", "dish"}
        ):
            interaction_steps.append(step)

    previous_event_end = -1
    index = 0
    while index < len(interaction_steps):
        start = interaction_steps[index]
        start_before = step_state_before(start).get("ai_held_object")
        start_after = step_state_after(start).get("ai_held_object")
        object_name = held_name(start_after) or held_name(start_before)
        start_total = int(start["total_step"])
        group = [start]
        index += 1
        while index < len(interaction_steps):
            later = interaction_steps[index]
            if int(later["total_step"]) - start_total > 12:
                break
            later_before = step_state_before(later).get("ai_held_object")
            later_after = step_state_after(later).get("ai_held_object")
            later_object = held_name(later_after) or held_name(later_before)
            if later_object != object_name:
                break
            group.append(later)
            index += 1
        if len(group) < 6:
            continue
        picked_up_tiles: dict[tuple[int, int], int] = {}
        for step in group:
            after = step_state_after(step)
            tile = pos_tuple(after.get("ai_pos"))
            if tile is None:
                continue
            picked_up_tiles[tile] = picked_up_tiles.get(tile, 0) + 1
        if max(picked_up_tiles.values()) < 3:
            continue
        loop_tile = max(picked_up_tiles, key=picked_up_tiles.get)
        first_total = int(group[0]["total_step"])
        if first_total <= previous_event_end:
            continue
        end_total = int(group[-1]["total_step"])
        rewards = [float(step.get("environment_reward") or 0.0) for step in group]
        if any(reward > 0.0 for reward in rewards):
            continue
        previous_event_end = end_total
        events.append(
            candidate_event(
                event_type="AI_pick_drop_loop",
                start_timestep=first_total,
                end_timestep=end_total,
                evidence={
                    "reason": (
                        "AI picked up and dropped the same object on the same "
                        "tile at least three times in a short window without "
                        "delivery reward."
                    ),
                    "object_name": object_name,
                    "loop_tile": list(loop_tile),
                    "pick_drop_count": picked_up_tiles[loop_tile],
                    "duration_steps": end_total - first_total + 1,
                    "interaction_timesteps": [
                        int(step["total_step"]) for step in group
                    ],
                    "held_sequence": [
                        [
                            held_name(step_state_before(step).get("ai_held_object")),
                            held_name(step_state_after(step).get("ai_held_object")),
                        ]
                        for step in group
                    ],
                    "ai_actions": [step.get("ai_action_name") for step in group],
                    "ai_positions": [
                        step_state_after(step).get("ai_pos") for step in group
                    ],
                },
                severity=0.7,
                confidence=0.85,
                actor=ACTOR_AI,
                event_valence=VALENCE_NEGATIVE_PROBLEM,
                related_subgoal=step_ai_subgoal(group[0]),
                condition_features=extract_condition_features(group[0]),
            )
        )
    return events


def detect_ai_held_unneeded_object_too_long(window: list[dict]) -> list[dict]:
    """Detect when AI keeps holding an ingredient that the recipe no longer needs.

    Real trajectories interrupt holding with short pick/drop actions, so a
    contiguous-streak threshold never fires.  Count qualifying frames inside a
    sliding window instead: at least HELD_UNNEEDED_MIN_FRAMES frames of holding
    an unneeded ingredient within HELD_UNNEEDED_WINDOW_STEPS span.
    """

    events = []
    qualifying: list[dict] = []
    previous_event_end = -1
    for step in window:
        after = step_state_after(step)
        held = held_name(after.get("ai_held_object"))
        if held in {"tomato", "onion"} and recipe_needs(step, held) is False:
            qualifying.append(step)
        cutoff = int(step["total_step"]) - HELD_UNNEEDED_WINDOW_STEPS
        qualifying = [
            candidate
            for candidate in qualifying
            if int(candidate["total_step"]) > cutoff
        ]
        if len(qualifying) < HELD_UNNEEDED_MIN_FRAMES:
            continue
        first_total = int(qualifying[0]["total_step"])
        if first_total <= previous_event_end:
            continue
        previous_event_end = int(qualifying[-1]["total_step"])
        events.append(held_unneeded_event(qualifying))
    return events


def detect_ai_put_object_on_unhelpful_counter(window: list[dict]) -> list[dict]:
    """Detect ingredient drops far from any pot.

    The drop may be staging (pre-positioning an ingredient for a later pot)
    or a temporary put-down while yielding to the human.  Neither intent can
    be read from trajectory facts alone, so the event is neutral context:
    polarity comes from human feedback and review, not from a static valence.
    """

    events = []
    recent_drops: dict[tuple[str, tuple[int, int]], int] = {}
    for step in window:
        before = step_state_before(step)
        after = step_state_after(step)
        before_held = held_name(before.get("ai_held_object"))
        after_held = held_name(after.get("ai_held_object"))
        if before_held not in {"tomato", "onion", "dish"} or after_held is not None:
            continue
        if step.get("ai_action_name") != "interact":
            continue
        dropped_positions = object_positions(after, before_held)
        pot_positions_ = pot_all_positions(after)
        if not dropped_positions or not pot_positions_:
            continue
        nearest = min(
            manhattan(obj_pos, pot_pos)
            for obj_pos in dropped_positions
            for pot_pos in pot_positions_
            if manhattan(obj_pos, pot_pos) is not None
        )
        if nearest <= 3:
            continue
        event_step = int(step["total_step"])
        duplicate_key = (before_held, dropped_positions[0])
        previous_step = recent_drops.get(duplicate_key)
        if previous_step is not None and event_step - previous_step <= 8:
            continue
        recent_drops[duplicate_key] = event_step
        events.append(
            candidate_event(
                event_type="AI_put_object_on_unhelpful_counter",
                start_timestep=event_step,
                end_timestep=event_step,
                evidence={
                    "reason": (
                        "AI put down an object far from the pot area. "
                        "This may be staging or a temporary yield drop; "
                        "human feedback decides the polarity."
                    ),
                    "object": before_held,
                    "dropped_positions": [list(pos) for pos in dropped_positions],
                    "pot_positions": [list(pos) for pos in pot_positions_],
                    "nearest_pot_distance": nearest,
                    "ai_pos": after.get("ai_pos"),
                    "drop_may_be_staging_or_yield": True,
                },
                severity=0.35,
                confidence=0.4,
                actor=ACTOR_AI,
                event_valence=VALENCE_NEUTRAL_CONTEXT,
                related_subgoal=step_ai_subgoal(step),
                condition_features=extract_condition_features(step),
            )
        )
    return events


def detect_ai_missed_plate_pickup_opportunity(window: list[dict]) -> list[dict]:
    """Detect ready/cooking soup windows where AI could help by getting a dish."""

    events = []
    streak = []
    for step in window:
        after = step_state_after(step)
        pot_states = after.get("pot_states") or {}
        soup_needs_plate = bool(pot_positions(pot_states, ("cooking", "ready")))
        ai_holding = held_name(after.get("ai_held_object"))
        human_holding = held_name(after.get("human_held_object"))
        if soup_needs_plate and ai_holding is None and human_holding != "dish":
            if step_ai_subgoal(step) not in {"GET_DISH", "PICKUP_SOUP", "SERVE_SOUP"}:
                streak.append(step)
                continue
        if len(streak) >= 3:
            events.append(missed_plate_event(streak))
        streak = []
    if len(streak) >= 3:
        events.append(missed_plate_event(streak))
    return events


def detect_ai_missed_useful_counter_object(window: list[dict]) -> list[dict]:
    """Detect when AI ignores a closer useful loose object on a counter.

    The event is restricted to cases where the AI already selected the matching
    pickup subgoal. This avoids treating every visible staged object as an
    obligation to pick it up.
    """

    events = []
    pickup_event_steps: set[int] = set()
    streak: list[dict] = []
    signature = None

    def flush() -> None:
        nonlocal streak, signature
        event = missed_useful_counter_event(streak)
        if event is not None and not any(
            abs(step - int(event["end_timestep"])) <= 2
            for step in pickup_event_steps
        ):
            events.append(event)
        streak = []
        signature = None

    for step in window:
        before = step_state_before(step)
        after = step_state_after(step)
        before_held = held_name(before.get("ai_held_object"))
        after_held = held_name(after.get("ai_held_object"))
        players = before.get("players") or []
        ai_player = players[0] if players and isinstance(players[0], dict) else {}
        ai_before = pos_tuple(before.get("ai_pos"))
        orientation = pos_tuple(ai_player.get("orientation"))
        pickup_target = (
            add_pos(ai_before, orientation)
            if ai_before is not None and orientation is not None
            else None
        )
        terrain = (before.get("layout_features") or {}).get("terrain") or []
        expected_symbol = {
            "tomato": "T",
            "onion": "O",
            "dish": "D",
        }.get(after_held)
        before_step = {
            **step,
            "state_facts": before,
            "ai_condition_features": {},
            "extra": {"state_before": before},
        }
        before_conditions = extract_condition_features(before_step)
        picked_from_dispenser = (
            step.get("ai_action_name") == "interact"
            and before_held is None
            and after_held in {"tomato", "onion", "dish"}
            and pickup_target is not None
            and terrain_at(terrain, pickup_target) == expected_symbol
        )
        counter_source_advantage = (
            before_conditions.get("useful_counter_object_closer_than_dispenser")
            is True
            or before_conditions.get(
                "useful_counter_object_lower_task_cost_than_dispenser"
            )
            is True
        )
        counter_staging_advantage = (
            before_conditions.get(
                "useful_counter_object_closer_to_pot_than_dispenser"
            )
            is True
        )
        ignored_closer_counter_object = (
            picked_from_dispenser
            and before_conditions.get("useful_counter_object_type") == after_held
            and (counter_source_advantage or counter_staging_advantage)
        )
        if ignored_closer_counter_object:
            event_step = int(step["total_step"])
            pickup_event_steps.add(event_step)
            alternative_subgoal = (
                "GET_DISH"
                if after_held == "dish"
                else "GET_USEFUL_INGREDIENT"
            )
            events.append(
                candidate_event(
                    event_type="AI_missed_useful_counter_object",
                    start_timestep=event_step,
                    end_timestep=event_step,
                    evidence={
                        "reason": (
                            "AI picked from a dispenser even though a same-type "
                            "staged counter object was available and closer to "
                            "the AI, cheaper for the full task, or closer to the pot."
                        ),
                        "detection_mode": (
                            "dispenser_pickup_despite_source_cost_advantage"
                            if counter_source_advantage
                            else "dispenser_pickup_despite_object_staged_near_pot"
                        ),
                        "object_type": after_held,
                        "object_position": before_conditions.get(
                            "useful_counter_object_position"
                        ),
                        "counter_distance": before_conditions.get(
                            "useful_counter_object_distance"
                        ),
                        "matching_dispenser_distance": before_conditions.get(
                            "matching_dispenser_distance"
                        ),
                        "counter_total_task_distance": before_conditions.get(
                            "useful_counter_total_task_distance"
                        ),
                        "dispenser_total_task_distance": before_conditions.get(
                            "matching_dispenser_total_task_distance"
                        ),
                        "pickup_target": list(pickup_target),
                        "ai_pos": list(ai_before),
                        "ai_action": step.get("ai_action_name"),
                        "ai_subgoal": step_ai_subgoal(step),
                    },
                    severity=0.65,
                    confidence=0.9 if counter_source_advantage else 0.65,
                    actor=ACTOR_AI,
                    event_valence=VALENCE_MISSED_OPPORTUNITY,
                    related_subgoal=step_ai_subgoal(step),
                    alternative_subgoals=[alternative_subgoal],
                    condition_features=before_conditions,
                )
            )

        conditions = extract_condition_features(step)
        object_type = conditions.get("useful_counter_object_type")
        object_position = conditions.get("useful_counter_object_position")
        expected_subgoal = {
            "tomato": "GET_TOMATO",
            "onion": "GET_ONION",
            "dish": "GET_DISH",
        }.get(object_type)
        step_signature = (
            object_type,
            tuple(object_position) if isinstance(object_position, list) else None,
        )
        qualifies = (
            conditions.get("ai_empty_handed") is True
            and conditions.get("useful_counter_object_available") is True
            and (
                conditions.get("useful_counter_object_closer_than_dispenser")
                is True
                or conditions.get(
                    "useful_counter_object_lower_task_cost_than_dispenser"
                )
                is True
                or conditions.get(
                    "useful_counter_object_closer_to_pot_than_dispenser"
                )
                is True
            )
            and expected_subgoal is not None
            and step_ai_subgoal(step) == expected_subgoal
        )
        if not qualifies:
            flush()
            continue
        if signature is not None and step_signature != signature:
            flush()
        signature = step_signature
        streak.append(step)

    flush()
    return events


def detect_ai_missed_labor_division_opportunity(window: list[dict]) -> list[dict]:
    """Detect two conservative forms of duplicated or uncovered team work."""

    events = []
    streak: list[dict] = []
    opportunity_kind = None

    def classify(step: dict) -> tuple[str, str] | None:
        conditions = extract_condition_features(step)
        subgoal = step_ai_subgoal(step)
        if (
            conditions.get("human_holding_last_needed_ingredient") is True
            and conditions.get("ai_empty_handed") is True
            and subgoal not in {"GET_DISH", "PICKUP_SOUP", "SERVE_SOUP"}
        ):
            return "human_covers_last_ingredient", "GET_DISH"
        if (
            conditions.get("human_has_dish") is True
            and conditions.get("human_inferred_subgoal") == "PICKUP_SOUP"
            and conditions.get("ai_empty_handed") is True
            and subgoal == "GET_DISH"
        ):
            # Conservative: the human is already holding a dish with the pot
            # cooking/ready (inferred PICKUP_SOUP), so the AI re-selecting the
            # dish task duplicates an already-covered role.  A human who is
            # merely walking toward the dish is not yet committed to the role.
            return "duplicate_dish_task", "GET_USEFUL_INGREDIENT"
        return None

    def flush() -> None:
        nonlocal streak, opportunity_kind
        if len(streak) >= 2 and opportunity_kind is not None:
            events.append(labor_division_event(streak, opportunity_kind))
        streak = []
        opportunity_kind = None

    for step in window:
        result = classify(step)
        if result is None:
            flush()
            continue
        kind, _preferred = result
        if opportunity_kind is not None and kind != opportunity_kind:
            flush()
        opportunity_kind = kind
        streak.append(step)

    flush()
    return events


def held_unneeded_event(streak: list[dict]) -> dict:
    first = streak[0]
    last = streak[-1]
    last_after = step_state_after(last)
    return candidate_event(
        event_type="AI_held_unneeded_object_too_long",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "reason": "AI held an ingredient for several steps after the recipe no longer needed it.",
            "duration_steps": len(streak),
            "ai_held_object": last_after.get("ai_held_object"),
            "ai_actions": [step.get("ai_action_name") for step in streak],
            "ai_positions": [step_state_after(step).get("ai_pos") for step in streak],
        },
        severity=min(1.0, 0.3 + len(streak) * 0.05),
        confidence=0.6,
        actor=ACTOR_AI,
        event_valence=VALENCE_NEGATIVE_PROBLEM,
        related_subgoal=step_ai_subgoal(first),
        condition_features=extract_condition_features(first),
    )


def missed_plate_event(streak: list[dict]) -> dict:
    first = streak[0]
    last = streak[-1]
    last_after = step_state_after(last)
    return candidate_event(
        event_type="AI_missed_plate_pickup_opportunity",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "reason": "Soup was cooking or ready while AI had free hands, but AI did not choose the dish-related subgoal.",
            "duration_steps": len(streak),
            "pot_states": last_after.get("pot_states"),
            "ai_actions": [step.get("ai_action_name") for step in streak],
            "ai_subgoals": [step_ai_subgoal(step) for step in streak],
            "ai_positions": [step_state_after(step).get("ai_pos") for step in streak],
        },
        severity=min(1.0, 0.3 + len(streak) * 0.05),
        confidence=0.6,
        actor=ACTOR_AI,
        event_valence=VALENCE_MISSED_OPPORTUNITY,
        related_subgoal=step_ai_subgoal(first),
        condition_features=extract_condition_features(first),
    )


def missed_useful_counter_event(streak: list[dict]) -> dict | None:
    if len(streak) < 3:
        return None
    first = streak[0]
    last = streak[-1]
    first_conditions = extract_condition_features(first)
    last_conditions = extract_condition_features(last)
    first_counter_distance = first_conditions.get("useful_counter_object_distance")
    last_counter_distance = last_conditions.get("useful_counter_object_distance")
    first_dispenser_distance = first_conditions.get("matching_dispenser_distance")
    last_dispenser_distance = last_conditions.get("matching_dispenser_distance")
    counter_progress = (
        first_counter_distance - last_counter_distance
        if isinstance(first_counter_distance, int)
        and isinstance(last_counter_distance, int)
        else None
    )
    dispenser_progress = (
        first_dispenser_distance - last_dispenser_distance
        if isinstance(first_dispenser_distance, int)
        and isinstance(last_dispenser_distance, int)
        else None
    )
    if counter_progress is not None and counter_progress > 0:
        return None

    return candidate_event(
        event_type="AI_missed_useful_counter_object",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "reason": (
                "A useful loose object was closer than its dispenser, but the "
                "AI selected the matching pickup subgoal without approaching it."
            ),
            "duration_steps": len(streak),
            "object_type": last_conditions.get("useful_counter_object_type"),
            "object_position": last_conditions.get("useful_counter_object_position"),
            "counter_distances": [
                extract_condition_features(step).get(
                    "useful_counter_object_distance"
                )
                for step in streak
            ],
            "matching_dispenser_distances": [
                extract_condition_features(step).get("matching_dispenser_distance")
                for step in streak
            ],
            "counter_total_task_distances": [
                extract_condition_features(step).get(
                    "useful_counter_total_task_distance"
                )
                for step in streak
            ],
            "dispenser_total_task_distances": [
                extract_condition_features(step).get(
                    "matching_dispenser_total_task_distance"
                )
                for step in streak
            ],
            "counter_progress": counter_progress,
            "dispenser_progress": dispenser_progress,
            "ai_actions": [step.get("ai_action_name") for step in streak],
            "ai_subgoals": [step_ai_subgoal(step) for step in streak],
            "ai_positions": [
                step_state_after(step).get("ai_pos") for step in streak
            ],
        },
        severity=min(1.0, 0.3 + len(streak) * 0.05),
        confidence=0.7 if dispenser_progress and dispenser_progress > 0 else 0.6,
        actor=ACTOR_AI,
        event_valence=VALENCE_MISSED_OPPORTUNITY,
        related_subgoal=step_ai_subgoal(first),
        alternative_subgoals=[
            (
                "GET_DISH"
                if first_conditions.get("useful_counter_object_type") == "dish"
                else "GET_USEFUL_INGREDIENT"
            )
        ],
        condition_features=first_conditions,
    )


def labor_division_event(streak: list[dict], opportunity_kind: str) -> dict:
    first = streak[0]
    last = streak[-1]
    conditions = extract_condition_features(first)
    if opportunity_kind == "human_covers_last_ingredient":
        preferred_subgoal = "GET_DISH"
        reason = (
            "Human held the last needed ingredient while AI had free hands, "
            "but AI did not prepare a dish."
        )
    else:
        preferred_subgoal = "GET_USEFUL_INGREDIENT"
        reason = (
            "Human already held a dish with the pot cooking/ready (inferred "
            "PICKUP_SOUP) while AI also selected the dish task instead of "
            "complementary work."
        )
    return candidate_event(
        event_type="AI_missed_labor_division_opportunity",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "reason": reason,
            "opportunity_kind": opportunity_kind,
            "duration_steps": len(streak),
            "preferred_subgoal": preferred_subgoal,
            "human_inferred_subgoal": conditions.get("human_inferred_subgoal"),
            "needed_ingredient": conditions.get("needed_ingredient"),
            "ai_actions": [step.get("ai_action_name") for step in streak],
            "ai_subgoals": [step_ai_subgoal(step) for step in streak],
            "ai_positions": [
                step_state_after(step).get("ai_pos") for step in streak
            ],
            "human_positions": [
                step_state_after(step).get("human_pos") for step in streak
            ],
        },
        severity=min(1.0, 0.35 + len(streak) * 0.05),
        confidence=0.75,
        actor=ACTOR_AI,
        event_valence=VALENCE_MISSED_OPPORTUNITY,
        related_subgoal=step_ai_subgoal(first),
        alternative_subgoals=[preferred_subgoal],
        condition_features=conditions,
    )


def prep_wait_event(streak: list[dict]) -> dict:
    first = streak[0]
    last = streak[-1]
    last_after = step_state_after(last)
    ai_positions = [
        pos_tuple(step_state_after(step).get("ai_pos"))
        for step in streak
    ]
    human_positions = [
        pos_tuple(step_state_after(step).get("human_pos"))
        for step in streak
    ]
    return candidate_event(
        event_type="AI_failed_to_prepare_ingredient_while_waiting",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "reason": "Pot was cooking while human held a dish, but AI stayed empty-handed instead of preparing another ingredient.",
            "duration_steps": len(streak),
            "pot_states": last_after.get("pot_states"),
            "ai_actions": [step.get("ai_action_name") for step in streak],
            "human_actions": [step.get("human_action_name") for step in streak],
            "ai_positions": [list(pos) if pos else None for pos in ai_positions],
            "human_positions": [list(pos) if pos else None for pos in human_positions],
            "ai_held_object": last_after.get("ai_held_object"),
            "human_held_object": last_after.get("human_held_object"),
        },
        severity=min(1.0, 0.3 + len(streak) * 0.05),
        confidence=0.65,
        actor=ACTOR_AI,
        event_valence=VALENCE_MISSED_OPPORTUNITY,
        related_subgoal=step_ai_subgoal(first),
        condition_features=extract_condition_features(first),
    )


def ready_pot_event(streak: list[dict]) -> dict:
    first = streak[0]
    last = streak[-1]
    last_after = step_state_after(last)
    ready_positions = pot_positions(
        last_after.get("pot_states"),
        ("cooking", "ready"),
    )
    ai_pos = pos_tuple(last_after.get("ai_pos"))
    distances = [manhattan(ai_pos, pot_pos) for pot_pos in ready_positions]
    nearest_ready_pot_dist = min(
        [distance for distance in distances if distance is not None],
        default=None,
    )
    return candidate_event(
        event_type="AI_ignored_ready_or_nearly_ready_pot",
        start_timestep=int(first["total_step"]),
        end_timestep=int(last["total_step"]),
        evidence={
            "ready_pot_positions": [list(pos) for pos in ready_positions],
            "detected_pot_states": ["cooking", "ready"],
            "ai_pos": list(ai_pos) if ai_pos else None,
            "ai_held_object": last_after.get("ai_held_object"),
            "nearest_ready_pot_dist": nearest_ready_pot_dist,
            "ai_actions": [step.get("ai_action_name") for step in streak],
        },
        severity=min(1.0, 0.25 + len(streak) * 0.1),
        confidence=0.55 if nearest_ready_pot_dist is None else 0.7,
        actor=ACTOR_AI,
        event_valence=VALENCE_MISSED_OPPORTUNITY,
        related_subgoal=step_ai_subgoal(first),
        condition_features=extract_condition_features(first),
    )


def detect_candidate_events(window: list[dict]) -> list[dict]:
    events: list[dict] = []
    events.extend(detect_coordination_decision_events(window))
    events.extend(detect_successfully_delivered_soup(window))
    events.extend(detect_ai_successfully_picked_up_soup(window))
    events.extend(detect_ai_successfully_put_ingredient_into_pot(window))
    events.extend(detect_ai_blocked_human_path(window))
    events.extend(detect_ai_failed_to_yield_or_clear_path(window))
    events.extend(detect_ai_ignored_ready_or_nearly_ready_pot(window))
    events.extend(detect_ai_failed_to_prepare_ingredient_while_waiting(window))
    events.extend(detect_ai_missed_plate_pickup_opportunity(window))
    events.extend(detect_ai_missed_useful_counter_object(window))
    events.extend(detect_ai_missed_labor_division_opportunity(window))
    events.extend(detect_ai_pick_drop_loop(window))
    events.extend(detect_ai_held_unneeded_object_too_long(window))
    events.extend(detect_ai_put_object_on_unhelpful_counter(window))
    return events
