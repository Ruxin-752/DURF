"""Candidate event detectors for saved trajectory logs.

These detectors are intentionally conservative. They turn factual trajectory
records into candidate events for later semantic attribution, not final labels.
"""

from __future__ import annotations

from .schemas import candidate_event


ACTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


DEFAULT_LOOKBACK_STEPS = 25


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
                    missing_required_facts=missing,
                )
            ]

    return []


def detect_ai_ignored_ready_or_nearly_ready_pot(window: list[dict]) -> list[dict]:
    """Detect ready-pot windows where the AI does not interact with the pot."""

    events: list[dict] = []
    streak: list[dict] = []

    for step in window:
        after = step_state_after(step)
        missing = missing_facts(step, ["ai_pos", "pot_states"])
        if missing:
            streak = []
            continue

        ready_positions = pot_positions(after.get("pot_states"), ("ready",))
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
    )


def ready_pot_event(streak: list[dict]) -> dict:
    first = streak[0]
    last = streak[-1]
    last_after = step_state_after(last)
    ready_positions = pot_positions(last_after.get("pot_states"), ("ready",))
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
            "ai_pos": list(ai_pos) if ai_pos else None,
            "ai_held_object": last_after.get("ai_held_object"),
            "nearest_ready_pot_dist": nearest_ready_pot_dist,
            "ai_actions": [step.get("ai_action_name") for step in streak],
        },
        severity=min(1.0, 0.25 + len(streak) * 0.1),
        confidence=0.55 if nearest_ready_pot_dist is None else 0.7,
    )


def detect_candidate_events(window: list[dict]) -> list[dict]:
    events: list[dict] = []
    events.extend(detect_ai_blocked_human_path(window))
    events.extend(detect_ai_ignored_ready_or_nearly_ready_pot(window))
    events.extend(detect_ai_failed_to_prepare_ingredient_while_waiting(window))
    return events
