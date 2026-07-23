"""Build first-pass attribution previews from feedback and candidate events."""

from __future__ import annotations

from .event_detectors import DEFAULT_LOOKBACK_STEPS, detect_candidate_events, recent_window
from .feedback_type_router import polarity_from_feedback, route_feedback_type
from .schemas import attribution_result
from .subgoal_preferences import infer_subgoal_preferences


EVENT_KEYWORDS = {
    "AI_blocked_human_path": (
        "block",
        "blocked",
        "blocking",
        "stuck",
        "in my way",
        "path",
    ),
    "AI_ignored_ready_or_nearly_ready_pot": (
        "pot",
        "soup",
        "ready",
        "serve",
        "dish",
    ),
    "AI_failed_to_prepare_ingredient_while_waiting": (
        "prepare",
        "prep",
        "food",
        "cook",
        "cooked",
        "waiting",
        "wait",
        "while i wait",
        "next",
        "put",
        "place",
        "beside",
        "next to",
    ),
    "AI_pick_drop_loop": (
        "pick",
        "drop",
        "pick it",
        "drop it",
        "pick up",
        "put down",
        "again",
        "repeat",
        "repeatedly",
        "consistantly",
        "consistently",
        "loop",
    ),
    "AI_held_unneeded_object_too_long": (
        "holding",
        "held",
        "unneeded",
        "not needed",
        "don't need",
        "wrong ingredient",
    ),
    "AI_put_object_on_unhelpful_counter": (
        "put it down",
        "drop it",
        "counter",
        "far",
        "near",
        "beside",
        "near the pot",
        "next to the pot",
        "bring it near",
    ),
    "AI_missed_plate_pickup_opportunity": (
        "plate",
        "dish",
        "closer to the pot",
        "turn",
        "ask me",
        "serve",
    ),
    "AI_successfully_delivered_soup": (
        "good delivery",
        "delivery",
        "deliver",
        "served",
        "serve",
        "good job",
        "nice",
    ),
    "Human_successfully_delivered_soup": (
        "delivery",
        "deliver",
        "served",
        "serve",
        "good job",
        "nice",
    ),
    "Team_successfully_delivered_soup": (
        "delivery",
        "deliver",
        "served",
        "serve",
        "good job",
        "nice",
    ),
    "AI_successfully_picked_up_soup": (
        "pick up soup",
        "got soup",
        "soup",
    ),
    "AI_successfully_put_ingredient_into_pot": (
        "put into pot",
        "put in the pot",
        "ingredient into pot",
        "good ingredient",
        "nice ingredient",
        "tomato",
        "onion",
    ),
}

EVENT_PREFERENCES = {
    "AI_blocked_human_path": "avoid_blocking_human_path",
    "AI_ignored_ready_or_nearly_ready_pot": "handle_ready_pot_when_possible",
    "AI_failed_to_prepare_ingredient_while_waiting": "prepare_ingredients_during_cooking_wait",
    "AI_pick_drop_loop": "avoid_repeated_pick_drop",
    "AI_held_unneeded_object_too_long": "put_down_unneeded_object",
    "AI_put_object_on_unhelpful_counter": "stage_objects_near_pot",
    "AI_missed_plate_pickup_opportunity": "pick_up_plate_when_useful",
    "AI_successfully_delivered_soup": "deliver_soup_when_ready",
    "AI_successfully_picked_up_soup": "pick_up_soup_when_ready",
    "AI_successfully_put_ingredient_into_pot": "put_needed_ingredient_into_pot",
}

POSITIVE_VALENCE = "positive_progress"
NEGATIVE_VALENCES = {"negative_problem", "missed_opportunity"}


def feedback_event_id(feedback: dict) -> str:
    return (
        f"{feedback.get('source')}:{feedback.get('total_step')}:"
        f"{feedback.get('timestamp_utc')}"
    )


def candidate_overlaps_feedback(candidate: dict, feedback_total_step: int | None) -> bool:
    if feedback_total_step is None:
        return True
    start = candidate.get("start_timestep")
    end = candidate.get("end_timestep")
    if start is None or end is None:
        return False
    return int(start) <= int(feedback_total_step) <= int(end)


def candidate_near_feedback(
    candidate: dict,
    *,
    feedback_total_step: int | None,
    lookback_steps: int,
) -> bool:
    if feedback_total_step is None:
        return True
    start = candidate.get("start_timestep")
    end = candidate.get("end_timestep")
    if start is None or end is None:
        return False
    return (
        int(end) >= int(feedback_total_step) - lookback_steps
        and int(start) <= int(feedback_total_step)
    )


def candidate_recency_distance(candidate: dict, feedback_total_step: int | None) -> int:
    if feedback_total_step is None:
        return 0
    end = candidate.get("end_timestep")
    if end is None:
        return 999999
    return abs(int(feedback_total_step) - int(end))


def event_keyword_score(event_type: str, feedback_text: str | None) -> int:
    lowered = (feedback_text or "").lower()
    if not lowered:
        return 0
    return sum(1 for keyword in EVENT_KEYWORDS.get(event_type, ()) if keyword in lowered)


def event_valence(event: dict) -> str:
    return str(event.get("event_valence") or "neutral_context")


def event_actor(event: dict) -> str:
    return str(event.get("actor") or "unknown")


def polarity_valence_bonus(feedback_polarity: str, event: dict) -> float:
    valence = event_valence(event)
    actor = event_actor(event)
    if feedback_polarity == "negative":
        if valence == POSITIVE_VALENCE:
            return -4.0
        if valence in NEGATIVE_VALENCES:
            return 1.0
    if feedback_polarity == "positive":
        if valence == POSITIVE_VALENCE:
            return 1.5
        if valence in NEGATIVE_VALENCES:
            return -1.0
    if actor != "ai" and valence != POSITIVE_VALENCE:
        return -0.5
    return 0.0


def select_candidate_event(
    *,
    feedback: dict,
    candidate_events: list[dict],
    lookback_steps: int,
) -> tuple[dict | None, list[dict], float, bool, str]:
    """Pick a first-pass target event using time proximity and keywords.

    This is deliberately a baseline. It should be replaced or supplemented by
    LLM semantic attribution, but it gives us a deterministic comparison point.
    """

    feedback_text = feedback.get("feedback_text")
    feedback_value = feedback.get("feedback_value")
    feedback_total_step = feedback.get("total_step")
    feedback_polarity = polarity_from_feedback(feedback_text, feedback_value)
    nearby = [
        event
        for event in candidate_events
        if candidate_near_feedback(
            event,
            feedback_total_step=feedback_total_step,
            lookback_steps=lookback_steps,
        )
    ]
    if not nearby:
        return None, [], 0.2, True, "No candidate event found near this feedback."

    scored = []
    for event in nearby:
        event_type = event.get("event_type", "")
        keyword_score = event_keyword_score(event_type, feedback_text)
        if feedback_polarity == "negative" and event_valence(event) == POSITIVE_VALENCE:
            keyword_score = 0
        overlap_bonus = 2 if candidate_overlaps_feedback(event, feedback_total_step) else 0
        confidence = float(event.get("confidence") or 0.0)
        distance_penalty = candidate_recency_distance(event, feedback_total_step) * 0.01
        score = (
            keyword_score * 3
            + overlap_bonus
            + confidence
            + polarity_valence_bonus(feedback_polarity, event)
            - distance_penalty
        )
        scored.append((score, keyword_score, event))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_keyword_score, best_event = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else None
    ambiguous = runner_up_score is not None and best_score - runner_up_score < 1.0
    if best_keyword_score >= 2:
        ambiguous = False

    if best_keyword_score > 0:
        confidence = min(0.85, 0.45 + best_keyword_score * 0.15)
        reason = "Matched feedback keywords to a nearby candidate event."
    else:
        return (
            None,
            nearby,
            0.25,
            True,
            "Nearby candidate events exist, but none semantically matched this feedback.",
        )

    return best_event, nearby, confidence, ambiguous, reason


def key_conditions_from_event(event: dict | None) -> dict:
    if not event:
        return {}

    event_type = event.get("event_type")
    evidence = event.get("evidence", {})
    if event_type == "AI_blocked_human_path":
        return {
            "human_attempted_target": evidence.get("human_attempted_target"),
            "ai_before": evidence.get("ai_before"),
            "ai_after": evidence.get("ai_after"),
            "narrow_corridor": evidence.get("narrow_corridor"),
        }
    if event_type == "AI_ignored_ready_or_nearly_ready_pot":
        return {
            "ready_pot_positions": evidence.get("ready_pot_positions"),
            "ai_pos": evidence.get("ai_pos"),
            "ai_held_object": evidence.get("ai_held_object"),
            "nearest_ready_pot_dist": evidence.get("nearest_ready_pot_dist"),
        }
    if event_type == "AI_failed_to_prepare_ingredient_while_waiting":
        return {
            "duration_steps": evidence.get("duration_steps"),
            "pot_states": evidence.get("pot_states"),
            "ai_actions": evidence.get("ai_actions"),
            "ai_positions": evidence.get("ai_positions"),
            "human_held_object": evidence.get("human_held_object"),
        }
    if event_type == "AI_pick_drop_loop":
        return {
            "duration_steps": evidence.get("duration_steps"),
            "interaction_timesteps": evidence.get("interaction_timesteps"),
            "held_sequence": evidence.get("held_sequence"),
            "ai_positions": evidence.get("ai_positions"),
        }
    if event_type == "AI_put_object_on_unhelpful_counter":
        return {
            "object": evidence.get("object"),
            "dropped_positions": evidence.get("dropped_positions"),
            "nearest_pot_distance": evidence.get("nearest_pot_distance"),
        }
    if event_type == "AI_held_unneeded_object_too_long":
        return {
            "duration_steps": evidence.get("duration_steps"),
            "ai_held_object": evidence.get("ai_held_object"),
            "ai_positions": evidence.get("ai_positions"),
        }
    if event_type == "AI_missed_plate_pickup_opportunity":
        return {
            "duration_steps": evidence.get("duration_steps"),
            "pot_states": evidence.get("pot_states"),
            "ai_subgoals": evidence.get("ai_subgoals"),
            "ai_positions": evidence.get("ai_positions"),
        }
    return evidence


def build_preview_attribution(
    *,
    feedback: dict,
    trajectory: list[dict],
    candidate_events: list[dict] | None = None,
    lookback_steps: int = DEFAULT_LOOKBACK_STEPS,
) -> dict:
    feedback_text = feedback.get("feedback_text")
    feedback_value = feedback.get("feedback_value")
    feedback_total_step = feedback.get("total_step")
    window = recent_window(
        trajectory,
        feedback_total_step=feedback_total_step,
        lookback_steps=lookback_steps,
    )
    candidates = candidate_events
    if candidates is None:
        candidates = detect_candidate_events(window)

    feedback_type = route_feedback_type(feedback_text, feedback_value)
    polarity = polarity_from_feedback(feedback_text, feedback_value)
    target, nearby_candidates, confidence, needs_clarification, selection_reason = (
        select_candidate_event(
            feedback=feedback,
            candidate_events=candidates,
            lookback_steps=lookback_steps,
        )
    )

    target_event = target.get("event_type") if target else None
    condition_features = target.get("condition_features") if target else {}
    preferred_subgoals, rejected_subgoals = infer_subgoal_preferences(
        target_event=target_event,
    )
    preference = None
    if target_event and polarity in {"negative", "unknown", "positive"}:
        preference = EVENT_PREFERENCES.get(target_event)

    clarification = None
    if needs_clarification:
        if nearby_candidates:
            event_names = sorted({event.get("event_type") for event in nearby_candidates})
            clarification = (
                "Which event did this feedback refer to? Candidates: "
                + ", ".join(str(name) for name in event_names)
            )
        else:
            clarification = (
                "What behavior was this feedback referring to? "
                "No candidate event was detected near the feedback time."
            )

    missing = sorted(
        {
            fact
            for event in nearby_candidates
            for fact in event.get("missing_required_facts", [])
        }
    )
    proposed_schema_update = []
    if missing:
        proposed_schema_update.append(
            {
                "field_name": "state_facts",
                "reason": "Current session logs lack facts needed for reliable event attribution.",
                "missing_fields": missing,
                "status": "system_required",
            }
        )

    target_time_window = None
    if target:
        target_time_window = [
            int(target["start_timestep"]),
            int(target["end_timestep"]),
        ]
    elif window:
        target_time_window = [
            int(window[0]["total_step"]),
            int(window[-1]["total_step"]),
        ]

    return attribution_result(
        feedback_event_id=feedback_event_id(feedback),
        feedback_type=feedback_type,
        target_time_window=target_time_window,
        target_event=target_event,
        polarity=polarity,
        preference=preference,
        key_conditions=key_conditions_from_event(target),
        condition_features=condition_features,
        preferred_subgoals=preferred_subgoals,
        rejected_subgoals=rejected_subgoals,
        candidate_events=nearby_candidates,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=clarification,
        proposed_schema_update=proposed_schema_update,
        notes=(
            "Preview attribution only. "
            f"{selection_reason} "
            "This is a deterministic baseline before LLM semantic attribution."
        ),
    )
