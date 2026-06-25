"""Build first-pass attribution previews from feedback and candidate events."""

from __future__ import annotations

from .event_detectors import DEFAULT_LOOKBACK_STEPS, detect_candidate_events, recent_window
from .feedback_type_router import polarity_from_feedback, route_feedback_type
from .schemas import attribution_result


EVENT_KEYWORDS = {
    "AI_blocked_human_path": (
        "block",
        "blocked",
        "blocking",
        "stuck",
        "in my way",
        "path",
        "堵",
        "挡",
        "卡",
        "路",
        "别堵",
    ),
    "AI_ignored_ready_or_nearly_ready_pot": (
        "pot",
        "soup",
        "ready",
        "serve",
        "dish",
        "锅",
        "汤",
        "菜",
        "盘",
        "好了",
        "不拿",
        "没拿",
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
        "准备",
        "备菜",
        "放",
        "旁边",
        "等",
    ),
}

EVENT_PREFERENCES = {
    "AI_blocked_human_path": "avoid_blocking_human_path",
    "AI_ignored_ready_or_nearly_ready_pot": "handle_ready_pot_when_possible",
    "AI_failed_to_prepare_ingredient_while_waiting": "prepare_ingredients_during_cooking_wait",
}


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
    feedback_total_step = feedback.get("total_step")
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
        keyword_score = event_keyword_score(event.get("event_type", ""), feedback_text)
        overlap_bonus = 2 if candidate_overlaps_feedback(event, feedback_total_step) else 0
        confidence = float(event.get("confidence") or 0.0)
        distance_penalty = candidate_recency_distance(event, feedback_total_step) * 0.01
        score = keyword_score * 3 + overlap_bonus + confidence - distance_penalty
        scored.append((score, keyword_score, event))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_keyword_score, best_event = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else None
    ambiguous = runner_up_score is not None and best_score - runner_up_score < 1.0

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
    preference = None
    if target_event and polarity in {"negative", "unknown"}:
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
