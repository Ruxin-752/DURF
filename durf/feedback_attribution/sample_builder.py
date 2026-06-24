"""Build first-pass attribution samples from converted session logs."""

from __future__ import annotations

from .event_detectors import detect_candidate_events, recent_window
from .feedback_type_router import polarity_from_feedback, route_feedback_type
from .schemas import attribution_result


def build_preview_attribution(
    *,
    feedback: dict,
    trajectory: list[dict],
    lookback_steps: int = 8,
) -> dict:
    feedback_text = feedback.get("feedback_text")
    feedback_value = feedback.get("feedback_value")
    feedback_total_step = feedback.get("total_step")
    window = recent_window(
        trajectory,
        feedback_total_step=feedback_total_step,
        lookback_steps=lookback_steps,
    )
    candidates = detect_candidate_events(window)
    feedback_type = route_feedback_type(feedback_text, feedback_value)
    polarity = polarity_from_feedback(feedback_text, feedback_value)

    target_event = None
    preference = None
    confidence = 0.2
    needs_clarification = True
    clarification = (
        "Does this feedback refer to AI blocking the human path, "
        "ignoring a task, or another collaboration issue?"
    )

    if candidates:
        blocking = next(
            (event for event in candidates if event["event_type"] == "AI_blocked_human_path"),
            candidates[0],
        )
        target_event = blocking["event_type"]
        if target_event == "AI_blocked_human_path" and polarity == "negative":
            preference = "avoid_blocking_human_path"

    missing = sorted(
        {
            fact
            for event in candidates
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

    if feedback_text and target_event:
        confidence = 0.35 if missing else 0.7

    return attribution_result(
        feedback_event_id=f"{feedback.get('source')}:{feedback.get('total_step')}:{feedback.get('timestamp_utc')}",
        feedback_type=feedback_type,
        target_time_window=[
            int(window[0]["total_step"]),
            int(window[-1]["total_step"]),
        ]
        if window
        else None,
        target_event=target_event,
        polarity=polarity,
        preference=preference,
        key_conditions={},
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=clarification,
        proposed_schema_update=proposed_schema_update,
        notes=(
            "Preview attribution only. Current CSV logs do not yet contain full "
            "state facts, so this result should not be used as training data."
        ),
    )
