"""LLM semantic attribution over program-detected candidate events.

The LLM is deliberately placed after factual extraction:

trajectory facts -> candidate events -> LLM semantic selection

It should not invent low-level game facts. It may choose among candidate events,
ask for clarification, or propose schema/event additions for human review.
"""

from __future__ import annotations

import json
from typing import Any

from durf.group_a.deepseek_chat import chat_once_detailed

from .sample_builder import (
    candidate_near_feedback,
    feedback_event_id,
    key_conditions_from_event,
)
from .schemas import ATTRIBUTOR_LLM, attribution_result
from .subgoal_preferences import (
    COORDINATION_SUBGOALS,
    TASK_HU_SUBGOALS,
    infer_subgoal_preferences,
)


SYSTEM_PROMPT = """You are an attribution module for a human-AI Overcooked study.

Your job is semantic attribution, not reward design and not factual detection.

You will receive:
1. one human feedback message,
2. nearby program-detected candidate events,
3. a short recent trajectory summary.

Rules:
- Choose target_event only from candidate_events[].event_type, or null.
- Use candidate_events[].event_valence to distinguish positive progress,
  negative problems, missed opportunities, and neutral context.
- For negative feedback, prefer negative_problem or missed_opportunity events
  unless the text explicitly praises a positive_progress event.
- Use candidate_events[].actor. Human/team-only progress events are context
  and should not become Hu labels for changing AI behavior unless the feedback
  clearly describes the AI's role.
- Do not invent positions, actions, pot states, or events.
- preferred_subgoals and rejected_subgoals MUST be names from
  subgoal_vocabulary, and MUST both appear in the "candidates" list of the
  trajectory step the feedback is about (the step inside target_time_window).
  A preference is a choice between options the AI actually had at that
  moment; naming a subgoal that was not on offer makes the label meaningless.
  If the person means "fetch an ingredient" without saying which, use
  GET_USEFUL_INGREDIENT -- it is resolved to whichever ingredient fetch was in
  the candidates.  Never pair a task subgoal with a coordination option.
- If the feedback refers to something not represented by candidate events,
  set target_event=null and propose_schema_update for human review.
- Do not bundle state conditions into event names. Events describe only AI
  behavior patterns; state facts belong in key_conditions or condition_features.
  For example, propose "AI_missed_plate_pickup_opportunity" plus condition
  "human_has_last_needed_ingredient", not
  "AI_should_get_dish_when_human_has_last_ingredient".
- If multiple candidates are plausible, set needs_clarification=true and ask
  one short clarification question.
- Return JSON only. No markdown, no explanation outside JSON.
"""


LLM_STATED_PREFERENCE = "llm_stated_preference"

SUBGOAL_VOCABULARY_HINT = {
    "task": list(TASK_HU_SUBGOALS) + ["WAIT"],
    "coordination": list(COORDINATION_SUBGOALS),
    "alias": {
        "GET_USEFUL_INGREDIENT": "either GET_TOMATO or GET_ONION, whichever was a candidate",
    },
    "rule": "preferred/rejected subgoals must be in the step's candidates; never mix task and coordination",
}


OUTPUT_SCHEMA_HINT = {
    "feedback_type": "evaluative|imperative|descriptive|unknown",
    "target_event": "one event_type from candidate_events or null",
    "target_time_window": [0, 0],
    "polarity": "positive|negative|neutral|unknown",
    "preference": "short snake_case preference or null",
    "key_conditions": {},
    "preferred_subgoals": [],
    "rejected_subgoals": [],
    "confidence": 0.0,
    "needs_clarification": False,
    "clarification_question": None,
    "proposed_schema_update": [],
    "rationale": "one short sentence",
}


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def nearby_events(
    *,
    feedback: dict,
    candidate_events: list[dict],
    lookback_steps: int,
) -> list[dict]:
    feedback_total_step = feedback.get("total_step")
    return [
        event
        for event in candidate_events
        if candidate_near_feedback(
            event,
            feedback_total_step=feedback_total_step,
            lookback_steps=lookback_steps,
        )
    ]


def recent_trajectory_summary(
    trajectory: list[dict],
    *,
    feedback_total_step: int | None,
    lookback_steps: int,
) -> list[dict]:
    if feedback_total_step is None:
        window = trajectory[-lookback_steps:]
    else:
        start = max(0, int(feedback_total_step) - lookback_steps)
        window = [
            step
            for step in trajectory
            if start <= int(step.get("total_step") or -1) <= int(feedback_total_step)
        ]
    summary = []
    for step in window:
        facts = step.get("state_facts") or {}
        summary.append(
            {
                "total_step": step.get("total_step"),
                "ai_action": step.get("ai_action_name"),
                "ai_subgoal": step.get("ai_subgoal"),
                "ai_event": step.get("ai_event"),
                "human_action": step.get("human_action_name"),
                "reward": step.get("environment_reward"),
                "ai_pos": facts.get("ai_pos"),
                "human_pos": facts.get("human_pos"),
                "ai_held_object": facts.get("ai_held_object"),
                "human_held_object": facts.get("human_held_object"),
                "pot_states": facts.get("pot_states"),
                # The options the AI actually had at this step.  Without this
                # the model named subgoals from memory: on the first full
                # corpus 84% of its task pairs had a side that was never on
                # offer at the decision they were attached to.
                "candidates": [
                    c.get("subgoal")
                    for c in (step.get("ai_subgoal_candidates") or [])
                    if isinstance(c, dict) and c.get("subgoal") and c.get("feasible", True)
                ],
            }
        )
    return summary


def compact_candidate_event(event: dict) -> dict:
    return {
        "event_type": event.get("event_type"),
        "actor": event.get("actor"),
        "event_valence": event.get("event_valence"),
        "start_timestep": event.get("start_timestep"),
        "end_timestep": event.get("end_timestep"),
        "confidence": event.get("confidence"),
        "severity": event.get("severity"),
        "related_subgoal": event.get("related_subgoal"),
        "condition_features": event.get("condition_features"),
        "evidence": event.get("evidence"),
    }


def build_llm_messages(
    *,
    feedback: dict,
    trajectory: list[dict],
    candidate_events: list[dict],
    baseline_attribution: dict,
    lookback_steps: int,
) -> list[dict[str, str]]:
    payload = {
        "feedback": {
            "feedback_text": feedback.get("feedback_text"),
            "feedback_value": feedback.get("feedback_value"),
            "role": feedback.get("role"),
            "total_step": feedback.get("total_step"),
        },
        "candidate_events": [
            compact_candidate_event(event) for event in candidate_events
        ],
        "recent_trajectory": recent_trajectory_summary(
            trajectory,
            feedback_total_step=feedback.get("total_step"),
            lookback_steps=lookback_steps,
        ),
        "rule_based_baseline": {
            "target_event": baseline_attribution.get("target_event"),
            "confidence": baseline_attribution.get("confidence"),
            "needs_clarification": baseline_attribution.get("needs_clarification"),
            "notes": baseline_attribution.get("notes"),
        },
        "subgoal_vocabulary": SUBGOAL_VOCABULARY_HINT,
        "output_schema": OUTPUT_SCHEMA_HINT,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(stripped[start : end + 1])


def normalize_llm_result(
    *,
    feedback: dict,
    parsed: dict,
    nearby_candidate_events: list[dict],
    baseline_attribution: dict,
) -> dict:
    allowed_by_type = {
        event.get("event_type"): event for event in nearby_candidate_events
    }
    requested_event = parsed.get("target_event")
    target = allowed_by_type.get(requested_event)

    proposed_schema_update = parsed.get("proposed_schema_update") or []
    needs_clarification = bool(parsed.get("needs_clarification"))
    clarification_question = parsed.get("clarification_question")
    if requested_event and target is None:
        needs_clarification = True
        clarification_question = (
            clarification_question
            or "The feedback seems to refer to an event outside the current candidate list."
        )

    target_time_window = parsed.get("target_time_window")
    if target is not None:
        target_time_window = [
            int(target["start_timestep"]),
            int(target["end_timestep"]),
        ]
    elif not isinstance(target_time_window, list):
        target_time_window = baseline_attribution.get("target_time_window")

    key_conditions = parsed.get("key_conditions")
    if not isinstance(key_conditions, dict) or not key_conditions:
        key_conditions = key_conditions_from_event(target)

    condition_features = parsed.get("condition_features")
    if not isinstance(condition_features, dict) or not condition_features:
        condition_features = target.get("condition_features") if target else {}

    preferred_subgoals, rejected_subgoals = infer_subgoal_preferences(
        target_event=target.get("event_type") if target else None,
        preferred_subgoals=parsed.get("preferred_subgoals"),
        rejected_subgoals=parsed.get("rejected_subgoals"),
        observed_subgoal=target.get("related_subgoal") if target else None,
        alternative_subgoals=target.get("alternative_subgoals") if target else None,
        event_valence=target.get("event_valence") if target else None,
    )

    rationale = parsed.get("rationale") or "No rationale returned."
    # A pair the LLM reads off the sentence itself, with no detected event
    # behind it, is a stated preference: the dataset builder keeps only labels
    # that are event-grounded OR carry a preference_source.  (First full LLM
    # corpus, 2026-09-05: every task-level answer came back event-free and all
    # 194 co-available task pairs were dropped here, silently.)
    preference_source = (
        LLM_STATED_PREFERENCE
        if target is None and preferred_subgoals and rejected_subgoals
        else None
    )
    return attribution_result(
        preference_source=preference_source,
        feedback_event_id=feedback_event_id(feedback),
        feedback_type=parsed.get("feedback_type") or baseline_attribution.get("feedback_type"),
        target_time_window=target_time_window,
        target_event=target.get("event_type") if target else None,
        polarity=parsed.get("polarity") or baseline_attribution.get("polarity"),
        preference=parsed.get("preference") or baseline_attribution.get("preference"),
        key_conditions=key_conditions,
        condition_features=condition_features,
        preferred_subgoals=preferred_subgoals,
        rejected_subgoals=rejected_subgoals,
        candidate_events=nearby_candidate_events,
        confidence=clamp_confidence(parsed.get("confidence")),
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        proposed_schema_update=proposed_schema_update,
        notes=f"LLM semantic attribution. Rationale: {rationale}",
    )


def run_llm_attribution(
    *,
    feedback: dict,
    trajectory: list[dict],
    candidate_events: list[dict],
    baseline_attribution: dict,
    lookback_steps: int,
) -> tuple[dict, dict]:
    nearby = nearby_events(
        feedback=feedback,
        candidate_events=candidate_events,
        lookback_steps=lookback_steps,
    )
    messages = build_llm_messages(
        feedback=feedback,
        trajectory=trajectory,
        candidate_events=nearby,
        baseline_attribution=baseline_attribution,
        lookback_steps=lookback_steps,
    )
    # temperature 0: the attribution has to be reproducible from the audit
    # log, and the chat UI's 0.2 is not (it is left untouched for the UI).
    chat = chat_once_detailed(messages, temperature=0.0)
    raw_response = chat.text
    parsed = extract_json_object(raw_response)
    attribution = normalize_llm_result(
        feedback=feedback,
        parsed=parsed,
        nearby_candidate_events=nearby,
        baseline_attribution=baseline_attribution,
    )
    attribution["attributor"] = ATTRIBUTOR_LLM
    attribution["model_id"] = chat.model_returned or chat.model_requested
    attribution["prompt_hash"] = chat.prompt_hash
    audit = {
        "feedback_event_id": feedback_event_id(feedback),
        "messages": messages,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "used_candidate_event_count": len(nearby),
        "status": "ok",
        "attributor": ATTRIBUTOR_LLM,
        "model_requested": chat.model_requested,
        "model_returned": chat.model_returned,
        "system_fingerprint": chat.system_fingerprint,
        "prompt_hash": chat.prompt_hash,
        "temperature": chat.temperature,
        "cached": chat.cached,
        "attempts": chat.attempts,
    }
    return attribution, audit
