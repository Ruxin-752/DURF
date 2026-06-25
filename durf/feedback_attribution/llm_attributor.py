"""LLM semantic attribution over program-detected candidate events.

The LLM is deliberately placed after factual extraction:

trajectory facts -> candidate events -> LLM semantic selection

It should not invent low-level game facts. It may choose among candidate events,
ask for clarification, or propose schema/event additions for human review.
"""

from __future__ import annotations

import json
from typing import Any

from durf.group_a.deepseek_chat import chat_once

from .sample_builder import (
    candidate_near_feedback,
    feedback_event_id,
    key_conditions_from_event,
)
from .schemas import attribution_result


SYSTEM_PROMPT = """You are an attribution module for a human-AI Overcooked study.

Your job is semantic attribution, not reward design and not factual detection.

You will receive:
1. one human feedback message,
2. nearby program-detected candidate events,
3. a short recent trajectory summary.

Rules:
- Choose target_event only from candidate_events[].event_type, or null.
- Do not invent positions, actions, pot states, or events.
- If the feedback refers to something not represented by candidate events,
  set target_event=null and propose_schema_update.
- If multiple candidates are plausible, set needs_clarification=true and ask
  one short clarification question.
- Return JSON only. No markdown, no explanation outside JSON.
"""


OUTPUT_SCHEMA_HINT = {
    "feedback_type": "evaluative|imperative|descriptive|unknown",
    "target_event": "one event_type from candidate_events or null",
    "target_time_window": [0, 0],
    "polarity": "positive|negative|neutral|unknown",
    "preference": "short snake_case preference or null",
    "key_conditions": {},
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
                "human_action": step.get("human_action_name"),
                "reward": step.get("environment_reward"),
                "ai_pos": facts.get("ai_pos"),
                "human_pos": facts.get("human_pos"),
                "ai_held_object": facts.get("ai_held_object"),
                "human_held_object": facts.get("human_held_object"),
                "pot_states": facts.get("pot_states"),
            }
        )
    return summary


def compact_candidate_event(event: dict) -> dict:
    return {
        "event_type": event.get("event_type"),
        "start_timestep": event.get("start_timestep"),
        "end_timestep": event.get("end_timestep"),
        "confidence": event.get("confidence"),
        "severity": event.get("severity"),
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

    rationale = parsed.get("rationale") or "No rationale returned."
    return attribution_result(
        feedback_event_id=feedback_event_id(feedback),
        feedback_type=parsed.get("feedback_type") or baseline_attribution.get("feedback_type"),
        target_time_window=target_time_window,
        target_event=target.get("event_type") if target else None,
        polarity=parsed.get("polarity") or baseline_attribution.get("polarity"),
        preference=parsed.get("preference") or baseline_attribution.get("preference"),
        key_conditions=key_conditions,
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
    raw_response = chat_once(messages)
    parsed = extract_json_object(raw_response)
    attribution = normalize_llm_result(
        feedback=feedback,
        parsed=parsed,
        nearby_candidate_events=nearby,
        baseline_attribution=baseline_attribution,
    )
    audit = {
        "feedback_event_id": feedback_event_id(feedback),
        "messages": messages,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "used_candidate_event_count": len(nearby),
        "status": "ok",
    }
    return attribution, audit
