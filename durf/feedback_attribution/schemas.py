"""Shared schemas for the feedback attribution pipeline.

The first implementation uses plain dictionaries for easy JSONL logging, but
the helper constructors below keep field names stable across scripts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def trajectory_step(
    *,
    source: str,
    timestamp_utc: str,
    episode: int,
    episode_step: int,
    total_step: int,
    layout: str,
    ai_action: int | None,
    ai_action_name: str | None,
    human_action: int | None,
    human_action_name: str | None,
    environment_reward: float | None,
    episode_reward: float | None,
    done: bool | None,
    ai_subgoal: str | None = None,
    ai_event: str | None = None,
    ai_condition_features: dict[str, Any] | None = None,
    ai_subgoal_candidates: list[dict[str, Any]] | None = None,
    task_decision: dict[str, Any] | None = None,
    coordination_decision: dict[str, Any] | None = None,
    state_facts: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    protocol_version: str = "protocol-v2",
) -> dict[str, Any]:
    """Create a normalized trajectory step.

    Current CSV logs do not yet include full state facts such as positions,
    held objects, or pot states. Those fields are included as null so later
    event detectors can depend on a stable schema.
    """

    return {
        "record_type": "trajectory_step",
        "source": source,
        "protocol_version": protocol_version,
        "timestamp_utc": timestamp_utc,
        "episode": episode,
        "episode_step": episode_step,
        "total_step": total_step,
        "layout": layout,
        "ai_action": ai_action,
        "ai_action_name": ai_action_name,
        "ai_subgoal": ai_subgoal,
        "ai_condition_features": ai_condition_features or {},
        "ai_subgoal_candidates": ai_subgoal_candidates or [],
        "task_decision": task_decision or {},
        "coordination_decision": coordination_decision or {},
        "ai_event": ai_event,
        "human_action": human_action,
        "human_action_name": human_action_name,
        "environment_reward": environment_reward,
        "episode_reward": episode_reward,
        "done": done,
        "state_facts": state_facts or {
            "ai_pos": None,
            "human_pos": None,
            "ai_held_object": None,
            "human_held_object": None,
            "pot_states": None,
            "layout_features": None,
        },
        "extra": extra or {},
    }


def feedback_event(
    *,
    source: str,
    timestamp_utc: str,
    episode: int,
    episode_step: int,
    total_step: int,
    feedback_text: str | None,
    feedback_value: int | None,
    role: str | None,
    extra: dict[str, Any] | None = None,
    user_id: str | None = None,
    protocol_version: str = "protocol-v2",
) -> dict[str, Any]:
    return {
        "record_type": "feedback_event",
        "source": source,
        "timestamp_utc": timestamp_utc,
        "episode": episode,
        "episode_step": episode_step,
        "total_step": total_step,
        "feedback_text": feedback_text,
        "feedback_value": feedback_value,
        "role": role,
        "extra": extra or {},
        "user_id": user_id,
        "protocol_version": protocol_version,
    }


def candidate_event(
    *,
    event_type: str,
    start_timestep: int | None,
    end_timestep: int | None,
    evidence: dict[str, Any],
    severity: float | None,
    confidence: float,
    actor: str = "ai",
    event_valence: str = "neutral_context",
    related_subgoal: str | None = None,
    alternative_subgoals: list[str] | None = None,
    condition_features: dict[str, Any] | None = None,
    missing_required_facts: list[str] | None = None,
    source: str = "synthetic_sim_human",
    label_status: str = "automatic",
    protocol_version: str = "protocol-v2",
) -> dict[str, Any]:
    return {
        "record_type": "candidate_event",
        "event_type": event_type,
        "actor": actor,
        "event_valence": event_valence,
        "start_timestep": start_timestep,
        "end_timestep": end_timestep,
        "evidence": evidence,
        "related_subgoal": related_subgoal,
        "alternative_subgoals": alternative_subgoals or [],
        "condition_features": condition_features or {},
        "severity": severity,
        "confidence": confidence,
        "missing_required_facts": missing_required_facts or [],
        "source": source,
        "label_status": label_status,
        "protocol_version": protocol_version,
    }


def attribution_result(
    *,
    feedback_event_id: str,
    feedback_type: str,
    target_time_window: list[int] | None,
    target_event: str | None,
    polarity: str,
    preference: str | None,
    key_conditions: dict[str, Any],
    condition_features: dict[str, Any] | None = None,
    preferred_subgoals: list[str] | None = None,
    rejected_subgoals: list[str] | None = None,
    candidate_events: list[dict[str, Any]] | None = None,
    confidence: float,
    needs_clarification: bool,
    clarification_question: str | None,
    proposed_schema_update: list[dict[str, Any]],
    notes: str,
    source: str = "synthetic_sim_human",
    label_status: str = "automatic",
    user_id: str | None = None,
    decision_level: str | None = None,
    preference_source: str | None = None,
    preference_overridden_event: str | None = None,
    protocol_version: str = "protocol-v2",
) -> dict[str, Any]:
    return {
        "record_type": "attribution_result",
        "created_at": utc_now(),
        "feedback_event_id": feedback_event_id,
        "feedback_type": feedback_type,
        "target_time_window": target_time_window,
        "target_event": target_event,
        "polarity": polarity,
        "preference": preference,
        "key_conditions": key_conditions,
        "condition_features": condition_features or {},
        "preferred_subgoals": preferred_subgoals or [],
        "rejected_subgoals": rejected_subgoals or [],
        "candidate_events": candidate_events or [],
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "proposed_schema_update": proposed_schema_update,
        "notes": notes,
        "source": source,
        "label_status": label_status,
        "user_id": user_id,
        "decision_level": decision_level,
        "preference_source": preference_source,
        # Set when an explicit policy statement was filed instead of the
        # nearest detected event because they belong to different decision
        # domains (see sample_builder.build_preview_attribution).
        "preference_overridden_event": preference_overridden_event,
        "protocol_version": protocol_version,
    }
