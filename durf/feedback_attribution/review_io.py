"""Data interface for post-session human review.

The review layer is deliberately separate from automatic attribution.  It is
used for pilot annotation, schema debugging, and gold-label construction.  The
saved decisions do not automatically rewrite LLM outputs or train Hu unless a
later script explicitly consumes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .condition_features import extract_condition_features, latest_step_at_or_before
from .io_utils import read_jsonl, write_jsonl
from .sample_builder import candidate_near_feedback, feedback_event_id


REVIEW_ITEMS_FILE = "review_items.jsonl"
REVIEW_DECISIONS_FILE = "review_decisions.jsonl"


def item_id_for_feedback(feedback: dict[str, Any]) -> str:
    return feedback_event_id(feedback)


def by_feedback_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("feedback_event_id")): record for record in records}


def read_review_decisions(session_dir: Path) -> dict[str, dict[str, Any]]:
    decisions = read_jsonl(session_dir / REVIEW_DECISIONS_FILE)
    return {str(decision.get("feedback_event_id")): decision for decision in decisions}


def write_review_decisions(
    session_dir: Path,
    decisions_by_id: dict[str, dict[str, Any]],
) -> int:
    decisions = sorted(
        decisions_by_id.values(),
        key=lambda item: (
            int(item.get("feedback_total_step") or 0),
            str(item.get("feedback_event_id") or ""),
        ),
    )
    return write_jsonl(session_dir / REVIEW_DECISIONS_FILE, decisions)


def upsert_review_decision(session_dir: Path, decision: dict[str, Any]) -> int:
    decisions = read_review_decisions(session_dir)
    decisions[str(decision["feedback_event_id"])] = decision
    return write_review_decisions(session_dir, decisions)


def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    facts = step.get("state_facts") or {}
    return {
        "total_step": step.get("total_step"),
        "episode_step": step.get("episode_step"),
        "ai_action": step.get("ai_action_name"),
        "ai_subgoal": step.get("ai_subgoal"),
        "human_action": step.get("human_action_name"),
        "reward": step.get("environment_reward"),
        "ai_pos": facts.get("ai_pos"),
        "human_pos": facts.get("human_pos"),
        "ai_held_object": facts.get("ai_held_object"),
        "human_held_object": facts.get("human_held_object"),
        "pot_states": facts.get("pot_states"),
        "condition_features": extract_condition_features(step),
        "candidate_subgoals": step.get("ai_subgoal_candidates") or [],
    }


def recent_steps(
    trajectory: list[dict[str, Any]],
    *,
    total_step: int | None,
    lookback_steps: int,
) -> list[dict[str, Any]]:
    if not trajectory:
        return []
    if total_step is None:
        window = trajectory[-lookback_steps:]
    else:
        start = int(total_step) - lookback_steps
        window = [
            step
            for step in trajectory
            if start <= int(step.get("total_step") or -1) <= int(total_step)
        ]
    return [compact_step(step) for step in window]


def nearby_candidate_events(
    feedback: dict[str, Any],
    candidate_events: list[dict[str, Any]],
    *,
    lookback_steps: int,
) -> list[dict[str, Any]]:
    total_step = feedback.get("total_step")
    events = [
        event
        for event in candidate_events
        if candidate_near_feedback(
            event,
            feedback_total_step=total_step,
            lookback_steps=lookback_steps,
        )
    ]
    return sorted(
        events,
        key=lambda event: (
            int(event.get("end_timestep") or 0),
            float(event.get("confidence") or 0.0),
        ),
        reverse=True,
    )


def nearby_probe_hits(
    feedback: dict[str, Any],
    probe_hits: list[dict[str, Any]],
    *,
    lookback_steps: int,
) -> list[dict[str, Any]]:
    total_step = feedback.get("total_step")
    if total_step is None:
        return []
    start = int(total_step) - lookback_steps
    hits = [
        hit
        for hit in probe_hits
        if start <= int(hit.get("total_step") or -1) <= int(total_step)
    ]
    return sorted(hits, key=lambda hit: int(hit.get("total_step") or 0), reverse=True)


def schema_reviews_for_feedback(
    schema_updates: list[dict[str, Any]],
    feedback_id: str,
) -> list[dict[str, Any]]:
    return [
        update
        for update in schema_updates
        if str(update.get("feedback_event_id")) == feedback_id
    ]


def default_decision_for_item(item: dict[str, Any]) -> dict[str, Any]:
    attribution = item.get("attribution") or {}
    provenance = item.get("provenance") or {}
    condition_features = (
        provenance.get("condition_features")
        or attribution.get("condition_features")
        or item.get("condition_at_feedback")
        or {}
    )
    preferred = provenance.get("preferred_subgoals") or attribution.get("preferred_subgoals") or []
    rejected = provenance.get("rejected_subgoals") or attribution.get("rejected_subgoals") or []
    target_event = attribution.get("target_event")
    needs_clarification = bool(attribution.get("needs_clarification"))
    usable = bool(target_event and preferred and rejected and not needs_clarification)
    return {
        "record_type": "review_decision",
        "feedback_event_id": item.get("feedback_event_id"),
        "feedback_total_step": item.get("feedback_total_step"),
        "feedback_text": item.get("feedback_text"),
        "decision": "accept" if usable else "needs_revision",
        "approved_time_window": attribution.get("target_time_window"),
        "approved_event": target_event,
        "approved_condition_overrides": condition_features,
        "approved_preference": {
            "preferred_subgoals": preferred,
            "rejected_subgoals": rejected,
        },
        "use_for_hu_training": usable,
        "schema_action": "none",
        "approved_schema_update": None,
        "review_notes": "",
    }


def build_review_items(
    session_dir: Path,
    *,
    lookback_steps: int = 50,
) -> list[dict[str, Any]]:
    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    feedback_events = read_jsonl(session_dir / "feedback_events.jsonl")
    candidate_events = read_jsonl(session_dir / "candidate_events.jsonl")
    attributions = by_feedback_id(read_jsonl(session_dir / "attribution_preview.jsonl"))
    provenance = by_feedback_id(read_jsonl(session_dir / "hu_attribution_provenance.jsonl"))
    schema_updates = read_jsonl(session_dir / "schema_updates.jsonl")
    probe_hits = read_jsonl(session_dir / "probe_hits.jsonl")
    existing_decisions = read_review_decisions(session_dir)

    items = []
    for feedback in feedback_events:
        feedback_id = item_id_for_feedback(feedback)
        total_step = feedback.get("total_step")
        condition_step = latest_step_at_or_before(trajectory, total_step)
        attribution = attributions.get(feedback_id, {})
        scoped_candidate_events = attribution.get("candidate_events") or (
            nearby_candidate_events(
                feedback,
                candidate_events,
                lookback_steps=lookback_steps,
            )
        )
        item = {
            "record_type": "review_item",
            "feedback_event_id": feedback_id,
            "feedback_text": feedback.get("feedback_text"),
            "feedback_value": feedback.get("feedback_value"),
            "feedback_role": feedback.get("role"),
            "feedback_total_step": total_step,
            "episode": feedback.get("episode"),
            "episode_step": feedback.get("episode_step"),
            "attribution": attribution,
            "provenance": provenance.get(feedback_id, {}),
            "schema_reviews": schema_reviews_for_feedback(schema_updates, feedback_id),
            "nearby_candidate_events": scoped_candidate_events,
            "nearby_probe_hits": nearby_probe_hits(
                feedback,
                probe_hits,
                lookback_steps=lookback_steps,
            ),
            "recent_steps": recent_steps(
                trajectory,
                total_step=total_step,
                lookback_steps=min(lookback_steps, 20),
            ),
            "condition_at_feedback": (
                extract_condition_features(condition_step) if condition_step else {}
            ),
        }
        item["default_decision"] = default_decision_for_item(item)
        item["existing_decision"] = existing_decisions.get(feedback_id)
        items.append(item)
    return items


def export_review_items(session_dir: Path, *, lookback_steps: int = 50) -> dict[str, int]:
    items = build_review_items(session_dir, lookback_steps=lookback_steps)
    item_count = write_jsonl(session_dir / REVIEW_ITEMS_FILE, items)
    return {
        "review_items": item_count,
        "existing_decisions": len(read_review_decisions(session_dir)),
    }


def parse_json_field(text: str, fallback: Any) -> Any:
    stripped = text.strip()
    if not stripped:
        return fallback
    return json.loads(stripped)


def parse_csv_field(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]
