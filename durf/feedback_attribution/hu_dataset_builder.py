"""Build Hu provenance records and pairwise subgoal preference samples.

This module separates two representations:

* provenance: full evidence chain for researchers and audits.
* training samples: minimal pairwise labels consumed by Hu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .condition_features import (
    extract_condition_features,
    latest_step_at_or_before,
    null_condition_features,
)
from .io_utils import read_jsonl, write_jsonl
from .sample_builder import feedback_event_id
from .subgoal_preferences import infer_subgoal_preferences


CONDITION_TOKENS = {
    "when",
    "while",
    "if",
    "because",
    "after",
    "before",
    "near",
    "closer",
    "holding",
    "has",
    "have",
    "last",
    "needed",
}


def clean_event_hint(raw_name: str, event_hint: str) -> str | None:
    lower = raw_name.lower()
    if "dish" in lower or "plate" in lower:
        return "AI_missed_plate_pickup_opportunity"
    if "counter" in lower or "table" in lower or "prepared" in lower:
        return "AI_missed_useful_counter_object"
    if "closer" in lower or "faster" in lower or "relative" in lower:
        return "AI_missed_labor_division_opportunity"
    if "yield" in lower or "block" in lower or "path" in lower:
        return "AI_blocked_human_path"
    if "ingredient" in lower:
        return "AI_missed_useful_ingredient_pickup"
    return event_hint or None


def feedback_map(feedback_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {feedback_event_id(feedback): feedback for feedback in feedback_events}


def matching_candidate_event(attribution: dict[str, Any]) -> dict[str, Any] | None:
    target_event = attribution.get("target_event")
    target_window = attribution.get("target_time_window")
    for event in attribution.get("candidate_events") or []:
        if event.get("event_type") != target_event:
            continue
        if not target_window:
            return event
        if [
            event.get("start_timestep"),
            event.get("end_timestep"),
        ] == target_window:
            return event
    return None


def condition_for_attribution(
    attribution: dict[str, Any],
    trajectory: list[dict[str, Any]],
    feedback: dict[str, Any] | None,
) -> dict[str, bool | None]:
    condition = attribution.get("condition_features")
    if isinstance(condition, dict) and condition:
        merged = null_condition_features()
        merged.update({key: condition.get(key) for key in merged})
        return merged

    event = matching_candidate_event(attribution)
    if event and isinstance(event.get("condition_features"), dict):
        merged = null_condition_features()
        merged.update({key: event["condition_features"].get(key) for key in merged})
        return merged

    target_step = None
    if attribution.get("target_time_window"):
        target_step = attribution["target_time_window"][-1]
    elif feedback:
        target_step = feedback.get("total_step")
    return extract_condition_features(
        latest_step_at_or_before(trajectory, target_step)
    )


def build_provenance_record(
    *,
    attribution: dict[str, Any],
    feedback: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    user_id: str,
) -> dict[str, Any]:
    event = matching_candidate_event(attribution)
    condition_features = condition_for_attribution(attribution, trajectory, feedback)
    preferred_subgoals, rejected_subgoals = infer_subgoal_preferences(
        target_event=attribution.get("target_event"),
        preferred_subgoals=attribution.get("preferred_subgoals"),
        rejected_subgoals=attribution.get("rejected_subgoals"),
    )
    return {
        "record_type": "hu_attribution_provenance",
        "user_id": user_id,
        "feedback_event_id": attribution.get("feedback_event_id"),
        "feedback_text": feedback.get("feedback_text") if feedback else None,
        "feedback_role": feedback.get("role") if feedback else None,
        "layout": (feedback.get("extra") or {}).get("layout") if feedback else None,
        "target_event": attribution.get("target_event"),
        "event_actor": event.get("actor") if event else None,
        "event_valence": event.get("event_valence") if event else None,
        "target_time_window": attribution.get("target_time_window"),
        "event_evidence": event.get("evidence") if event else {},
        "condition_features": condition_features,
        "preferred_subgoals": preferred_subgoals,
        "rejected_subgoals": rejected_subgoals,
        "confidence": attribution.get("confidence"),
        "needs_clarification": attribution.get("needs_clarification"),
        "clarification_question": attribution.get("clarification_question"),
        "proposed_schema_update": attribution.get("proposed_schema_update") or [],
        "notes": attribution.get("notes"),
    }


def build_training_samples(
    provenance_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for provenance in provenance_records:
        if provenance.get("needs_clarification"):
            continue
        if not provenance.get("target_event"):
            continue
        if provenance.get("event_actor") not in {None, "ai"}:
            continue
        preferred = provenance.get("preferred_subgoals") or []
        rejected = provenance.get("rejected_subgoals") or []
        if not preferred or not rejected:
            continue
        for preferred_subgoal in preferred:
            for rejected_subgoal in rejected:
                if preferred_subgoal == rejected_subgoal:
                    continue
                samples.append(
                    {
                        "record_type": "hu_pairwise_subgoal_preference",
                        "sample_id": (
                            f"{provenance.get('user_id')}:"
                            f"{provenance.get('feedback_event_id')}:"
                            f"{preferred_subgoal}>{rejected_subgoal}"
                        ),
                        "user_id": provenance.get("user_id"),
                        "layout": provenance.get("layout"),
                        "condition_features": provenance.get("condition_features"),
                        "preferred_subgoal": preferred_subgoal,
                        "rejected_subgoal": rejected_subgoal,
                        "source_event": provenance.get("target_event"),
                        "source_feedback_id": provenance.get("feedback_event_id"),
                    }
                )
    return samples


def normalize_raw_schema_update(update: Any) -> dict[str, Any]:
    if isinstance(update, dict):
        raw_name = (
            update.get("event")
            or update.get("event_type")
            or update.get("proposed_event")
            or update.get("name")
            or update.get("field_name")
            or update.get("schema_update")
            or update.get("description")
            or ""
        )
        raw_update = update
    else:
        raw_name = str(update)
        raw_update = update
    return {
        "raw_update": raw_update,
        "raw_name": str(raw_name),
    }


def split_event_condition_hint(raw_name: str) -> dict[str, Any]:
    """Turn an LLM schema suggestion into an audit-friendly review proposal.

    LLM suggestions often bundle behavior and condition, e.g.
    ``AI_should_get_dish_when_human_has_last_ingredient``.  We keep the raw
    suggestion for traceability, but make the review boundary explicit:

    event = behavior pattern, condition = state facts, preference = subgoals.
    """

    normalized = raw_name.strip()
    lower = normalized.lower()
    boundary = None
    for token in ("_when_", "_while_", "_if_", "_because_", "_after_", "_before_"):
        index = lower.find(token)
        if index >= 0:
            boundary = (index, token)
            break

    if boundary is None:
        event_hint = normalized
        condition_hint = ""
    else:
        index, token = boundary
        event_hint = normalized[:index]
        condition_hint = normalized[index + len(token) :]
    event_hint = clean_event_hint(normalized, event_hint)

    words = {
        word
        for word in lower.replace("-", "_").split("_")
        if word and word in CONDITION_TOKENS
    }
    is_condition_bundled = boundary is not None or bool(words)
    return {
        "event_hint": event_hint or None,
        "condition_hint": condition_hint or None,
        "condition_bundled_in_name": is_condition_bundled,
        "review_note": (
            "Do not accept this raw name directly as an event. Split behavior "
            "into candidate_event, state facts into condition_features, and "
            "subgoal ranking into preference labels."
            if is_condition_bundled
            else "Review whether this is a behavior event, a condition key, or a subgoal preference."
        ),
    }


def build_schema_review_records(
    provenance_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for provenance in provenance_records:
        for update in provenance.get("proposed_schema_update") or []:
            normalized = normalize_raw_schema_update(update)
            split_hint = split_event_condition_hint(normalized["raw_name"])
            records.append(
                {
                    "record_type": "schema_update_review",
                    "status": "needs_human_review",
                    "feedback_event_id": provenance.get("feedback_event_id"),
                    "feedback_text": provenance.get("feedback_text"),
                    "target_event": provenance.get("target_event"),
                    "condition_features_at_feedback": provenance.get("condition_features"),
                    "preferred_subgoals": provenance.get("preferred_subgoals") or [],
                    "rejected_subgoals": provenance.get("rejected_subgoals") or [],
                    "raw_schema_update": normalized["raw_update"],
                    "raw_name": normalized["raw_name"],
                    "suggested_decomposition": {
                        **split_hint,
                        "event_rule": "candidate_event should name only the AI behavior pattern.",
                        "condition_rule": "condition_features should hold state facts such as human_has_last_needed_ingredient.",
                        "preference_rule": "preferred/rejected subgoals are separate Hu labels.",
                    },
                }
            )
    return records


def build_hu_dataset(session_dir: Path, *, user_id: str) -> dict[str, int]:
    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    feedback_events = read_jsonl(session_dir / "feedback_events.jsonl")
    attributions = read_jsonl(session_dir / "attribution_preview.jsonl")
    feedback_by_id = feedback_map(feedback_events)

    provenance_records = [
        build_provenance_record(
            attribution=attribution,
            feedback=feedback_by_id.get(attribution.get("feedback_event_id")),
            trajectory=trajectory,
            user_id=user_id,
        )
        for attribution in attributions
    ]
    training_samples = build_training_samples(provenance_records)

    provenance_count = write_jsonl(
        session_dir / "hu_attribution_provenance.jsonl",
        provenance_records,
    )
    sample_count = write_jsonl(
        session_dir / "hu_subgoal_preferences.jsonl",
        training_samples,
    )

    schema_updates = build_schema_review_records(provenance_records)
    schema_update_count = write_jsonl(session_dir / "schema_updates.jsonl", schema_updates)

    summary = {
        "session": str(session_dir),
        "user_id": user_id,
        "provenance_records": provenance_count,
        "hu_training_samples": sample_count,
        "schema_updates": schema_update_count,
    }
    (session_dir / "hu_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--user-id", default="PILOT01")
    args = parser.parse_args()
    summary = build_hu_dataset(args.session, user_id=args.user_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
