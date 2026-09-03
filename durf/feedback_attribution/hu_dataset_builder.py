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
    decision_condition_features,
    latest_step_at_or_before,
    null_condition_features,
)
from .io_utils import read_jsonl, write_jsonl
from .review_io import read_review_decisions
from .sample_builder import feedback_event_id
from .subgoal_preferences import (
    COORDINATION_SUBGOALS,
    infer_subgoal_preferences,
    resolve_useful_ingredient,
)

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


def apply_review_decision(
    attribution: dict[str, Any],
    review_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the effective attribution used to build Hu provenance.

    Review files remain separate audit artifacts. When a decision exists, its
    approved values override the automatic guess for dataset construction only.
    """
    effective = dict(attribution)
    if not review_decision:
        return effective

    approved_preference = review_decision.get("approved_preference") or {}
    if review_decision.get("approved_event"):
        effective["target_event"] = review_decision["approved_event"]
    if review_decision.get("approved_time_window"):
        effective["target_time_window"] = review_decision["approved_time_window"]
    if isinstance(review_decision.get("approved_condition_overrides"), dict):
        effective["condition_features"] = review_decision[
            "approved_condition_overrides"
        ]
    if "preferred_subgoals" in approved_preference:
        effective["preferred_subgoals"] = approved_preference[
            "preferred_subgoals"
        ]
    if "rejected_subgoals" in approved_preference:
        effective["rejected_subgoals"] = approved_preference[
            "rejected_subgoals"
        ]
    # A human-approved training record no longer inherits an automatic
    # clarification flag.
    if review_decision.get("use_for_hu_training"):
        effective["needs_clarification"] = False
    return effective


def condition_for_attribution(
    attribution: dict[str, Any],
    trajectory: list[dict[str, Any]],
    feedback: dict[str, Any] | None,
) -> dict[str, Any]:
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
    # A label is about the decision made at the target step, so use the
    # pre-action condition the runtime actually scored with (see
    # decision_condition_features), not the post-action snapshot.
    return decision_condition_features(
        latest_step_at_or_before(trajectory, target_step)
    )


def decision_step_for_attribution(
    attribution: dict[str, Any],
    event: dict[str, Any] | None,
    feedback: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The trajectory step whose decision a label is about: the event's first
    step when there is one, else the start of the attributed window, else the
    feedback step."""
    target_step = None
    if event and event.get("start_timestep") is not None:
        target_step = event.get("start_timestep")
    elif attribution.get("target_time_window"):
        target_step = attribution["target_time_window"][0]
    elif feedback:
        target_step = feedback.get("total_step")
    if target_step is None:
        return None
    return latest_step_at_or_before(trajectory, int(target_step))


def build_provenance_record(
    *,
    attribution: dict[str, Any],
    feedback: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    user_id: str,
    review_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attribution = apply_review_decision(attribution, review_decision)
    event = matching_candidate_event(attribution)
    condition_features = condition_for_attribution(attribution, trajectory, feedback)
    preferred_subgoals, rejected_subgoals = infer_subgoal_preferences(
        target_event=attribution.get("target_event"),
        preferred_subgoals=attribution.get("preferred_subgoals"),
        rejected_subgoals=attribution.get("rejected_subgoals"),
        observed_subgoal=event.get("related_subgoal") if event else None,
        alternative_subgoals=event.get("alternative_subgoals") if event else None,
        event_valence=event.get("event_valence") if event else None,
    )
    # Attribution-level names that the runtime never offers as candidates
    # (GET_USEFUL_INGREDIENT) are resolved against the candidate set the
    # decision actually had; a pair that cannot be expressed there is dropped
    # and the unresolved name is kept in the provenance for audit.
    attributed_preferred = list(preferred_subgoals)
    attributed_rejected = list(rejected_subgoals)
    decision_step = decision_step_for_attribution(attribution, event, feedback, trajectory)
    decision_candidates = (
        (decision_step or {}).get("ai_subgoal_candidates") or []
    )
    preferred_subgoals, unresolved_preferred = resolve_useful_ingredient(
        preferred_subgoals, decision_candidates
    )
    rejected_subgoals, unresolved_rejected = resolve_useful_ingredient(
        rejected_subgoals, decision_candidates
    )
    unresolved_subgoals = [*unresolved_preferred, *unresolved_rejected]
    event_evidence = event.get("evidence") if event else {}
    explicit_level = (
        event_evidence.get("decision_level")
        if isinstance(event_evidence, dict)
        else None
    )
    all_labeled_subgoals = {
        *preferred_subgoals,
        *rejected_subgoals,
    }
    if explicit_level in {"task", "coordination"}:
        decision_level = explicit_level
    elif all_labeled_subgoals and all_labeled_subgoals.issubset(
        set(COORDINATION_SUBGOALS)
    ):
        decision_level = "coordination"
    else:
        decision_level = "task"
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
        "event_evidence": event_evidence,
        "decision_level": decision_level,
        "preference_source": attribution.get("preference_source"),
        "source_decision_id": (
            event_evidence.get("decision_id")
            if isinstance(event_evidence, dict)
            else None
        ),
        "condition_features": condition_features,
        "preferred_subgoals": preferred_subgoals,
        "rejected_subgoals": rejected_subgoals,
        "attributed_preferred_subgoals": attributed_preferred,
        "attributed_rejected_subgoals": attributed_rejected,
        "unresolved_subgoals": unresolved_subgoals,
        "decision_candidate_set": [
            candidate.get("subgoal") if isinstance(candidate, dict) else candidate
            for candidate in decision_candidates
        ],
        "confidence": attribution.get("confidence"),
        "needs_clarification": attribution.get("needs_clarification"),
        "clarification_question": attribution.get("clarification_question"),
        "proposed_schema_update": attribution.get("proposed_schema_update") or [],
        "notes": attribution.get("notes"),
        "reviewed": bool(review_decision),
        "review_decision": (
            review_decision.get("decision") if review_decision else None
        ),
        "use_for_hu_training": (
            bool(review_decision.get("use_for_hu_training"))
            if review_decision
            else None
        ),
        "source": attribution.get("source", "unknown"),
        "label_status": attribution.get("label_status", "automatic"),
        "protocol_version": attribution.get("protocol_version", "protocol-v2"),
    }


def build_training_samples(
    provenance_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for provenance in provenance_records:
        if (
            provenance.get("reviewed")
            and not provenance.get("use_for_hu_training")
        ):
            continue
        if provenance.get("needs_clarification") and not provenance.get(
            "preference_source"
        ):
            continue
        # A direct policy preference may be valid without an observed event.
        # Keep event-grounded and event-free labels distinguishable in the
        # provenance and report them separately during evaluation.
        if not provenance.get("target_event") and not provenance.get(
            "preference_source"
        ):
            continue
        if provenance.get("event_actor") not in {None, "ai"}:
            continue
        preferred = provenance.get("preferred_subgoals") or []
        rejected = provenance.get("rejected_subgoals") or []
        # Coordination is a binary domain: a single-sided label identifies the
        # other option as the rejected/preferred side by elimination.
        if provenance.get("decision_level") == "coordination":
            if preferred and not rejected:
                rejected = [
                    subgoal
                    for subgoal in COORDINATION_SUBGOALS
                    if subgoal not in preferred
                ]
            elif rejected and not preferred:
                preferred = [
                    subgoal
                    for subgoal in COORDINATION_SUBGOALS
                    if subgoal not in rejected
                ]
        if not preferred or not rejected:
            continue
        for preferred_subgoal in preferred:
            for rejected_subgoal in rejected:
                if preferred_subgoal == rejected_subgoal:
                    continue
                pair_is_coordination = {
                    preferred_subgoal,
                    rejected_subgoal,
                }.issubset(set(COORDINATION_SUBGOALS))
                decision_level = provenance.get("decision_level")
                if decision_level not in {"task", "coordination"}:
                    decision_level = (
                        "coordination" if pair_is_coordination else "task"
                    )
                if (
                    decision_level == "coordination"
                    and not pair_is_coordination
                ):
                    continue
                if (
                    decision_level == "task"
                    and (
                        preferred_subgoal in COORDINATION_SUBGOALS
                        or rejected_subgoal in COORDINATION_SUBGOALS
                    )
                ):
                    continue
                samples.append(
                    {
                        "record_type": "hu_pairwise_subgoal_preference",
                        "sample_id": (
                            f"{provenance.get('user_id')}:"
                            f"{provenance.get('feedback_event_id')}:"
                            f"{decision_level}:"
                            f"{preferred_subgoal}>{rejected_subgoal}"
                        ),
                        "user_id": provenance.get("user_id"),
                        "layout": provenance.get("layout"),
                        "condition_features": provenance.get("condition_features"),
                        "decision_level": decision_level,
                        "preferred_subgoal": preferred_subgoal,
                        "rejected_subgoal": rejected_subgoal,
                        "source_event": provenance.get("target_event"),
                        "preference_source": provenance.get("preference_source"),
                        "source_feedback_id": provenance.get("feedback_event_id"),
                        "source_decision_id": provenance.get(
                            "source_decision_id"
                        ),
                        "source": provenance.get("source", "unknown"),
                        "label_status": (
                            "reviewed"
                            if provenance.get("reviewed")
                            else "automatic"
                        ),
                        "label_source": (
                            "human_review"
                            if provenance.get("reviewed")
                            else "automatic_attribution"
                        ),
                        "protocol_version": provenance.get(
                            "protocol_version", "protocol-v2"
                        ),
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
    review_by_id = read_review_decisions(session_dir)

    provenance_records = [
        build_provenance_record(
            attribution=attribution,
            feedback=feedback_by_id.get(attribution.get("feedback_event_id")),
            trajectory=trajectory,
            user_id=user_id,
            review_decision=review_by_id.get(
                str(attribution.get("feedback_event_id"))
            ),
        )
        for attribution in attributions
    ]
    training_samples = build_training_samples(provenance_records)

    # --- protocol-v2: user isolation check ---
    foreign_user_ids = {
        sample["user_id"]
        for sample in training_samples
        if sample.get("user_id") != user_id
    }
    if foreign_user_ids:
        raise ValueError(
            f"User isolation violation in {session_dir}: "
            f"expected user_id={user_id}, "
            f"found foreign user_ids={sorted(foreign_user_ids)}. "
            f"Real human feedback must not be mixed across participants."
        )
    # --- end user isolation check ---

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
        "protocol_version": "protocol-v2",
        "provenance_records": provenance_count,
        "hu_training_samples": sample_count,
        "schema_updates": schema_update_count,
        "review_decisions_consumed": sum(
            bool(record.get("reviewed")) for record in provenance_records
        ),
        "reviewed_training_records": sum(
            bool(
                record.get("reviewed")
                and record.get("use_for_hu_training")
            )
            for record in provenance_records
        ),
        "event_grounded_training_records": sum(
            1
            for sample in training_samples
            if sample.get("source_event") and not sample.get("preference_source")
        ),
        "direct_preference_training_records": sum(
            1 for sample in training_samples if sample.get("preference_source")
        ),
        "source_distribution": {
            "synthetic_sim_human": sum(
                1 for r in provenance_records
                if r.get("source", "") == "synthetic_sim_human"
            ),
            "human_feedback": sum(
                1 for r in provenance_records
                if r.get("source", "") == "human_feedback"
            ),
        },
        "label_status_distribution": {
            "reviewed": sum(
                1 for r in provenance_records
                if r.get("reviewed")
            ),
            "automatic": sum(
                1 for r in provenance_records
                if not r.get("reviewed")
            ),
        },
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
