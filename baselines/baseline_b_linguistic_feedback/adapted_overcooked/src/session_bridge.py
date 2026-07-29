"""Convert attributed human-AI sessions into Baseline B feedback examples."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .feedback_form_classifier import classify_feedback
from .trajectory_featurizer import featurize_trajectory_steps, read_jsonl


VALID_FEEDBACK_TYPES = {"evaluative", "imperative", "descriptive"}
EVENT_FEATURES = {
    "AI_blocked_human_path": {
        "blocks_human_path": 1.0,
        "human_wait_cost": 1.0,
        "frustrates_human": 1.0,
    },
    "AI_ignored_ready_or_nearly_ready_pot": {
        "soup_ready": 1.0,
        "dish_needed_for_ready_soup": 1.0,
        "delays_serving": 1.0,
    },
    "AI_failed_to_prepare_ingredient_while_waiting": {
        "time_cost": 1.0,
        "moves_away_from_needed_object": 1.0,
        "delays_serving": 1.0,
    },
}
POLARITY_SCORES = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}


def feedback_event_id(feedback: dict) -> str:
    return (
        f"{feedback.get('source')}:{feedback.get('total_step')}:"
        f"{feedback.get('timestamp_utc')}"
    )


def _merge_features(*vectors: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for vector in vectors:
        for feature, value in vector.items():
            merged[str(feature)] = merged.get(str(feature), 0.0) + float(value)
    return {feature: value for feature, value in merged.items() if value}


def _steps_in_window(
    trajectory: list[dict],
    target_time_window: list[int] | None,
    feedback_total_step: int,
) -> list[dict]:
    if target_time_window and len(target_time_window) == 2:
        start, end = (int(target_time_window[0]), int(target_time_window[1]))
    else:
        start = end = int(feedback_total_step)
    return [
        step
        for step in trajectory
        if start <= int(step.get("total_step", -1)) <= end
    ]


def build_session_feedback_examples(
    session_dir: str | Path,
    *,
    include_ambiguous: bool = False,
) -> list[dict]:
    """Build provenance-rich examples from converted and attributed JSONL files."""

    session_dir = Path(session_dir)
    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    feedback_events = read_jsonl(session_dir / "feedback_events.jsonl")
    attributions = read_jsonl(session_dir / "attribution_preview.jsonl")
    attribution_by_id = {
        str(item.get("feedback_event_id")): item
        for item in attributions
        if item.get("feedback_event_id")
    }
    examples = []

    for feedback in feedback_events:
        text = feedback.get("feedback_text")
        if feedback.get("role") != "human_language" or not isinstance(text, str) or not text.strip():
            continue
        event_id = feedback_event_id(feedback)
        attribution = attribution_by_id.get(event_id)
        if not attribution:
            continue
        if attribution.get("needs_clarification") and not include_ambiguous:
            continue

        feedback_type = attribution.get("feedback_type")
        if feedback_type not in VALID_FEEDBACK_TYPES:
            feedback_type = classify_feedback(text)
        window = attribution.get("target_time_window")
        steps = _steps_in_window(
            trajectory,
            window,
            int(feedback.get("total_step") or 0),
        )
        target_event = attribution.get("target_event")
        features = _merge_features(
            featurize_trajectory_steps(steps),
            EVENT_FEATURES.get(target_event, {}),
        )
        if not features:
            continue

        digest = hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:12]
        feedback_extra = feedback.get("extra") or {}
        layout = feedback_extra.get("layout")
        if not layout and steps:
            layout = steps[-1].get("layout")
        example = {
            "feedback_id": f"session_{session_dir.name}_{digest}",
            "text": text.strip(),
            "expected_feedback_type": feedback_type,
            "trajectory_features": features,
            "session_id": session_dir.name,
            "timestamp_utc": feedback.get("timestamp_utc"),
            "episode": feedback.get("episode"),
            "total_step": feedback.get("total_step"),
            "layout": layout,
            "source": feedback_extra.get("source") or feedback.get("source"),
            "provenance": {
                "feedback_event_id": event_id,
                "target_time_window": window,
                "target_event": target_event,
                "attribution_confidence": attribution.get("confidence"),
                "needs_clarification": attribution.get("needs_clarification"),
                "online_update_id": feedback_extra.get("update_id"),
                "online_feedback_mode": feedback_extra.get("feedback_mode"),
                "effective_precision": feedback_extra.get("effective_precision"),
                "checkpoint_sha256": feedback_extra.get("checkpoint_sha256"),
                "before_subgoal": feedback_extra.get("before_subgoal"),
                "after_subgoal": feedback_extra.get("after_subgoal"),
                "reference_type": feedback_extra.get("reference_type"),
            },
        }
        polarity = attribution.get("polarity")
        if polarity in POLARITY_SCORES:
            example["attributed_sentiment_score"] = POLARITY_SCORES[polarity]
        examples.append(example)

    return examples
