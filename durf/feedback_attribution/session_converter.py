"""Convert current pygame session CSV logs into first-pass JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import as_bool, as_float, as_int, as_json, read_csv, write_jsonl
from .schemas import feedback_event, trajectory_step


def convert_trajectory(session_dir: Path) -> list[dict]:
    rows = read_csv(session_dir / "trajectory.csv")
    records = []
    for row in rows:
        records.append(
            trajectory_step(
                source="trajectory.csv",
                timestamp_utc=row.get("timestamp_utc", ""),
                episode=as_int(row.get("episode")) or 0,
                episode_step=as_int(row.get("episode_step")) or 0,
                total_step=as_int(row.get("total_step")) or 0,
                layout=row.get("layout", ""),
                ai_action=as_int(row.get("ai_action")),
                ai_action_name=row.get("ai_action_name"),
                human_action=as_int(row.get("human_action")),
                human_action_name=row.get("human_action_name"),
                environment_reward=as_float(row.get("environment_reward")),
                episode_reward=as_float(row.get("episode_reward")),
                done=as_bool(row.get("done")),
                ai_subgoal=row.get("ai_subgoal") or None,
                ai_event=row.get("ai_event") or None,
                ai_condition_features=as_json(
                    row.get("ai_condition_features_json")
                )
                or {},
                ai_subgoal_candidates=as_json(
                    row.get("ai_subgoal_candidates_json")
                )
                or [],
                task_decision=as_json(row.get("task_decision_json")) or {},
                coordination_decision=as_json(
                    row.get("coordination_decision_json")
                )
                or {},
                state_facts=as_json(row.get("state_after_json")),
                extra={
                    "predict_ms": as_float(row.get("predict_ms")),
                    "environment_step_ms": as_float(row.get("environment_step_ms")),
                    "state_before": as_json(row.get("state_before_json")),
                },
            )
        )
    return records


def convert_feedback(session_dir: Path) -> list[dict]:
    records = []
    for row in read_csv(session_dir / "feedback.csv"):
        records.append(
            feedback_event(
                source="feedback.csv",
                timestamp_utc=row.get("timestamp_utc", ""),
                episode=as_int(row.get("episode")) or 0,
                episode_step=as_int(row.get("episode_step")) or 0,
                total_step=as_int(row.get("total_step")) or 0,
                feedback_text=None,
                feedback_value=as_int(row.get("feedback")),
                role="human_scalar",
                extra={
                    "last_ai_action": as_int(row.get("last_ai_action")),
                    "last_ai_action_name": row.get("last_ai_action_name"),
                },
            )
        )

    for row in read_csv(session_dir / "chat_messages.csv"):
        if row.get("role") != "user":
            continue
        records.append(
            feedback_event(
                source="chat_messages.csv",
                timestamp_utc=row.get("timestamp_utc", ""),
                episode=as_int(row.get("episode")) or 0,
                episode_step=as_int(row.get("episode_step")) or 0,
                total_step=as_int(row.get("total_step")) or 0,
                feedback_text=row.get("content"),
                feedback_value=None,
                role="human_language",
                extra={"layout": row.get("layout")},
            )
        )

    for row in read_csv(session_dir / "annotations.csv"):
        records.append(
            feedback_event(
                source="annotations.csv",
                timestamp_utc=row.get("timestamp_utc", ""),
                episode=as_int(row.get("episode")) or 0,
                episode_step=as_int(row.get("episode_step")) or 0,
                total_step=as_int(row.get("total_step")) or 0,
                feedback_text=row.get("annotation_text"),
                feedback_value=None,
                role="human_annotation",
                extra={
                    "layout": row.get("layout"),
                    "scope_hint": row.get("annotation_scope_hint"),
                    "last_ai_action": as_int(row.get("last_ai_action")),
                    "last_ai_action_name": row.get("last_ai_action_name"),
                    "episode_reward": as_float(row.get("episode_reward")),
                    "state_facts": as_json(row.get("state_json")),
                    "recent_replay": as_json(row.get("recent_replay_json")),
                },
            )
        )

    records.sort(key=lambda item: (item["total_step"], item["timestamp_utc"]))
    return records


def convert_session(session_dir: Path) -> dict[str, int]:
    trajectory = convert_trajectory(session_dir)
    feedback = convert_feedback(session_dir)
    return {
        "trajectory": write_jsonl(session_dir / "trajectory.jsonl", trajectory),
        "feedback": write_jsonl(session_dir / "feedback_events.jsonl", feedback),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()

    counts = convert_session(args.session)
    print(f"Wrote {counts['trajectory']} trajectory records")
    print(f"Wrote {counts['feedback']} feedback records")
    print(f"Session: {args.session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
