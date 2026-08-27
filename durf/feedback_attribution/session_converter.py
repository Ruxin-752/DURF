"""Convert current pygame session CSV logs into first-pass JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .io_utils import (
    as_bool,
    as_float,
    as_int,
    as_json,
    read_csv,
    read_jsonl,
    write_jsonl,
)
from .schemas import feedback_event, trajectory_step


CONVERTER_SCHEMA_VERSION = 2


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session_id(session_dir: Path) -> str:
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict) and manifest.get("session_id"):
            return str(manifest["session_id"])
    return session_dir.name


def _stable_feedback_event_id(
    session_dir: Path,
    row: dict,
    *,
    kind: str,
    occurrence: int,
) -> str:
    """Keep live IDs, and deterministically backfill IDs for legacy CSVs."""

    existing = str(row.get("feedback_event_id") or "").strip()
    if existing:
        return existing
    identity = {
        "session_id": _session_id(session_dir),
        "kind": kind,
        "timestamp_utc": row.get("timestamp_utc") or "",
        "episode": as_int(row.get("episode")) or 0,
        "episode_step": as_int(row.get("episode_step")) or 0,
        "total_step": as_int(row.get("total_step")) or 0,
        "content": (row.get("content") or row.get("feedback") or "").strip(),
        "occurrence": occurrence,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "feedback_" + hashlib.sha256(encoded).hexdigest()[:24]


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
                state_facts=as_json(row.get("state_after_json")),
                extra={
                    "predict_ms": as_float(row.get("predict_ms")),
                    "environment_step_ms": as_float(row.get("environment_step_ms")),
                    "state_before": as_json(row.get("state_before_json")),
                    "ai_subgoal": row.get("ai_subgoal"),
                    "ai_event": row.get("ai_event"),
                    "ai_mode": row.get("ai_mode"),
                    "comfort_feedback_mode": row.get("comfort_feedback_mode"),
                },
            )
        )
    return records


def convert_feedback(
    session_dir: Path,
    *,
    return_audit: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    records = []
    online_updates = read_jsonl(session_dir / "feedback_updates.jsonl")
    used_update_indices: set[int] = set()
    match_counts: Counter[str] = Counter()

    def legacy_key(record: dict, *, csv_row: bool = False) -> tuple[int, int, str]:
        episode = (
            as_int(record.get("episode"))
            if csv_row
            else int(record.get("episode", 0) or 0)
        )
        step = (
            as_int(record.get("total_step"))
            if csv_row
            else int(record.get("total_step", 0) or 0)
        )
        text_field = "content" if csv_row else "text"
        text = str(record.get(text_field) or "").strip()
        return int(episode or 0), int(step or 0), text

    def matching_update(row: dict) -> tuple[dict | None, str | None]:
        event_id = str(row.get("feedback_event_id") or "").strip()
        if event_id:
            for index, update in enumerate(online_updates):
                if index in used_update_indices:
                    continue
                if str(update.get("feedback_event_id") or "").strip() == event_id:
                    used_update_indices.add(index)
                    match_counts["feedback_event_id"] += 1
                    return update, "feedback_event_id"

        key = legacy_key(row, csv_row=True)
        for index, update in enumerate(online_updates):
            if index in used_update_indices:
                continue
            if legacy_key(update) == key:
                used_update_indices.add(index)
                match_counts["episode_step_text"] += 1
                return update, "episode_step_text"
        match_counts["unmatched_chat"] += 1
        return None, None

    scalar_occurrences: Counter[tuple] = Counter()
    for row_index, row in enumerate(read_csv(session_dir / "feedback.csv"), start=1):
        identity = (
            row.get("timestamp_utc"),
            row.get("episode"),
            row.get("episode_step"),
            row.get("total_step"),
            row.get("feedback"),
        )
        scalar_occurrences[identity] += 1
        event_id = _stable_feedback_event_id(
            session_dir,
            row,
            kind="human_scalar",
            occurrence=scalar_occurrences[identity],
        )
        records.append(
            feedback_event(
                feedback_event_id=event_id,
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
                    "source_row": row_index,
                },
            )
        )

    language_occurrences: Counter[tuple] = Counter()
    chat_rows = read_csv(session_dir / "chat_messages.csv")
    language_rows = 0
    for row_index, row in enumerate(chat_rows, start=1):
        if row.get("role") != "user":
            continue
        language_rows += 1
        identity = (
            row.get("timestamp_utc"),
            row.get("episode"),
            row.get("episode_step"),
            row.get("total_step"),
            row.get("content"),
        )
        language_occurrences[identity] += 1
        event_id = _stable_feedback_event_id(
            session_dir,
            row,
            kind="human_language",
            occurrence=language_occurrences[identity],
        )
        update, match_rule = matching_update(row)
        online = update or {}
        raw_reference_types = online.get("reference_types")
        if isinstance(raw_reference_types, str):
            reference_types = [raw_reference_types]
        elif isinstance(raw_reference_types, list):
            reference_types = [
                str(value) for value in raw_reference_types if str(value).strip()
            ]
        else:
            reference_types = []
        singular_reference = online.get("reference_type")
        if singular_reference and str(singular_reference) not in reference_types:
            reference_types.append(str(singular_reference))
        records.append(
            feedback_event(
                feedback_event_id=event_id,
                source="chat_messages.csv",
                timestamp_utc=row.get("timestamp_utc", ""),
                episode=as_int(row.get("episode")) or 0,
                episode_step=as_int(row.get("episode_step")) or 0,
                total_step=as_int(row.get("total_step")) or 0,
                feedback_text=row.get("content"),
                feedback_value=None,
                role="human_language",
                extra={
                    "layout": row.get("layout"),
                    "source": online.get("source", "human_live"),
                    "update_id": online.get("update_id"),
                    "feedback_mode": online.get("mode"),
                    "effective_precision": max(
                        (
                            float(item.get("effective_precision", 0.0))
                            for item in online.get("observations", [])
                        ),
                        default=None,
                    ),
                    "checkpoint_sha256": online.get("checkpoint_sha256"),
                    "before_subgoal": online.get("before_subgoal"),
                    "after_subgoal": online.get("after_subgoal"),
                    "reference_type": (
                        str(singular_reference)
                        if singular_reference
                        else reference_types[0] if len(reference_types) == 1 else None
                    ),
                    "reference_types": reference_types,
                    "update_match_rule": match_rule,
                    "update_match_status": "matched" if update else "unmatched",
                    "source_row": row_index,
                },
            )
        )

    records.sort(key=lambda item: (item["total_step"], item["timestamp_utc"]))
    update_key_counts = Counter(legacy_key(update) for update in online_updates)
    audit = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "session_id": _session_id(session_dir),
        "online_update_count": len(online_updates),
        "human_language_count": language_rows,
        "match_counts": dict(sorted(match_counts.items())),
        "matched_update_count": len(used_update_indices),
        "unmatched_update_count": len(online_updates) - len(used_update_indices),
        "unmatched_update_ids": [
            update.get("update_id") or update.get("feedback_event_id") or f"row_{index + 1}"
            for index, update in enumerate(online_updates)
            if index not in used_update_indices
        ],
        "duplicate_legacy_update_keys": sum(
            1 for count in update_key_counts.values() if count > 1
        ),
        "generated_feedback_event_ids": sum(
            1
            for record in records
            if str(record.get("feedback_event_id") or "").startswith("feedback_")
        ),
        "unique_feedback_event_ids": len(
            {record.get("feedback_event_id") for record in records}
        ),
    }
    if return_audit:
        return records, audit
    return records


def convert_session(session_dir: Path) -> dict:
    trajectory = convert_trajectory(session_dir)
    feedback, feedback_audit = convert_feedback(session_dir, return_audit=True)
    trajectory_path = session_dir / "trajectory.jsonl"
    feedback_path = session_dir / "feedback_events.jsonl"
    counts = {
        "trajectory": write_jsonl(trajectory_path, trajectory),
        "feedback": write_jsonl(feedback_path, feedback),
    }
    audit = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "session_id": _session_id(session_dir),
        "inputs": {
            name: {
                "path": name,
                "sha256": _file_sha256(session_dir / name),
            }
            for name in (
                "trajectory.csv",
                "feedback.csv",
                "chat_messages.csv",
                "feedback_updates.jsonl",
            )
        },
        "outputs": {
            "trajectory.jsonl": {
                "count": counts["trajectory"],
                "sha256": _file_sha256(trajectory_path),
            },
            "feedback_events.jsonl": {
                "count": counts["feedback"],
                "sha256": _file_sha256(feedback_path),
            },
        },
        "feedback_matching": feedback_audit,
    }
    audit_path = session_dir / "session_conversion_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts["audit"] = str(audit_path)
    return counts


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
