"""Print a compact behavior/learning summary for one human-AI session."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from durf.baseline.runtime import REPO_ROOT


DEFAULT_SESSIONS_ROOT = REPO_ROOT / "outputs" / "human_ai_sessions"


def latest_session(root: str | Path = DEFAULT_SESSIONS_ROOT) -> Path:
    candidates = [
        path
        for path in Path(root).resolve().iterdir()
        if path.is_dir() and (path / "trajectory.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No sessions with trajectory.csv under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _feedback_records(path: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    invalid = 0
    if not path.is_file():
        return records, invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            invalid += 1
    return records, invalid


def _longest_wait(rows: list[dict]) -> dict:
    best = {"length": 0, "start_step": None, "end_step": None}
    run_start = None
    run_length = 0
    for index, row in enumerate(rows, start=1):
        step = int(row.get("total_step") or index)
        if row.get("ai_subgoal") == "WAIT":
            if run_length == 0:
                run_start = step
            run_length += 1
            if run_length > best["length"]:
                best = {
                    "length": run_length,
                    "start_step": run_start,
                    "end_step": step,
                }
        else:
            run_start = None
            run_length = 0
    return best


def _immediate_stash_repick(rows: list[dict]) -> dict:
    """Count adjacent INTERACT pairs that undo a stash immediately."""

    by_resource: Counter[str] = Counter()
    steps: list[int] = []
    for previous, current in zip(rows, rows[1:]):
        if (
            previous.get("ai_subgoal") == "STASH_HELD_OBJECT"
            and str(previous.get("ai_action")) == "5"
            and str(current.get("ai_subgoal") or "").startswith("GET_")
            and str(current.get("ai_action")) == "5"
        ):
            resource = str(current.get("ai_subgoal"))[4:].lower()
            by_resource[resource] += 1
            steps.append(int(current.get("total_step") or len(steps) + 1))
    return {
        "count": sum(by_resource.values()),
        "by_resource": dict(sorted(by_resource.items())),
        "steps": steps,
    }


def summarize_session(session_dir: str | Path) -> dict:
    session = Path(session_dir).resolve()
    trajectory_path = session / "trajectory.csv"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Missing trajectory.csv: {trajectory_path}")

    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = _json(session / "session_manifest.json")
    feedback, invalid_feedback = _feedback_records(session / "feedback_updates.jsonl")
    return summarize_records(
        session,
        rows,
        manifest,
        feedback,
        invalid_feedback=invalid_feedback,
    )


def summarize_records(
    session: str | Path,
    rows: list[dict],
    manifest: dict,
    feedback: list[dict],
    *,
    invalid_feedback: int = 0,
) -> dict:
    """Summarize already-loaded records; useful for tests and other tooling."""

    session = Path(session).resolve()
    statuses = Counter(str(row.get("status") or "unknown") for row in feedback)
    subgoals = Counter(row.get("ai_subgoal") or "unknown" for row in rows)
    steps = len(rows)
    mode = manifest.get("feedback_mode") or (
        rows[0].get("comfort_feedback_mode") if rows else None
    )
    learning_enabled = manifest.get("online_learning_enabled")
    if learning_enabled is None:
        learning_enabled = mode not in {None, "", "frozen"}
    reward = sum(float(row.get("environment_reward") or 0.0) for row in rows)
    wait = subgoals.get("WAIT", 0)
    wait_streak = _longest_wait(rows)
    stash_repick = _immediate_stash_repick(rows)
    alerts = []
    if mode == "frozen" and feedback:
        alerts.append("FROZEN control: feedback was recorded but not learned.")
    if wait_streak["length"] >= 10:
        alerts.append(
            f"Long WAIT streak: {wait_streak['length']} steps "
            f"({wait_streak['start_step']}-{wait_streak['end_step']})."
        )
    if stash_repick["count"]:
        alerts.append(
            f"Immediate STASH->GET reversal: {stash_repick['count']} times."
        )
    if feedback and not statuses.get("updated"):
        alerts.append("No submitted feedback produced an update.")
    if invalid_feedback:
        alerts.append(f"Invalid feedback JSONL rows: {invalid_feedback}.")

    return {
        "session": str(session),
        "ai_mode": manifest.get("ai_mode") or (rows[0].get("ai_mode") if rows else None),
        "feedback_mode": mode,
        "online_learning_enabled": bool(learning_enabled),
        "teacher_id": manifest.get("teacher_id"),
        "seed": manifest.get("seed"),
        "steps": steps,
        "reward": reward,
        "reward_events": sum(
            float(row.get("environment_reward") or 0.0) != 0.0 for row in rows
        ),
        "wait_steps": wait,
        "wait_fraction": wait / steps if steps else 0.0,
        "longest_wait_streak": wait_streak,
        "immediate_stash_repick": stash_repick,
        "feedback_records": len(feedback),
        "feedback_status_counts": dict(sorted(statuses.items())),
        "updates_applied": statuses.get("updated", 0),
        "alerts": alerts,
    }


def format_summary(summary: dict) -> str:
    streak = summary["longest_wait_streak"]
    statuses = summary["feedback_status_counts"] or {"none": 0}
    lines = [
        f"Session: {summary['session']}",
        (
            f"Mode: {summary['ai_mode']}/{summary['feedback_mode']} | "
            f"online learning: {'ON' if summary['online_learning_enabled'] else 'OFF'}"
        ),
        (
            f"Steps: {summary['steps']} | reward: {summary['reward']:g} "
            f"({summary['reward_events']} events)"
        ),
        (
            f"WAIT: {summary['wait_steps']}/{summary['steps']} "
            f"({summary['wait_fraction']:.1%}) | longest: {streak['length']} "
            f"steps ({streak['start_step']}-{streak['end_step']})"
        ),
        (
            "Immediate STASH->GET reversals: "
            f"{summary['immediate_stash_repick']['count']}"
        ),
        f"Feedback statuses: {json.dumps(statuses, ensure_ascii=False, sort_keys=True)}",
    ]
    lines.extend(f"WARNING: {alert}" for alert in summary["alerts"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", type=Path, help="Session directory; default: latest")
    parser.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    session = args.session.resolve() if args.session else latest_session(args.sessions_root)
    summary = summarize_session(session)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
