"""Compare matched Hu-general and Hu-user simulation rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


NEGATIVE_COORDINATION = (
    "blocking me",
    "in my way",
    "step aside",
    "didn't need to move",
    "should have stayed",
)
POSITIVE_COORDINATION = (
    "squeeze past",
    "held still",
    "let me through",
    "finish what you were doing",
    "keep going",
    "stick with it",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def summarize(root: Path) -> dict[str, Any]:
    rewards: list[float] = []
    decisions: dict[str, dict[str, Any]] = {}
    feedback = Counter()
    events = Counter()
    session_dirs = sorted({path.parent for path in root.rglob("trajectory.csv")})

    for session in session_dirs:
        with (session / "trajectory.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_episode: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_episode.setdefault(row["episode"], []).append(row)
            raw_decision = row.get("coordination_decision_json", "")
            if raw_decision and raw_decision != "{}":
                decision = json.loads(raw_decision)
                decisions[decision["decision_id"]] = decision
        rewards.extend(float(episode[-1]["episode_reward"]) for episode in by_episode.values())

        chat_path = session / "chat_messages.csv"
        if chat_path.exists():
            with chat_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    content = row["content"].lower()
                    if any(phrase in content for phrase in NEGATIVE_COORDINATION):
                        feedback["negative_coordination"] += 1
                    elif any(phrase in content for phrase in POSITIVE_COORDINATION):
                        feedback["positive_coordination"] += 1
                    else:
                        feedback["task_or_other"] += 1

        for event in _read_jsonl(session / "candidate_events.jsonl"):
            events[event["event_type"]] += 1

    selected = Counter(decision["selected"] for decision in decisions.values())
    decision_total = sum(selected.values())
    return {
        "sessions": len(session_dirs),
        "episodes": len(rewards),
        "episode_rewards": rewards,
        "mean_episode_reward": mean(rewards) if rewards else None,
        "sample_sd_episode_reward": _sample_sd(rewards),
        "coordination_decisions": decision_total,
        "coordination_selection": dict(sorted(selected.items())),
        "yield_rate": selected["YIELD"] / decision_total if decision_total else None,
        "feedback": dict(sorted(feedback.items())),
        "candidate_events": dict(sorted(events.items())),
    }


def _relative_change(before: float, after: float) -> float | None:
    return (after - before) / before if before else None


def compare(general: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    g_feedback = general["feedback"]
    u_feedback = user["feedback"]
    g_events = general["candidate_events"]
    u_events = user["candidate_events"]
    return {
        "reward_delta": user["mean_episode_reward"] - general["mean_episode_reward"],
        "reward_relative_change": _relative_change(
            general["mean_episode_reward"], user["mean_episode_reward"]
        ),
        "yield_rate_delta": user["yield_rate"] - general["yield_rate"],
        "negative_coordination_feedback_delta": (
            u_feedback.get("negative_coordination", 0)
            - g_feedback.get("negative_coordination", 0)
        ),
        "negative_coordination_feedback_relative_change": _relative_change(
            g_feedback.get("negative_coordination", 0),
            u_feedback.get("negative_coordination", 0),
        ),
        "positive_coordination_feedback_delta": (
            u_feedback.get("positive_coordination", 0)
            - g_feedback.get("positive_coordination", 0)
        ),
        "blocked_path_event_delta": (
            u_events.get("AI_blocked_human_path", 0)
            - g_events.get("AI_blocked_human_path", 0)
        ),
        "blocked_path_event_relative_change": _relative_change(
            g_events.get("AI_blocked_human_path", 0),
            u_events.get("AI_blocked_human_path", 0),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-root", type=Path, required=True)
    parser.add_argument("--user-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args()

    general = summarize(args.general_root)
    user = summarize(args.user_root)
    result = {
        "status": "matched_sim_runtime_comparison",
        "general": general,
        "user": user,
        "comparison": compare(general, user),
        "limitations": [
            "The simulated human uses deterministic persona templates.",
            "Only four matched random seeds and eight episodes per condition were run.",
            "Candidate-event counts are automatic detector outputs, not human gold labels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.matrix:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        matrix["task_and_collaboration"]["hu_online_comparison"] = result
        matrix["task_and_collaboration"].pop("reason", None)
        limitations = matrix.get("limitations", [])
        limitations = [
            item for item in limitations if not item.startswith("No matched online rollout")
        ]
        limitations.extend(result["limitations"])
        matrix["limitations"] = limitations
        args.matrix.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    print(json.dumps(result["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
