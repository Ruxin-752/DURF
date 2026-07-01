"""Probe-state evaluation for the learned Overcooked feature weights."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .feature_schema import read_json
from .reward_weight_model import RewardWeightModel


DEFAULT_PROBE_STATES_PATH = Path(__file__).resolve().parents[1] / "data" / "probe_states.json"


def load_probe_states(path: str | Path = DEFAULT_PROBE_STATES_PATH) -> list[dict]:
    probes = read_json(path)
    if not isinstance(probes, list):
        raise ValueError(f"Probe states must be a JSON list: {path}")
    return probes


def choose_action(
    probe: dict,
    model: RewardWeightModel,
) -> dict:
    scored_actions = []
    for action in probe.get("available_actions", []):
        features = action.get("features", {})
        score = model.score_action(features)
        scored_actions.append(
            {
                "action_id": action.get("action_id"),
                "score": score,
                "features": features,
            }
        )
    if not scored_actions:
        raise ValueError(f"Probe has no available actions: {probe.get('probe_id')}")
    scored_actions.sort(key=lambda item: item["score"], reverse=True)
    return {
        "chosen_action": scored_actions[0]["action_id"],
        "chosen_score": scored_actions[0]["score"],
        "action_scores": scored_actions,
    }


def evaluate_probes(
    probes: list[dict],
    weights: dict[str, float],
) -> dict:
    model = RewardWeightModel(weights)
    results = []
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for probe in probes:
        choice = choose_action(probe, model)
        acceptable_actions = probe.get("acceptable_actions") or [probe.get("expected_action")]
        correct = choice["chosen_action"] in acceptable_actions
        category = probe.get("category", "unknown")
        category_counts[category]["total"] += 1
        category_counts[category]["correct"] += int(correct)
        results.append(
            {
                "probe_id": probe.get("probe_id"),
                "probe_name": probe.get("probe_name"),
                "category": category,
                "expected_action": probe.get("expected_action"),
                "acceptable_actions": acceptable_actions,
                "chosen_action": choice["chosen_action"],
                "chosen_score": choice["chosen_score"],
                "correct": correct,
                "action_scores": choice["action_scores"],
            }
        )

    total = len(results)
    correct = sum(1 for result in results if result["correct"])
    category_accuracy = {
        category: {
            **counts,
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for category, counts in sorted(category_counts.items())
    }
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "results": results,
    }
