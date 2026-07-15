"""Probe-state evaluation for the learned Overcooked feature weights."""

from __future__ import annotations

from collections import defaultdict
import math
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
    *,
    tie_tolerance: float = 1e-9,
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
    scored_actions.sort(
        key=lambda item: (-item["score"], str(item["action_id"])),
    )
    best_score = scored_actions[0]["score"]
    tied_actions = [
        item["action_id"]
        for item in scored_actions
        if math.isclose(item["score"], best_score, abs_tol=tie_tolerance)
    ]
    return {
        "chosen_action": tied_actions[0] if len(tied_actions) == 1 else None,
        "chosen_score": best_score,
        "is_tie": len(tied_actions) > 1,
        "tied_actions": tied_actions,
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
        acceptable_scores = [
            action["score"]
            for action in choice["action_scores"]
            if action["action_id"] in acceptable_actions
        ]
        unacceptable_scores = [
            action["score"]
            for action in choice["action_scores"]
            if action["action_id"] not in acceptable_actions
        ]
        expected_score = max(acceptable_scores) if acceptable_scores else None
        best_incorrect_score = max(unacceptable_scores) if unacceptable_scores else None
        margin = (
            expected_score - best_incorrect_score
            if expected_score is not None and best_incorrect_score is not None
            else None
        )
        correct = not choice["is_tie"] and choice["chosen_action"] in acceptable_actions
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
                "is_tie": choice["is_tie"],
                "tied_actions": choice["tied_actions"],
                "expected_score": expected_score,
                "best_incorrect_score": best_incorrect_score,
                "margin": margin,
                "correct": correct,
                "action_scores": choice["action_scores"],
            }
        )

    total = len(results)
    correct = sum(1 for result in results if result["correct"])
    tie_count = sum(1 for result in results if result["is_tie"])
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
        "tie_count": tie_count,
        "mean_margin": (
            sum(result["margin"] for result in results if result["margin"] is not None)
            / sum(1 for result in results if result["margin"] is not None)
            if any(result["margin"] is not None for result in results)
            else None
        ),
        "category_accuracy": category_accuracy,
        "results": results,
    }
