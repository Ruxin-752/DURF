"""Probe-state evaluation for the learned Overcooked feature weights."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .belief_model import GaussianBelief, score_action
from .feature_schema import read_json


DEFAULT_PROBE_STATES_PATH = Path(__file__).resolve().parents[1] / "data" / "probe_states.json"


def load_probe_states(path: str | Path = DEFAULT_PROBE_STATES_PATH) -> list[dict]:
    probes = read_json(path)
    if not isinstance(probes, list):
        raise ValueError(f"Probe states must be a JSON list: {path}")
    return probes


def choose_action(
    probe: dict,
    weights: dict[str, float],
    *,
    tie_tolerance: float = 1e-9,
) -> dict:
    scored_actions = []
    for action in probe.get("available_actions", []):
        features = action.get("features", {})
        score = score_action(weights, features)
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
    results = []
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for probe in probes:
        choice = choose_action(probe, weights)
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


def evaluate_probes_sampled(
    probes: list[dict],
    belief: GaussianBelief,
    *,
    n_samples: int = 500,
    seed: int = 0,
) -> dict:
    """Paper-style stochastic action selection under the posterior belief.

    Mirrors the original ``BaseAgent.execute_trajectories``: draw ``n_samples``
    weight hypotheses from the belief, pick the argmax action under each, and
    report how often the chosen action is acceptable (expected accuracy) plus
    the per-action selection probability.
    """

    rng = np.random.default_rng(seed)
    samples = belief.sample(n_samples, rng)
    feature_index = {feature: position for position, feature in enumerate(belief.features)}

    results = []
    expected_correct = 0.0
    for probe in probes:
        actions = probe.get("available_actions", [])
        if not actions:
            raise ValueError(f"Probe has no available actions: {probe.get('probe_id')}")

        action_matrix = np.zeros((len(actions), len(belief.features)), dtype=float)
        for row, action in enumerate(actions):
            for feature, value in action.get("features", {}).items():
                position = feature_index.get(str(feature))
                if position is not None:
                    action_matrix[row, position] = float(value)

        # values[sample, action]
        values = samples @ action_matrix.T
        chosen_per_sample = np.argmax(values, axis=1)
        counts = np.bincount(chosen_per_sample, minlength=len(actions))
        selection_probability = counts / n_samples

        acceptable_actions = probe.get("acceptable_actions") or [probe.get("expected_action")]
        acceptable_prob = float(
            sum(
                selection_probability[row]
                for row, action in enumerate(actions)
                if action.get("action_id") in acceptable_actions
            )
        )
        expected_correct += acceptable_prob
        top_row = int(np.argmax(counts))
        results.append(
            {
                "probe_id": probe.get("probe_id"),
                "category": probe.get("category", "unknown"),
                "expected_action": probe.get("expected_action"),
                "acceptable_actions": acceptable_actions,
                "most_selected_action": actions[top_row].get("action_id"),
                "acceptable_probability": acceptable_prob,
                "selection_probability": {
                    action.get("action_id"): float(selection_probability[row])
                    for row, action in enumerate(actions)
                },
            }
        )

    total = len(results)
    return {
        "n_samples": n_samples,
        "seed": seed,
        "expected_accuracy": expected_correct / total if total else 0.0,
        "total": total,
        "results": results,
    }
