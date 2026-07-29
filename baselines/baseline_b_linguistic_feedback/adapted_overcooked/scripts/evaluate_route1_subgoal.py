"""Make Route 1 visible with grouped held-out subgoal and type ablations.

This evaluates the full paper-style path:

    text -> feedback-form classification -> type-specific grounding
         -> Literal/PseudoPragmatic Gaussian update -> subgoal re-ranking

``oracle`` uses corpus annotations; ``inferred`` removes gold type, grounding,
and valence labels while retaining the runtime fact of which just-executed
subgoal the human is commenting on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import empty_weights, load_features, read_json  # noqa: E402
from src.route1_online import OnlineRoute1Learner  # noqa: E402
from src.subgoal_featurizer import featurize_subgoal  # noqa: E402
from src.subgoal_reranker import choose_subgoal  # noqa: E402
from src.subgoal_teacher import load_gold_weights  # noqa: E402


DEFAULT_FEEDBACK = ROOT / "data" / "synthetic_feedback.validated.json"
DEFAULT_FROZEN = ROOT / "outputs" / "route2" / "learned_comfort_weights.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "route1_subgoal_ablation.json"
FEEDBACK_TYPES = ("evaluative", "imperative", "descriptive")
ROLE_REFERENCE_TYPE = {
    "praise_best": "trajectory",
    "criticize_alt": "action_spatial",
    "command_best": "action_spatial",
    "describe_alt": "feature",
    "describe_behavior_alt": "action_behavioral",
}


def _make_group_folds(group_ids: list[str], n_folds: int, seed: int) -> list[dict]:
    """Torch-free grouped folds for the Route 1-only evaluator."""

    groups = sorted(set(group_ids))
    random.Random(seed).shuffle(groups)
    n_folds = max(1, min(n_folds, len(groups)))
    return [
        {"fold": fold, "test_groups": groups[fold::n_folds]}
        for fold in range(n_folds)
    ]


def _unique_scenarios(examples: list[dict]) -> list[dict]:
    scenarios: dict[str, dict] = {}
    for example in examples:
        scenarios.setdefault(
            example["group_id"],
            {
                "group_id": example["group_id"],
                "context": example["context"],
                "feasible_subgoals": example["feasible_subgoals"],
                "acceptable_subgoals": example.get("acceptable_subgoals"),
                "expected_subgoal": example.get("expected_subgoal"),
            },
        )
    return list(scenarios.values())


def _accuracy(weights: dict[str, float], scenarios: list[dict], lambda_pref: float) -> float:
    correct = 0
    for scenario in scenarios:
        choice = choose_subgoal(
            weights,
            scenario["context"],
            scenario["feasible_subgoals"],
            lambda_pref=lambda_pref,
        )
        acceptable = scenario.get("acceptable_subgoals") or [scenario["expected_subgoal"]]
        correct += int(not choice["is_tie"] and choice["chosen_subgoal"] in acceptable)
    return correct / len(scenarios) if scenarios else 0.0


def _runtime_decision(example: dict) -> dict:
    """Reconstruct information available when feedback comments on a subgoal."""

    ranking = [
        {
            "subgoal": subgoal,
            "features": featurize_subgoal(example["context"], subgoal),
        }
        for subgoal in example["feasible_subgoals"]
    ]
    return {
        "chosen_subgoal": example.get("referenced_subgoal"),
        "ranking": ranking,
    }


def _fit_route1(
    examples: list[dict],
    features: list[str],
    *,
    mode: str,
    interpretation: str,
    feedback_type: str | None,
) -> tuple[dict[str, float], dict]:
    learner = OnlineRoute1Learner(features, mode=mode)
    traces = []
    selected = [
        example
        for example in examples
        if feedback_type is None
        or example.get("expected_feedback_type") == feedback_type
    ]
    for example in selected:
        traces.append(
            learner.update(
                example["text"],
                decision=_runtime_decision(example),
                oracle_feedback=example,
                interpretation=interpretation,
            )
        )
    updated = [trace for trace in traces if trace["status"] == "updated"]
    type_correct = [
        trace.get("feedback_type") == example.get("expected_feedback_type")
        for trace, example in zip(traces, selected)
    ]
    reference_correct = []
    for trace, example in zip(traces, selected):
        expected_reference = example.get("reference_type") or ROLE_REFERENCE_TYPE.get(
            example.get("role")
        )
        predicted = [
            value for value in trace.get("reference_types", []) if value is not None
        ]
        if expected_reference is not None and predicted:
            reference_correct.append(expected_reference in predicted)
    return learner.weights(), {
        "examples": len(selected),
        "updated": len(updated),
        "grounding_rate": len(updated) / len(selected) if selected else 0.0,
        "type_accuracy": (
            sum(type_correct) / len(type_correct)
            if interpretation == "inferred" and type_correct
            else None
        ),
        "reference_type_accuracy": (
            sum(reference_correct) / len(reference_correct)
            if interpretation == "inferred" and reference_correct
            else None
        ),
    }


def evaluate(
    feedback: list[dict],
    *,
    n_folds: int,
    seed: int,
    lambda_pref: float,
    frozen_weights: dict[str, float],
) -> dict:
    features = load_features()
    groups = [example["group_id"] for example in feedback]
    folds = _make_group_folds(groups, n_folds=n_folds, seed=seed)
    gold = load_gold_weights()
    zero = empty_weights(features)
    runs = []

    for fold in folds:
        test_groups = set(fold["test_groups"])
        train = [example for example in feedback if example["group_id"] not in test_groups]
        test = [example for example in feedback if example["group_id"] in test_groups]
        scenarios = _unique_scenarios(test)
        fold_result = {
            "fold": fold["fold"],
            "test_groups": len(test_groups),
            "test_scenarios": len(scenarios),
            "baselines": {
                "zero": _accuracy(zero, scenarios, lambda_pref),
                "gold": _accuracy(gold, scenarios, lambda_pref),
                "route2_frozen": _accuracy(frozen_weights, scenarios, lambda_pref),
            },
            "route1": {},
        }
        for mode in ("route1-literal", "route1-pseudopragmatic"):
            for interpretation in ("oracle", "inferred"):
                for feedback_type in (None, *FEEDBACK_TYPES):
                    label = feedback_type or "all"
                    weights, diagnostics = _fit_route1(
                        train,
                        features,
                        mode=mode,
                        interpretation=interpretation,
                        feedback_type=feedback_type,
                    )
                    fold_result["route1"][f"{mode}/{interpretation}/{label}"] = {
                        "accuracy": _accuracy(weights, scenarios, lambda_pref),
                        **diagnostics,
                    }
        runs.append(fold_result)

    keys = list(runs[0]["route1"]) if runs else []
    summary = {
        "baselines": {
            key: mean(run["baselines"][key] for run in runs)
            for key in ("zero", "gold", "route2_frozen")
        },
        "route1": {
            key: {
                "accuracy": mean(run["route1"][key]["accuracy"] for run in runs),
                "grounding_rate": mean(
                    run["route1"][key]["grounding_rate"] for run in runs
                ),
                "type_accuracy": (
                    mean(
                        run["route1"][key]["type_accuracy"]
                        for run in runs
                        if run["route1"][key]["type_accuracy"] is not None
                    )
                    if any(
                        run["route1"][key]["type_accuracy"] is not None for run in runs
                    )
                    else None
                ),
                "reference_type_accuracy": (
                    mean(
                        run["route1"][key]["reference_type_accuracy"]
                        for run in runs
                        if run["route1"][key]["reference_type_accuracy"] is not None
                    )
                    if any(
                        run["route1"][key]["reference_type_accuracy"] is not None
                        for run in runs
                    )
                    else None
                ),
            }
            for key in keys
        },
    }
    return {
        "n_folds": len(folds),
        "seed": seed,
        "lambda_pref": lambda_pref,
        "summary": summary,
        "folds": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--frozen-weights", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = evaluate(
        read_json(args.feedback),
        n_folds=args.n_folds,
        seed=args.seed,
        lambda_pref=args.lambda_pref,
        frozen_weights=read_json(args.frozen_weights),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Mean grouped held-out subgoal accuracy:")
    for label, accuracy in result["summary"]["baselines"].items():
        print(f"  {label:24s} {accuracy * 100:5.1f}%")
    for label, metrics in result["summary"]["route1"].items():
        print(
            f"  {label:48s} {metrics['accuracy'] * 100:5.1f}% "
            f"(grounded {metrics['grounding_rate'] * 100:5.1f}%)"
        )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
