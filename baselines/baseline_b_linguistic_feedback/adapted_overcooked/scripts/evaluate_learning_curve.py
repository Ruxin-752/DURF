"""Online learning-curve evaluation (aligned with ``aaai_model_evaluation``).

The paper evaluates a learner by streaming feedback one interaction at a time
and, after each update, re-deciding on a fixed benchmark set to measure a
"percent of max score" learning curve, averaged over runs with confidence
intervals, and compared across the literal / pseudo-pragmatic learners.

We reproduce that protocol in the offline Overcooked setting:

- the benchmark set is the probe states (each probe = one decision problem);
- after each feedback we record the deterministic probe accuracy and the
  paper-style *sampled* expected accuracy (draw weight hypotheses, argmax);
- feedback order is shuffled across seeds and results are aggregated to a
  mean +/- 95% CI curve;
- literal vs pseudo-pragmatic are compared against a random-action baseline.

This is an evaluation-protocol reproduction only; it performs Bayesian
inference, not model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import (  # noqa: E402
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_PRAGMATIC_PRECISION,
    DEFAULT_PRAGMATIC_VALENCE,
    DEFAULT_PRECISION_SCALE,
    DEFAULT_VALENCE_SCALE,
)
from src.feedback_observations import build_feedback_observations  # noqa: E402
from src.feature_schema import (  # noqa: E402
    collect_action_feature_library,
    load_features,
    read_json,
)
from src.feedback_form_classifier import classify_feedback  # noqa: E402
from src.probe_evaluator import (  # noqa: E402
    DEFAULT_PROBE_STATES_PATH,
    evaluate_probes,
    evaluate_probes_sampled,
    load_probe_states,
)
from src.reward_weight_model import BayesianRewardLearner  # noqa: E402


def _t_value(df: int) -> float:
    """Two-sided 95% t multiplier (falls back to the normal approximation)."""

    if df <= 0:
        return 0.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def random_baseline_accuracy(probes: list[dict]) -> float:
    """Expected accuracy of picking a uniformly random available action."""

    fractions = []
    for probe in probes:
        actions = probe.get("available_actions", [])
        acceptable = probe.get("acceptable_actions") or [probe.get("expected_action")]
        if not actions:
            continue
        hits = sum(1 for action in actions if action.get("action_id") in acceptable)
        fractions.append(hits / len(actions))
    return sum(fractions) / len(fractions) if fractions else 0.0


def single_run(
    feedback_examples: list[dict],
    probes: list[dict],
    features: list[str],
    action_library: dict[str, dict[str, float]],
    *,
    mode: str,
    seed: int,
    n_samples: int,
    max_steps: int | None,
) -> tuple[list[float], list[float]]:
    """One online pass: return (deterministic_accuracy, sampled_accuracy) per step."""

    order = list(feedback_examples)
    random.Random(seed).shuffle(order)
    if max_steps is not None:
        order = order[:max_steps]

    pragmatic = mode == "pseudopragmatic"
    learner = BayesianRewardLearner(
        features,
        valence_scale=DEFAULT_VALENCE_SCALE,
        precision_scale=DEFAULT_PRECISION_SCALE,
        pragmatic_valence=DEFAULT_PRAGMATIC_VALENCE if pragmatic else None,
        pragmatic_precision=DEFAULT_PRAGMATIC_PRECISION if pragmatic else None,
    )

    acc_curve = [evaluate_probes(probes, learner.as_dict())["overall_accuracy"]]
    exp_curve = [
        evaluate_probes_sampled(probes, learner.belief, n_samples=n_samples, seed=seed)[
            "expected_accuracy"
        ]
    ]

    for feedback in order:
        feedback_type = feedback.get("expected_feedback_type") or classify_feedback(
            feedback.get("text")
        )
        for sub in build_feedback_observations(
            feedback, feedback_type=feedback_type, action_feature_library=action_library
        ):
            learner.update(sub["target_features"], sub["valence"])
        acc_curve.append(evaluate_probes(probes, learner.as_dict())["overall_accuracy"])
        exp_curve.append(
            evaluate_probes_sampled(
                probes, learner.belief, n_samples=n_samples, seed=seed
            )["expected_accuracy"]
        )

    return acc_curve, exp_curve


def _aggregate(curves: list[list[float]]) -> dict:
    matrix = np.asarray(curves, dtype=float)  # [seeds, steps]
    n = matrix.shape[0]
    mean = matrix.mean(axis=0)
    if n > 1:
        sem = matrix.std(axis=0, ddof=1) / math.sqrt(n)
    else:
        sem = np.zeros_like(mean)
    half = _t_value(n - 1) * sem
    return {
        "mean": mean.tolist(),
        "ci_low": (mean - half).tolist(),
        "ci_high": (mean + half).tolist(),
    }


def run_learning_curve(
    *,
    feedback_examples: list[dict],
    probes: list[dict],
    features: list[str],
    mode: str,
    seeds: list[int],
    n_samples: int,
    max_steps: int | None,
) -> dict:
    action_library = collect_action_feature_library(probes)
    acc_runs, exp_runs = [], []
    for seed in seeds:
        acc_curve, exp_curve = single_run(
            feedback_examples,
            probes,
            features,
            action_library,
            mode=mode,
            seed=seed,
            n_samples=n_samples,
            max_steps=max_steps,
        )
        acc_runs.append(acc_curve)
        exp_runs.append(exp_curve)
    return {
        "mode": mode,
        "seeds": seeds,
        "steps": list(range(len(acc_runs[0]))),
        "deterministic_accuracy": _aggregate(acc_runs),
        "sampled_expected_accuracy": _aggregate(exp_runs),
    }


def _summary_rows(curve: dict, milestones: list[int]) -> list[tuple[int, float, float]]:
    steps = curve["steps"]
    mean = curve["deterministic_accuracy"]["mean"]
    half_low = curve["deterministic_accuracy"]["ci_low"]
    rows = []
    for milestone in milestones:
        if milestone < len(steps):
            rows.append((milestone, mean[milestone], mean[milestone] - half_low[milestone]))
    if steps:
        rows.append((steps[-1], mean[-1], mean[-1] - half_low[-1]))
    return rows


def print_summary(results: dict, random_baseline: float) -> None:
    print(f"Random-action baseline accuracy: {random_baseline * 100:.1f}%")
    milestones = [1, 5, 10]
    for mode, curve in results.items():
        print(f"\n[{mode}] deterministic probe accuracy (mean +/- 95% CI half-width):")
        for step, mean, half in _summary_rows(curve, milestones):
            label = "final" if step == curve["steps"][-1] else f"step {step}"
            print(f"  {label:>8}: {mean * 100:5.1f}% +/- {half * 100:4.1f}")


def write_csv(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "step",
                "det_acc_mean",
                "det_acc_ci_low",
                "det_acc_ci_high",
                "sampled_acc_mean",
                "sampled_acc_ci_low",
                "sampled_acc_ci_high",
            ]
        )
        for mode, curve in results.items():
            det = curve["deterministic_accuracy"]
            samp = curve["sampled_expected_accuracy"]
            for i, step in enumerate(curve["steps"]):
                writer.writerow(
                    [
                        mode,
                        step,
                        det["mean"][i],
                        det["ci_low"][i],
                        det["ci_high"][i],
                        samp["mean"][i],
                        samp["ci_low"][i],
                        samp["ci_high"][i],
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--seeds", type=int, default=8, help="Number of shuffled runs.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["literal", "pseudopragmatic"],
        choices=["literal", "pseudopragmatic"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "learning_curve.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs" / "learning_curve.csv")
    args = parser.parse_args()

    feedback_examples = read_json(args.feedback)
    probes = load_probe_states(args.probe_states)
    features = load_features()
    seeds = list(range(args.seeds))

    results = {}
    for mode in args.modes:
        results[mode] = run_learning_curve(
            feedback_examples=feedback_examples,
            probes=probes,
            features=features,
            mode=mode,
            seeds=seeds,
            n_samples=args.n_samples,
            max_steps=args.max_steps,
        )

    random_baseline = random_baseline_accuracy(probes)
    payload = {
        "random_baseline_accuracy": random_baseline,
        "n_seeds": len(seeds),
        "n_samples": args.n_samples,
        "curves": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, results)

    print_summary(results, random_baseline)
    print(f"\nCurve JSON: {args.output}")
    print(f"Curve CSV:  {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
