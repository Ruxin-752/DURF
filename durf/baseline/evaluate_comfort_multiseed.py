"""Select comfort strength on dev seeds, then report untouched test seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from durf.baseline.comfort_reward import ADAPTED_ROOT
from durf.baseline.evaluate_comfort_subgoal import rollout
from durf.baseline.runtime import RING_TOMATO_ONION_H0_LAYOUT


def _summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one rollout is required")
    total_steps = sum(row["steps"] for row in rows)
    wait_steps = sum(row.get("subgoal_counts", {}).get("WAIT", 0) for row in rows)
    return {
        "n": len(rows),
        "mean_soup_reward": mean(row["soup_reward"] for row in rows),
        "mean_comfort_per_step": mean(row["comfort_per_step"] for row in rows),
        "mean_discomfort_rate": mean(row["discomfort_rate"] for row in rows),
        "worst_soup_reward": min(row["soup_reward"] for row in rows),
        "worst_discomfort_rate": max(row["discomfort_rate"] for row in rows),
        "longest_wait_streak": max(row["longest_wait_streak"] for row in rows),
        "wait_rate": wait_steps / total_steps if total_steps else 0.0,
        "zero_soup_runs": sum(row["soup_reward"] <= 0 for row in rows),
    }


def _quantile(values: list[float], q: float) -> float:
    """Deterministic linearly interpolated quantile (stdlib-only)."""

    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate(
    *,
    layout: str,
    horizon: int,
    dev_seeds: list[int],
    test_seeds: list[int],
    lambdas: list[float],
) -> dict:
    all_seeds = sorted(set(dev_seeds + test_seeds))
    baseline = {
        seed: rollout("h0_rule", layout, seed, horizon, lambda_pref=0.0)
        for seed in all_seeds
    }
    candidates = []
    for value in lambdas:
        rows = [
            rollout("comfort_subgoal", layout, seed, horizon, lambda_pref=value)
            for seed in dev_seeds
        ]
        summary = _summary(rows)
        baseline_soup = mean(baseline[seed]["soup_reward"] for seed in dev_seeds)
        summary["soup_retention"] = (
            summary["mean_soup_reward"] / baseline_soup if baseline_soup else 1.0
        )
        comfort_deltas = [
            row["comfort_per_step"] - baseline[seed]["comfort_per_step"]
            for seed, row in zip(dev_seeds, rows)
        ]
        summary["comfort_delta_p10"] = _quantile(comfort_deltas, 0.10)
        baseline_dev = _summary([baseline[seed] for seed in dev_seeds])
        constraints = {
            "soup_retention": summary["soup_retention"] >= 0.95,
            "worst_soup_retention": summary["worst_soup_reward"]
            >= 0.90 * baseline_dev["worst_soup_reward"],
            "worst_discomfort": summary["worst_discomfort_rate"]
            <= baseline_dev["worst_discomfort_rate"],
            "wait_streak": summary["longest_wait_streak"] <= 3,
            "no_zero_soup": summary["zero_soup_runs"] == 0,
        }
        summary["hard_constraints"] = constraints
        summary["hard_constraints_passed"] = all(constraints.values())
        candidates.append({"lambda_pref": value, "summary": summary, "runs": rows})

    eligible = [
        candidate
        for candidate in candidates
        if candidate["summary"]["hard_constraints_passed"]
    ]
    if not eligible:
        raise ValueError(
            "no lambda satisfies soup, worst-discomfort, WAIT, and no-zero hard constraints"
        )
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["summary"]["comfort_delta_p10"],
            candidate["summary"]["mean_comfort_per_step"],
            -candidate["summary"]["mean_discomfort_rate"],
            -candidate["lambda_pref"],
        ),
    )
    selected_lambda = float(selected["lambda_pref"])
    test_comfort = [
        rollout(
            "comfort_subgoal",
            layout,
            seed,
            horizon,
            lambda_pref=selected_lambda,
        )
        for seed in test_seeds
    ]
    test_baseline = [baseline[seed] for seed in test_seeds]
    comfort_summary = _summary(test_comfort)
    baseline_summary = _summary(test_baseline)
    comfort_summary["soup_retention"] = (
        comfort_summary["mean_soup_reward"] / baseline_summary["mean_soup_reward"]
        if baseline_summary["mean_soup_reward"]
        else 1.0
    )
    return {
        "layout": layout,
        "horizon": horizon,
        "selection_rule": (
            "maximize dev per-seed comfort delta p10 subject to mean/worst soup, "
            "worst discomfort, WAIT streak, and no-zero-soup hard constraints"
        ),
        "dev_seeds": dev_seeds,
        "test_seeds": test_seeds,
        "dev_candidates": candidates,
        "selected_lambda": selected_lambda,
        "untouched_test": {
            "baseline": baseline_summary,
            "comfort": comfort_summary,
            "baseline_runs": test_baseline,
            "comfort_runs": test_comfort,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=RING_TOMATO_ONION_H0_LAYOUT)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--dev-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ADAPTED_ROOT / "outputs" / "route2" / "comfort_multiseed_strict.json",
    )
    args = parser.parse_args()
    result = evaluate(
        layout=args.layout,
        horizon=args.horizon,
        dev_seeds=args.dev_seeds,
        test_seeds=args.test_seeds,
        lambdas=args.lambdas,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected lambda on dev: {result['selected_lambda']}")
    print(f"Untouched test: {result['untouched_test']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
