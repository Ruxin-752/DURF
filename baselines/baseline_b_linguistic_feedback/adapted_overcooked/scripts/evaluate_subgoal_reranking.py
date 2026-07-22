"""Evaluate subgoal re-ranking with the reward learned from language feedback.

This closes the loop that makes Baseline B *subgoal-compatible*:

1. learn a reward-weight belief from the linguistic feedback (same pipeline);
2. featurize each of H0's feasible subgoals in a probe state;
3. re-rank the subgoals with the learned weights and check that the
   human-comfortable subgoal wins.

It reports learned vs zero vs hand-authored-prior weights so we can see that
the language feedback (not just the prior) drives the subgoal choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import run_pipeline  # noqa: E402
from src.feature_schema import (  # noqa: E402
    DEFAULT_WEIGHTS_PATH,
    empty_weights,
    load_features,
    load_weights,
    read_json,
    write_json,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_reranker import evaluate_subgoal_probes  # noqa: E402


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "feedback_examples.json"
DEFAULT_SUBGOAL_PROBES_PATH = ROOT / "data" / "subgoal_probe_states.json"


def print_summary(evaluation: dict, label: str) -> None:
    accuracy = evaluation["overall_accuracy"] * 100
    print(
        f"[{label}] subgoal accuracy: "
        f"{evaluation['correct']}/{evaluation['total']} = {accuracy:.1f}% "
        f"(ties={evaluation['tie_count']})"
    )
    for result in evaluation["results"]:
        mark = "OK" if result["correct"] else "FAIL"
        chosen = result["chosen_subgoal"] or "TIE"
        print(
            f"  [{mark}] {result['probe_id']}: "
            f"chosen={chosen} acceptable={result['acceptable_subgoals']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--subgoal-probes", type=Path, default=DEFAULT_SUBGOAL_PROBES_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--mode",
        choices=("literal", "pseudopragmatic"),
        default="literal",
    )
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    feedback = read_json(args.feedback)
    probe_states = load_probe_states(args.probe_states)
    subgoal_probes = read_json(args.subgoal_probes)
    initial_weights = load_weights(args.weights)

    pipeline = run_pipeline(
        feedback_examples=feedback,
        probe_states=probe_states,
        initial_weights=initial_weights,
        pragmatic_valence=-30.0 if args.mode == "pseudopragmatic" else None,
        pragmatic_precision=2.0 if args.mode == "pseudopragmatic" else None,
    )
    learned_weights = pipeline["learned_weights"]

    learned_eval = evaluate_subgoal_probes(
        subgoal_probes, learned_weights, lambda_pref=args.lambda_pref
    )
    zero_eval = evaluate_subgoal_probes(
        subgoal_probes, empty_weights(load_features()), lambda_pref=args.lambda_pref
    )
    initial_eval = evaluate_subgoal_probes(
        subgoal_probes, initial_weights, lambda_pref=args.lambda_pref
    )

    result = {
        "mode": args.mode,
        "lambda_pref": args.lambda_pref,
        "learned": learned_eval,
        "zero_weights": zero_eval,
        "initial_weights": initial_eval,
    }
    if args.output:
        write_json(args.output, result)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(learned_eval, "learned")
        print_summary(zero_eval, "zero")
        print_summary(initial_eval, "initial")

    return 0 if learned_eval["correct"] == learned_eval["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
