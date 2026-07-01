"""Evaluate static Overcooked probe states with feature weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import DEFAULT_WEIGHTS_PATH, load_weights, write_json  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, evaluate_probes, load_probe_states  # noqa: E402


def print_summary(evaluation: dict) -> None:
    correct = evaluation["correct"]
    total = evaluation["total"]
    accuracy = evaluation["overall_accuracy"] * 100
    print(f"Probe Accuracy: {correct}/{total} = {accuracy:.1f}%")
    for result in evaluation["results"]:
        mark = "OK" if result["correct"] else "FAIL"
        print(
            f"[{mark}] {result['probe_id']}: "
            f"chosen={result['chosen_action']} expected={result['expected_action']} "
            f"score={result['chosen_score']:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    probes = load_probe_states(args.probe_states)
    weights = load_weights(args.weights)
    evaluation = evaluate_probes(probes, weights)

    if args.output:
        write_json(args.output, evaluation)
    if args.json:
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    else:
        print_summary(evaluation)
    return 0 if evaluation["correct"] == evaluation["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
