"""Run Baseline B linguistic feedback weight learning and probe evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import (  # noqa: E402
    DEFAULT_WEIGHTS_PATH,
    collect_action_feature_library,
    load_weights,
    read_json,
    write_json,
)
from src.feedback_form_classifier import classify_feedback  # noqa: E402
from src.overcooked_grounding import ground_feedback  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, evaluate_probes, load_probe_states  # noqa: E402
from src.reward_weight_model import RewardWeightModel  # noqa: E402
from src.sentiment_extractor import desired_action_sentiment, extract_sentiment  # noqa: E402


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "feedback_examples.json"


def effective_sentiment_score(
    *,
    feedback_type: str,
    sentiment: dict,
    target_features: dict[str, float],
    feedback: dict,
) -> float:
    if feedback_type == "imperative" and feedback.get("target_action"):
        return desired_action_sentiment(feedback_type, target_features)
    return float(sentiment["sentiment_score"])


def run_pipeline(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    initial_weights: dict[str, float],
    learning_rate: float,
) -> dict:
    model = RewardWeightModel(initial_weights, learning_rate=learning_rate)
    action_library = collect_action_feature_library(probe_states)
    updates = []

    for feedback in feedback_examples:
        feedback_type = classify_feedback(feedback.get("text"))
        sentiment = extract_sentiment(feedback.get("text"))
        grounding = ground_feedback(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_library,
        )
        target_features = grounding["target_features"]
        sentiment_score = effective_sentiment_score(
            feedback_type=feedback_type,
            sentiment=sentiment,
            target_features=target_features,
            feedback=feedback,
        )
        delta = model.update(target_features, sentiment_score)
        updates.append(
            {
                "feedback_id": feedback.get("feedback_id"),
                "text": feedback.get("text"),
                "feedback_type": feedback_type,
                "sentiment": sentiment["sentiment"],
                "sentiment_score": sentiment_score,
                "grounding_source": grounding["grounding_source"],
                "target_features": target_features,
                "weight_delta": delta,
            }
        )

    evaluation = evaluate_probes(probe_states, model.as_dict())
    return {
        "updates": updates,
        "learned_weights": model.as_dict(),
        "top_weights": model.top_weights(),
        "probe_evaluation": evaluation,
    }


def print_summary(result: dict) -> None:
    print("Updated feature weights:")
    for feature, weight in result["top_weights"]:
        print(f"  {feature}: {weight:.2f}")
    evaluation = result["probe_evaluation"]
    correct = evaluation["correct"]
    total = evaluation["total"]
    accuracy = evaluation["overall_accuracy"] * 100
    print(f"\nProbe Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("\nFeedback updates:")
    for update in result["updates"]:
        changed = ", ".join(sorted(update["weight_delta"])) or "no change"
        print(
            f"  {update['feedback_id']}: "
            f"type={update['feedback_type']} sentiment={update['sentiment_score']:+.1f} "
            f"changed={changed}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "baseline_b_pipeline_result.json",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    feedback_examples = read_json(args.feedback)
    probe_states = load_probe_states(args.probe_states)
    initial_weights = load_weights(args.weights)
    result = run_pipeline(
        feedback_examples=feedback_examples,
        probe_states=probe_states,
        initial_weights=initial_weights,
        learning_rate=args.learning_rate,
    )
    write_json(args.output, result)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)
        print(f"\nResult JSON: {args.output}")
    return 0 if result["probe_evaluation"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
