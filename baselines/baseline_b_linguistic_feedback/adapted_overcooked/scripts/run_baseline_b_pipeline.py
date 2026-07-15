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
    empty_weights,
    load_weights,
    read_json,
    validate_feedback_examples,
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
    if feedback.get("attributed_sentiment_score") is not None:
        return float(feedback["attributed_sentiment_score"])
    if feedback_type == "imperative" and feedback.get("target_action"):
        return desired_action_sentiment(feedback_type, target_features)
    return float(sentiment["sentiment_score"])


def learn_from_feedback(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    initial_weights: dict[str, float],
    learning_rate: float,
) -> tuple[RewardWeightModel, list[dict]]:
    model = RewardWeightModel(initial_weights, learning_rate=learning_rate)
    action_library = collect_action_feature_library(probe_states)
    updates = []

    for feedback in feedback_examples:
        feedback_type = feedback.get("expected_feedback_type") or classify_feedback(
            feedback.get("text")
        )
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
    return model, updates


def evaluate_leave_one_probe_out(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    initial_weights: dict[str, float],
    learning_rate: float,
) -> dict:
    results = []
    for probe in probe_states:
        probe_id = probe.get("probe_id")
        training_examples = [
            feedback
            for feedback in feedback_examples
            if feedback.get("probe_id") != probe_id
        ]
        model, _ = learn_from_feedback(
            feedback_examples=training_examples,
            probe_states=probe_states,
            initial_weights=initial_weights,
            learning_rate=learning_rate,
        )
        evaluation = evaluate_probes([probe], model.as_dict())
        probe_result = evaluation["results"][0]
        results.append(
            {
                **probe_result,
                "excluded_probe_id": probe_id,
                "training_examples": len(training_examples),
                "excluded_examples": len(feedback_examples) - len(training_examples),
            }
        )

    total = len(results)
    correct = sum(int(result["correct"]) for result in results)
    tie_count = sum(int(result["is_tie"]) for result in results)
    margins = [result["margin"] for result in results if result["margin"] is not None]
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "tie_count": tie_count,
        "mean_margin": sum(margins) / len(margins) if margins else None,
        "results": results,
    }


def run_pipeline(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    initial_weights: dict[str, float],
    learning_rate: float,
) -> dict:
    validate_feedback_examples(feedback_examples, probe_states=probe_states)
    model, updates = learn_from_feedback(
        feedback_examples=feedback_examples,
        probe_states=probe_states,
        initial_weights=initial_weights,
        learning_rate=learning_rate,
    )

    learned_evaluation = evaluate_probes(probe_states, model.as_dict())
    zero_evaluation = evaluate_probes(
        probe_states,
        empty_weights(list(initial_weights)),
    )
    initial_evaluation = evaluate_probes(probe_states, initial_weights)
    holdout_evaluation = evaluate_leave_one_probe_out(
        feedback_examples=feedback_examples,
        probe_states=probe_states,
        initial_weights=initial_weights,
        learning_rate=learning_rate,
    )
    classification_correct = sum(
        classify_feedback(feedback.get("text")) == feedback.get("expected_feedback_type")
        for feedback in feedback_examples
    )
    grounding_counts: dict[str, int] = {}
    for update in updates:
        source = update["grounding_source"].split(":", 1)[0]
        grounding_counts[source] = grounding_counts.get(source, 0) + 1

    return {
        "updates": updates,
        "learned_weights": model.as_dict(),
        "top_weights": model.top_weights(),
        "pipeline_metrics": {
            "feedback_examples": len(feedback_examples),
            "classification_correct": classification_correct,
            "classification_accuracy": (
                classification_correct / len(feedback_examples)
                if feedback_examples
                else 0.0
            ),
            "grounding_source_counts": dict(sorted(grounding_counts.items())),
        },
        "probe_evaluation": learned_evaluation,
        "probe_evaluations": {
            "zero_weights": zero_evaluation,
            "initial_weights": initial_evaluation,
            "learned_weights": learned_evaluation,
            "leave_one_probe_out": holdout_evaluation,
        },
    }


def print_summary(result: dict) -> None:
    print("Updated feature weights:")
    for feature, weight in result["top_weights"]:
        print(f"  {feature}: {weight:.2f}")
    evaluation = result["probe_evaluation"]
    correct = evaluation["correct"]
    total = evaluation["total"]
    accuracy = evaluation["overall_accuracy"] * 100
    print(
        f"\nLearned Probe Accuracy: {correct}/{total} = {accuracy:.1f}% "
        f"(ties={evaluation['tie_count']})"
    )
    print("Evaluation baselines:")
    for name in ("zero_weights", "initial_weights", "leave_one_probe_out"):
        baseline = result["probe_evaluations"][name]
        print(
            f"  {name}: {baseline['correct']}/{baseline['total']} "
            f"= {baseline['overall_accuracy'] * 100:.1f}% "
            f"(ties={baseline['tie_count']})"
        )
    metrics = result["pipeline_metrics"]
    print(
        "Classifier agreement: "
        f"{metrics['classification_correct']}/{metrics['feedback_examples']} "
        f"= {metrics['classification_accuracy'] * 100:.1f}%"
    )
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
