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
from src.overcooked_grounding import features_from_keywords, ground_feedback  # noqa: E402
from src.feature_schema import load_features  # noqa: E402
from src.text_analysis import limited_punc_tokenization  # noqa: E402
from src.probe_evaluator import (  # noqa: E402
    DEFAULT_PROBE_STATES_PATH,
    evaluate_probes,
    evaluate_probes_sampled,
    load_probe_states,
)
from src.reward_weight_model import BayesianRewardLearner  # noqa: E402
from src.sentiment_extractor import (  # noqa: E402
    desired_action_sentiment,
    extract_sentiment,
    modified_vader_observation,
)


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "feedback_examples.json"

# Paper defaults (science/agents/agents.py ExpLiteralLearner / ExpPseudoPragmatic).
DEFAULT_VALENCE_SCALE = 30.0
DEFAULT_PRECISION_SCALE = 2.0
DEFAULT_PRAGMATIC_VALENCE = -30.0
DEFAULT_PRAGMATIC_PRECISION = 2.0


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


def build_feedback_observations(
    feedback: dict,
    *,
    feedback_type: str,
    action_feature_library: dict[str, dict[str, float]],
) -> list[dict]:
    """Decompose one feedback utterance into per-phrase Gaussian sub-observations.

    Mirrors the paper's ``observations_from_utterance``: an utterance is split
    on punctuation (``limited_punc_tokenization``) and each phrase becomes its
    own observation with its own VADER valence. Feedback that carries an
    explicit grounding (``target_features`` / ``trajectory_features`` /
    ``target_action``) is treated as a single whole-utterance reference, since
    that authored vector -- not the text -- is the effective reference here.
    """

    text = feedback.get("text") or feedback.get("feedback_text") or ""
    grounding = ground_feedback(
        feedback,
        feedback_type=feedback_type,
        action_feature_library=action_feature_library,
    )

    if grounding["grounding_source"] != "keyword_features":
        sentiment = extract_sentiment(text)
        valence = effective_sentiment_score(
            feedback_type=feedback_type,
            sentiment=sentiment,
            target_features=grounding["target_features"],
            feedback=feedback,
        )
        return [
            {
                "target_features": grounding["target_features"],
                "valence": valence,
                "phrase": text,
                "grounding_source": grounding["grounding_source"],
            }
        ]

    sub_observations: list[dict] = []
    for phrase in limited_punc_tokenization(text):
        phrase_features = features_from_keywords(phrase)
        if phrase_features:
            sub_observations.append(
                {
                    "target_features": phrase_features,
                    "valence": modified_vader_observation(phrase),
                    "phrase": phrase,
                    "grounding_source": "keyword_features",
                }
            )
    if not sub_observations:
        sub_observations.append(
            {
                "target_features": features_from_keywords(text),
                "valence": modified_vader_observation(text),
                "phrase": text,
                "grounding_source": "keyword_features",
            }
        )
    return sub_observations


def learn_from_feedback(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    features: list[str],
    valence_scale: float = DEFAULT_VALENCE_SCALE,
    precision_scale: float = DEFAULT_PRECISION_SCALE,
    pragmatic_valence: float | None = None,
    pragmatic_precision: float | None = None,
) -> tuple[BayesianRewardLearner, list[dict]]:
    model = BayesianRewardLearner(
        features,
        valence_scale=valence_scale,
        precision_scale=precision_scale,
        pragmatic_valence=pragmatic_valence,
        pragmatic_precision=pragmatic_precision,
    )
    action_library = collect_action_feature_library(probe_states)
    updates = []

    for feedback in feedback_examples:
        feedback_type = feedback.get("expected_feedback_type") or classify_feedback(
            feedback.get("text")
        )
        sub_observations = build_feedback_observations(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_library,
        )

        before = model.as_dict()
        merged_features: dict[str, float] = {}
        valences: list[float] = []
        for sub in sub_observations:
            model.update(sub["target_features"], sub["valence"])
            for feature, value in sub["target_features"].items():
                merged_features[feature] = merged_features.get(feature, 0.0) + float(value)
            valences.append(sub["valence"])
        after = model.as_dict()

        delta = {
            feature: round(after[feature] - before[feature], 12)
            for feature in after
            if abs(after[feature] - before[feature]) > 1e-9
        }
        mean_valence = sum(valences) / len(valences) if valences else 0.0
        sentiment_label = (
            "positive" if mean_valence > 0 else "negative" if mean_valence < 0 else "neutral"
        )
        updates.append(
            {
                "feedback_id": feedback.get("feedback_id"),
                "text": feedback.get("text"),
                "feedback_type": feedback_type,
                "sentiment": sentiment_label,
                "sentiment_score": mean_valence,
                "n_observations": len(sub_observations),
                "grounding_source": sub_observations[0]["grounding_source"],
                "target_features": merged_features,
                "weight_delta": delta,
            }
        )
    return model, updates


def evaluate_leave_one_probe_out(
    *,
    feedback_examples: list[dict],
    probe_states: list[dict],
    features: list[str],
    valence_scale: float = DEFAULT_VALENCE_SCALE,
    precision_scale: float = DEFAULT_PRECISION_SCALE,
    pragmatic_valence: float | None = None,
    pragmatic_precision: float | None = None,
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
            features=features,
            valence_scale=valence_scale,
            precision_scale=precision_scale,
            pragmatic_valence=pragmatic_valence,
            pragmatic_precision=pragmatic_precision,
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
    learning_rate: float = 1.0,  # deprecated: point-estimate step size, ignored by Bayesian learner
    valence_scale: float = DEFAULT_VALENCE_SCALE,
    precision_scale: float = DEFAULT_PRECISION_SCALE,
    pragmatic_valence: float | None = None,
    pragmatic_precision: float | None = None,
    n_samples: int = 500,
    seed: int = 0,
) -> dict:
    validate_feedback_examples(feedback_examples, probe_states=probe_states)
    features = sorted(initial_weights) if initial_weights else load_features()
    learn_kwargs = dict(
        probe_states=probe_states,
        features=features,
        valence_scale=valence_scale,
        precision_scale=precision_scale,
        pragmatic_valence=pragmatic_valence,
        pragmatic_precision=pragmatic_precision,
    )
    model, updates = learn_from_feedback(
        feedback_examples=feedback_examples,
        **learn_kwargs,
    )

    learned_evaluation = evaluate_probes(probe_states, model.as_dict())
    learned_evaluation_sampled = evaluate_probes_sampled(
        probe_states, model.belief, n_samples=n_samples, seed=seed
    )
    zero_evaluation = evaluate_probes(
        probe_states,
        empty_weights(list(initial_weights)),
    )
    initial_evaluation = evaluate_probes(probe_states, initial_weights)
    holdout_evaluation = evaluate_leave_one_probe_out(
        feedback_examples=feedback_examples,
        **learn_kwargs,
    )
    classification_correct = sum(
        classify_feedback(feedback.get("text")) == feedback.get("expected_feedback_type")
        for feedback in feedback_examples
    )
    grounding_counts: dict[str, int] = {}
    for update in updates:
        source = update["grounding_source"].split(":", 1)[0]
        grounding_counts[source] = grounding_counts.get(source, 0) + 1

    learner_mode = "pseudopragmatic" if pragmatic_valence is not None else "literal"
    return {
        "learner": {
            "type": "BayesianRewardLearner",
            "mode": learner_mode,
            "valence_scale": valence_scale,
            "precision_scale": precision_scale,
            "pragmatic_valence": pragmatic_valence,
            "pragmatic_precision": pragmatic_precision,
        },
        "updates": updates,
        "learned_weights": model.as_dict(),
        "learned_weight_variance": model.belief.variance_as_dict(),
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
        "probe_evaluation_sampled": learned_evaluation_sampled,
        "probe_evaluations": {
            "zero_weights": zero_evaluation,
            "initial_weights": initial_evaluation,
            "learned_weights": learned_evaluation,
            "leave_one_probe_out": holdout_evaluation,
        },
    }


def print_summary(result: dict) -> None:
    learner = result.get("learner", {})
    print(
        f"Learner: {learner.get('type')} mode={learner.get('mode')} "
        f"(V{learner.get('valence_scale')},P{learner.get('precision_scale')})"
    )
    print("Posterior mean feature weights (top):")
    for feature, weight in result["top_weights"]:
        print(f"  {feature}: {weight:.2f}")
    evaluation = result["probe_evaluation"]
    correct = evaluation["correct"]
    total = evaluation["total"]
    accuracy = evaluation["overall_accuracy"] * 100
    print(
        f"\nLearned Probe Accuracy (posterior mean): {correct}/{total} = {accuracy:.1f}% "
        f"(ties={evaluation['tie_count']})"
    )
    sampled = result.get("probe_evaluation_sampled")
    if sampled:
        print(
            f"Learned Probe Accuracy (sampled policy, n={sampled['n_samples']}): "
            f"{sampled['expected_accuracy'] * 100:.1f}% expected"
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
    parser.add_argument(
        "--mode",
        choices=("literal", "pseudopragmatic"),
        default="literal",
        help="Bayesian learner variant from the paper.",
    )
    parser.add_argument("--valence-scale", type=float, default=DEFAULT_VALENCE_SCALE)
    parser.add_argument("--precision-scale", type=float, default=DEFAULT_PRECISION_SCALE)
    parser.add_argument("--pragmatic-valence", type=float, default=DEFAULT_PRAGMATIC_VALENCE)
    parser.add_argument("--pragmatic-precision", type=float, default=DEFAULT_PRAGMATIC_PRECISION)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
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
    pragmatic = args.mode == "pseudopragmatic"
    result = run_pipeline(
        feedback_examples=feedback_examples,
        probe_states=probe_states,
        initial_weights=initial_weights,
        valence_scale=args.valence_scale,
        precision_scale=args.precision_scale,
        pragmatic_valence=args.pragmatic_valence if pragmatic else None,
        pragmatic_precision=args.pragmatic_precision if pragmatic else None,
        n_samples=args.n_samples,
        seed=args.seed,
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
