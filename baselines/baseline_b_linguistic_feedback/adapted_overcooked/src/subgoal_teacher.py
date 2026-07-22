"""Gold rule-teacher over subgoals -- the free ground-truth for synthetic data.

We do not have a human corpus, but we *do* have a hand-authored comfort weight
vector ``w*`` (``data/gold_comfort_weights.json``). Together with the same
``subgoal_reranker`` used at runtime, ``w*`` defines the human-comfortable
choice in any decision context. This module:

1. labels a context -- which subgoal(s) are acceptable (``label_context``);
2. turns that labelling into *feedback intents* (``feedback_intents``) -- a
   ``(feedback_type, polarity, referenced feature vector, subgoal)`` tuple that
   the Phase 2 templates / LLM turn into natural language.

Because every generated ``(target_features, valence)`` is sign-consistent with
``w*``, training Route 2 to reproduce those observations distills the comfort
rule into a network that accepts free-form language. This is honest about its
limits: it validates *language generalization*, not the rule itself (the rule
is only sanity-checked against the small real held-out set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .feature_schema import read_json
from .subgoal_featurizer import SubgoalContext, featurize_subgoal
from .subgoal_reranker import score_subgoals

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_WEIGHTS_PATH = ROOT / "data" / "gold_comfort_weights.json"


def load_gold_weights(path: str | Path = DEFAULT_GOLD_WEIGHTS_PATH) -> dict[str, float]:
    """Load the hand-authored gold comfort/task weights ``w*``."""

    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Gold weights must be a JSON object: {path}")
    return {str(feature): float(value) for feature, value in raw.items()}


def label_context(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    feasible_subgoals: list[str],
    *,
    lambda_pref: float = 1.0,
    tie_tolerance: float = 1e-9,
) -> dict:
    """Return the gold subgoal labelling for a decision context.

    ``expected_subgoal`` is the argmax; ``acceptable_subgoals`` is the set tied
    at the top score (so probes / evaluations can accept genuine ties).
    """

    context = SubgoalContext.coerce(context)
    ranking = score_subgoals(weights, context, feasible_subgoals, lambda_pref=lambda_pref)
    top_score = ranking[0]["total_score"]
    acceptable = [
        item["subgoal"]
        for item in ranking
        if abs(item["total_score"] - top_score) <= tie_tolerance
    ]
    return {
        "expected_subgoal": ranking[0]["subgoal"],
        "acceptable_subgoals": acceptable,
        "ranking": ranking,
    }


@dataclass
class FeedbackIntent:
    """One synthetic feedback the teacher wants to express about a context."""

    feedback_type: str  # evaluative | imperative | descriptive
    polarity: float  # +1.0 praise / command-to-do ; -1.0 criticism / command-not-to-do
    subgoal: str  # the subgoal this feedback is about
    target_features: dict[str, float]  # sign-consistent reference vector
    referenced_features: list[str] = field(default_factory=list)
    role: str = ""  # "praise_best" | "command_best" | "criticize_alt" | "describe_alt"


def _features_by_sign(
    features: dict[str, float], weights: dict[str, float], *, positive: bool
) -> dict[str, float]:
    """Keep the features whose gold weight has the requested sign."""

    keep: dict[str, float] = {}
    for feature in features:
        weight = weights.get(feature, 0.0)
        if positive and weight > 0:
            keep[feature] = 1.0
        elif not positive and weight < 0:
            keep[feature] = 1.0
    return keep


def feedback_intents(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    feasible_subgoals: list[str],
    *,
    lambda_pref: float = 1.0,
    score_margin: float = 1e-6,
) -> list[FeedbackIntent]:
    """Enumerate the feedback intents a teacher would give for one context.

    - The best subgoal earns *positive* feedback grounded to its positive-sign
      features (praise + a command to do it).
    - Every strictly-worse feasible subgoal that has negative-sign features
      earns *negative* feedback grounded to those features (criticism +
      description of the bad behaviour).

    Grounding to sign-partitioned features guarantees each ``(target, valence)``
    pushes the belief toward ``w*``.
    """

    context = SubgoalContext.coerce(context)
    ranking = score_subgoals(weights, context, feasible_subgoals, lambda_pref=lambda_pref)
    best = ranking[0]
    intents: list[FeedbackIntent] = []

    best_features = featurize_subgoal(context, best["subgoal"])
    positive_target = _features_by_sign(best_features, weights, positive=True)
    if positive_target:
        referenced = sorted(positive_target)
        intents.append(
            FeedbackIntent(
                feedback_type="evaluative",
                polarity=1.0,
                subgoal=best["subgoal"],
                target_features=positive_target,
                referenced_features=referenced,
                role="praise_best",
            )
        )
        intents.append(
            FeedbackIntent(
                feedback_type="imperative",
                polarity=1.0,
                subgoal=best["subgoal"],
                target_features=positive_target,
                referenced_features=referenced,
                role="command_best",
            )
        )

    for item in ranking[1:]:
        if item["total_score"] >= best["total_score"] - score_margin:
            continue  # a genuine tie with the best: not a "worse" alternative
        alt_features = featurize_subgoal(context, item["subgoal"])
        negative_target = _features_by_sign(alt_features, weights, positive=False)
        if not negative_target:
            continue
        referenced = sorted(negative_target)
        intents.append(
            FeedbackIntent(
                feedback_type="evaluative",
                polarity=-1.0,
                subgoal=item["subgoal"],
                target_features=negative_target,
                referenced_features=referenced,
                role="criticize_alt",
            )
        )
        intents.append(
            FeedbackIntent(
                feedback_type="descriptive",
                polarity=-1.0,
                subgoal=item["subgoal"],
                target_features=negative_target,
                referenced_features=referenced,
                role="describe_alt",
            )
        )

    return intents
