"""Re-rank H0's feasible subgoals with a learned reward belief.

H0 already knows how to *complete the recipe*; it just has no notion of human
comfort. This module takes the feasible subgoals H0 proposes and re-ranks them
by the reward learned from language feedback, so among task-valid options the
agent prefers the human-comfortable one.

The paper-aligned decision score is exactly:

    total(subgoal) = w dot phi(state, subgoal)

Task and comfort subtotals are retained only for diagnostics. They never change
the total score or break ties; otherwise a hand-authored partition would
silently override the reward inferred from language.
"""

from __future__ import annotations

from .belief_model import score_action
from .subgoal_featurizer import SubgoalContext, featurize_subgoal


# Coordination / human-comfort features. Everything else is treated as task.
COMFORT_FEATURES: frozenset[str] = frozenset(
    {
        "respects_human_intent",
        "avoids_duplicate_human_task",
        "clears_human_shortest_path",
        "clears_human_path",
        "clears_serving_access",
        "blocks_human_path",
        "blocks_serving_route",
        "human_wait_cost",
        "duplicate_human_task",
        "steals_human_target",
        "crowds_human_target",
        "frustrates_human",
        "collision_risk",
        "complementary_to_human",
        "avoids_human_shortest_path",
        "cuts_in_front_of_human",
        "blocks_partner_on_ring",
    }
)


def _split_features(features: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    task = {f: v for f, v in features.items() if f not in COMFORT_FEATURES}
    comfort = {f: v for f, v in features.items() if f in COMFORT_FEATURES}
    return task, comfort


def score_subgoals(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    feasible_subgoals: list[str],
    *,
    lambda_pref: float = 1.0,
) -> list[dict]:
    """Score each feasible subgoal; returned list is sorted best-first."""

    if abs(float(lambda_pref) - 1.0) > 1e-12:
        raise ValueError(
            "paper-aligned subgoal scoring requires lambda_pref=1.0; "
            "change reward weights instead of rescaling a hand-authored partition"
        )

    context = SubgoalContext.coerce(context)

    scored = []
    for subgoal in feasible_subgoals:
        features = featurize_subgoal(context, subgoal)
        task_features, comfort_features = _split_features(features)
        task_score = score_action(weights, task_features)
        comfort_score = score_action(weights, comfort_features)
        total_score = score_action(weights, features)
        scored.append(
            {
                "subgoal": subgoal,
                "total_score": total_score,
                "task_score": task_score,
                "comfort_score": comfort_score,
                "features": features,
            }
        )

    # Exact reward ties remain ties. Stable sorting preserves the candidate
    # order for display only; a caller-provided H0 fallback makes the decision.
    scored.sort(key=lambda item: -item["total_score"])
    return scored


def choose_subgoal(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    feasible_subgoals: list[str],
    *,
    lambda_pref: float = 1.0,
    tie_tolerance: float = 1e-9,
    tie_fallback: str | None = None,
) -> dict:
    """Pick the best feasible subgoal under the learned reward.

    The choice is the argmax of `w dot phi(state, subgoal)` over the task-valid
    subgoals H0 proposed. Yielding to the human (choosing WAIT)
    is a legitimate outcome here; preventing the agent from stalling *forever*
    is a multi-step concern left to the live caller (e.g. H0 can cap the number
    of consecutive WAITs, at which point the human is no longer contesting the
    resource and the comfort bonus for waiting disappears).
    """

    if not feasible_subgoals:
        raise ValueError("choose_subgoal requires at least one feasible subgoal")

    scored = score_subgoals(
        weights, context, feasible_subgoals, lambda_pref=lambda_pref
    )
    top_score = scored[0]["total_score"]
    tied_rows = [
        item for item in scored if abs(item["total_score"] - top_score) <= tie_tolerance
    ]
    tied = [item["subgoal"] for item in tied_rows]
    is_tie = len(tied) > 1
    best = scored[0]
    decision_source = "reward_argmax"
    if is_tie:
        if tie_fallback in tied:
            best = next(item for item in tied_rows if item["subgoal"] == tie_fallback)
            decision_source = "h0_tie_fallback"
        else:
            decision_source = "unresolved_reward_tie"
    reward_margin = (
        float(top_score - scored[1]["total_score"]) if len(scored) > 1 else None
    )
    return {
        "chosen_subgoal": best["subgoal"],
        "is_tie": is_tie,
        "tied_subgoals": tied,
        "tie_fallback": tie_fallback,
        "decision_source": decision_source,
        "reward_margin": reward_margin,
        "score_formula": "w_dot_phi",
        "ranking": scored,
    }


def evaluate_subgoal_probes(
    probes: list[dict],
    weights: dict[str, float],
    *,
    lambda_pref: float = 1.0,
) -> dict:
    """Score subgoal-level probes with a (learned) reward weight vector."""

    results = []
    correct = 0
    tie_count = 0
    for probe in probes:
        context = probe["context"]
        feasible = probe["feasible_subgoals"]
        acceptable = probe.get("acceptable_subgoals") or [probe.get("expected_subgoal")]
        choice = choose_subgoal(weights, context, feasible, lambda_pref=lambda_pref)
        is_correct = (not choice["is_tie"]) and choice["chosen_subgoal"] in acceptable
        correct += int(is_correct)
        tie_count += int(choice["is_tie"])
        results.append(
            {
                "probe_id": probe.get("probe_id"),
                "expected_subgoal": probe.get("expected_subgoal"),
                "acceptable_subgoals": acceptable,
                "chosen_subgoal": choice["chosen_subgoal"],
                "is_tie": choice["is_tie"],
                "correct": is_correct,
                "ranking": [
                    {
                        "subgoal": item["subgoal"],
                        "total_score": item["total_score"],
                        "task_score": item["task_score"],
                        "comfort_score": item["comfort_score"],
                    }
                    for item in choice["ranking"]
                ],
            }
        )

    total = len(results)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "tie_count": tie_count,
        "lambda_pref": lambda_pref,
        "results": results,
    }
