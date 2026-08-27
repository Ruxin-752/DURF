"""Deterministic synthetic reward configurations for paper-aligned Route 2.

The real paper varied hidden reward functions. Until human teacher rewards are
available, these configurations provide the same supervised structure without
pretending that synthetic styles are people. Every vector uses the shared
53-dimensional Overcooked schema. Task-safety signs remain fixed while four
preference-strength families vary, plus one sign-flexible preparation axis
representing cautious, neutral, and proactive teachers.
"""

from __future__ import annotations

from itertools import product
import random

from .feature_schema import load_features
from .subgoal_teacher import load_gold_weights


PREFERENCE_GROUPS: dict[str, frozenset[str]] = {
    "path_coordination": frozenset(
        {
            "avoids_human_shortest_path",
            "blocks_human_path",
            "blocks_partner_on_ring",
            "blocks_serving_route",
            "clears_human_path",
            "clears_human_shortest_path",
            "clears_serving_access",
            "collision_risk",
            "cuts_in_front_of_human",
            "human_wait_cost",
        }
    ),
    "division_of_labor": frozenset(
        {
            "avoids_duplicate_human_task",
            "complementary_to_human",
            "duplicate_human_task",
        }
    ),
    "target_respect": frozenset(
        {
            "crowds_human_target",
            "frustrates_human",
            "respects_human_intent",
            "steals_human_target",
        }
    ),
    "efficiency": frozenset(
        {
            "delays_serving",
            "distance_cost",
            "moves_away_from_needed_object",
            "supports_serving",
        }
    ),
}

LEVELS = (0.25, 1.0, 1.75)

# These are factual action features, not task-safety constraints. Letting their
# reward signs vary is what makes the same cooking-state candidate set support
# genuinely different hidden preferences instead of hard-coding prefetching as
# universally good. Current-task progress still has separate safety features
# such as adds_needed_* and matches_current_order.
PROACTIVITY_FEATURES: frozenset[str] = frozenset(
    {
        "ingredient_tomato",
        "ingredient_onion",
        "pick_tomato",
        "pick_onion",
        "pick_dish",
        "moves_toward_needed_object",
    }
)
PROACTIVITY_LEVELS = (-1.0, 0.0, 1.0)
PROACTIVITY_LABELS = {
    -1.0: "cautious",
    0.0: "neutral",
    1.0: "proactive",
}


def _complete_gold(features: list[str]) -> dict[str, float]:
    raw = load_gold_weights()
    return {feature: float(raw.get(feature, 0.0)) for feature in features}


def sample_reward_configurations(
    *,
    n: int = 36,
    features: list[str] | None = None,
) -> list[dict]:
    """Return one canonical gold vector plus deterministic structured variants."""

    max_configurations = (len(LEVELS) ** len(PREFERENCE_GROUPS)) * len(
        PROACTIVITY_LEVELS
    )
    if n < 1 or n > max_configurations:
        raise ValueError(
            f"n must be between 1 and {max_configurations} "
            "(gold plus structured preference variants)"
        )
    ordered = list(features or load_features())
    gold = _complete_gold(ordered)
    configurations = [
        {
            "reward_config_id": "reward_00_gold",
            "profile": "canonical_gold",
            "multipliers": {
                **{name: 1.0 for name in PREFERENCE_GROUPS},
                "proactivity": 1.0,
            },
            "preparation_style": PROACTIVITY_LABELS[1.0],
            "weights": dict(gold),
        }
    ]
    multiplier_candidates = [
        (*tuple(float(value) for value in values), float(proactivity))
        for values in product(LEVELS, repeat=len(PREFERENCE_GROUPS))
        for proactivity in PROACTIVITY_LEVELS
        if not (
            tuple(values) == tuple(1.0 for _ in PREFERENCE_GROUPS)
            and float(proactivity) == 1.0
        )
    ]
    random.Random(7321).shuffle(multiplier_candidates)
    for multipliers in multiplier_candidates[: n - 1]:
        by_group = dict(zip(PREFERENCE_GROUPS, multipliers[:-1]))
        proactivity = float(multipliers[-1])
        by_group["proactivity"] = proactivity
        weights = dict(gold)
        for group, group_features in PREFERENCE_GROUPS.items():
            multiplier = by_group[group]
            for feature in group_features:
                weights[feature] = gold.get(feature, 0.0) * multiplier
        for feature in PROACTIVITY_FEATURES:
            weights[feature] = gold.get(feature, 0.0) * proactivity
        config_index = len(configurations)
        configurations.append(
            {
                "reward_config_id": f"reward_{config_index:02d}",
                "profile": "structured_preference_variant",
                "multipliers": by_group,
                "preparation_style": PROACTIVITY_LABELS[proactivity],
                "weights": weights,
            }
        )
    return configurations


def validate_reward_configurations(configurations: list[dict]) -> None:
    """Reject incomplete, duplicate, non-finite, or unsafe configurations."""

    import math

    features = load_features()
    feature_set = set(features)
    gold = _complete_gold(features)
    identifiers: set[str] = set()
    vectors: set[tuple[float, ...]] = set()
    for config in configurations:
        identifier = str(config.get("reward_config_id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError(f"invalid or duplicate reward_config_id: {identifier!r}")
        identifiers.add(identifier)
        multipliers = config.get("multipliers") or {}
        proactivity = float(multipliers.get("proactivity", 1.0))
        if proactivity not in PROACTIVITY_LABELS:
            raise ValueError(f"{identifier} has invalid proactivity {proactivity}")
        expected_style = PROACTIVITY_LABELS[proactivity]
        if config.get("preparation_style") != expected_style:
            raise ValueError(
                f"{identifier} preparation_style must be {expected_style!r}"
            )
        weights = config.get("weights") or {}
        if set(weights) != feature_set:
            raise ValueError(f"{identifier} does not cover the complete feature schema")
        vector = tuple(float(weights[feature]) for feature in features)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{identifier} contains a non-finite weight")
        if vector in vectors:
            raise ValueError(f"{identifier} duplicates another reward vector")
        vectors.add(vector)
        for feature, base in gold.items():
            if feature in PROACTIVITY_FEATURES:
                continue
            value = float(weights[feature])
            if base != 0.0 and value != 0.0 and (base > 0) != (value > 0):
                raise ValueError(f"{identifier} inverts the safety sign of {feature}")
