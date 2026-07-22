"""Feasible-subgoal enumeration + a single learned-reward decision core.

Offline Baseline B so far only knew how to *score* a subgoal once someone handed
it a candidate list (the hand-authored ``feasible_subgoals`` in the probe files).
To drive a live agent we need two things the probes used to supply by hand:

1. ``enumerate_feasible_subgoals(context)`` -- derive the task-valid candidate
   set directly from the state, mirroring ``durf.baseline.h0_planner`` but
   returning the *set* of options (H0 commits to one; the reranker needs all of
   them). ``WAIT`` is always offered as a legitimate yield.
2. ``plan_subgoal(weights, context)`` -- the one function the runtime calls:
   enumerate -> featurize -> re-rank with the learned reward -> pick.

This module stays free of any Overcooked runtime import so it can be unit-tested
offline; the live loop only has to translate its own state into a
``SubgoalContext`` (or a state-facts dict) and read back a subgoal name.
"""

from __future__ import annotations

from collections import Counter

from .subgoal_featurizer import SubgoalContext
from .subgoal_reranker import choose_subgoal, score_subgoals
from .subgoal_schema import (
    INGREDIENT_PICKUP_SUBGOALS,
    INGREDIENT_POT_SUBGOALS,
    is_valid_subgoal,
)


WAIT = "WAIT"


def _missing_ingredients(recipe: list[str], current: list[str]) -> list[str]:
    missing = Counter(recipe)
    missing.subtract(Counter(current))
    ordered: list[str] = []
    for ingredient in recipe:
        if missing[ingredient] > 0:
            ordered.append(ingredient)
            missing[ingredient] -= 1
    return ordered


def _distinct(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def enumerate_feasible_subgoals(context: SubgoalContext | dict) -> list[str]:
    """Task-valid candidate subgoals for a state, mirroring H0's rule logic.

    The result is the set H0 *could* legitimately pursue (not the single choice
    H0 would commit to), so the learned reward can re-rank among real options.
    ``WAIT`` is always included as a valid way to yield a contested resource.
    Order is deterministic: task-productive options first, ``WAIT`` last.
    """

    context = SubgoalContext.coerce(context)
    missing = _missing_ingredients(context.recipe, context.pot_ingredients)
    held = context.agent_holding

    candidates: list[str] = []

    if held in ("tomato", "onion"):
        # Holding an ingredient: the only productive move is potting it, and
        # only if the pot still needs it.
        if held in missing:
            candidates.append(INGREDIENT_POT_SUBGOALS[held])
    elif held == "dish":
        # Holding a dish is only useful once a soup is ready to be plated.
        if context.soup_ready:
            candidates.append("PICKUP_SOUP")
    elif held == "soup":
        candidates.append("SERVE_SOUP")
    else:
        # Empty-handed. If a soup is ready, fetching a dish supports serving;
        # otherwise fetch each still-missing ingredient.
        if context.soup_ready:
            candidates.append("GET_DISH")
        else:
            for ingredient in _distinct(missing):
                pickup = INGREDIENT_PICKUP_SUBGOALS.get(ingredient)
                if pickup is not None:
                    candidates.append(pickup)

    candidates = _distinct(candidates)
    candidates.append(WAIT)
    return _distinct(candidates)


def plan_subgoal(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    *,
    lambda_pref: float = 1.0,
    feasible_subgoals: list[str] | None = None,
    tie_tolerance: float = 1e-9,
) -> dict:
    """Enumerate + re-rank + choose in one call (the runtime entry point).

    Pass ``feasible_subgoals`` to override enumeration (e.g. when the live H0
    already computed a set that respects motion feasibility); otherwise the set
    is derived from ``context`` via :func:`enumerate_feasible_subgoals`.
    """

    context = SubgoalContext.coerce(context)
    feasible = feasible_subgoals or enumerate_feasible_subgoals(context)
    invalid = [name for name in feasible if not is_valid_subgoal(name)]
    if invalid:
        raise ValueError(f"unknown subgoal(s) in feasible set: {invalid}")

    choice = choose_subgoal(
        weights,
        context,
        feasible,
        lambda_pref=lambda_pref,
        tie_tolerance=tie_tolerance,
    )
    choice["feasible_subgoals"] = feasible
    return choice


def rank_subgoals(
    weights: dict[str, float],
    context: SubgoalContext | dict,
    *,
    lambda_pref: float = 1.0,
    feasible_subgoals: list[str] | None = None,
) -> list[dict]:
    """Convenience: full best-first ranking over the enumerated candidate set."""

    context = SubgoalContext.coerce(context)
    feasible = feasible_subgoals or enumerate_feasible_subgoals(context)
    return score_subgoals(weights, context, feasible, lambda_pref=lambda_pref)
