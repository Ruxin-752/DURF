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


def _human_covers_held_ingredient(
    context: SubgoalContext, held: str, missing: list[str]
) -> bool:
    """Whether the human-held twin covers all remaining units of ``held``."""

    remaining = int(Counter(missing)[held])
    if remaining <= 0 or context.human_holding != held:
        return False
    committed = context.human_committed_units
    if committed is None:
        committed = 1
    return int(committed) >= remaining


def _prefetch_ingredient_deficits(context: SubgoalContext) -> list[str]:
    """Next-order ingredient units not already staged or held by either actor."""

    remaining = Counter(context.recipe)
    for ingredient in _distinct(context.recipe):
        remaining[ingredient] -= context.staged_units(ingredient)
        remaining[ingredient] -= int(context.agent_holding == ingredient)
        remaining[ingredient] -= int(context.human_holding == ingredient)

    deficits: list[str] = []
    for ingredient in context.recipe:
        if remaining[ingredient] > 0:
            deficits.append(ingredient)
            remaining[ingredient] -= 1
    return deficits


def _open_work_not_covered_by_human(
    context: SubgoalContext, missing: list[str]
) -> list[str]:
    """Open-pot units still needing the AI after the human-held commitment.

    A staged counter object remains a useful pickup while a pot is open.  But
    when the human already holds every remaining unit of that ingredient, an
    empty AI must not immediately re-pick the object it just staged.  This is
    resource accounting, not a preference or feedback-to-action rule.
    """

    remaining = Counter(missing)
    held = context.human_holding
    if held in remaining:
        committed = context.human_committed_units
        if committed is None:
            committed = 1
        remaining[held] = max(0, remaining[held] - int(committed))

    uncovered: list[str] = []
    for ingredient in missing:
        if remaining[ingredient] > 0:
            uncovered.append(ingredient)
            remaining[ingredient] -= 1
    return uncovered


def _dish_supply(context: SubgoalContext) -> int:
    """Dishes already staged or committed by either actor."""

    dish_supply = context.staged_units("dish")
    dish_supply += int(context.agent_holding == "dish")
    dish_supply += int(context.human_holding == "dish")
    return dish_supply


def _active_soups_need_another_dish(context: SubgoalContext) -> bool:
    """Whether ready/cooking soup demand exceeds staged and held dish supply."""

    active_soups = context.ready_soup_count + context.cooking_soup_count
    return active_soups > _dish_supply(context)


def enumerate_feasible_subgoals(context: SubgoalContext | dict) -> list[str]:
    """Task-valid candidate subgoals for a state, mirroring H0's rule logic.

    The result is the set H0 *could* legitimately pursue (not the single choice
    H0 would commit to), so the learned reward can re-rank among real options.
    ``WAIT`` is always included as a valid way to yield a contested resource.
    Order is deterministic: task-productive options first, ``WAIT`` last.
    """

    context = SubgoalContext.coerce(context)
    missing = context.open_missing_ingredients
    held = context.agent_holding

    candidates: list[str] = []

    if held in ("tomato", "onion"):
        # Potting advances the task. If the human holds the same ingredient
        # and has already covered every remaining unit, STASH is also a real
        # option so learned coordination weights can choose instead of having
        # duplicate work forced by candidate enumeration.
        if held in missing:
            candidates.append(INGREDIENT_POT_SUBGOALS[held])
            if _human_covers_held_ingredient(context, held, missing):
                candidates.append("STASH_HELD_OBJECT")
        else:
            candidates.append("STASH_HELD_OBJECT")
    elif held == "dish":
        # Holding a dish is only useful once a soup is ready to be plated.
        if context.soup_ready:
            candidates.append("PICKUP_SOUP")
        else:
            candidates.append("STASH_HELD_OBJECT")
    elif held == "soup":
        candidates.append("SERVE_SOUP")
    else:
        # Empty-handed work is the union across pots: a ready pot can need a
        # dish while a second open pot simultaneously needs ingredients.
        uncovered_open_work = _open_work_not_covered_by_human(context, missing)
        # When the human-held ingredient covers the complete remaining open
        # recipe, that pot is an imminent soup. Preparing exactly one dish is
        # bounded downstream work; it avoids duplicate ingredient collection
        # without treating the human commitment as "the AI has no work".
        imminent_soup_count = int(bool(missing and not uncovered_open_work))
        active_soup_count = context.ready_soup_count + context.cooking_soup_count
        dish_work_uncovered = bool(
            (context.soup_ready or imminent_soup_count)
            and active_soup_count + imminent_soup_count > _dish_supply(context)
        )
        if dish_work_uncovered:
            candidates.append("GET_DISH")
        for ingredient in _distinct(uncovered_open_work):
            pickup = INGREDIENT_PICKUP_SUBGOALS.get(ingredient)
            if pickup is not None:
                candidates.append(pickup)

        if context.soup_cooking and not context.soup_ready and not missing:
            # Counter objects and held resources are completed preparation,
            # so only enumerate deficits. This prevents STASH -> immediate
            # re-pick loops without assigning an action any fixed preference.
            if _active_soups_need_another_dish(context):
                candidates.append("GET_DISH")
            for ingredient in _distinct(_prefetch_ingredient_deficits(context)):
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
    tie_fallback: str | None = None,
) -> dict:
    """Enumerate + re-rank + choose in one call (the runtime entry point).

    Pass ``feasible_subgoals`` to override enumeration (e.g. when the live H0
    already computed a set that respects motion feasibility); otherwise the set
    is derived from ``context`` via :func:`enumerate_feasible_subgoals`.
    """

    context = SubgoalContext.coerce(context)
    feasible = (
        enumerate_feasible_subgoals(context)
        if feasible_subgoals is None
        else list(feasible_subgoals)
    )
    invalid = [name for name in feasible if not is_valid_subgoal(name)]
    if invalid:
        raise ValueError(f"unknown subgoal(s) in feasible set: {invalid}")

    choice = choose_subgoal(
        weights,
        context,
        feasible,
        lambda_pref=lambda_pref,
        tie_tolerance=tie_tolerance,
        tie_fallback=tie_fallback,
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
    feasible = (
        enumerate_feasible_subgoals(context)
        if feasible_subgoals is None
        else list(feasible_subgoals)
    )
    if not feasible:
        raise ValueError("rank_subgoals requires at least one feasible subgoal")
    return score_subgoals(weights, context, feasible, lambda_pref=lambda_pref)
