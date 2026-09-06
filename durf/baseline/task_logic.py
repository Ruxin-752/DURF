"""Small, dependency-free helpers for recipe-aware baseline task logic."""

from __future__ import annotations

from collections import Counter


def unstaged_ingredients(
    recipe: list[str],
    staged_ingredients: list[str],
) -> list[str]:
    """Every distinct recipe item not already prepared, in recipe order.

    The single-value ``next_unstaged_ingredient`` below is the head of this
    list.  Returning only the head is what collapsed "fetch the tomato or the
    onion first?" into a non-choice: the candidate generator never saw the
    second option, so no preference could ever express it.
    """
    remaining = Counter(recipe)
    remaining.subtract(Counter(staged_ingredients))
    out: list[str] = []
    for ingredient in recipe:
        if remaining[ingredient] > 0 and ingredient not in out:
            out.append(ingredient)
    return out


def next_unstaged_ingredient(
    recipe: list[str],
    staged_ingredients: list[str],
) -> str | None:
    """Return the next recipe item not already prepared for the next cycle."""
    pending = unstaged_ingredients(recipe, staged_ingredients)
    return pending[0] if pending else None
