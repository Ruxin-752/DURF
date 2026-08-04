"""Small, dependency-free helpers for recipe-aware baseline task logic."""

from __future__ import annotations

from collections import Counter


def next_unstaged_ingredient(
    recipe: list[str],
    staged_ingredients: list[str],
) -> str | None:
    """Return the next recipe item not already prepared for the next cycle."""
    remaining = Counter(recipe)
    remaining.subtract(Counter(staged_ingredients))
    for ingredient in recipe:
        if remaining[ingredient] > 0:
            return ingredient
    return None
