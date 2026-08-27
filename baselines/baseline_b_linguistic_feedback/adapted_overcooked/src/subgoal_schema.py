"""H0 subgoal vocabulary, shared with `durf.baseline.h0_planner`.

The reward learner (Baseline B) and the H0 planner must agree on the subgoal
set so that a reward learned from language feedback can re-rank exactly the
subgoals H0 is able to execute. These names mirror
`durf/baseline/h0_planner.py::SUBGOALS`; they are duplicated here (rather than
imported) to keep the offline Baseline B package free of Overcooked runtime
dependencies.
"""

from __future__ import annotations


SUBGOALS: tuple[str, ...] = (
    "GET_TOMATO",
    "PUT_TOMATO_IN_POT",
    "GET_ONION",
    "PUT_ONION_IN_POT",
    "GET_DISH",
    "PICKUP_SOUP",
    "SERVE_SOUP",
    "STASH_HELD_OBJECT",
    "YIELD_PATH",
    "WAIT",
)
SUBGOAL_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(SUBGOALS)}

# The resource each subgoal competes for. Used to detect duplication / stealing
# against the human's current target.
SUBGOAL_RESOURCE: dict[str, str | None] = {
    "GET_TOMATO": "tomato",
    "PUT_TOMATO_IN_POT": "tomato",
    "GET_ONION": "onion",
    "PUT_ONION_IN_POT": "onion",
    "GET_DISH": "dish",
    "PICKUP_SOUP": "soup",
    "SERVE_SOUP": "serving",
    "STASH_HELD_OBJECT": None,
    "YIELD_PATH": None,
    "WAIT": None,
}

# Subgoals that move the agent along the dish -> pot -> serving-counter route.
SERVING_SUBGOALS: frozenset[str] = frozenset(
    {"GET_DISH", "PICKUP_SOUP", "SERVE_SOUP"}
)

INGREDIENT_PICKUP_SUBGOALS: dict[str, str] = {
    "tomato": "GET_TOMATO",
    "onion": "GET_ONION",
}
INGREDIENT_POT_SUBGOALS: dict[str, str] = {
    "tomato": "PUT_TOMATO_IN_POT",
    "onion": "PUT_ONION_IN_POT",
}


def is_valid_subgoal(name: str) -> bool:
    return name in SUBGOAL_TO_INDEX
