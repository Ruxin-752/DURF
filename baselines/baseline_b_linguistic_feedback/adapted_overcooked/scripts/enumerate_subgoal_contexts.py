"""Enumerate subgoal-decision contexts -- the free, unlimited state supply.

Systematically sweeps recipe progress x pot status x agent holding x human
intent, keeps only the *decision points* where H0 has >= 2 feasible subgoals
(so the learned comfort reward actually gets to choose), de-duplicates, and
labels each with the gold rule-teacher. The result feeds Phase 2 language
synthesis.

This replaces the paper's human "reward configs": instead of collecting human
teachers, we procedurally cover the state space of the ring layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import write_json  # noqa: E402
from src.human_intent import RESOURCE_TO_INTENT  # noqa: E402
from src.subgoal_featurizer import SubgoalContext  # noqa: E402
from src.subgoal_planner import enumerate_feasible_subgoals  # noqa: E402
from src.subgoal_teacher import label_context, load_gold_weights  # noqa: E402


DEFAULT_RECIPE = ["tomato", "tomato", "onion"]

# (pot_ingredients, pot_status) pairs consistent with the ring recipe.
POT_STATES: tuple[tuple[list[str], str], ...] = (
    ([], "empty"),
    (["tomato"], "partial"),
    (["tomato", "tomato"], "partial"),
    (["tomato", "tomato", "onion"], "cooking"),
    (["tomato", "tomato", "onion"], "ready"),
)

AGENT_HOLDINGS: tuple[str | None, ...] = (None, "tomato", "onion", "dish", "soup")

# Human intents (which resource the human is competing for); None = human idle.
HUMAN_RESOURCES: tuple[str | None, ...] = (None, "tomato", "onion", "dish", "soup", "serving")


def _slug(context: SubgoalContext) -> str:
    pot = "-".join(context.pot_ingredients) or "empty"
    hold = context.agent_holding or "empty"
    human = context.human_intent or "idle"
    return f"pot[{pot}]_{context.pot_status}_hold[{hold}]_human[{human}]"


def enumerate_contexts(
    weights: dict[str, float],
    *,
    recipe: list[str] | None = None,
    lambda_pref: float = 1.0,
) -> list[dict]:
    recipe = list(recipe or DEFAULT_RECIPE)
    seen: set[str] = set()
    contexts: list[dict] = []

    for pot_ingredients, pot_status in POT_STATES:
        for agent_holding in AGENT_HOLDINGS:
            for human_resource in HUMAN_RESOURCES:
                human_intent = (
                    RESOURCE_TO_INTENT.get(human_resource) if human_resource else None
                )
                context = SubgoalContext(
                    recipe=list(recipe),
                    pot_ingredients=list(pot_ingredients),
                    pot_status=pot_status,
                    agent_holding=agent_holding,
                    human_holding=None,
                    human_intent=human_intent,
                )
                feasible = enumerate_feasible_subgoals(context)
                if len(feasible) < 2:
                    continue
                slug = _slug(context)
                if slug in seen:
                    continue
                seen.add(slug)

                label = label_context(
                    weights, context, feasible, lambda_pref=lambda_pref
                )
                contexts.append(
                    {
                        "scenario_id": f"ctx_{len(contexts):04d}_{slug}",
                        "context": {
                            "recipe": context.recipe,
                            "pot_ingredients": context.pot_ingredients,
                            "pot_status": context.pot_status,
                            "agent_holding": context.agent_holding,
                            "human_holding": context.human_holding,
                            "human_intent": context.human_intent,
                        },
                        "feasible_subgoals": feasible,
                        "expected_subgoal": label["expected_subgoal"],
                        "acceptable_subgoals": label["acceptable_subgoals"],
                    }
                )
    return contexts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "synth" / "contexts.json"
    )
    args = parser.parse_args()

    weights = load_gold_weights()
    contexts = enumerate_contexts(weights, lambda_pref=args.lambda_pref)
    write_json(args.output, contexts)

    print(f"Enumerated {len(contexts)} decision-point contexts (>=2 feasible subgoals).")
    by_expected: dict[str, int] = {}
    for entry in contexts:
        by_expected[entry["expected_subgoal"]] = (
            by_expected.get(entry["expected_subgoal"], 0) + 1
        )
    print("Gold expected-subgoal distribution:")
    for subgoal, count in sorted(by_expected.items()):
        print(f"  {subgoal}: {count}")
    print(f"Contexts JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
