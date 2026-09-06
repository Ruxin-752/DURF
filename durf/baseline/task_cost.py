"""Ground the task prior in a quantity that means something: steps to delivery.

The legacy ``task_score`` (100 for SERVE_SOUP, 70 for GET_TOMATO, 0 for WAIT)
is a priority encoding lifted from the rule teacher.  Its gaps carry no unit,
so "how many task points may we give up for the user's preference?" has no
answerable form, and the observed gaps are lumpy (10, 50, 60, 70, 90) rather
than continuous.

This module estimates, for each candidate subgoal, how many timesteps the agent
would still need to deliver the next soup if it committed to that subgoal.
Lower is better, so the task prior becomes ``-estimated_steps`` and the
satisficing tolerance is denominated in timesteps: "we allow the agent to spend
at most N extra steps to honour the user's preference."

Two deliberate simplifications, both of which keep the estimate a *task* prior
rather than a coordination judgement:

* the partner is not modelled -- the estimate answers "how much work is left",
  not "who will do it".  Partner facts reach the decision only as condition
  features consumed by the preference model.
* the plan is serial and greedy over the remaining recipe. It is an ordering
  signal, not a solver.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_COOK_TIME = 20
# Anything that cannot lead to a delivery on its own still has to be comparable
# with the options that can, so idling is priced as "the work is all still ahead
# of you, plus the step you just spent".
IDLE_STEP_COST = 1.0


@dataclass(frozen=True)
class FeatureMap:
    """Feature positions and the pairwise walk-and-interact costs between them."""

    pot: tuple
    dish: tuple
    serving: tuple
    tomato: tuple
    onion: tuple
    pairwise: dict

    def cost(self, source: str, target: str) -> float:
        return self.pairwise[(source, target)]


def build_feature_map(mdp, motion_planner) -> FeatureMap:
    features = {
        "pot": tuple(mdp.get_pot_locations()),
        "dish": tuple(mdp.get_dish_dispenser_locations()),
        "serving": tuple(mdp.get_serving_locations()),
        "tomato": tuple(mdp.get_tomato_dispenser_locations()),
        "onion": tuple(mdp.get_onion_dispenser_locations()),
    }
    pairwise: dict = {}
    for source, source_positions in features.items():
        for target, target_positions in features.items():
            if not source_positions or not target_positions:
                pairwise[(source, target)] = float("inf")
                continue
            if source == target:
                pairwise[(source, target)] = 0.0
                continue
            pairwise[(source, target)] = float(
                motion_planner.min_cost_between_features(
                    list(source_positions), list(target_positions), manhattan_if_fail=True
                )
            )
    return FeatureMap(
        pot=features["pot"], dish=features["dish"], serving=features["serving"],
        tomato=features["tomato"], onion=features["onion"], pairwise=pairwise,
    )


def target_recipe(mdp) -> list[str]:
    """The layout's recipe -- the same source the candidate generator uses
    (collect_rule_teacher_dataset.target_recipe), so both price the same
    job.  Reading it off ``state.all_orders`` instead silently substituted
    Recipe.ALL_RECIPES[0] on states built outside the environment, which made
    a held, needed tomato look unneeded and dropping it look cheap."""
    orders = getattr(mdp, "start_all_orders", None) or []
    if not orders:
        return ["onion", "onion", "onion"]
    order = orders[0]
    if isinstance(order, dict):
        return list(order.get("ingredients", []))
    return list(getattr(order, "ingredients", order))


def missing_ingredients(state, mdp) -> list[str]:
    """Ingredients the current order still needs in the pot, as a flat list."""
    recipe = target_recipe(mdp)
    in_pot: list[str] = []
    for position in mdp.get_pot_locations():
        soup = state.objects.get(tuple(position))
        if soup is not None and getattr(soup, "name", None) == "soup":
            in_pot.extend(list(getattr(soup, "ingredients", []) or []))
    remaining = list(recipe)
    for ingredient in in_pot:
        if ingredient in remaining:
            remaining.remove(ingredient)
    return remaining


def pot_status(state, mdp) -> tuple[bool, bool, float]:
    """(is_cooking, is_ready, cook steps still to elapse)."""
    for position in mdp.get_pot_locations():
        soup = state.objects.get(tuple(position))
        if soup is None or getattr(soup, "name", None) != "soup":
            continue
        if getattr(soup, "is_ready", False):
            return False, True, 0.0
        if getattr(soup, "is_cooking", False):
            cook_time = float(getattr(soup, "cook_time", DEFAULT_COOK_TIME) or DEFAULT_COOK_TIME)
            elapsed = float(getattr(soup, "_cooking_tick", 0) or 0)
            return True, False, max(0.0, cook_time - elapsed)
    return False, False, float(DEFAULT_COOK_TIME)


def _finish(
    features: FeatureMap,
    first_hop,
    held: str | None,
    missing: list[str],
    cook_left: float,
) -> float:
    """Steps left to deliver, taking the cheapest order over the missing items.

    ``first_hop(target)`` gives the cost from wherever the agent currently is to
    that feature; later legs are feature-to-feature lookups.  The remaining
    ingredients are tried in every order (at most 3! = 6) so the estimate does
    not depend on the order the recipe happens to list them in -- otherwise two
    candidates could be scored against different plans.
    """
    best = float("inf")
    for order in _permutations(missing):
        best = min(best, _plan_cost(features, first_hop, held, list(order), cook_left))
    return best


def _permutations(items: list[str]) -> list[tuple]:
    if len(items) <= 1:
        return [tuple(items)]
    out = []
    for index, item in enumerate(items):
        for rest in _permutations(items[:index] + items[index + 1:]):
            out.append((item,) + rest)
    return out


def _plan_cost(
    features: FeatureMap,
    first_hop,
    held: str | None,
    missing: list[str],
    cook_left: float,
) -> float:
    total = 0.0
    here = None          # None means "still at the agent's own position"

    def hop(target: str) -> float:
        return first_hop(target) if here is None else features.cost(here, target)

    if held in ("tomato", "onion") and held in missing:
        total += hop("pot")
        here = "pot"
        missing.remove(held)
        held = None
    elif held in ("tomato", "onion"):
        # An ingredient no pot needs has to be put down before anything else
        # can be picked up.  A free counter is almost always adjacent, so this
        # is priced as one step rather than routed.
        total += IDLE_STEP_COST
        held = None

    for ingredient in missing:
        total += hop(ingredient)
        here = ingredient
        total += features.cost(ingredient, "pot")
        here = "pot"

    total += cook_left

    if held == "soup":
        return total + hop("serving")
    if held != "dish":
        total += hop("dish")
        here = "dish"
    total += hop("pot")
    here = "pot"
    return total + features.cost("pot", "serving")


_SUBGOAL_ANCHOR = {
    "GET_TOMATO": ("tomato", "tomato", None),
    "GET_ONION": ("onion", "onion", None),
    "GET_DISH": ("dish", "dish", None),
    "PUT_TOMATO_IN_POT": ("pot", None, "tomato"),
    "PUT_ONION_IN_POT": ("pot", None, "onion"),
    "PICKUP_SOUP": ("pot", "soup", None),
    "SERVE_SOUP": ("serving", None, None),
}
# Subgoals that leave the world exactly as it is.  They matter to the
# satisficing rule (collect_rule_teacher_dataset.choose_task_candidate): a
# preference for one of these over a state-changing optimum is offered again,
# unchanged, at the very next tick, so any per-decision tolerance compounds
# without bound.  PUT_DOWN_OBJECT is deliberately NOT here -- it frees the
# hands, the situation moves on -- and is priced on its own plan below.
WAITING_SUBGOALS = {"WAIT", "WAIT_NEAR_POT", "CONTINUE_CURRENT_SUBGOAL"}
# Subgoals with no _SUBGOAL_ANCHOR: priced relative to the best active plan
# (waiting) or on an explicit "drop it, then carry on" plan (PUT_DOWN_OBJECT).
IDLE_SUBGOALS = WAITING_SUBGOALS | {"PUT_DOWN_OBJECT"}


def _estimate_active(subgoal, *, features, motion_planner, player, state, mdp):
    missing = missing_ingredients(state, mdp)
    _, is_ready, cook_left = pot_status(state, mdp)

    def from_player(target: str) -> float:
        positions = list(getattr(features, target))
        if not positions:
            return float("inf")
        return float(motion_planner.min_cost_to_feature(player.pos_and_or, positions))

    if subgoal == "SERVE_SOUP":
        return from_player("serving")

    effect = _SUBGOAL_ANCHOR.get(subgoal)
    if effect is None and subgoal == "GET_USEFUL_INGREDIENT":
        ingredient = missing[0] if missing else "tomato"
        effect = (ingredient, ingredient, None)
    if effect is None:
        return None

    anchor, becomes_held, consumes = effect
    reach = from_player(anchor)
    remaining = list(missing)
    if consumes and consumes in remaining:
        remaining.remove(consumes)
    if subgoal == "PICKUP_SOUP" and not is_ready:
        reach += cook_left
        cook_left = 0.0
    return reach + _finish(
        features, lambda target: features.cost(anchor, target),
        becomes_held, remaining, cook_left,
    )


def attach_step_costs(
    candidates, *, features, motion_planner, player, state, mdp
) -> None:
    """Set ``candidate.step_cost`` on each CandidateSubgoal in place.

    Idle candidates are priced off the cheapest *real* action (see
    ``estimate_candidate_costs``).  A candidate the task layer itself rates
    below waiting -- negative ``task_score``, e.g. a second dish while the
    partner already carries one -- is not a real action by the task's own
    judgement, so it must not become the anchor that makes waiting look cheap.
    It still gets its own cost so a preference can reach it through the band.
    """
    anchored = [c for c in candidates if float(c.task_score) >= 0.0]
    costs = estimate_candidate_costs(
        [candidate.subgoal for candidate in anchored],
        features=features, motion_planner=motion_planner,
        player=player, state=state, mdp=mdp,
    )
    active_costs = [v for k, v in costs.items() if k not in IDLE_SUBGOALS and v is not None]
    best_active = min(active_costs) if active_costs else None
    for candidate in candidates:
        if float(candidate.task_score) < 0.0 and candidate.subgoal not in IDLE_SUBGOALS:
            # A candidate the task layer rates below waiting is REDUNDANT work:
            # a second dish while the partner already carries one.  The
            # partner-blind estimator prices it as if it led to the next
            # delivery (28 steps in the first live session -- cheaper than the
            # real optimum at 38), which put it inside the tolerance band and
            # sent the agent for a second plate after one generic "go get the
            # plate".  Price it like PUT_DOWN_OBJECT instead: its own steps,
            # and then the whole real job is still ahead.
            own = _estimate_active(
                candidate.subgoal, features=features, motion_planner=motion_planner,
                player=player, state=state, mdp=mdp,
            )
            if own is None:
                value = None
            elif best_active is None:
                value = own
            else:
                value = float(own) + float(best_active)
        else:
            value = costs.get(candidate.subgoal)
        candidate.step_cost = float(value) if value is not None else None


def estimate_candidate_costs(
    subgoals: list[str], *, features, motion_planner, player, state, mdp
) -> dict:
    """Steps-to-delivery for a whole candidate set.

    Idling is priced off the best available action rather than off an imagined
    optimal plan: standing still costs one step and gains nothing, so it can
    never look cheaper than acting.  Pricing it against the optimal plan instead
    would quietly reward WAIT whenever the candidate generator failed to offer
    the move that plan needs -- an estimator must not paper over that.
    """
    costs: dict = {}
    for subgoal in subgoals:
        if subgoal in IDLE_SUBGOALS:
            continue
        value = _estimate_active(
            subgoal, features=features, motion_planner=motion_planner,
            player=player, state=state, mdp=mdp,
        )
        if value is not None:
            costs[subgoal] = value

    best_active = min(costs.values()) if costs else None
    missing = missing_ingredients(state, mdp)
    _, _, cook_left = pot_status(state, mdp)
    held = getattr(getattr(player, "held_object", None), "name", None)

    def from_player(target: str) -> float:
        positions = list(getattr(features, target))
        return (float(motion_planner.min_cost_to_feature(player.pos_and_or, positions))
                if positions else float("inf"))

    for subgoal in subgoals:
        if subgoal not in IDLE_SUBGOALS:
            continue
        if subgoal == "PUT_DOWN_OBJECT":
            # Drop what is held (one step), then the whole job is still ahead
            # -- including re-fetching the item if the pot needed it.  Priced
            # off best_active + 1 this read as "as good as acting" whenever
            # anything else was on the table, which is exactly wrong for
            # dropping a needed ingredient.
            costs[subgoal] = IDLE_STEP_COST + _finish(
                features, from_player, None, list(missing), cook_left)
        elif best_active is not None:
            costs[subgoal] = best_active + IDLE_STEP_COST
        else:
            costs[subgoal] = IDLE_STEP_COST + _finish(
                features, from_player, held, list(missing), cook_left)
    return costs
