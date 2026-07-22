"""Bridge the learned comfort reward into the live Overcooked env.

The Route 2 stage exports a frozen comfort weight vector ``w`` over the shared
53-dim feature schema (``outputs/route2/learned_comfort_weights.json``). This
module turns ``w`` into a per-step scalar shaping term for PPO:

    r_comfort(state) = coeff * ( w . phi_comfort(context, ai_subgoal) )

where the AI's committed subgoal is what H0 would pick, ``phi`` is the same
featurizer used in learning, and only the coordination/comfort feature
partition is scored (task progress is already rewarded by the env). This keeps
training and runtime on ONE feature schema.

Runtime dependency note: only numpy-level modules from the offline Baseline B
package are imported (featurizer / reranker / schema / score_action). No torch
or nltk is needed here, so it is safe to import inside the RLlib/tf training
process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTED_ROOT = (
    REPO_ROOT
    / "baselines"
    / "baseline_b_linguistic_feedback"
    / "adapted_overcooked"
)
if str(ADAPTED_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTED_ROOT))

from src.belief_model import score_action  # noqa: E402
from src.human_intent import infer_human_intent  # noqa: E402
from src.subgoal_featurizer import SubgoalContext, featurize_subgoal  # noqa: E402
from src.subgoal_reranker import COMFORT_FEATURES  # noqa: E402

from durf.baseline.h0_planner import (  # noqa: E402
    pot_ingredients,
    plan_h0_subgoal,
    target_recipe,
)


DEFAULT_WEIGHTS_PATH = ADAPTED_ROOT / "outputs" / "route2" / "learned_comfort_weights.json"


def _held_name(player) -> str | None:
    held = getattr(player, "held_object", None)
    return getattr(held, "name", None) if held is not None else None


def _pot_snapshot(state, mdp) -> tuple[list[str], str]:
    """Return (ingredients, status) of the most-progressed pot."""

    pot_states = mdp.get_pot_states(state)
    ready = set(mdp.get_ready_pots(pot_states))
    cooking = set(mdp.get_cooking_pots(pot_states))

    best_ingredients: list[str] = []
    best_status = "empty"
    best_rank = -1
    status_rank = {"empty": 0, "partial": 1, "cooking": 2, "ready": 3}
    for pot_pos in mdp.get_pot_locations():
        ingredients = pot_ingredients(state, pot_pos)
        if pot_pos in ready:
            status = "ready"
        elif pot_pos in cooking:
            status = "cooking"
        elif ingredients:
            status = "partial"
        else:
            status = "empty"
        if status_rank[status] > best_rank:
            best_rank = status_rank[status]
            best_status = status
            best_ingredients = ingredients
    return best_ingredients, best_status


def context_from_state(state, mdp, ai_index: int = 0) -> SubgoalContext:
    """Build a :class:`SubgoalContext` from a live Overcooked state."""

    human_index = 1 - ai_index if len(state.players) == 2 else (ai_index + 1) % len(state.players)
    ai_player = state.players[ai_index]
    human_player = state.players[human_index]

    ingredients, status = _pot_snapshot(state, mdp)
    human_holding = _held_name(human_player)
    human_intent = infer_human_intent(human_holding=human_holding)

    return SubgoalContext(
        recipe=list(target_recipe(mdp)),
        pot_ingredients=list(ingredients),
        pot_status=status,
        agent_holding=_held_name(ai_player),
        human_holding=human_holding,
        human_intent=human_intent,
    )


class ComfortReward:
    """Frozen learned comfort reward, evaluated per step for PPO shaping."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        coeff: float = 1.0,
    ) -> None:
        if weights is None:
            weights = self.load_weights(weights_path)
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.comfort_weights = {
            feature: value
            for feature, value in self.weights.items()
            if feature in COMFORT_FEATURES
        }
        self.coeff = float(coeff)

    @staticmethod
    def load_weights(path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Comfort weights must be a JSON object: {path}")
        return {str(k): float(v) for k, v in raw.items()}

    def comfort_score(self, state, mdp, ai_index: int = 0) -> float:
        """Raw ``w . phi_comfort`` for the AI's committed subgoal (no coeff)."""

        context = context_from_state(state, mdp, ai_index=ai_index)
        subgoal = plan_h0_subgoal(state, mdp, player_index=ai_index)
        phi = featurize_subgoal(context, subgoal)
        comfort_phi = {f: v for f, v in phi.items() if f in COMFORT_FEATURES}
        return float(score_action(self.comfort_weights, comfort_phi))

    def shaping(self, state, mdp, ai_index: int = 0) -> float:
        """Comfort shaping term ``coeff * comfort_score`` for one step."""

        return self.coeff * self.comfort_score(state, mdp, ai_index=ai_index)
