"""Live H0 + learned-comfort subgoal agent (Path A).

The direct realization of "adapt to H0/subgoal": at every step the agent lets
H0 propose the task-valid feasible subgoals, re-ranks them with the reward
learned from language feedback, and executes the chosen subgoal through H0's
motion planner. Comfort therefore drives *which subgoal* is pursued -- not just
how smoothly a fixed subgoal is executed.

    state --context_from_state--> SubgoalContext
          --enumerate_feasible_subgoals--> feasible (H0 safety floor)
          --plan_subgoal(w, ...)--> chosen subgoal (comfort reranked)
          --execute_subgoal--> action (H0 motion planner)

Default path is pure rules + a frozen weight vector (no torch). Optionally,
``update_from_feedback(text)`` blends a Route 2 per-utterance prediction into
``w`` so live chat feedback can reshape preferences without retraining.
"""

from __future__ import annotations

from pathlib import Path

from durf.baseline.comfort_reward import ADAPTED_ROOT, context_from_state
from durf.baseline.h0_planner import execute_subgoal

from src.subgoal_planner import enumerate_feasible_subgoals, plan_subgoal  # noqa: E402

STAY = 4
DEFAULT_WEIGHTS_PATH = ADAPTED_ROOT / "outputs" / "route2" / "learned_comfort_weights.json"
GOLD_WEIGHTS_PATH = ADAPTED_ROOT / "data" / "gold_comfort_weights.json"
DEFAULT_MODEL_PATH = ADAPTED_ROOT / "outputs" / "route2" / "model.pt"


def load_comfort_weights(path: str | Path | None = None) -> dict[str, float]:
    """Load learned comfort weights, falling back to the gold teacher weights."""

    import json

    candidate = Path(path) if path else DEFAULT_WEIGHTS_PATH
    if not candidate.exists():
        candidate = GOLD_WEIGHTS_PATH
    with candidate.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k): float(v) for k, v in raw.items()}


class ComfortSubgoalAgent:
    """Choose subgoals with the learned reward; execute them through H0."""

    def __init__(
        self,
        motion_planner,
        *,
        weights: dict[str, float] | None = None,
        weights_path: str | Path | None = None,
        lambda_pref: float = 1.0,
        ai_index: int = 0,
        model_path: str | Path | None = None,
        online_blend: float = 0.35,
    ) -> None:
        self.motion_planner = motion_planner
        self.mdp = motion_planner.mdp
        self.weights = weights if weights is not None else load_comfort_weights(weights_path)
        self.base_weights = dict(self.weights)
        self.lambda_pref = float(lambda_pref)
        self.ai_index = int(ai_index)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.online_blend = float(online_blend)
        self._route2 = None  # lazy (model, vocab, features)
        self.last_feedback_update: str = ""

    def plan(self, state) -> dict:
        """Return the full re-ranking decision for the current state."""

        context = context_from_state(state, self.mdp, ai_index=self.ai_index)
        feasible = enumerate_feasible_subgoals(context)
        decision = plan_subgoal(
            self.weights, context, lambda_pref=self.lambda_pref, feasible_subgoals=feasible
        )
        decision["context"] = context
        return decision

    def act(self, state) -> tuple[int, str]:
        """Return ``(action_index, chosen_subgoal)`` for the AI player."""

        decision = self.plan(state)
        subgoal = decision["chosen_subgoal"]
        action = execute_subgoal(
            state, self.motion_planner, subgoal, player_index=self.ai_index
        )
        return int(action), subgoal

    def _ensure_route2(self):
        if self._route2 is not None:
            return self._route2
        if not self.model_path.exists():
            raise FileNotFoundError(f"Route 2 checkpoint not found: {self.model_path}")
        from src.neural_inference import load_checkpoint

        model, vocab, features, _use_fc = load_checkpoint(self.model_path)
        self._route2 = (model, vocab, features)
        return self._route2

    def update_from_feedback(self, text: str, *, blend: float | None = None) -> dict[str, float]:
        """Blend a Route 2 per-utterance ``w_hat`` into the live comfort weights.

        ``w <- (1-α)·w + α·w_hat``. Requires torch (Route 2 checkpoint). Returns
        the predicted ``w_hat``. No-op on empty text.
        """

        cleaned = (text or "").strip()
        if not cleaned:
            return {}
        from src.neural_inference import predict_reward_vector

        model, vocab, features = self._ensure_route2()
        w_hat = predict_reward_vector(model, vocab, features, cleaned)
        alpha = self.online_blend if blend is None else float(blend)
        alpha = max(0.0, min(1.0, alpha))
        for feature, value in w_hat.items():
            current = self.weights.get(feature, 0.0)
            self.weights[feature] = (1.0 - alpha) * current + alpha * float(value)
        self.last_feedback_update = cleaned
        return w_hat

    def reset_weights(self) -> None:
        """Restore the frozen export used at construction time."""

        self.weights = dict(self.base_weights)
        self.last_feedback_update = ""
