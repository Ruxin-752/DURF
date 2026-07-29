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

Four explicit feedback modes are available: frozen weights, paper-style Route
1 Literal, Route 1 PseudoPragmatic, and Route 2 neural blending. Route 1 keeps
an auditable Gaussian posterior and makes the feedback-form
classification/grounding/update visible in the live decision loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from durf.baseline.comfort_reward import ADAPTED_ROOT, context_from_state
from durf.baseline.h0_planner import execute_subgoal

from src.subgoal_planner import enumerate_feasible_subgoals, plan_subgoal  # noqa: E402
from src.feature_schema import load_features  # noqa: E402
from src.route1_online import OnlineRoute1Learner, ROUTE1_MODES  # noqa: E402
from src.trajectory_featurizer import featurize_trajectory_steps  # noqa: E402

STAY = 4
FEEDBACK_MODES = ("frozen", "route1-literal", "route1-pseudopragmatic", "route2")
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
        feedback_mode: str = "route2",
        route1_prior: str = "zero",
        route1_lookback: int = 25,
        human_feedback_precision: float = 4.0,
        learner_state_path: str | Path | None = None,
        resume_learner_state: bool = False,
        max_consecutive_wait: int = 3,
        subgoal_commitment_steps: int = 2,
        switch_margin: float = 0.05,
        deadlock_patience: int = 2,
    ) -> None:
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"Unknown feedback mode {feedback_mode!r}; choose from {FEEDBACK_MODES}")
        if route1_prior not in {"zero", "frozen"}:
            raise ValueError("route1_prior must be 'zero' or 'frozen'")
        if route1_lookback <= 0:
            raise ValueError("route1_lookback must be positive")
        if human_feedback_precision <= 0:
            raise ValueError("human_feedback_precision must be positive")
        if max_consecutive_wait < 0:
            raise ValueError("max_consecutive_wait must be non-negative")
        if subgoal_commitment_steps < 0:
            raise ValueError("subgoal_commitment_steps must be non-negative")
        if switch_margin < 0:
            raise ValueError("switch_margin must be non-negative")
        if deadlock_patience < 1:
            raise ValueError("deadlock_patience must be positive")
        self.motion_planner = motion_planner
        self.mdp = motion_planner.mdp
        self.weights = weights if weights is not None else load_comfort_weights(weights_path)
        frozen_weights = dict(self.weights)
        self.lambda_pref = float(lambda_pref)
        self.ai_index = int(ai_index)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.online_blend = float(online_blend)
        self.feedback_mode = feedback_mode
        self.route1_prior = route1_prior
        self.route1_lookback = int(route1_lookback)
        self.human_feedback_precision = float(human_feedback_precision)
        self.learner_state_path = (
            Path(learner_state_path) if learner_state_path is not None else None
        )
        self.max_consecutive_wait = int(max_consecutive_wait)
        self.subgoal_commitment_steps = int(subgoal_commitment_steps)
        self.switch_margin = float(switch_margin)
        self.deadlock_patience = int(deadlock_patience)
        self.consecutive_waits = 0
        self.committed_subgoal: str | None = None
        self.commitment_age = 0
        self._last_state_signature: str | None = None
        self._last_executed_action: int | None = None
        self._unchanged_state_steps = 0
        self._deadlock_detected = False
        self._route2 = None  # lazy (model, vocab, features)
        self._route1 = None
        self._route1_prior_mean = (
            frozen_weights if feedback_mode in ROUTE1_MODES and route1_prior == "frozen" else None
        )
        if feedback_mode in ROUTE1_MODES:
            self._route1 = OnlineRoute1Learner(
                load_features(),
                mode=feedback_mode,
                prior_mean=self._route1_prior_mean,
                source_precision={"human_live": self.human_feedback_precision},
            )
            self.weights = self._route1.weights()
        self.base_weights = dict(self.weights)
        self.last_decision: dict | None = None
        self.decision_history: list[dict] = []
        self.trajectory_history: list[dict] = []
        self.feedback_history: list[dict] = []
        self.last_feedback_update: str = ""
        if resume_learner_state:
            if self.learner_state_path is None:
                raise ValueError("resume_learner_state requires learner_state_path")
            self.load_learner_state(self.learner_state_path)

    @staticmethod
    def _state_signature(state) -> str:
        """Stable signature of actual player/object state, excluding timestep."""

        players = []
        for player in getattr(state, "players", ()):
            held = getattr(player, "held_object", None)
            players.append(
                (
                    tuple(getattr(player, "position", ())),
                    tuple(getattr(player, "orientation", ())),
                    getattr(held, "name", None),
                )
            )
        objects = []
        for position, obj in sorted(getattr(state, "objects", {}).items()):
            objects.append(
                (
                    tuple(position),
                    getattr(obj, "name", None),
                    tuple(
                        getattr(ingredient, "name", ingredient)
                        for ingredient in getattr(obj, "ingredients", ())
                    ),
                    bool(getattr(obj, "is_cooking", False)),
                    bool(getattr(obj, "is_ready", False)),
                )
            )
        return repr((players, objects))

    def _observe_execution_progress(self, state) -> None:
        signature = self._state_signature(state)
        unchanged = (
            self._last_state_signature is not None
            and signature == self._last_state_signature
            and self._last_executed_action is not None
        )
        self._unchanged_state_steps = self._unchanged_state_steps + 1 if unchanged else 0
        self._deadlock_detected = self._unchanged_state_steps >= self.deadlock_patience

    @staticmethod
    def _score_for(decision: dict, subgoal: str) -> float | None:
        item = next(
            (row for row in decision.get("ranking", []) if row.get("subgoal") == subgoal),
            None,
        )
        return float(item["total_score"]) if item is not None else None

    def _stabilize_decision(self, decision: dict, *, force_switch: bool = False) -> dict:
        raw_choice = decision["chosen_subgoal"]
        current = self.committed_subgoal
        feasible = decision.get("feasible_subgoals", [])
        reason = "raw_argmax"

        if force_switch:
            alternatives = [
                row["subgoal"]
                for row in decision.get("ranking", [])
                if row["subgoal"] != "WAIT" and row["subgoal"] != current
            ]
            if alternatives:
                decision["chosen_subgoal"] = alternatives[0]
                reason = "deadlock_nonwait_fallback"
            elif raw_choice == "WAIT":
                nonwait = [name for name in feasible if name != "WAIT"]
                if nonwait:
                    decision["chosen_subgoal"] = nonwait[0]
                    reason = "deadlock_nonwait_fallback"
        elif current in feasible and raw_choice != current:
            current_score = self._score_for(decision, current)
            challenger_score = self._score_for(decision, raw_choice)
            if self.commitment_age < self.subgoal_commitment_steps:
                decision["chosen_subgoal"] = current
                reason = "subgoal_commitment"
            elif (
                current_score is not None
                and challenger_score is not None
                and challenger_score < current_score + self.switch_margin
            ):
                decision["chosen_subgoal"] = current
                reason = "switch_hysteresis"

        decision["raw_chosen_subgoal"] = raw_choice
        decision["policy_switch"] = (
            current is not None and decision["chosen_subgoal"] != current
        )
        decision["stability_reason"] = reason
        decision["commitment_age"] = self.commitment_age
        return decision

    def plan(self, state) -> dict:
        """Return the full re-ranking decision for the current state."""

        context = context_from_state(state, self.mdp, ai_index=self.ai_index)
        feasible = enumerate_feasible_subgoals(context)
        wait_guard_applied = (
            self.max_consecutive_wait > 0
            and self.consecutive_waits >= self.max_consecutive_wait
            and any(subgoal != "WAIT" for subgoal in feasible)
        )
        if wait_guard_applied:
            feasible = [subgoal for subgoal in feasible if subgoal != "WAIT"]
        decision = plan_subgoal(
            self.weights, context, lambda_pref=self.lambda_pref, feasible_subgoals=feasible
        )
        decision = self._stabilize_decision(
            decision, force_switch=self._deadlock_detected
        )
        decision["context"] = context
        decision["wait_guard_applied"] = wait_guard_applied
        decision["deadlock_detected"] = self._deadlock_detected
        decision["unchanged_state_steps"] = self._unchanged_state_steps
        self.last_decision = decision
        self.decision_history.append(decision)
        del self.decision_history[:-self.route1_lookback]
        return decision

    def act(self, state) -> tuple[int, str]:
        """Return ``(action_index, chosen_subgoal)`` for the AI player."""

        self._observe_execution_progress(state)
        decision = self.plan(state)
        subgoal = decision["chosen_subgoal"]
        action = execute_subgoal(
            state, self.motion_planner, subgoal, player_index=self.ai_index
        )
        if (
            (self._deadlock_detected or (subgoal == "WAIT" and decision["wait_guard_applied"]))
            and (subgoal == "WAIT" or int(action) == STAY)
        ):
            for row in decision.get("ranking", []):
                fallback = row["subgoal"]
                if fallback in {"WAIT", subgoal}:
                    continue
                fallback_action = execute_subgoal(
                    state, self.motion_planner, fallback, player_index=self.ai_index
                )
                subgoal, action = fallback, fallback_action
                decision["chosen_subgoal"] = fallback
                decision["stability_reason"] = "executable_nonwait_fallback"
                if int(action) != STAY:
                    break
        decision["policy_switch"] = (
            self.committed_subgoal is not None
            and subgoal != self.committed_subgoal
        )
        if subgoal == self.committed_subgoal:
            self.commitment_age += 1
        else:
            self.committed_subgoal = subgoal
            self.commitment_age = 1
        self._last_state_signature = self._state_signature(state)
        self._last_executed_action = int(action)
        decision["executed_action"] = int(action)
        self.consecutive_waits = self.consecutive_waits + 1 if subgoal == "WAIT" else 0
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

    def _decision_after_update(self) -> str | None:
        if not self.last_decision:
            return None
        context = self.last_decision.get("context")
        feasible = [item["subgoal"] for item in self.last_decision.get("ranking", [])]
        if context is None or not feasible:
            return None
        decision = plan_subgoal(
            self.weights,
            context,
            lambda_pref=self.lambda_pref,
            feasible_subgoals=feasible,
        )
        return self._stabilize_decision(decision)["chosen_subgoal"]

    @staticmethod
    def _annotate_policy_switch(trace: dict) -> None:
        accepted = trace.get("status") == "updated"
        before = trace.get("before_subgoal")
        after = trace.get("after_subgoal")
        trace["policy_switch"] = bool(accepted and before != after)
        trace["accepted_but_no_policy_switch"] = bool(
            accepted and before is not None and before == after
        )

    def _recent_trajectory_features(self) -> tuple[dict[str, float], str]:
        """Ground evaluative feedback to recent executed steps, with a plan fallback."""

        executed = featurize_trajectory_steps(self.trajectory_history)
        if executed:
            return executed, "executed_steps"

        merged: dict[str, float] = {}
        for decision in self.decision_history:
            chosen = decision.get("chosen_subgoal")
            selected = next(
                (
                    item
                    for item in decision.get("ranking", [])
                    if item.get("subgoal") == chosen
                ),
                None,
            )
            for feature, value in (selected or {}).get("features", {}).items():
                merged[str(feature)] = merged.get(str(feature), 0.0) + float(value)
        return merged, "selected_subgoals"

    def record_transition(
        self,
        *,
        state_before: dict,
        state_after: dict,
        ai_action_name: str,
        human_action_name: str,
        environment_reward: float,
    ) -> None:
        """Append one normalized live step for paper-style trajectory grounding."""

        self.trajectory_history.append(
            {
                "ai_action_name": ai_action_name,
                "human_action_name": human_action_name,
                "environment_reward": float(environment_reward),
                "state_facts": state_after,
                "extra": {"state_before": state_before},
            }
        )
        del self.trajectory_history[:-self.route1_lookback]

    def update_from_feedback(
        self,
        text: str,
        *,
        blend: float | None = None,
        oracle_feedback: dict | None = None,
        interpretation: str = "inferred",
        source: str = "human_live",
        confidence: float = 1.0,
    ) -> dict:
        """Update the selected Route 1/Route 2 learner and return an audit trace."""

        cleaned = (text or "").strip()
        if not cleaned:
            return {"status": "empty", "mode": self.feedback_mode, "text": ""}

        before_subgoal = (
            self.last_decision.get("chosen_subgoal") if self.last_decision else None
        )
        if self.feedback_mode == "frozen":
            trace = {
                "status": "ignored",
                "mode": "frozen",
                "text": cleaned,
                "before_subgoal": before_subgoal,
                "after_subgoal": before_subgoal,
            }
            self.feedback_history.append(trace)
            self.last_feedback_update = cleaned
            return trace

        if self.feedback_mode in ROUTE1_MODES:
            if self._route1 is None:
                raise RuntimeError("Route 1 learner was not initialized")
            trajectory_features, trajectory_source = self._recent_trajectory_features()
            trace = self._route1.update(
                cleaned,
                decision=self.last_decision,
                trajectory_features=trajectory_features,
                oracle_feedback=oracle_feedback,
                interpretation=interpretation,
                source=source,
                confidence=confidence,
            )
            trace["trajectory_window"] = (
                len(self.trajectory_history)
                if self.trajectory_history
                else len(self.decision_history)
            )
            trace["trajectory_source"] = trajectory_source
            self.weights = self._route1.weights()
            trace["before_subgoal"] = before_subgoal
            trace["after_subgoal"] = self._decision_after_update()
            self._annotate_policy_switch(trace)
            self.last_feedback_update = cleaned
            if trace["status"] == "updated" and self.learner_state_path is not None:
                saved = self.save_learner_state(self.learner_state_path)
                trace["checkpoint_path"] = str(saved)
                trace["checkpoint_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
            self.feedback_history.append(trace)
            return trace

        from src.neural_inference import predict_reward_vector

        model, vocab, features = self._ensure_route2()
        w_hat = predict_reward_vector(model, vocab, features, cleaned)
        alpha = self.online_blend if blend is None else float(blend)
        alpha = max(0.0, min(1.0, alpha))
        before_weights = dict(self.weights)
        for feature, value in w_hat.items():
            current = self.weights.get(feature, 0.0)
            self.weights[feature] = (1.0 - alpha) * current + alpha * float(value)
        deltas = {
            feature: self.weights[feature] - before_weights.get(feature, 0.0)
            for feature in self.weights
            if abs(self.weights[feature] - before_weights.get(feature, 0.0)) > 1e-9
        }
        changed = sorted(deltas, key=lambda feature: abs(deltas[feature]), reverse=True)
        trace = {
            "status": "updated",
            "mode": "route2",
            "interpretation": "neural",
            "text": cleaned,
            "blend": alpha,
            "source": source,
            "before_subgoal": before_subgoal,
            "after_subgoal": self._decision_after_update(),
            "weight_delta": {feature: deltas[feature] for feature in changed},
            "top_changes": [
                {
                    "feature": feature,
                    "before": before_weights.get(feature, 0.0),
                    "after": self.weights[feature],
                    "delta": deltas[feature],
                }
                for feature in changed[:12]
            ],
        }
        self._annotate_policy_switch(trace)
        self.feedback_history.append(trace)
        self.last_feedback_update = cleaned
        return trace

    def learner_state_dict(self) -> dict:
        """Return a versioned online learner checkpoint."""

        payload = {
            "version": 1,
            "feedback_mode": self.feedback_mode,
            "route1_prior": self.route1_prior,
            "human_feedback_precision": self.human_feedback_precision,
            "base_weights_sha256": hashlib.sha256(
                json.dumps(self.base_weights, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        if self._route1 is not None:
            payload["route1"] = self._route1.state_dict()
        else:
            payload["weights"] = dict(self.weights)
        return payload

    def save_learner_state(self, path: str | Path | None = None) -> Path:
        """Atomically save the complete posterior for exact process resume."""

        target = Path(path) if path is not None else self.learner_state_path
        if target is None:
            raise ValueError("learner state path is not configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.learner_state_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def load_learner_state(self, path: str | Path | None = None) -> None:
        """Restore a compatible learner checkpoint."""

        target = Path(path) if path is not None else self.learner_state_path
        if target is None:
            raise ValueError("learner state path is not configured")
        state = json.loads(target.read_text(encoding="utf-8"))
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported comfort learner checkpoint version")
        if state.get("feedback_mode") != self.feedback_mode:
            raise ValueError("checkpoint feedback mode does not match agent")
        if self._route1 is not None:
            self._route1.load_state_dict(state["route1"])
            self.weights = self._route1.weights()
        else:
            self.weights = {
                str(key): float(value)
                for key, value in (state.get("weights") or {}).items()
            }

    def reset_weights(self) -> None:
        """Restore the initial frozen vector or Route 1 prior."""

        if self._route1 is not None:
            self._route1.reset(prior_mean=self._route1_prior_mean)
            self.weights = self._route1.weights()
        else:
            self.weights = dict(self.base_weights)
        self.reset_context()
        self.feedback_history.clear()
        self.last_feedback_update = ""

    def reset_context(self) -> None:
        """Clear temporal grounding state while preserving the learned posterior."""

        self.last_decision = None
        self.decision_history.clear()
        self.trajectory_history.clear()
        self.consecutive_waits = 0
        self.committed_subgoal = None
        self.commitment_age = 0
        self._last_state_signature = None
        self._last_executed_action = None
        self._unchanged_state_steps = 0
        self._deadlock_detected = False

