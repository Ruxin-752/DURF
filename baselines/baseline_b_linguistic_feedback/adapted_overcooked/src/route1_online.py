"""Online paper-style Route 1 updates for the live subgoal agent."""

from __future__ import annotations

import copy
import re

import numpy as np

from .feedback_form_classifier import classify_feedback
from .feedback_observations import (
    build_feedback_observations,
    without_privileged_labels,
)
from .reward_weight_model import BayesianRewardLearner


ROUTE1_LITERAL = "route1-literal"
ROUTE1_PSEUDOPRAGMATIC = "route1-pseudopragmatic"
ROUTE1_MODES = frozenset({ROUTE1_LITERAL, ROUTE1_PSEUDOPRAGMATIC})
DEFAULT_SOURCE_PRECISION = {
    "synthetic": 1.0,
    "template": 1.0,
    "offline_human": 2.0,
    "human_live": 4.0,
}
STATE_VERSION = 1

_SUBGOAL_ALIASES: dict[str, tuple[str, ...]] = {
    "GET_TOMATO": ("tomato", "get tomato", "grab tomato", "take tomato"),
    "PUT_TOMATO_IN_POT": ("tomato", "put tomato", "pot tomato"),
    "GET_ONION": ("onion", "get onion", "grab onion", "take onion"),
    "PUT_ONION_IN_POT": ("onion", "put onion", "pot onion"),
    "GET_DISH": ("dish", "plate", "get dish", "grab dish"),
    "PICKUP_SOUP": ("soup", "pick up soup", "plate soup"),
    "SERVE_SOUP": ("serve", "deliver", "serve soup"),
    "WAIT": ("wait", "stay", "hold back", "pause", "step aside"),
}


def infer_target_action(
    text: str,
    action_feature_library: dict[str, dict[str, float]],
) -> str | None:
    """Infer the commanded feasible subgoal from simple lexical aliases."""

    lowered = text.lower()
    scored: list[tuple[int, int, str]] = []
    for order, action in enumerate(action_feature_library):
        aliases = _SUBGOAL_ALIASES.get(action, ())
        score = max((len(alias) for alias in aliases if alias in lowered), default=0)
        if score:
            scored.append((score, -order, action))
    return max(scored)[2] if scored else None


def _semantic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "a", "an", "the", "please", "just", "really", "you", "could", "would"
    }


def _semantic_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _semantic_tokens(left), _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return float(left.strip().lower() == right.strip().lower())
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class OnlineRoute1Learner:
    """Classify, ground, and absorb live feedback with a Gaussian posterior."""

    def __init__(
        self,
        features: list[str],
        *,
        mode: str = ROUTE1_LITERAL,
        prior_mean: dict[str, float] | None = None,
        valence_scale: float = 30.0,
        precision_scale: float = 2.0,
        pragmatic_valence: float = -30.0,
        pragmatic_precision: float = 2.0,
        source_precision: dict[str, float] | None = None,
        minimum_input_confidence: float = 0.5,
        minimum_reference_confidence: float = 0.55,
        minimum_grounding_confidence: float = 0.6,
        minimum_valence_confidence: float = 0.6,
        dedup_window: int = 8,
        semantic_dedup_threshold: float = 0.8,
        max_abs_delta: float = 8.0,
        max_update_kl: float = 2.0,
        pragmatic_confidence_threshold: float = 0.8,
    ) -> None:
        if mode not in ROUTE1_MODES:
            raise ValueError(f"Unknown Route 1 mode: {mode}")
        self.mode = mode
        self.source_precision = {
            **DEFAULT_SOURCE_PRECISION,
            **(source_precision or {}),
        }
        if any(value <= 0 for value in self.source_precision.values()):
            raise ValueError("source precision multipliers must be positive")
        self.update_count = 0
        self.feedback_count = 0
        self.minimum_input_confidence = float(minimum_input_confidence)
        self.minimum_reference_confidence = float(minimum_reference_confidence)
        self.minimum_grounding_confidence = float(minimum_grounding_confidence)
        self.minimum_valence_confidence = float(minimum_valence_confidence)
        self.dedup_window = int(dedup_window)
        self.semantic_dedup_threshold = float(semantic_dedup_threshold)
        self.max_abs_delta = float(max_abs_delta)
        self.max_update_kl = float(max_update_kl)
        self.pragmatic_confidence_threshold = float(pragmatic_confidence_threshold)
        self._recent_feedback: list[tuple[int, str]] = []
        pseudo = mode == ROUTE1_PSEUDOPRAGMATIC
        self.learner = BayesianRewardLearner(
            features,
            valence_scale=valence_scale,
            precision_scale=precision_scale,
            pragmatic_valence=pragmatic_valence if pseudo else None,
            pragmatic_precision=pragmatic_precision if pseudo else None,
            prior_mean=prior_mean,
        )

    def update(
        self,
        text: str,
        *,
        decision: dict | None = None,
        trajectory_features: dict[str, float] | None = None,
        oracle_feedback: dict | None = None,
        interpretation: str = "inferred",
        source: str = "synthetic",
        confidence: float = 1.0,
        precision_multiplier: float | None = None,
    ) -> dict:
        """Apply one utterance and return a complete auditable update trace.

        ``inferred`` ignores gold type/grounding/valence fields. Runtime state
        still supplies the subgoal that was just chosen and the feasible action
        library, matching the paper's trajectory/action reference context.
        ``oracle`` is reserved for controlled evaluation with annotated data.
        """

        if interpretation not in {"inferred", "oracle"}:
            raise ValueError("interpretation must be 'inferred' or 'oracle'")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        cleaned = (text or "").strip()
        if not cleaned:
            return {"status": "empty", "mode": self.mode, "text": ""}
        self.feedback_count += 1
        self._recent_feedback = [
            item
            for item in self._recent_feedback
            if self.feedback_count - item[0] <= self.dedup_window
        ]
        duplicate = next(
            (
                previous
                for _index, previous in self._recent_feedback
                if _semantic_similarity(cleaned, previous) >= self.semantic_dedup_threshold
            ),
            None,
        )
        if duplicate is not None:
            return {
                "status": "rejected_duplicate",
                "mode": self.mode,
                "text": cleaned,
                "duplicate_of": duplicate,
                "update_id": self.update_count,
            }
        if confidence < self.minimum_input_confidence:
            return {
                "status": "rejected_input_confidence",
                "mode": self.mode,
                "text": cleaned,
                "input_confidence": confidence,
                "required_confidence": self.minimum_input_confidence,
                "update_id": self.update_count,
            }

        supplied = dict(oracle_feedback or {})
        supplied["text"] = cleaned
        feedback = supplied if interpretation == "oracle" else without_privileged_labels(supplied)
        feedback_type = (
            str(feedback.get("expected_feedback_type"))
            if interpretation == "oracle" and feedback.get("expected_feedback_type")
            else classify_feedback(cleaned)
        )

        ranking = list((decision or {}).get("ranking") or [])
        action_library = {
            str(item["subgoal"]): {
                str(feature): float(value)
                for feature, value in (item.get("features") or {}).items()
            }
            for item in ranking
        }
        if interpretation == "inferred":
            # Supply every runtime referent candidate, but let the independent
            # phrase reference classifier choose which grounding path consumes
            # it. Speech act is metadata, not a proxy for reference type.
            if trajectory_features:
                feedback["trajectory_features"] = {
                    str(feature): float(value)
                    for feature, value in trajectory_features.items()
                }
            elif decision:
                chosen = decision.get("chosen_subgoal")
                if chosen in action_library:
                    feedback["trajectory_features"] = action_library[chosen]
            target_action = infer_target_action(cleaned, action_library)
            if target_action:
                feedback["target_action"] = target_action

        observations = build_feedback_observations(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_library,
            prefer_explicit_reference=interpretation == "oracle",
        )
        rejection_reasons: list[dict] = []
        if interpretation == "inferred":
            for index, observation in enumerate(observations):
                if observation.get("reference_abstained") or float(
                    observation.get("reference_confidence", 1.0)
                ) < self.minimum_reference_confidence:
                    rejection_reasons.append({"observation": index, "stage": "reference"})
                if observation.get("grounding_abstained") or float(
                    observation.get("grounding_confidence", 0.0)
                ) < self.minimum_grounding_confidence:
                    rejection_reasons.append({"observation": index, "stage": "grounding"})
                if float(
                    observation.get("valence_confidence", 1.0)
                ) < self.minimum_valence_confidence:
                    rejection_reasons.append({"observation": index, "stage": "valence"})
        if rejection_reasons:
            stages = sorted({reason["stage"] for reason in rejection_reasons})
            return {
                "status": "rejected_low_confidence",
                "trace_state": "rejected_" + "_and_".join(stages),
                "mode": self.mode,
                "interpretation": interpretation,
                "text": cleaned,
                "feedback_type": feedback_type,
                "rejection_reasons": rejection_reasons,
                "candidate_observations": observations,
                "update_id": self.update_count,
            }
        before = self.learner.as_dict()
        before_variance = self.learner.belief.variance_as_dict()
        before_mean_array = self.learner.belief.mean.copy()
        before_covariance_array = self.learner.belief.covariance.copy()
        merged_delta: dict[str, float] = {}
        grounded = []
        source_multiplier = (
            float(precision_multiplier)
            if precision_multiplier is not None
            else float(self.source_precision.get(source, 1.0))
        )
        for observation in observations:
            if not observation["target_features"]:
                continue
            grounding_confidence = float(observation.get("grounding_confidence", 1.0))
            reference_confidence = float(observation.get("reference_confidence", 1.0))
            valence_confidence = float(observation.get("valence_confidence", 1.0))
            effective_multiplier = (
                source_multiplier
                * confidence
                * grounding_confidence
                * reference_confidence
                * valence_confidence
            )
            if effective_multiplier < 0.15:
                continue
            delta = self.learner.update(
                observation["target_features"],
                observation["valence"],
                precision_multiplier=effective_multiplier,
                allow_pragmatic=(
                    self.mode != ROUTE1_PSEUDOPRAGMATIC
                    or min(
                        confidence,
                        grounding_confidence,
                        reference_confidence,
                        valence_confidence,
                    ) >= self.pragmatic_confidence_threshold
                ),
            )
            for feature, value in delta.items():
                merged_delta[feature] = merged_delta.get(feature, 0.0) + float(value)
            grounded.append(
                {
                    **observation,
                    "effective_precision": (
                        self.learner.precision_scale * effective_multiplier
                    ),
                    "precision_multiplier": effective_multiplier,
                }
            )

        constraint = self.learner.constrain_posterior_update(
            before_mean_array,
            before_covariance_array,
            max_abs_delta=self.max_abs_delta,
            max_kl=self.max_update_kl,
        )
        after = self.learner.as_dict()
        after_variance = self.learner.belief.variance_as_dict()
        merged_delta = {
            feature: after[feature] - before[feature]
            for feature in after
            if abs(after[feature] - before[feature]) > 1e-9
        }
        changed = sorted(merged_delta, key=lambda feature: abs(merged_delta[feature]), reverse=True)
        if grounded:
            self.update_count += 1
            self._recent_feedback.append((self.feedback_count, cleaned))
        return {
            "status": "updated" if grounded else "rejected_ungrounded",
            "trace_state": (
                "posterior_updated_capped"
                if grounded and constraint["capped"]
                else "posterior_updated"
                if grounded
                else "no_grounded_observation"
            ),
            "mode": self.mode,
            "interpretation": interpretation,
            "text": cleaned,
            "feedback_type": feedback_type,
            "reference_types": [
                observation.get("reference_type") for observation in grounded
            ],
            "source": source,
            "source_precision_multiplier": source_multiplier,
            "input_confidence": confidence,
            "update_id": self.update_count,
            "target_action": feedback.get("target_action"),
            "observations": grounded,
            "update_constraint": constraint,
            "weight_delta": {feature: merged_delta[feature] for feature in changed},
            "top_changes": [
                {
                    "feature": feature,
                    "before": before[feature],
                    "after": after[feature],
                    "delta": merged_delta[feature],
                    "variance_before": before_variance[feature],
                    "variance_after": after_variance[feature],
                }
                for feature in changed[:12]
            ],
        }

    def weights(self) -> dict[str, float]:
        return self.learner.as_dict()

    def state_dict(self) -> dict:
        """Return the complete, versioned posterior needed for exact resume."""

        belief = self.learner.belief
        return {
            "version": STATE_VERSION,
            "mode": self.mode,
            "features": list(belief.features),
            "mean": belief.mean.tolist(),
            "covariance": belief.covariance.tolist(),
            "update_count": self.update_count,
            "feedback_count": self.feedback_count,
            "recent_feedback": copy.deepcopy(self._recent_feedback),
            "source_precision": copy.deepcopy(self.source_precision),
            "hyperparameters": {
                "valence_scale": self.learner.valence_scale,
                "precision_scale": self.learner.precision_scale,
                "pragmatic_valence": self.learner.pragmatic_valence,
                "pragmatic_precision": self.learner.pragmatic_precision,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore an exact posterior, rejecting incompatible feature schemas."""

        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported Route 1 state version")
        if state.get("mode") != self.mode:
            raise ValueError("Route 1 checkpoint mode does not match learner mode")
        features = list(state.get("features") or [])
        if features != self.learner.features:
            raise ValueError("Route 1 checkpoint feature schema does not match")
        mean = np.asarray(state.get("mean"), dtype=float)
        covariance = np.asarray(state.get("covariance"), dtype=float)
        n = len(features)
        if mean.shape != (n,) or covariance.shape != (n, n):
            raise ValueError("invalid Route 1 checkpoint posterior dimensions")
        self.learner.belief.mean = mean
        self.learner.belief.covariance = covariance
        self.update_count = int(state.get("update_count", 0))
        self.feedback_count = int(state.get("feedback_count", self.update_count))
        self._recent_feedback = [
            (int(index), str(text))
            for index, text in (state.get("recent_feedback") or [])
        ]
        self.source_precision.update(
            {str(key): float(value) for key, value in (state.get("source_precision") or {}).items()}
        )

    def reset(self, *, prior_mean: dict[str, float] | None = None) -> None:
        """Reset the posterior while preserving the selected learner mode."""

        self.__init__(
            self.learner.features,
            mode=self.mode,
            prior_mean=prior_mean,
            valence_scale=self.learner.valence_scale,
            precision_scale=self.learner.precision_scale,
            pragmatic_valence=(
                float(self.learner.pragmatic_valence)
                if self.learner.pragmatic_valence is not None
                else -30.0
            ),
            pragmatic_precision=(
                float(self.learner.pragmatic_precision)
                if self.learner.pragmatic_precision is not None
                else 2.0
            ),
            source_precision=self.source_precision,
            minimum_input_confidence=self.minimum_input_confidence,
            minimum_reference_confidence=self.minimum_reference_confidence,
            minimum_grounding_confidence=self.minimum_grounding_confidence,
            minimum_valence_confidence=self.minimum_valence_confidence,
            dedup_window=self.dedup_window,
            semantic_dedup_threshold=self.semantic_dedup_threshold,
            max_abs_delta=self.max_abs_delta,
            max_update_kl=self.max_update_kl,
            pragmatic_confidence_threshold=self.pragmatic_confidence_threshold,
        )
