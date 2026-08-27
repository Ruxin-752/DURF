"""Online paper-style Route 1 updates for the live subgoal agent."""

from __future__ import annotations

import copy
import re

import numpy as np

from .feedback_form_classifier import FEEDBACK_TYPES, predict_feedback_form
from .feedback_observations import (
    build_feedback_observations,
    without_privileged_labels,
)
from .reward_weight_model import BayesianRewardLearner
from .subgoal_featurizer import (
    LIVE_DECISION_FEATURES,
    audit_live_feature_coverage,
    candidate_varying_features,
)
from .text_analysis import limited_punc_tokenization


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


def _normalize_feedback_form_prediction(phrase: str, value: dict) -> dict:
    """Normalize the exact UI/model prediction consumed by Route 1."""

    row = dict(value or {})
    feedback_type = str(row.get("feedback_type") or "unknown").lower()
    raw_probabilities = row.get("probabilities")
    probabilities: dict[str, float] = {}
    if isinstance(raw_probabilities, dict):
        for label, score in raw_probabilities.items():
            normalized_label = str(label)
            if normalized_label not in FEEDBACK_TYPES:
                continue
            try:
                normalized_score = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(normalized_score) and 0.0 <= normalized_score <= 1.0:
                probabilities[normalized_label] = normalized_score
    raw_confidence = row.get("confidence")
    if raw_confidence is None and feedback_type in probabilities:
        raw_confidence = probabilities[feedback_type]
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = float("nan")
    try:
        threshold = float(row.get("confidence_threshold"))
    except (TypeError, ValueError):
        threshold = 0.55
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        threshold = 0.55
    confidence_is_valid = bool(np.isfinite(confidence) and 0.0 <= confidence <= 1.0)
    abstained = (
        bool(row.get("abstained", False))
        or feedback_type not in FEEDBACK_TYPES
        or not confidence_is_valid
        or confidence < threshold
    )
    return {
        **row,
        "text": phrase,
        "feedback_type": feedback_type,
        "confidence": confidence if confidence_is_valid else None,
        "probabilities": probabilities,
        "confidence_threshold": threshold,
        "abstained": abstained,
    }


def _feedback_form_predictions(
    text: str,
    supplied: dict | None,
) -> list[dict]:
    """Return phrase-level ``f_G`` predictions from one authoritative source."""

    if supplied:
        supplied_rows = list(supplied.get("phrases") or [])
        if not supplied_rows:
            supplied_rows = [{**supplied, "text": text}]
        predictions = []
        for row in supplied_rows:
            if not isinstance(row, dict):
                continue
            phrase = str(row.get("text") or row.get("phrase") or "").strip()
            if phrase:
                predictions.append(_normalize_feedback_form_prediction(phrase, row))
        return predictions

    return [
        _normalize_feedback_form_prediction(phrase, dict(predict_feedback_form(phrase)))
        for phrase in limited_punc_tokenization(text)
    ]


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
        self.live_feature_coverage = audit_live_feature_coverage(features)
        self.live_feature_mask = (
            set(LIVE_DECISION_FEATURES)
            if self.live_feature_coverage["schema_size"] == 53
            and self.live_feature_coverage["decision_feature_count"]
            == len(LIVE_DECISION_FEATURES)
            else None
        )
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
        recent_events: list[dict] | None = None,
        total_step: int | None = None,
        oracle_feedback: dict | None = None,
        feedback_form_prediction: dict | None = None,
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
        if interpretation == "oracle":
            feedback_type = str(
                feedback.get("expected_feedback_type") or "unknown"
            ).lower()
            form_predictions = [
                {
                    "text": cleaned,
                    "feedback_type": feedback_type,
                    "confidence": 1.0,
                    "probabilities": {feedback_type: 1.0},
                    "classifier": "oracle_feedback_form",
                    "confidence_threshold": 0.0,
                    "abstained": False,
                }
            ]
        else:
            form_predictions = _feedback_form_predictions(
                cleaned, feedback_form_prediction
            )
            distinct_forms = {
                row["feedback_type"]
                for row in form_predictions
                if row["feedback_type"] in FEEDBACK_TYPES
            }
            feedback_type = (
                next(iter(distinct_forms))
                if len(distinct_forms) == 1
                else "mixed"
                if len(distinct_forms) > 1
                else "unknown"
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
            # Supply every runtime referent candidate.  The three-way ``f_G``
            # prediction selects the grounding branch; the five-way reference
            # prediction may only refine a compatible subtype inside it.
            if trajectory_features:
                feedback["trajectory_features"] = {
                    str(feature): float(value)
                    for feature, value in trajectory_features.items()
                }
            elif decision:
                chosen = decision.get("chosen_subgoal")
                if chosen in action_library:
                    feedback["trajectory_features"] = action_library[chosen]
            if recent_events:
                feedback["recent_events"] = [dict(event) for event in recent_events]
                feedback["total_step"] = int(
                    total_step
                    if total_step is not None
                    else max(int(event.get("total_step", 0)) for event in recent_events)
                )
            target_action = infer_target_action(cleaned, action_library)
            if target_action:
                feedback["target_action"] = target_action

        observations = build_feedback_observations(
            feedback,
            feedback_type=feedback_type,
            action_feature_library=action_library,
            prefer_explicit_reference=interpretation == "oracle",
            live_feature_mask=self.live_feature_mask,
            feedback_form_predictions=(
                None if interpretation == "oracle" else form_predictions
            ),
        )
        rejection_reasons: list[dict] = []
        if interpretation == "inferred":
            for index, observation in enumerate(observations):
                form_confidence = observation.get("feedback_form_confidence")
                form_threshold = observation.get(
                    "feedback_form_confidence_threshold"
                )
                try:
                    form_confidence_value = float(form_confidence)
                except (TypeError, ValueError):
                    form_confidence_value = float("nan")
                try:
                    form_threshold_value = float(form_threshold)
                except (TypeError, ValueError):
                    form_threshold_value = 0.55
                if (
                    observation.get("feedback_form_abstained")
                    or not np.isfinite(form_confidence_value)
                    or form_confidence_value < form_threshold_value
                ):
                    rejection_reasons.append(
                        {"observation": index, "stage": "feedback_form"}
                    )
                reference_confidence = float(
                    observation.get("reference_confidence", 1.0)
                )
                grounding_confidence = float(
                    observation.get("grounding_confidence", 0.0)
                )
                grounding_is_strong = (
                    not observation.get("grounding_abstained")
                    and grounding_confidence
                    >= max(self.minimum_grounding_confidence, 0.6)
                )
                reference_is_weak = (
                    observation.get("reference_abstained")
                    or reference_confidence < self.minimum_reference_confidence
                )
                # Strong literal grounding can rescue a conservative reference
                # abstention. The raw confidence still gates Pragmatic updates.
                reference_grounding_override = bool(
                    reference_is_weak and grounding_is_strong
                )
                observation["reference_grounding_override"] = (
                    reference_grounding_override
                )
                if reference_is_weak and not reference_grounding_override:
                    rejection_reasons.append({"observation": index, "stage": "reference"})
                if (
                    observation.get("grounding_abstained")
                    or grounding_confidence < self.minimum_grounding_confidence
                ):
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
                "feedback_types": [
                    row["feedback_type"] for row in form_predictions
                ],
                "feedback_form_prediction": copy.deepcopy(
                    feedback_form_prediction
                ),
                "feedback_form_predictions": copy.deepcopy(form_predictions),
                "rejection_reasons": rejection_reasons,
                "candidate_observations": observations,
                "recent_event_count": len(feedback.get("recent_events") or []),
                "live_feature_coverage": self.live_feature_coverage,
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
            form_confidence = float(
                observation.get("feedback_form_confidence", 1.0)
            )
            literal_reference_confidence = (
                max(reference_confidence, self.minimum_reference_confidence)
                if observation.get("reference_grounding_override")
                else reference_confidence
            )
            effective_multiplier = (
                source_multiplier
                * confidence
                * form_confidence
                * grounding_confidence
                * literal_reference_confidence
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
                        form_confidence,
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
                    "feedback_form_confidence": form_confidence,
                    "literal_reference_confidence": literal_reference_confidence,
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
            "feedback_types": [row["feedback_type"] for row in form_predictions],
            "feedback_form_prediction": copy.deepcopy(feedback_form_prediction),
            "feedback_form_predictions": copy.deepcopy(form_predictions),
            "reference_types": [
                observation.get("reference_type") for observation in grounded
            ],
            "effective_reference_types": [
                observation.get("effective_reference_type") for observation in grounded
            ],
            "source": source,
            "source_precision_multiplier": source_multiplier,
            "input_confidence": confidence,
            "update_id": self.update_count,
            "target_action": feedback.get("target_action"),
            "recent_event_count": len(feedback.get("recent_events") or []),
            "current_candidate_varying_features": sorted(
                candidate_varying_features(list(action_library.values()))
            ),
            "live_feature_coverage": self.live_feature_coverage,
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
        """Atomically restore an exact, compatible Route 1 posterior."""

        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported Route 1 state version")
        if state.get("mode") != self.mode:
            raise ValueError("Route 1 checkpoint mode does not match learner mode")
        features = list(state.get("features") or [])
        if features != self.learner.features:
            raise ValueError("Route 1 checkpoint feature schema does not match")
        try:
            mean = np.asarray(state.get("mean"), dtype=float)
            covariance = np.asarray(state.get("covariance"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Route 1 checkpoint posterior") from exc
        n = len(features)
        if mean.shape != (n,) or covariance.shape != (n, n):
            raise ValueError("invalid Route 1 checkpoint posterior dimensions")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Route 1 checkpoint posterior must be finite")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
            raise ValueError("Route 1 checkpoint covariance must be symmetric")
        try:
            if np.min(np.linalg.eigvalsh(covariance)) <= 0:
                raise ValueError("Route 1 checkpoint covariance must be positive definite")
        except np.linalg.LinAlgError as exc:
            raise ValueError("Route 1 checkpoint covariance is invalid") from exc

        try:
            update_count = int(state.get("update_count", 0))
            feedback_count = int(state.get("feedback_count", update_count))
        except (TypeError, ValueError) as exc:
            raise ValueError("Route 1 checkpoint counters are invalid") from exc
        if update_count < 0 or feedback_count < update_count:
            raise ValueError("Route 1 checkpoint counters are invalid")

        try:
            recent_feedback = []
            for item in state.get("recent_feedback") or []:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError
                index, text = int(item[0]), str(item[1])
                if index < 0 or index > feedback_count or not text.strip():
                    raise ValueError
                recent_feedback.append((index, text))
        except (TypeError, ValueError) as exc:
            raise ValueError("Route 1 checkpoint recent feedback is invalid") from exc

        raw_source_precision = state.get("source_precision")
        if not isinstance(raw_source_precision, dict) or not raw_source_precision:
            raise ValueError("Route 1 checkpoint source precision is missing")
        try:
            source_precision = {
                str(key): float(value) for key, value in raw_source_precision.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("Route 1 checkpoint source precision is invalid") from exc
        if any(
            not key or value <= 0 or not np.isfinite(value)
            for key, value in source_precision.items()
        ):
            raise ValueError("Route 1 checkpoint source precision is invalid")

        expected_hyperparameters = {
            "valence_scale": self.learner.valence_scale,
            "precision_scale": self.learner.precision_scale,
            "pragmatic_valence": self.learner.pragmatic_valence,
            "pragmatic_precision": self.learner.pragmatic_precision,
        }
        saved_hyperparameters = state.get("hyperparameters")
        if not isinstance(saved_hyperparameters, dict):
            raise ValueError("Route 1 checkpoint hyperparameters are missing")
        for name, expected in expected_hyperparameters.items():
            saved = saved_hyperparameters.get(name)
            if expected is None:
                if saved is not None:
                    raise ValueError(
                        "Route 1 checkpoint hyperparameters do not match learner"
                    )
                continue
            try:
                saved_value = float(saved)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Route 1 checkpoint hyperparameters are invalid"
                ) from exc
            if not np.isfinite(saved_value) or not np.isclose(
                saved_value, float(expected), rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "Route 1 checkpoint hyperparameters do not match learner"
                )

        # Commit only after every field has passed validation.  Callers can
        # safely catch a resume error and continue using the current learner.
        self.learner.belief.mean = mean.copy()
        self.learner.belief.covariance = covariance.copy()
        self.update_count = update_count
        self.feedback_count = feedback_count
        self._recent_feedback = recent_feedback
        self.source_precision = source_precision

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
