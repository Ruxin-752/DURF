"""Bayesian reward-feature learner for Baseline B.

This replaces the earlier point-estimate updater with a faithful reproduction
of the paper's ``MultivariateNormalLearner`` (see the original
``science/agents/agents.py``): reward weights are a Gaussian belief that is
updated by conjugate Bayesian observations, not by a single additive step.

Two learner variants from the paper are supported through one class:

- **Literal** (``pragmatic_valence=None``): only the mentioned reference
  features are updated.
- **PseudoPragmatic** (``pragmatic_valence`` set, e.g. ``-30``): unmentioned
  features are additionally nudged toward the implicature valence.
"""

from __future__ import annotations

import numpy as np

from .belief_model import GaussianBelief, PRIOR_VARIANCE, score_action
from .observations import build_observations


class BayesianRewardLearner:
    def __init__(
        self,
        features: list[str],
        *,
        valence_scale: float = 30.0,
        precision_scale: float = 2.0,
        pragmatic_valence: float | None = None,
        pragmatic_precision: float | None = None,
        initial_variance: float = PRIOR_VARIANCE,
        prior_mean: dict[str, float] | None = None,
    ) -> None:
        self.features = sorted(features)
        self.belief = GaussianBelief.prior(self.features, var=initial_variance)
        if prior_mean:
            mean_dict = self.belief.mean_as_dict()
            mean_dict.update({str(k): float(v) for k, v in prior_mean.items()})
            self.belief.mean = np.array(
                [mean_dict[feature] for feature in self.features], dtype=float
            )

        self.valence_scale = float(valence_scale)
        self.precision_scale = float(precision_scale)
        self.pragmatic_valence = pragmatic_valence
        self.pragmatic_precision = pragmatic_precision

    def update(
        self,
        target_features: dict[str, int | float],
        valence: float,
        *,
        precision_multiplier: float = 1.0,
        allow_pragmatic: bool = True,
    ) -> dict[str, float]:
        """Absorb one feedback and return the change in posterior mean."""

        before = self.belief.mean.copy()
        observations = build_observations(
            target_features,
            valence,
            self.features,
            valence_scale=self.valence_scale,
            precision_scale=self.precision_scale,
            pragmatic_valence=self.pragmatic_valence if allow_pragmatic else None,
            pragmatic_precision=self.pragmatic_precision if allow_pragmatic else None,
            precision_multiplier=precision_multiplier,
        )
        for observation in observations:
            self.belief = self.belief.multiply_observation(
                observation.reference_vector,
                observation.valence,
                observation.precision,
            )

        after = self.belief.mean
        delta: dict[str, float] = {}
        for position, feature in enumerate(self.features):
            change = float(after[position] - before[position])
            if abs(change) > 1e-9:
                delta[feature] = change
        return delta

    @staticmethod
    def _gaussian_kl(
        mean: np.ndarray,
        covariance: np.ndarray,
        base_mean: np.ndarray,
        base_covariance: np.ndarray,
        *,
        base_precision: np.ndarray | None = None,
        base_logdet: float | None = None,
    ) -> float:
        """KL(N(mean,covariance) || N(base_mean,base_covariance))."""

        n = len(mean)
        if base_precision is None:
            base_precision = np.linalg.inv(base_covariance)
        difference = base_mean - mean
        sign_new, logdet_new = np.linalg.slogdet(covariance)
        if base_logdet is None:
            sign_base, computed_logdet = np.linalg.slogdet(base_covariance)
            if sign_base <= 0:
                return float("inf")
            base_logdet = float(computed_logdet)
        if sign_new <= 0:
            return float("inf")
        return 0.5 * float(
            np.trace(base_precision @ covariance)
            + difference.T @ base_precision @ difference
            - n
            + base_logdet
            - logdet_new
        )

    def constrain_posterior_update(
        self,
        before_mean: np.ndarray,
        before_covariance: np.ndarray,
        *,
        max_abs_delta: float | None,
        max_kl: float | None,
    ) -> dict:
        """Limit one externally defined update transaction by delta and KL."""

        proposed_mean = self.belief.mean.copy()
        proposed_covariance = self.belief.covariance.copy()
        delta = proposed_mean - before_mean
        alpha = 1.0
        largest = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs_delta is not None and largest > max_abs_delta:
            alpha = min(alpha, float(max_abs_delta) / largest)

        def candidate(scale: float) -> tuple[np.ndarray, np.ndarray]:
            return (
                before_mean + scale * (proposed_mean - before_mean),
                before_covariance + scale * (proposed_covariance - before_covariance),
            )

        base_precision = np.linalg.inv(before_covariance)
        sign_base, base_logdet = np.linalg.slogdet(before_covariance)
        if sign_base <= 0:
            raise ValueError("pre-update covariance must be positive definite")

        def transaction_kl(mean: np.ndarray, covariance: np.ndarray) -> float:
            return self._gaussian_kl(
                mean,
                covariance,
                before_mean,
                before_covariance,
                base_precision=base_precision,
                base_logdet=float(base_logdet),
            )

        if max_kl is not None:
            mean, covariance = candidate(alpha)
            if transaction_kl(mean, covariance) > max_kl:
                low, high = 0.0, alpha
                # Eight bisection steps bound the scale within 1/256 while
                # avoiding dozens of 53x53 log-determinants per feedback.
                for _ in range(8):
                    middle = (low + high) / 2.0
                    mid_mean, mid_covariance = candidate(middle)
                    if transaction_kl(mid_mean, mid_covariance) <= max_kl:
                        low = middle
                    else:
                        high = middle
                alpha = low
        self.belief.mean, self.belief.covariance = candidate(alpha)
        return {
            "capped": alpha < 1.0 - 1e-9,
            "scale": float(alpha),
            "kl": transaction_kl(self.belief.mean, self.belief.covariance),
            "max_abs_delta": float(
                np.max(np.abs(self.belief.mean - before_mean))
            ) if delta.size else 0.0,
        }

    def score_action(self, action_features: dict[str, int | float]) -> float:
        return score_action(self.belief.mean_as_dict(), action_features)

    def as_dict(self) -> dict[str, float]:
        return dict(sorted(self.belief.mean_as_dict().items()))

    def sample_beliefs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.belief.sample(n, rng)

    def top_weights(self, limit: int = 12) -> list[tuple[str, float]]:
        non_zero = [
            (feature, weight)
            for feature, weight in self.belief.mean_as_dict().items()
            if abs(weight) > 1e-9
        ]
        return sorted(non_zero, key=lambda item: abs(item[1]), reverse=True)[:limit]
