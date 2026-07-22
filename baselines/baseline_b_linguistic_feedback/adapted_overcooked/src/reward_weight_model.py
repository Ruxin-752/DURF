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
    ) -> dict[str, float]:
        """Absorb one feedback and return the change in posterior mean."""

        before = self.belief.mean.copy()
        observations = build_observations(
            target_features,
            valence,
            self.features,
            valence_scale=self.valence_scale,
            precision_scale=self.precision_scale,
            pragmatic_valence=self.pragmatic_valence,
            pragmatic_precision=self.pragmatic_precision,
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
