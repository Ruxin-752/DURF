"""Gaussian belief over reward-feature weights.

Faithful reproduction of the paper *Learning Rewards from Linguistic Feedback*
(Sumers et al., AAAI 2021). This mirrors ``science/agents/beliefs.py`` in the
original repository: reward weights ``w`` are represented as a multivariate
normal belief, and each piece of linguistic feedback is absorbed as a
conjugate Gaussian (Bayesian linear regression) observation.

The original prior is ``N(0, var=25)`` per feature (precision ``1/25``); the
same prior is used here so the Overcooked adaptation stays comparable to the
paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0


@dataclass
class GaussianBelief:
    """Multivariate normal belief over an ordered feature-weight vector."""

    features: list[str]
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def prior(
        cls,
        features: list[str],
        *,
        mean: float = PRIOR_MEAN,
        var: float = PRIOR_VARIANCE,
    ) -> "GaussianBelief":
        ordered = sorted(features)
        mean_vector = np.full(len(ordered), float(mean), dtype=float)
        covariance = np.identity(len(ordered), dtype=float) * float(var)
        return cls(ordered, mean_vector, covariance)

    @property
    def precision(self) -> np.ndarray:
        return np.linalg.inv(self.covariance)

    def multiply_observation(
        self,
        reference_vector: np.ndarray,
        valence: float,
        obs_precision: float,
    ) -> "GaussianBelief":
        """Absorb one observation by multiplying two Gaussian factors.

        This reproduces the *active* experiment update in the paper
        (``science/agents/agents.py`` line 184: ``belief_state.multiply(obs)``),
        where the observation is the Gaussian factor
        ``N(mean = r * valence, precision = obs_precision * r r^T)``:

            Lambda' = Lambda + p * r r^T
            mu'     = Sigma' (Lambda mu + (p r r^T)(r * valence))

        Note the information term uses the full precision-matrix product
        ``obs_precision * (r . r) * valence * r`` (not ``p * valence * r``),
        matching ``MultivariateNormal.multiply``.
        """

        reference_vector = np.asarray(reference_vector, dtype=float)
        prior_precision = self.precision

        obs_precision_matrix = np.outer(reference_vector, reference_vector) * obs_precision
        new_precision = prior_precision + obs_precision_matrix
        new_covariance = np.linalg.inv(new_precision)

        obs_mean = reference_vector * valence
        information = prior_precision @ self.mean + obs_precision_matrix @ obs_mean
        new_mean = new_covariance @ information

        return GaussianBelief(list(self.features), new_mean, new_covariance)

    def update_from_observation(
        self,
        reference_vector: np.ndarray,
        valence: float,
        obs_precision: float,
    ) -> "GaussianBelief":
        """Bayesian-linear-regression conjugate update (the paper's *inactive*,
        commented-out ``MultivariateNormal.update_from_observation``).

        Treats the feedback as a BLR datapoint ``valence ~= r . w``:

            Lambda' = Lambda + p * r r^T
            mu'     = Sigma' (Lambda mu + p * valence * r)

        Kept for comparison; the experiment learners use
        :meth:`multiply_observation` instead.
        """

        reference_vector = np.asarray(reference_vector, dtype=float)
        prior_precision = self.precision

        new_precision = prior_precision + np.outer(reference_vector, reference_vector) * obs_precision
        new_covariance = np.linalg.inv(new_precision)

        information = prior_precision @ self.mean + reference_vector * valence * obs_precision
        new_mean = new_covariance @ information

        return GaussianBelief(list(self.features), new_mean, new_covariance)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` weight hypotheses from the current belief."""

        return rng.multivariate_normal(self.mean, self.covariance, size=n)

    def mean_as_dict(self) -> dict[str, float]:
        return {feature: float(value) for feature, value in zip(self.features, self.mean)}

    def variance_as_dict(self) -> dict[str, float]:
        variances = np.diag(self.covariance)
        return {feature: float(value) for feature, value in zip(self.features, variances)}

    def score_action(self, action_features: dict[str, float]) -> float:
        return score_action(self.mean_as_dict(), action_features)


def score_action(weights: dict[str, float], action_features: dict[str, float]) -> float:
    """Linear reward ``r(s, a) = w . phi(s, a)`` for one action."""

    return sum(
        float(weights.get(str(feature), 0.0)) * float(value)
        for feature, value in action_features.items()
    )
