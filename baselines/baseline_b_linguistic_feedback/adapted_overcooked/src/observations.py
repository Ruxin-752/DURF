"""Turn grounded feedback into Gaussian belief observations.

Faithful reproduction of ``science/observations/observations.py`` from the
paper repo. Each feedback becomes one (or two) conjugate-Gaussian observations:

- a *literal* observation: the grounded reference features should carry reward
  equal to the (scaled) sentiment valence;
- an optional *pragmatic* observation (PseudoPragmatic learner): features that
  were **not** mentioned are nudged toward a fixed implicature valence, mirroring
  the paper's ``pragmatic_valence`` / inverse-reference term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .text_analysis import normalize_reference_vector


@dataclass
class Observation:
    reference_vector: np.ndarray
    valence: float
    precision: float
    kind: str


def reference_vector(
    target_features: dict[str, float],
    features: list[str],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Reference vector over the ordered feature list.

    The paper normalizes reference vectors to sum to 1 (see
    ``reference_vector_from_features`` / ``reference_vector_from_trajectory``);
    ``normalize=True`` reproduces that convention.
    """

    index = {feature: position for position, feature in enumerate(features)}
    vector = np.zeros(len(features), dtype=float)
    for feature, value in target_features.items():
        position = index.get(str(feature))
        if position is not None:
            vector[position] = float(value)
    if normalize:
        vector = normalize_reference_vector(vector)
    return vector


def build_observations(
    target_features: dict[str, float],
    valence: float,
    features: list[str],
    *,
    valence_scale: float = 30.0,
    precision_scale: float = 2.0,
    pragmatic_valence: float | None = None,
    pragmatic_precision: float | None = None,
    precision_multiplier: float = 1.0,
) -> list[Observation]:
    """Build literal (+ optional pragmatic) observations for one feedback.

    ``valence`` is the raw sentiment in ``[-1, 1]``; it is multiplied by
    ``valence_scale`` (paper default 30). ``pragmatic_valence`` is an absolute
    valence applied to unmentioned features (paper's ExpPseudoPragmatic uses
    ``-30``).
    """

    if precision_multiplier <= 0:
        raise ValueError("precision_multiplier must be positive")
    ref = reference_vector(target_features, features)
    observations: list[Observation] = []

    if not ref.any():
        return observations

    observations.append(
        Observation(
            ref,
            float(valence) * float(valence_scale),
            float(precision_scale) * float(precision_multiplier),
            "literal",
        )
    )

    if pragmatic_valence is not None:
        inverse = (ref == 0).astype(float)
        total = inverse.sum()
        if total > 0:
            inverse = inverse / total
            prag_precision = (
                precision_scale if pragmatic_precision is None else pragmatic_precision
            )
            observations.append(
                Observation(
                    inverse,
                    float(pragmatic_valence),
                    float(prag_precision) * float(precision_multiplier),
                    "pragmatic",
                )
            )

    return observations
