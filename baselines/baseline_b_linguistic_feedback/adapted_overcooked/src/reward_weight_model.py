"""Linear feature-weight reward model for Baseline B."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeightModel:
    weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 1.0

    def score_action(self, action_features: dict[str, int | float]) -> float:
        return sum(
            self.weights.get(str(feature), 0.0) * float(value)
            for feature, value in action_features.items()
        )

    def update(
        self,
        target_features: dict[str, int | float],
        sentiment_score: float,
    ) -> dict[str, float]:
        delta: dict[str, float] = {}
        for feature, value in target_features.items():
            feature = str(feature)
            change = self.learning_rate * float(sentiment_score) * float(value)
            if change == 0:
                continue
            self.weights[feature] = self.weights.get(feature, 0.0) + change
            delta[feature] = change
        return delta

    def as_dict(self) -> dict[str, float]:
        return dict(sorted(self.weights.items()))

    def top_weights(self, limit: int = 12) -> list[tuple[str, float]]:
        non_zero = [
            (feature, weight)
            for feature, weight in self.weights.items()
            if abs(weight) > 1e-9
        ]
        return sorted(non_zero, key=lambda item: abs(item[1]), reverse=True)[:limit]
