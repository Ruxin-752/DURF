"""Linear Hu-v0 model for condition-aware subgoal preference scoring.

The model learns from pairwise labels:

    Hu(user, condition, preferred_subgoal)
    >
    Hu(user, condition, rejected_subgoal)

It is deliberately simple and interpretable.  The low-level task executor still
handles movement and object interaction; Hu-v0 only scores high-level subgoals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from durf.feedback_attribution.condition_features import (
    CONDITION_KEYS,
    COORDINATION_CONDITION_KEYS,
    TASK_CONDITION_KEYS,
)
from durf.feedback_attribution.io_utils import read_jsonl
from durf.feedback_attribution.subgoal_preferences import (
    COORDINATION_SUBGOALS,
    HU_SUBGOALS,
    TASK_HU_SUBGOALS,
)


TASK_DECISION_LEVEL = "task"
COORDINATION_DECISION_LEVEL = "coordination"
DECISION_LEVELS = (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL)
UNKNOWN_CONDITION_VALUE = 0.0
TRUE_CONDITION_VALUE = 1.0
FALSE_CONDITION_VALUE = -1.0


@dataclass(frozen=True)
class PairwiseSample:
    user_id: str
    layout: str | None
    condition_features: dict[str, Any]
    preferred_subgoal: str
    rejected_subgoal: str
    decision_level: str = TASK_DECISION_LEVEL
    source_feedback_id: str | None = None
    source_event: str | None = None
    source_decision_id: str | None = None


def condition_value(value: Any) -> float:
    if value is True:
        return TRUE_CONDITION_VALUE
    if value is False:
        return FALSE_CONDITION_VALUE
    return UNKNOWN_CONDITION_VALUE


def normalize_conditions(
    condition_features: dict[str, Any] | None,
    condition_keys: tuple[str, ...] = CONDITION_KEYS,
) -> np.ndarray:
    condition_features = condition_features or {}
    return np.asarray(
        [condition_value(condition_features.get(key)) for key in condition_keys],
        dtype=np.float32,
    )


def resolve_dataset_path(path: Path) -> Path:
    if path.is_dir():
        return path / "hu_subgoal_preferences.jsonl"
    return path


def infer_decision_level(
    preferred_subgoal: str,
    rejected_subgoal: str,
    explicit_level: Any = None,
) -> str | None:
    level = str(explicit_level or "").strip().lower()
    if level in DECISION_LEVELS:
        return level
    pair = {preferred_subgoal, rejected_subgoal}
    if pair.issubset(set(COORDINATION_SUBGOALS)):
        return COORDINATION_DECISION_LEVEL
    if pair.issubset(set(TASK_HU_SUBGOALS)):
        return TASK_DECISION_LEVEL
    return None


def load_pairwise_samples(paths: list[Path]) -> list[PairwiseSample]:
    samples: list[PairwiseSample] = []
    for raw_path in paths:
        path = resolve_dataset_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for record in read_jsonl(path):
            if record.get("record_type") != "hu_pairwise_subgoal_preference":
                continue
            preferred = str(record.get("preferred_subgoal") or "")
            rejected = str(record.get("rejected_subgoal") or "")
            if not preferred or not rejected or preferred == rejected:
                continue
            if preferred not in HU_SUBGOALS or rejected not in HU_SUBGOALS:
                continue
            decision_level = infer_decision_level(
                preferred,
                rejected,
                record.get("decision_level"),
            )
            if decision_level is None:
                continue
            samples.append(
                PairwiseSample(
                    user_id=str(record.get("user_id") or "UNKNOWN_USER"),
                    layout=record.get("layout"),
                    condition_features=dict(record.get("condition_features") or {}),
                    preferred_subgoal=preferred,
                    rejected_subgoal=rejected,
                    decision_level=decision_level,
                    source_feedback_id=record.get("source_feedback_id"),
                    source_event=record.get("source_event"),
                    source_decision_id=record.get("source_decision_id"),
                )
            )
    return samples


class LinearSubgoalReranker:
    """Interpretable pairwise ranking model.

    Score decomposition:

        score(user, condition, subgoal)
        = global_subgoal_bias[subgoal]
        + user_subgoal_bias[user, subgoal]
        + sum_k condition_weight[k, subgoal] * condition[k]

    The training objective is logistic pairwise ranking.
    """

    def __init__(
        self,
        *,
        subgoals: tuple[str, ...] = HU_SUBGOALS,
        condition_keys: tuple[str, ...] = CONDITION_KEYS,
        users: tuple[str, ...] = (),
        seed: int = 0,
    ) -> None:
        self.subgoals = tuple(subgoals)
        self.condition_keys = tuple(condition_keys)
        self.users = tuple(users)
        self.subgoal_to_index = {name: index for index, name in enumerate(self.subgoals)}
        self.user_to_index = {name: index for index, name in enumerate(self.users)}
        rng = np.random.default_rng(seed)
        self.global_subgoal_bias = np.zeros(len(self.subgoals), dtype=np.float32)
        self.condition_weights = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(len(self.condition_keys), len(self.subgoals)),
        ).astype(np.float32)
        self.user_subgoal_bias = np.zeros(
            (len(self.users), len(self.subgoals)),
            dtype=np.float32,
        )

    @classmethod
    def from_samples(
        cls,
        samples: list[PairwiseSample],
        seed: int = 0,
        *,
        subgoals: tuple[str, ...] = HU_SUBGOALS,
        condition_keys: tuple[str, ...] = CONDITION_KEYS,
    ):
        users = tuple(sorted({sample.user_id for sample in samples}))
        return cls(
            subgoals=subgoals,
            condition_keys=condition_keys,
            users=users,
            seed=seed,
        )

    def score(self, user_id: str, condition_features: dict[str, Any], subgoal: str) -> float:
        if subgoal not in self.subgoal_to_index:
            raise KeyError(f"Unknown subgoal: {subgoal}")
        subgoal_index = self.subgoal_to_index[subgoal]
        conditions = normalize_conditions(condition_features, self.condition_keys)
        value = float(self.global_subgoal_bias[subgoal_index])
        value += float(np.dot(conditions, self.condition_weights[:, subgoal_index]))
        user_index = self.user_to_index.get(user_id)
        if user_index is not None:
            value += float(self.user_subgoal_bias[user_index, subgoal_index])
        return value

    def rank_subgoals(
        self,
        *,
        user_id: str,
        condition_features: dict[str, Any],
        candidate_subgoals: list[str] | None = None,
    ) -> list[dict[str, float | str]]:
        candidates = candidate_subgoals or list(self.subgoals)
        scored = [
            {
                "subgoal": subgoal,
                "preference_score": self.score(user_id, condition_features, subgoal),
            }
            for subgoal in candidates
            if subgoal in self.subgoal_to_index
        ]
        return sorted(scored, key=lambda item: float(item["preference_score"]), reverse=True)

    def pair_margin(self, sample: PairwiseSample) -> float:
        return self.score(
            sample.user_id,
            sample.condition_features,
            sample.preferred_subgoal,
        ) - self.score(
            sample.user_id,
            sample.condition_features,
            sample.rejected_subgoal,
        )

    def train(
        self,
        samples: list[PairwiseSample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        seed: int = 0,
    ) -> dict[str, list[float]]:
        if not samples:
            raise ValueError("Cannot train Hu-v0 without pairwise samples")
        history = {"loss": [], "accuracy": []}
        rng = np.random.default_rng(seed)
        order = np.arange(len(samples))
        for _epoch in range(epochs):
            rng.shuffle(order)
            loss_sum = 0.0
            correct = 0
            for index in order:
                sample = samples[int(index)]
                preferred_index = self.subgoal_to_index[sample.preferred_subgoal]
                rejected_index = self.subgoal_to_index[sample.rejected_subgoal]
                conditions = normalize_conditions(
                    sample.condition_features,
                    self.condition_keys,
                )
                margin = self.pair_margin(sample)
                correct += int(margin > 0.0)
                # softplus(-margin)
                loss_sum += float(np.logaddexp(0.0, -margin))
                grad_margin = -1.0 / (1.0 + float(np.exp(margin)))

                self.global_subgoal_bias[preferred_index] -= learning_rate * (
                    grad_margin + l2 * self.global_subgoal_bias[preferred_index]
                )
                self.global_subgoal_bias[rejected_index] -= learning_rate * (
                    -grad_margin + l2 * self.global_subgoal_bias[rejected_index]
                )

                self.condition_weights[:, preferred_index] -= learning_rate * (
                    grad_margin * conditions
                    + l2 * self.condition_weights[:, preferred_index]
                )
                self.condition_weights[:, rejected_index] -= learning_rate * (
                    -grad_margin * conditions
                    + l2 * self.condition_weights[:, rejected_index]
                )

                user_index = self.user_to_index.get(sample.user_id)
                if user_index is not None:
                    self.user_subgoal_bias[user_index, preferred_index] -= learning_rate * (
                        grad_margin
                        + l2 * self.user_subgoal_bias[user_index, preferred_index]
                    )
                    self.user_subgoal_bias[user_index, rejected_index] -= learning_rate * (
                        -grad_margin
                        + l2 * self.user_subgoal_bias[user_index, rejected_index]
                    )

            history["loss"].append(loss_sum / len(samples))
            history["accuracy"].append(correct / len(samples))
        return history

    def evaluate(self, samples: list[PairwiseSample]) -> dict[str, Any]:
        if not samples:
            return {"samples": 0, "pairwise_accuracy": None, "mean_margin": None}
        margins = np.asarray([self.pair_margin(sample) for sample in samples], dtype=np.float32)
        return {
            "samples": len(samples),
            "pairwise_accuracy": float((margins > 0.0).mean()),
            "mean_margin": float(margins.mean()),
            "min_margin": float(margins.min()),
            "max_margin": float(margins.max()),
        }

    def top_condition_weights(self, limit: int = 20) -> list[dict[str, Any]]:
        entries = []
        for condition_index, condition_key in enumerate(self.condition_keys):
            for subgoal_index, subgoal in enumerate(self.subgoals):
                weight = float(self.condition_weights[condition_index, subgoal_index])
                entries.append(
                    {
                        "condition": condition_key,
                        "subgoal": subgoal,
                        "weight": weight,
                        "direction": "promotes" if weight >= 0.0 else "discourages",
                    }
                )
        entries.sort(key=lambda item: abs(float(item["weight"])), reverse=True)
        return entries[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "linear_pairwise_subgoal_reranker",
            "subgoals": list(self.subgoals),
            "condition_keys": list(self.condition_keys),
            "users": list(self.users),
            "global_subgoal_bias": self.global_subgoal_bias.tolist(),
            "condition_weights": self.condition_weights.tolist(),
            "user_subgoal_bias": self.user_subgoal_bias.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        model = cls(
            subgoals=tuple(data["subgoals"]),
            condition_keys=tuple(data["condition_keys"]),
            users=tuple(data.get("users") or ()),
        )
        model.global_subgoal_bias = np.asarray(
            data["global_subgoal_bias"],
            dtype=np.float32,
        )
        model.condition_weights = np.asarray(data["condition_weights"], dtype=np.float32)
        model.user_subgoal_bias = np.asarray(
            data.get("user_subgoal_bias") or [],
            dtype=np.float32,
        ).reshape((len(model.users), len(model.subgoals)))
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class HierarchicalHu:
    """One Hu interface with independent task and coordination heads."""

    def __init__(
        self,
        *,
        task_head: LinearSubgoalReranker | None,
        coordination_head: LinearSubgoalReranker | None,
    ) -> None:
        self.task_head = task_head
        self.coordination_head = coordination_head

    @classmethod
    def from_samples(cls, samples: list[PairwiseSample], seed: int = 0):
        task_samples = [
            sample
            for sample in samples
            if sample.decision_level == TASK_DECISION_LEVEL
        ]
        coordination_samples = [
            sample
            for sample in samples
            if sample.decision_level == COORDINATION_DECISION_LEVEL
        ]
        task_head = (
            LinearSubgoalReranker.from_samples(
                task_samples,
                seed=seed,
                subgoals=TASK_HU_SUBGOALS,
                condition_keys=TASK_CONDITION_KEYS,
            )
            if task_samples
            else None
        )
        coordination_head = (
            LinearSubgoalReranker.from_samples(
                coordination_samples,
                seed=seed + 1,
                subgoals=COORDINATION_SUBGOALS,
                condition_keys=COORDINATION_CONDITION_KEYS,
            )
            if coordination_samples
            else None
        )
        return cls(
            task_head=task_head,
            coordination_head=coordination_head,
        )

    def head_for(self, decision_level: str) -> LinearSubgoalReranker | None:
        if decision_level == TASK_DECISION_LEVEL:
            return self.task_head
        if decision_level == COORDINATION_DECISION_LEVEL:
            return self.coordination_head
        raise KeyError(f"Unknown decision level: {decision_level}")

    def score(
        self,
        decision_level: str,
        user_id: str,
        condition_features: dict[str, Any],
        candidate: str,
    ) -> float:
        head = self.head_for(decision_level)
        if head is None or candidate not in head.subgoal_to_index:
            return 0.0
        return head.score(user_id, condition_features, candidate)

    def train(
        self,
        samples: list[PairwiseSample],
        **kwargs,
    ) -> dict[str, dict[str, list[float]] | None]:
        history: dict[str, dict[str, list[float]] | None] = {
            TASK_DECISION_LEVEL: None,
            COORDINATION_DECISION_LEVEL: None,
        }
        for decision_level in DECISION_LEVELS:
            head = self.head_for(decision_level)
            domain_samples = [
                sample
                for sample in samples
                if sample.decision_level == decision_level
            ]
            if head is not None and domain_samples:
                history[decision_level] = head.train(domain_samples, **kwargs)
        return history

    def evaluate(self, samples: list[PairwiseSample]) -> dict[str, Any]:
        metrics = {}
        for decision_level in DECISION_LEVELS:
            head = self.head_for(decision_level)
            domain_samples = [
                sample
                for sample in samples
                if sample.decision_level == decision_level
            ]
            metrics[decision_level] = (
                head.evaluate(domain_samples)
                if head is not None
                else {
                    "samples": len(domain_samples),
                    "pairwise_accuracy": None,
                    "mean_margin": None,
                }
            )
        return metrics

    def top_condition_weights(self, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        return {
            TASK_DECISION_LEVEL: (
                self.task_head.top_condition_weights(limit)
                if self.task_head is not None
                else []
            ),
            COORDINATION_DECISION_LEVEL: (
                self.coordination_head.top_condition_weights(limit)
                if self.coordination_head is not None
                else []
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "hierarchical_linear_pairwise_hu",
            "task_head": self.task_head.to_dict() if self.task_head else None,
            "coordination_head": (
                self.coordination_head.to_dict()
                if self.coordination_head
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if data.get("model_type") == "linear_pairwise_subgoal_reranker":
            return cls(
                task_head=LinearSubgoalReranker.from_dict(data),
                coordination_head=None,
            )
        if data.get("model_type") != "hierarchical_linear_pairwise_hu":
            raise ValueError(f"Unsupported Hu model: {data.get('model_type')}")
        return cls(
            task_head=(
                LinearSubgoalReranker.from_dict(data["task_head"])
                if data.get("task_head")
                else None
            ),
            coordination_head=(
                LinearSubgoalReranker.from_dict(data["coordination_head"])
                if data.get("coordination_head")
                else None
            ),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
