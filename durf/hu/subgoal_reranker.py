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

# PerUserAdapter design switch: whether the task head keeps a per-user
# constant offset (user_bias_task) alongside its condition-dependent term.
# Default is off -- see PerUserAdapter.__init__ for the rationale.  This is
# a reversible flag, not an architecture deletion: flipping it back to True
# restores the original three-level decomposition for the task head.
DEFAULT_ENABLE_TASK_BIAS = False


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


# ---------------------------------------------------------------------------
# Protocol-v2: Per-user adapter with frozen Hu_general
# ---------------------------------------------------------------------------


class PerUserAdapter:
    """Protocol-v2 three-level user model on top of frozen Hu_general.

    Score decomposition:

        Hu_user(subgoal | condition)
            = Hu_general(condition, subgoal)                    [frozen]
            + user_bias[subgoal]                                 [learnable]
            + sum_k(condition_delta[k, subgoal] * condition[k])  [learnable]

    Only user_bias and condition_delta are trained.  Hu_general weights
    are never modified.
    """

    def __init__(
        self,
        *,
        hu_general: HierarchicalHu,
        user_id: str,
        enable_task_bias: bool = DEFAULT_ENABLE_TASK_BIAS,
    ) -> None:
        self.hu_general = hu_general
        self.user_id = user_id
        self._model_type = "per_user_adapter_v2"

        # Task-level user_bias is disabled by default: across the simulated
        # personas evaluated so far, user_bias_task stays within +/-0.06,
        # consistent with the design view that "unconditionally liking a
        # subgoal" is not a meaningful construct at the task level -- task
        # preference is inherently conditional on the situation (missing
        # ingredients, pot status, etc.).  Coordination-level yielding
        # tendencies behave differently and do show a stable per-user
        # offset (user_bias_coord ~= +/-1.7), so that head keeps its bias
        # term.  This flag is a reversible switch, not a deletion: when
        # disabled, user_bias_task is never updated during training (stays
        # exactly zero) and never contributes to the task-head score, but
        # the array, its shape, and the load/save code are all left intact
        # so a future dataset (e.g. real human feedback) can flip it back
        # on and re-test the assumption without a schema migration.
        self.enable_task_bias = bool(enable_task_bias)

        # -- user_bias (user x subgoal) --
        self.user_bias_task = np.zeros(len(TASK_HU_SUBGOALS), dtype=np.float32)
        self.user_bias_coord = np.zeros(
            len(COORDINATION_SUBGOALS), dtype=np.float32
        )
        self._task_subgoal_to_index = {
            name: idx for idx, name in enumerate(TASK_HU_SUBGOALS)
        }
        self._coord_subgoal_to_index = {
            name: idx for idx, name in enumerate(COORDINATION_SUBGOALS)
        }

        # -- condition_delta (condition x subgoal) --
        self.condition_delta_task = np.zeros(
            (len(CONDITION_KEYS), len(TASK_HU_SUBGOALS)), dtype=np.float32
        )
        self.condition_delta_coord = np.zeros(
            (len(COORDINATION_CONDITION_KEYS), len(COORDINATION_SUBGOALS)),
            dtype=np.float32,
        )
        self._task_cond_to_index = {
            name: idx for idx, name in enumerate(CONDITION_KEYS)
        }
        self._coord_cond_to_index = {
            name: idx for idx, name in enumerate(COORDINATION_CONDITION_KEYS)
        }

        # tracking: which conditions this user has observed
        self.observed_conditions_task: set[str] = set()
        self.observed_conditions_coord: set[str] = set()

    # -- Scoring -----------------------------------------------------------

    def score(
        self,
        decision_level: str,
        condition_features: dict[str, Any],
        subgoal: str,
    ) -> dict[str, float]:
        """Return decomposed dict: general_score, user_bias_score,
        condition_delta_score, final_score."""
        general = self.hu_general.score(
            decision_level, self.user_id, condition_features, subgoal
        )
        bias = self._user_bias_score(decision_level, subgoal)
        delta = self._condition_delta_score(
            decision_level, condition_features, subgoal
        )
        return {
            "general_score": float(general),
            "user_bias_score": float(bias),
            "condition_delta_score": float(delta),
            "final_score": float(general + bias + delta),
        }

    def _user_bias_score(self, decision_level: str, subgoal: str) -> float:
        if decision_level == TASK_DECISION_LEVEL:
            if not self.enable_task_bias:
                return 0.0
            idx = self._task_subgoal_to_index.get(subgoal)
            return float(self.user_bias_task[idx]) if idx is not None else 0.0
        idx = self._coord_subgoal_to_index.get(subgoal)
        return float(self.user_bias_coord[idx]) if idx is not None else 0.0

    def _condition_delta_score(
        self,
        decision_level: str,
        condition_features: dict[str, Any],
        subgoal: str,
    ) -> float:
        if decision_level == TASK_DECISION_LEVEL:
            cond_keys = CONDITION_KEYS
            delta = self.condition_delta_task
            c2i = self._task_cond_to_index
            s2i = self._task_subgoal_to_index
        else:
            cond_keys = COORDINATION_CONDITION_KEYS
            delta = self.condition_delta_coord
            c2i = self._coord_cond_to_index
            s2i = self._coord_subgoal_to_index

        subgoal_idx = s2i.get(subgoal)
        if subgoal_idx is None:
            return 0.0

        value = 0.0
        for cond_key in cond_keys:
            cond_idx = c2i.get(cond_key)
            if cond_idx is None:
                continue
            cv = condition_value(condition_features.get(cond_key))
            if cv == UNKNOWN_CONDITION_VALUE:
                continue
            value += delta[cond_idx, subgoal_idx] * cv
        return float(value)

    # -- Training ----------------------------------------------------------

    def pair_margin(
        self,
        decision_level: str,
        condition_features: dict[str, Any],
        preferred_subgoal: str,
        rejected_subgoal: str,
    ) -> float:
        pref = self.score(decision_level, condition_features, preferred_subgoal)
        rej = self.score(decision_level, condition_features, rejected_subgoal)
        return pref["final_score"] - rej["final_score"]

    def train(
        self,
        samples: list[PairwiseSample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2_bias: float = 1e-4,
        l2_delta: float = 1e-3,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Train user_bias and condition_delta. Hu_general is frozen."""
        if not samples:
            return {
                "history": {"loss": [], "accuracy": []},
                "observed_conditions": {"task": [], "coordination": []},
            }

        # Track observed conditions
        for sample in samples:
            cf = sample.condition_features or {}
            keys = (
                CONDITION_KEYS
                if sample.decision_level == TASK_DECISION_LEVEL
                else COORDINATION_CONDITION_KEYS
            )
            target = (
                self.observed_conditions_task
                if sample.decision_level == TASK_DECISION_LEVEL
                else self.observed_conditions_coord
            )
            for key in keys:
                if cf.get(key) is not None:
                    target.add(key)

        history: dict[str, Any] = {"loss": [], "accuracy": []}
        rng = np.random.default_rng(seed)
        order = np.arange(len(samples))

        for _ in range(epochs):
            rng.shuffle(order)
            loss_sum = 0.0
            correct = 0
            for index in order:
                sample = samples[int(index)]
                margin = self.pair_margin(
                    sample.decision_level,
                    sample.condition_features,
                    sample.preferred_subgoal,
                    sample.rejected_subgoal,
                )
                correct += int(margin > 0.0)
                loss_sum += float(np.logaddexp(0.0, -margin))
                g = -1.0 / (1.0 + float(np.exp(margin)))

                if sample.decision_level == TASK_DECISION_LEVEL:
                    bias_arr = self.user_bias_task
                    delta_mat = self.condition_delta_task
                    cond_keys = CONDITION_KEYS
                    c2i = self._task_cond_to_index
                    s2i = self._task_subgoal_to_index
                else:
                    bias_arr = self.user_bias_coord
                    delta_mat = self.condition_delta_coord
                    cond_keys = COORDINATION_CONDITION_KEYS
                    c2i = self._coord_cond_to_index
                    s2i = self._coord_subgoal_to_index

                pi = s2i[sample.preferred_subgoal]
                ri = s2i[sample.rejected_subgoal]

                update_bias = (
                    sample.decision_level != TASK_DECISION_LEVEL
                    or self.enable_task_bias
                )
                if update_bias:
                    bias_arr[pi] -= learning_rate * (g + l2_bias * bias_arr[pi])
                    bias_arr[ri] -= learning_rate * (-g + l2_bias * bias_arr[ri])

                for cond_key in cond_keys:
                    ci = c2i.get(cond_key)
                    if ci is None:
                        continue
                    cv = condition_value(
                        sample.condition_features.get(cond_key)
                    )
                    if cv == UNKNOWN_CONDITION_VALUE:
                        continue
                    delta_mat[ci, pi] -= learning_rate * (
                        g * cv + l2_delta * delta_mat[ci, pi]
                    )
                    delta_mat[ci, ri] -= learning_rate * (
                        -g * cv + l2_delta * delta_mat[ci, ri]
                    )

            history["loss"].append(loss_sum / len(samples))
            history["accuracy"].append(correct / len(samples))

        return {
            "history": history,
            "observed_conditions": {
                "task": sorted(self.observed_conditions_task),
                "coordination": sorted(self.observed_conditions_coord),
            },
        }

    def evaluate(self, samples: list[PairwiseSample]) -> dict[str, Any]:
        if not samples:
            return {}
        metrics: dict[str, Any] = {}
        for dl in DECISION_LEVELS:
            domain_samples = [s for s in samples if s.decision_level == dl]
            if not domain_samples:
                metrics[dl] = {
                    "samples": 0, "pairwise_accuracy": None,
                    "mean_margin": None,
                }
                continue
            margins = np.asarray(
                [
                    self.pair_margin(
                        s.decision_level,
                        s.condition_features,
                        s.preferred_subgoal,
                        s.rejected_subgoal,
                    )
                    for s in domain_samples
                ],
                dtype=np.float32,
            )
            metrics[dl] = {
                "samples": len(domain_samples),
                "pairwise_accuracy": float((margins > 0.0).mean()),
                "mean_margin": float(margins.mean()),
                "min_margin": float(margins.min()),
                "max_margin": float(margins.max()),
            }
        return metrics

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self._model_type,
            "user_id": self.user_id,
            "enable_task_bias": self.enable_task_bias,
            "hu_general": self.hu_general.to_dict(),
            "user_bias_task": self.user_bias_task.tolist(),
            "user_bias_coord": self.user_bias_coord.tolist(),
            "condition_delta_task": self.condition_delta_task.tolist(),
            "condition_delta_coord": self.condition_delta_coord.tolist(),
            "observed_conditions_task": sorted(
                self.observed_conditions_task
            ),
            "observed_conditions_coord": sorted(
                self.observed_conditions_coord
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        hu_general: HierarchicalHu | None = None,
    ):
        if hu_general is None:
            hu_general = HierarchicalHu.from_dict(data["hu_general"])
        # Backward compatibility: model files saved before this switch
        # existed were always trained with the task bias term enabled, so a
        # missing key means "on" (preserve exactly what was trained), not
        # "off" (the new default for freshly constructed adapters).
        enable_task_bias = data.get("enable_task_bias")
        if enable_task_bias is None:
            enable_task_bias = True
        adapter = cls(
            hu_general=hu_general,
            user_id=data["user_id"],
            enable_task_bias=bool(enable_task_bias),
        )
        adapter.user_bias_task = np.asarray(
            data.get("user_bias_task") or [], dtype=np.float32
        )
        adapter.user_bias_coord = np.asarray(
            data.get("user_bias_coord") or [], dtype=np.float32
        )
        adapter.condition_delta_task = np.asarray(
            data.get("condition_delta_task") or [],
            dtype=np.float32,
        ).reshape((len(CONDITION_KEYS), len(TASK_HU_SUBGOALS)))
        adapter.condition_delta_coord = np.asarray(
            data.get("condition_delta_coord") or [],
            dtype=np.float32,
        ).reshape(
            (len(COORDINATION_CONDITION_KEYS), len(COORDINATION_SUBGOALS))
        )
        adapter.observed_conditions_task = set(
            data.get("observed_conditions_task") or []
        )
        adapter.observed_conditions_coord = set(
            data.get("observed_conditions_coord") or []
        )
        return adapter

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, path: Path, *, hu_general: HierarchicalHu | None = None
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, hu_general=hu_general)


def load_runtime_hu(path: Path) -> HierarchicalHu | PerUserAdapter:
    """Load either a frozen general model or a protocol-v2 user adapter."""
    data = json.loads(path.read_text(encoding="utf-8"))
    model_type = data.get("model_type")
    if model_type == "per_user_adapter_v2":
        return PerUserAdapter.from_dict(data)
    if model_type == "hierarchical_linear_pairwise_hu":
        return HierarchicalHu.from_dict(data)
    raise ValueError(f"Unsupported Hu runtime model_type={model_type!r}: {path}")


def runtime_hu_score(
    model: HierarchicalHu | PerUserAdapter,
    decision_level: str,
    user_id: str,
    condition_features: dict[str, Any],
    subgoal: str,
) -> float:
    """Return one scalar score through a common runtime interface."""
    if isinstance(model, PerUserAdapter):
        return float(
            model.score(decision_level, condition_features, subgoal)[
                "final_score"
            ]
        )
    return float(
        model.score(decision_level, user_id, condition_features, subgoal)
    )
