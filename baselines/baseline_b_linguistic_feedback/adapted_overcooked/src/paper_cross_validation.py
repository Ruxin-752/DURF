"""Paper-style teacher and reward-configuration cross-validation for Route 2."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .evaluation_splits import canonical_sha256


CV_SCHEMA_VERSION = "route2-paper-cross-validation-v2"


def _teacher_id(example: dict) -> str:
    value = example.get("teacher_id")
    if value is None:
        raise ValueError(
            "paper-style cross-validation requires a stable teacher_id; "
            "row-hash cv_teacher_id proxies and implicit style fallbacks are invalid"
        )
    return str(value)


def _reward_id(example: dict) -> str:
    value = example.get("reward_config_id")
    if value is None:
        raise ValueError("paper-style cross-validation requires reward_config_id")
    return str(value)


def _components(examples: list[dict]) -> list[dict[str, set[str]]]:
    """Find teacher/reward bipartite components in possibly sparse corpora."""

    teacher_rewards: dict[str, set[str]] = defaultdict(set)
    reward_teachers: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        teacher, reward = _teacher_id(example), _reward_id(example)
        teacher_rewards[teacher].add(reward)
        reward_teachers[reward].add(teacher)

    remaining = set(teacher_rewards)
    output = []
    while remaining:
        teacher_queue = [min(remaining)]
        teachers: set[str] = set()
        rewards: set[str] = set()
        while teacher_queue:
            teacher = teacher_queue.pop()
            if teacher in teachers:
                continue
            teachers.add(teacher)
            remaining.discard(teacher)
            for reward in teacher_rewards[teacher]:
                if reward in rewards:
                    continue
                rewards.add(reward)
                teacher_queue.extend(reward_teachers[reward] - teachers)
        output.append({"teachers": teachers, "rewards": rewards})
    return sorted(output, key=lambda item: (-len(item["teachers"]), sorted(item["teachers"])))


def _assign_axes(examples: list[dict], *, n_folds: int, seed: int) -> tuple[dict, dict]:
    teachers = sorted({_teacher_id(example) for example in examples})
    rewards = sorted({_reward_id(example) for example in examples})
    if n_folds < 3:
        raise ValueError("paper-style train/dev/test rotation needs at least 3 folds")
    if len(teachers) < n_folds or len(rewards) < n_folds:
        raise ValueError(
            "paper-style folds need at least one teacher and reward per fold: "
            f"teachers={len(teachers)}, rewards={len(rewards)}, folds={n_folds}"
        )

    rng = random.Random(seed)
    teacher_fold: dict[str, int] = {}
    reward_fold: dict[str, int] = {}
    fold_teacher_counts = [0] * n_folds

    # Sparse generated corpora may contain disconnected declared-split blocks.
    # Spread teachers globally, then assign each component's rewards only to
    # folds where that component has a teacher, guaranteeing non-empty tests.
    for component in _components(examples):
        component_teachers = sorted(component["teachers"])
        rng.shuffle(component_teachers)
        for teacher in component_teachers:
            fold = min(range(n_folds), key=lambda index: (fold_teacher_counts[index], index))
            teacher_fold[teacher] = fold
            fold_teacher_counts[fold] += 1

        active_folds = sorted({teacher_fold[teacher] for teacher in component_teachers})
        component_rewards = sorted(component["rewards"])
        rng.shuffle(component_rewards)
        local_counts = {fold: 0 for fold in active_folds}
        for reward in component_rewards:
            fold = min(active_folds, key=lambda index: (local_counts[index], index))
            reward_fold[reward] = fold
            local_counts[fold] += 1

    if set(teacher_fold) != set(teachers) or set(reward_fold) != set(rewards):
        raise RuntimeError("cross-validation axis assignment is incomplete")
    return teacher_fold, reward_fold


def make_paper_cross_validation_manifest(
    examples: list[dict],
    *,
    n_folds: int = 10,
    seed: int = 42,
) -> dict:
    """Match the paper: validate on fold i and test on fold i+1."""

    components = _components(examples)
    teacher_fold, reward_fold = _assign_axes(examples, n_folds=n_folds, seed=seed)
    folds = []
    all_folds = set(range(n_folds))
    total_examples = len(examples)
    test_frequency = [0] * total_examples
    dev_frequency = [0] * total_examples
    for fold in range(n_folds):
        dev_fold = fold
        test_fold = (fold + 1) % n_folds
        train_folds = all_folds - {dev_fold, test_fold}
        train_indices: list[int] = []
        dev_indices: list[int] = []
        test_indices: list[int] = []
        excluded_indices: list[int] = []
        excluded_axis_pairs: dict[str, int] = defaultdict(int)
        for index, example in enumerate(examples):
            teacher_axis = teacher_fold[_teacher_id(example)]
            reward_axis = reward_fold[_reward_id(example)]
            if teacher_axis in train_folds and reward_axis in train_folds:
                train_indices.append(index)
            elif teacher_axis == dev_fold and reward_axis == dev_fold:
                dev_indices.append(index)
            elif teacher_axis == test_fold and reward_axis == test_fold:
                test_indices.append(index)
            else:
                excluded_indices.append(index)
                teacher_partition = (
                    "train"
                    if teacher_axis in train_folds
                    else "dev"
                    if teacher_axis == dev_fold
                    else "test"
                )
                reward_partition = (
                    "train"
                    if reward_axis in train_folds
                    else "dev"
                    if reward_axis == dev_fold
                    else "test"
                )
                excluded_axis_pairs[
                    f"teacher_{teacher_partition}__reward_{reward_partition}"
                ] += 1
        if not train_indices or not dev_indices or not test_indices:
            raise ValueError(
                f"fold {fold} is empty: train={len(train_indices)}, "
                f"dev={len(dev_indices)}, test={len(test_indices)}"
            )
        train_teachers = {_teacher_id(examples[index]) for index in train_indices}
        dev_teachers = {_teacher_id(examples[index]) for index in dev_indices}
        test_teachers = {_teacher_id(examples[index]) for index in test_indices}
        train_rewards = {_reward_id(examples[index]) for index in train_indices}
        dev_rewards = {_reward_id(examples[index]) for index in dev_indices}
        test_rewards = {_reward_id(examples[index]) for index in test_indices}
        if train_teachers & (dev_teachers | test_teachers) or dev_teachers & test_teachers:
            raise RuntimeError("teacher leakage in paper cross-validation fold")
        if train_rewards & (dev_rewards | test_rewards) or dev_rewards & test_rewards:
            raise RuntimeError("reward leakage in paper cross-validation fold")
        for index in test_indices:
            test_frequency[index] += 1
        for index in dev_indices:
            dev_frequency[index] += 1

        def normalized_text(index: int) -> str:
            return " ".join(str(examples[index].get("text") or "").casefold().split())

        def input_signature(index: int) -> tuple[str, tuple[float, ...]]:
            return (
                normalized_text(index),
                tuple(
                    round(float(value), 12)
                    for value in examples[index].get("feature_counts") or ()
                ),
            )

        train_texts = {normalized_text(index) for index in train_indices}
        train_inputs = {input_signature(index) for index in train_indices}
        text_overlap = sum(normalized_text(index) in train_texts for index in test_indices)
        input_overlap = sum(input_signature(index) in train_inputs for index in test_indices)
        folds.append(
            {
                "fold": fold,
                "dev_axis_fold": dev_fold,
                "test_axis_fold": test_fold,
                "train_indices": train_indices,
                "dev_indices": dev_indices,
                "test_indices": test_indices,
                "excluded_indices": excluded_indices,
                "train_teachers": sorted(train_teachers),
                "dev_teachers": sorted(dev_teachers),
                "test_teachers": sorted(test_teachers),
                "train_rewards": sorted(train_rewards),
                "dev_rewards": sorted(dev_rewards),
                "test_rewards": sorted(test_rewards),
                "coverage": {
                    "corpus_examples": total_examples,
                    "train_examples": len(train_indices),
                    "dev_examples": len(dev_indices),
                    "test_examples": len(test_indices),
                    "excluded_examples": len(excluded_indices),
                    "included_fraction": (
                        (len(train_indices) + len(dev_indices) + len(test_indices))
                        / total_examples
                        if total_examples
                        else 0.0
                    ),
                    "excluded_fraction": (
                        len(excluded_indices) / total_examples if total_examples else 0.0
                    ),
                    "excluded_axis_pairs": dict(sorted(excluded_axis_pairs.items())),
                },
                "heldout_overlap_audit": {
                    "test_exact_text_in_train": text_overlap,
                    "test_exact_text_in_train_fraction": text_overlap / len(test_indices),
                    "test_exact_input_in_train": input_overlap,
                    "test_exact_input_in_train_fraction": input_overlap / len(test_indices),
                },
            }
        )
    unique_test = sum(value > 0 for value in test_frequency)
    unique_dev = sum(value > 0 for value in dev_frequency)
    manifest = {
        "schema_version": CV_SCHEMA_VERSION,
        "policy": "paper_10fold_teacher_and_reward_holdout",
        "n_folds": n_folds,
        "seed": seed,
        "teacher_identity_field": "teacher_id",
        "teacher_fold": teacher_fold,
        "reward_fold": reward_fold,
        "teacher_reward_axis_graph": {
            "component_count": len(components),
            "components": [
                {
                    "teachers": sorted(component["teachers"]),
                    "rewards": sorted(component["rewards"]),
                }
                for component in components
            ],
        },
        "folds": folds,
        "coverage": {
            "corpus_examples": total_examples,
            "unique_test_examples": unique_test,
            "unique_test_fraction": unique_test / total_examples if total_examples else 0.0,
            "never_test_examples": total_examples - unique_test,
            "unique_dev_examples": unique_dev,
            "unique_dev_fraction": unique_dev / total_examples if total_examples else 0.0,
            "test_assignment_frequency_min": min(test_frequency) if test_frequency else 0,
            "test_assignment_frequency_max": max(test_frequency) if test_frequency else 0,
            "total_excluded_fold_assignments": sum(
                len(fold["excluded_indices"]) for fold in folds
            ),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def aggregate_vector_metrics(metrics: Iterable[dict]) -> dict:
    rows = list(metrics)
    examples = sum(int(row["examples"]) for row in rows)
    active = sum(int(row["active_target_count"]) for row in rows)
    return {
        "mse": sum(float(row["mse"]) * int(row["examples"]) for row in rows) / examples,
        "mean_cosine_similarity": sum(
            float(row["mean_cosine_similarity"]) * int(row["examples"]) for row in rows
        ) / examples,
        "active_sign_accuracy": sum(
            float(row["active_sign_accuracy"]) * int(row["active_target_count"])
            for row in rows
        ) / active,
        "active_target_count": active,
        "examples": examples,
    }
