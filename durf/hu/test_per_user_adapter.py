"""Regression tests for the task-head user_bias switch on PerUserAdapter.

Design decision under test: task-level "unconditional liking of a subgoal"
was judged not to be a meaningful construct (simulated personas trained with
the full three-level decomposition consistently learn a task-level user_bias
within +/-0.06, essentially noise, while the coordination-level bias learns a
real signal, +/-1.7). PerUserAdapter now defaults to disabling user_bias_task
while leaving user_bias_coord untouched. This is a reversible flag
(enable_task_bias), not a deletion, so old model files must still load with
their original behaviour intact.

Four things this suite has to nail down:
1. When disabled, a nonzero user_bias_task never reaches the task-head score.
2. When disabled, user_bias_task is never updated during training (frozen at
   exactly zero) while condition_delta_task keeps training normally.
3. Disabling the task head's bias does not touch the coordination head's
   bias in any way -- training or scoring.
4. A model file saved before this switch existed (no "enable_task_bias" key)
   must load as if the switch were on, to preserve exactly what was trained.
"""

from __future__ import annotations

import unittest

import numpy as np

from durf.feedback_attribution.condition_features import (
    COORDINATION_CONDITION_KEYS,
    TASK_CONDITION_KEYS,
)
from durf.feedback_attribution.subgoal_preferences import (
    COORDINATION_SUBGOALS,
    TASK_HU_SUBGOALS,
)
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    TASK_DECISION_LEVEL,
    HierarchicalHu,
    LinearSubgoalReranker,
    PairwiseSample,
    PerUserAdapter,
)


def make_frozen_hu_general() -> HierarchicalHu:
    task_head = LinearSubgoalReranker(
        subgoals=TASK_HU_SUBGOALS,
        condition_keys=TASK_CONDITION_KEYS,
    )
    coordination_head = LinearSubgoalReranker(
        subgoals=COORDINATION_SUBGOALS,
        condition_keys=COORDINATION_CONDITION_KEYS,
    )
    return HierarchicalHu(task_head=task_head, coordination_head=coordination_head)


def task_sample(preferred: str, rejected: str, **conditions) -> PairwiseSample:
    return PairwiseSample(
        user_id="probe_user",
        layout=None,
        condition_features=conditions,
        preferred_subgoal=preferred,
        rejected_subgoal=rejected,
        decision_level=TASK_DECISION_LEVEL,
    )


def coord_sample(preferred: str, rejected: str, **conditions) -> PairwiseSample:
    return PairwiseSample(
        user_id="probe_user",
        layout=None,
        condition_features=conditions,
        preferred_subgoal=preferred,
        rejected_subgoal=rejected,
        decision_level=COORDINATION_DECISION_LEVEL,
    )


class TaskBiasDisabledByDefaultTest(unittest.TestCase):
    def test_default_construction_disables_task_bias(self) -> None:
        adapter = PerUserAdapter(hu_general=make_frozen_hu_general(), user_id="u1")
        self.assertFalse(adapter.enable_task_bias)

    def test_disabled_bias_never_reaches_task_score(self) -> None:
        adapter = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=False
        )
        # Poke a large nonzero value directly into the array, as if it had
        # been loaded from an old model file or corrupted some other way.
        adapter.user_bias_task[:] = 999.0
        subgoal = TASK_HU_SUBGOALS[0]
        score = adapter.score(TASK_DECISION_LEVEL, {}, subgoal)
        self.assertEqual(score["user_bias_score"], 0.0)
        self.assertEqual(score["final_score"], score["general_score"] + score["condition_delta_score"])

    def test_coord_bias_is_unaffected_by_the_task_switch(self) -> None:
        adapter = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=False
        )
        adapter.user_bias_coord[:] = 1.739
        subgoal = COORDINATION_SUBGOALS[0]
        score = adapter.score(COORDINATION_DECISION_LEVEL, {}, subgoal)
        self.assertAlmostEqual(score["user_bias_score"], 1.739, places=5)


class TrainingFreezesTaskBiasTest(unittest.TestCase):
    def test_task_bias_stays_exactly_zero_after_training(self) -> None:
        adapter = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=False
        )
        samples = [
            task_sample("GET_TOMATO", "GET_ONION", ai_empty_handed=True),
            task_sample("GET_ONION", "GET_TOMATO", ai_empty_handed=False),
            task_sample("GET_TOMATO", "GET_ONION", ai_empty_handed=True),
        ] * 20  # enough passes for a nonzero gradient to show up if not frozen
        adapter.train(samples, epochs=50, learning_rate=0.05, seed=0)
        self.assertTrue(np.all(adapter.user_bias_task == 0.0))
        # condition_delta_task should NOT be frozen -- the whole point is
        # that only the constant offset is disabled, not the model's ability
        # to learn condition-dependent task preferences.
        self.assertFalse(np.all(adapter.condition_delta_task == 0.0))

    def test_task_bias_trains_normally_when_re_enabled(self) -> None:
        adapter = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=True
        )
        samples = [task_sample("GET_TOMATO", "GET_ONION")] * 40
        adapter.train(samples, epochs=50, learning_rate=0.05, seed=0)
        self.assertFalse(np.all(adapter.user_bias_task == 0.0))

    def test_coord_bias_trains_the_same_regardless_of_the_task_switch(self) -> None:
        samples = [
            coord_sample("YIELD", "CONTINUE_CURRENT_SUBGOAL", human_trying_to_pass=True)
        ] * 40

        adapter_off = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=False
        )
        adapter_off.train(samples, epochs=50, learning_rate=0.05, seed=0)

        adapter_on = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=True
        )
        adapter_on.train(samples, epochs=50, learning_rate=0.05, seed=0)

        np.testing.assert_allclose(
            adapter_off.user_bias_coord, adapter_on.user_bias_coord, rtol=1e-6
        )
        np.testing.assert_allclose(
            adapter_off.condition_delta_coord,
            adapter_on.condition_delta_coord,
            rtol=1e-6,
        )


class BackwardCompatibilityTest(unittest.TestCase):
    def test_old_model_file_without_flag_loads_as_enabled(self) -> None:
        adapter = PerUserAdapter(
            hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=True
        )
        adapter.user_bias_task[:] = 0.42
        data = adapter.to_dict()
        # Simulate a model file saved before this switch existed.
        del data["enable_task_bias"]

        reloaded = PerUserAdapter.from_dict(data, hu_general=adapter.hu_general)
        self.assertTrue(reloaded.enable_task_bias)
        subgoal = TASK_HU_SUBGOALS[0]
        score = reloaded.score(TASK_DECISION_LEVEL, {}, subgoal)
        self.assertAlmostEqual(score["user_bias_score"], 0.42, places=5)

    def test_round_trip_preserves_the_flag_either_way(self) -> None:
        for flag in (True, False):
            adapter = PerUserAdapter(
                hu_general=make_frozen_hu_general(), user_id="u1", enable_task_bias=flag
            )
            reloaded = PerUserAdapter.from_dict(
                adapter.to_dict(), hu_general=adapter.hu_general
            )
            self.assertEqual(reloaded.enable_task_bias, flag)


if __name__ == "__main__":
    unittest.main()
