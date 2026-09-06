"""Guard rails for candidate generation and task-layer decision making.

Two invariants this project relies on:

1. The partner never decides whether an option exists. A human standing in the
   corridor may make one route unusable, but deleting the option is exactly the
   wrong move: that is the moment a coordination preference needs both options
   on the table.
2. Task score and preference score are never added. The task prior satisfices
   first (within a tolerance stated in task points), and the learned preference
   only orders what survives -- so Hu's arbitrary output scale never has to be
   calibrated, and a zero tolerance reproduces the frozen backbone exactly.
"""

from __future__ import annotations

import unittest

from durf.baseline.collect_rule_teacher_dataset import (
    CandidateSubgoal,
    choose_task_candidate,
)


def candidate(subgoal: str, task_score: float, hu_score: float = 0.0) -> CandidateSubgoal:
    return CandidateSubgoal(
        subgoal=subgoal,
        task_score=task_score,
        action=0,
        reason="test",
        feasible=True,
        hu_score=hu_score,
    )


class TaskToleranceTest(unittest.TestCase):
    def test_zero_tolerance_reproduces_the_task_backbone(self) -> None:
        for scores in ([70.0, 0.0], [80.0, 60.0, 0.0], [95.0, 90.0, 70.0, 0.0]):
            candidates = [
                candidate(f"S{index}", value, hu_score=-value)  # preference points away
                for index, value in enumerate(scores)
            ]
            chosen = choose_task_candidate(candidates, task_tolerance=0.0)
            self.assertEqual(chosen.task_score, max(scores), scores)
            self.assertFalse(chosen.metadata["preference_override"])

    def test_preference_decides_inside_the_tolerance_band(self) -> None:
        candidates = [
            candidate("GET_DISH", 90.0, hu_score=0.0),
            candidate("GET_TOMATO", 70.0, hu_score=4.0),
        ]
        # 20-point gap: out of reach at 15, reachable at 25.
        self.assertEqual(
            choose_task_candidate(list(candidates), task_tolerance=15.0).subgoal,
            "GET_DISH",
        )
        chosen = choose_task_candidate(list(candidates), task_tolerance=25.0)
        self.assertEqual(chosen.subgoal, "GET_TOMATO")
        self.assertTrue(chosen.metadata["preference_override"])
        self.assertEqual(chosen.metadata["task_points_sacrificed"], 20.0)

    def test_high_stakes_decisions_stay_out_of_reach(self) -> None:
        """No plausible tolerance should let a preference cancel a delivery."""
        candidates = [
            candidate("SERVE_SOUP", 100.0, hu_score=-9.0),
            candidate("WAIT", 0.0, hu_score=9.0),
        ]
        for tolerance in (0.0, 10.0, 25.0, 50.0):
            self.assertEqual(
                choose_task_candidate(list(candidates), task_tolerance=tolerance).subgoal,
                "SERVE_SOUP",
                tolerance,
            )

    def test_only_the_order_of_hu_matters_not_its_scale(self) -> None:
        """Rescaling Hu must not change the decision -- that is the whole point."""
        for scale in (0.01, 1.0, 1000.0):
            candidates = [
                candidate("A", 80.0, hu_score=1.0 * scale),
                candidate("B", 75.0, hu_score=2.0 * scale),
            ]
            self.assertEqual(
                choose_task_candidate(candidates, task_tolerance=10.0).subgoal, "B", scale
            )

    def test_negative_tolerance_is_clamped(self) -> None:
        candidates = [candidate("A", 90.0, hu_score=0.0), candidate("B", 10.0, hu_score=5.0)]
        self.assertEqual(
            choose_task_candidate(candidates, task_tolerance=-30.0).subgoal, "A"
        )

    def test_ties_fall_back_to_the_task_prior(self) -> None:
        candidates = [candidate("A", 80.0, hu_score=2.0), candidate("B", 60.0, hu_score=2.0)]
        chosen = choose_task_candidate(candidates, task_tolerance=50.0)
        self.assertEqual(chosen.subgoal, "A")
        self.assertFalse(chosen.metadata["preference_override"])

    def test_infeasible_candidates_are_never_selected(self) -> None:
        blocked = candidate("BLOCKED", 99.0, hu_score=9.0)
        blocked.feasible = False
        candidates = [blocked, candidate("OK", 10.0)]
        self.assertEqual(
            choose_task_candidate(candidates, task_tolerance=50.0).subgoal, "OK"
        )

    def test_override_is_recorded_for_audit(self) -> None:
        candidates = [
            candidate("GET_DISH", 60.0, hu_score=0.0),
            candidate("GET_TOMATO", 50.0, hu_score=3.0),
        ]
        chosen = choose_task_candidate(candidates, task_tolerance=20.0)
        self.assertEqual(chosen.metadata["acceptable_set_size"], 2)
        self.assertEqual(chosen.metadata["task_tolerance"], 20.0)
        self.assertEqual(chosen.metadata["tolerance_unit"], "task_points")
        self.assertTrue(chosen.metadata["preference_override"])
        self.assertEqual(chosen.metadata["task_points_sacrificed"], 10.0)


def priced(subgoal: str, task_score: float, steps: float, hu_score: float = 0.0, supported: bool = True) -> CandidateSubgoal:
    made = candidate(subgoal, task_score, hu_score=hu_score)
    made.step_cost = steps
    made.hu_supported = supported
    return made


class StepToleranceTest(unittest.TestCase):
    """The band denominated in estimated extra steps to the next delivery
    (durf/baseline/task_cost.py), and the two rules that keep a per-decision
    band from being gamed when the same decision recurs.
    """

    def test_step_band_widens_around_the_backbones_pick_not_the_cheapest(self) -> None:
        # The estimator thinks GET_TOMATO is 9 steps dearer; the backbone's
        # ordering (GET_DISH first) is untouched, the band decides reach.
        candidates = [
            priced("GET_DISH", 60.0, steps=35.0, hu_score=-1.0),
            priced("GET_TOMATO", 50.0, steps=44.0, hu_score=+1.0),
        ]
        self.assertEqual(choose_task_candidate(candidates, step_tolerance=8.0).subgoal, "GET_DISH")
        chosen = choose_task_candidate(candidates, step_tolerance=9.0)
        self.assertEqual(chosen.subgoal, "GET_TOMATO")
        self.assertEqual(chosen.metadata["tolerance_unit"], "steps")
        self.assertEqual(chosen.metadata["steps_sacrificed"], 9.0)
        self.assertEqual(chosen.metadata["task_optimum_steps"], 35.0)

    def test_tolerance_zero_is_the_backbone_in_both_units(self) -> None:
        # Even when the estimator says the alternative is CHEAPER, tolerance 0
        # keeps the frozen pick: the estimator may widen the band, never
        # overrule the backbone.
        candidates = [
            priced("PUT_DOWN_OBJECT", 20.0, steps=61.0, hu_score=-5.0),
            priced("WAIT_NEAR_POT", 10.0, steps=52.0, hu_score=+5.0),
        ]
        for kwargs in ({"step_tolerance": None}, {"step_tolerance": 0.0}, {"task_tolerance": 0.0}):
            self.assertEqual(choose_task_candidate(candidates, **kwargs).subgoal, "PUT_DOWN_OBJECT", kwargs)

    def test_a_preference_may_choose_how_to_act_but_not_whether_to_act(self) -> None:
        """Waiting costs 'best action + 1' and leaves the world unchanged, so
        the same one-step sacrifice is offered again next tick: unbounded.
        Observed as a 0-reward episode. Waiting candidates therefore never
        displace a state-changing optimum, in either unit."""
        candidates = [
            priced("GET_TOMATO", 70.0, steps=62.0, hu_score=-9.0),
            priced("WAIT", 0.0, steps=63.0, hu_score=+9.0),
            priced("WAIT_NEAR_POT", 10.0, steps=63.0, hu_score=+9.0),
        ]
        self.assertEqual(choose_task_candidate(candidates, step_tolerance=50.0).subgoal, "GET_TOMATO")
        self.assertEqual(choose_task_candidate(candidates, task_tolerance=100.0).subgoal, "GET_TOMATO")

        # ...but when the backbone itself waits, how to wait is open.
        waiting = [
            priced("WAIT_NEAR_POT", 15.0, steps=40.0, hu_score=-1.0),
            priced("WAIT", 5.0, steps=40.0, hu_score=+1.0),
        ]
        self.assertEqual(choose_task_candidate(waiting, step_tolerance=1.0).subgoal, "WAIT")

        # Putting something down changes the world: it is an action, priced
        # once, and may be preferred inside the band.
        drop = [
            priced("PUT_TOMATO_IN_POT", 80.0, steps=45.0, hu_score=-1.0),
            priced("PUT_DOWN_OBJECT", 5.0, steps=47.0, hu_score=+1.0),
        ]
        self.assertEqual(choose_task_candidate(drop, step_tolerance=2.0).subgoal, "PUT_DOWN_OBJECT")

    def test_no_evidence_means_no_opinion(self) -> None:
        """A subgoal Hu has never seen in a label cannot be the reason a
        decision moves off the backbone -- neither a random-init residue nor
        'no opinion' may outrank a trained negative opinion."""
        candidates = [
            priced("GET_TOMATO", 70.0, steps=62.0, hu_score=-0.16, supported=True),
            priced("PUT_DOWN_OBJECT", 5.0, steps=63.0, hu_score=0.04, supported=False),
        ]
        chosen = choose_task_candidate(candidates, step_tolerance=5.0)
        self.assertEqual(chosen.subgoal, "GET_TOMATO")
        self.assertFalse(chosen.metadata["preference_override"])

        # The backbone's own pick is always eligible even when unlabelled.
        candidates = [
            priced("GET_DISH", 60.0, steps=35.0, hu_score=0.0, supported=False),
            priced("GET_TOMATO", 50.0, steps=40.0, hu_score=+1.0, supported=True),
        ]
        self.assertEqual(choose_task_candidate(candidates, step_tolerance=5.0).subgoal, "GET_TOMATO")

    def test_unpriced_candidates_stay_outside_the_step_band(self) -> None:
        candidates = [
            priced("GET_DISH", 60.0, steps=35.0, hu_score=-1.0),
            candidate("GET_TOMATO", 50.0, hu_score=+1.0),  # no step_cost
        ]
        self.assertEqual(choose_task_candidate(candidates, step_tolerance=50.0).subgoal, "GET_DISH")


if __name__ == "__main__":
    unittest.main()
