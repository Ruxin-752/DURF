"""The trainer fits whatever it is given; this gate is what stands between it
and the two label shapes protocol v2 S13 forbids.  The corpus audit that
motivated it: 4 554 labels, 201 training an unreachable subgoal, 6 pairing a
task subgoal with a coordination option."""

from __future__ import annotations

import unittest

from durf.feedback_attribution.label_validator import (
    filter_training_samples,
    rejection_summary,
    validate_training_sample,
)


def _sample(preferred, rejected, level="task", attributor="llm"):
    return {
        "preferred_subgoal": preferred,
        "rejected_subgoal": rejected,
        "decision_level": level,
        "attributor": attributor,
    }


class ValidateTrainingSampleTest(unittest.TestCase):
    def test_clean_task_pair_passes(self) -> None:
        self.assertEqual(validate_training_sample(_sample("GET_ONION", "GET_TOMATO")), [])

    def test_clean_coordination_pair_passes(self) -> None:
        self.assertEqual(
            validate_training_sample(
                _sample("YIELD", "CONTINUE_CURRENT_SUBGOAL", level="coordination")
            ),
            [],
        )

    def test_cross_domain_pair_is_rejected(self) -> None:
        problems = validate_training_sample(
            _sample("CONTINUE_CURRENT_SUBGOAL", "GET_TOMATO", level="task")
        )
        self.assertIn("cross_domain_pair", problems)

    def test_unreachable_attribution_level_name_is_rejected(self) -> None:
        """GET_USEFUL_INGREDIENT must be resolved to GET_TOMATO / GET_ONION
        before a pair is formed; a label on it trains a dimension nothing can
        ever select."""
        problems = validate_training_sample(_sample("GET_USEFUL_INGREDIENT", "WAIT"))
        self.assertTrue(any(p.startswith("preferred_unreachable_at_runtime") for p in problems))

    def test_two_task_subgoals_are_not_cross_domain(self) -> None:
        """PUT_DOWN_OBJECT vs GET_TOMATO is an odd pair but a legal one -- both
        are task subgoals.  A reviewer once counted it as cross-domain; it is not."""
        self.assertEqual(validate_training_sample(_sample("PUT_DOWN_OBJECT", "GET_TOMATO")), [])

    def test_missing_attributor_is_rejected(self) -> None:
        self.assertIn(
            "missing_attributor",
            validate_training_sample(_sample("GET_ONION", "GET_TOMATO", attributor=None)),
        )

    def test_identical_sides_are_rejected(self) -> None:
        self.assertIn(
            "preferred_equals_rejected",
            validate_training_sample(_sample("GET_DISH", "GET_DISH")),
        )

    def test_decision_level_must_match_the_pair(self) -> None:
        problems = validate_training_sample(
            _sample("GET_ONION", "GET_TOMATO", level="coordination")
        )
        self.assertTrue(any(p.startswith("decision_level_mismatch") for p in problems))


class FilterTest(unittest.TestCase):
    def test_filter_splits_and_summarises(self) -> None:
        kept, rejected = filter_training_samples(
            [
                _sample("GET_ONION", "GET_TOMATO"),
                _sample("GET_USEFUL_INGREDIENT", "WAIT"),
                _sample("YIELD", "GET_DISH", level="coordination"),
            ]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 2)
        summary = rejection_summary(rejected)
        self.assertEqual(summary.get("preferred_unreachable_at_runtime"), 1)
        self.assertEqual(summary.get("cross_domain_pair"), 1)


if __name__ == "__main__":
    unittest.main()
