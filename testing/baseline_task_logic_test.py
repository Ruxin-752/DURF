"""Regression tests for dependency-free baseline task decisions."""

from __future__ import annotations

import unittest

from durf.baseline.task_logic import next_unstaged_ingredient


class NextCyclePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = ["tomato", "tomato", "onion"]

    def test_empty_staging_starts_with_first_tomato(self):
        self.assertEqual(
            next_unstaged_ingredient(self.recipe, []),
            "tomato",
        )

    def test_staged_tomato_is_not_reselected_as_the_same_unit(self):
        self.assertEqual(
            next_unstaged_ingredient(self.recipe, ["tomato"]),
            "tomato",
        )

    def test_two_staged_tomatoes_advance_to_onion(self):
        self.assertEqual(
            next_unstaged_ingredient(
                self.recipe,
                ["tomato", "tomato"],
            ),
            "onion",
        )

    def test_complete_staging_stops_fetching_ingredients(self):
        self.assertIsNone(
            next_unstaged_ingredient(
                self.recipe,
                ["tomato", "onion", "tomato"],
            )
        )


if __name__ == "__main__":
    unittest.main()
