from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.enumerate_subgoal_contexts import enumerate_contexts  # noqa: E402
from scripts.generate_route2_teacher_corpus import (  # noqa: E402
    COOKING_PREPARATION_SUBGOALS,
    build_teacher_corpus,
    load_generation_contexts,
)
from src.feature_schema import load_features  # noqa: E402
from src.feedback_templates import is_single_sentence, render_templates  # noqa: E402
from src.reward_configurations import (  # noqa: E402
    PREFERENCE_GROUPS,
    PROACTIVITY_FEATURES,
    PROACTIVITY_LABELS,
    PROACTIVITY_LEVELS,
    sample_reward_configurations,
    validate_reward_configurations,
)
from src.subgoal_featurizer import SubgoalContext  # noqa: E402
from src.subgoal_teacher import FeedbackIntent, load_gold_weights  # noqa: E402


class ProactivityRewardConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = load_features()
        cls.configurations = sample_reward_configurations(
            n=36, features=cls.features
        )

    def test_three_preparation_styles_keep_the_53_dimensional_schema(self) -> None:
        validate_reward_configurations(self.configurations)
        self.assertEqual(len(self.features), 53)
        self.assertEqual(
            {config["preparation_style"] for config in self.configurations},
            set(PROACTIVITY_LABELS.values()),
        )
        for config in self.configurations:
            self.assertEqual(set(config["weights"]), set(self.features))

    def test_axis_crosses_zero_only_for_whitelisted_preference_features(self) -> None:
        configurations = sample_reward_configurations(n=243, features=self.features)
        aligned = {}
        for config in configurations:
            if all(config["multipliers"][group] == 1.0 for group in PREFERENCE_GROUPS):
                aligned[config["multipliers"]["proactivity"]] = config
        self.assertEqual(set(aligned), set(PROACTIVITY_LEVELS))

        gold = aligned[1.0]["weights"]
        for feature in PROACTIVITY_FEATURES:
            self.assertAlmostEqual(aligned[-1.0]["weights"][feature], -gold[feature])
            self.assertEqual(aligned[0.0]["weights"][feature], 0.0)
        self.assertLess(aligned[-1.0]["weights"]["blocks_human_path"], 0.0)
        self.assertLess(aligned[0.0]["weights"]["blocks_human_path"], 0.0)

    def test_validator_still_rejects_a_safety_sign_inversion(self) -> None:
        bad = copy.deepcopy(self.configurations[0])
        bad["reward_config_id"] = "bad_safety_sign"
        bad["weights"]["blocks_human_path"] *= -1.0
        with self.assertRaisesRegex(ValueError, "inverts the safety sign"):
            validate_reward_configurations([bad])

    def test_reward_set_contains_every_preparation_style(self) -> None:
        styles = {config["preparation_style"] for config in self.configurations}
        self.assertEqual(styles, set(PROACTIVITY_LABELS.values()))


class CookingPreparationLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            agent_holding=None,
        )

    @staticmethod
    def _intent(feedback_type: str, polarity: float) -> FeedbackIntent:
        return FeedbackIntent(
            feedback_type=feedback_type,
            polarity=polarity,
            subgoal="GET_ONION",
            target_features={"pick_onion": 1.0},
            referenced_features=["pick_onion"],
            role="preparation_test",
            reference_type="feature",
        )

    def test_positive_templates_explicitly_describe_next_round_preparation(self) -> None:
        for feedback_type in ("evaluative", "imperative", "descriptive"):
            texts = render_templates(self._intent(feedback_type, 1.0), self.context)
            self.assertTrue(any("next round" in text.lower() for text in texts))
            self.assertTrue(any("soup cooks" in text.lower() for text in texts))
            self.assertTrue(all(is_single_sentence(text) for text in texts))

    def test_negative_templates_explicitly_describe_early_pickup(self) -> None:
        for feedback_type in ("evaluative", "imperative", "descriptive"):
            texts = render_templates(self._intent(feedback_type, -1.0), self.context)
            self.assertTrue(any("early" in text.lower() for text in texts))
            self.assertTrue(any("soup cooks" in text.lower() for text in texts))
            self.assertTrue(all(is_single_sentence(text) for text in texts))


class ProactivityTeacherCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contexts = enumerate_contexts(load_gold_weights())
        cls.configurations = sample_reward_configurations(n=36)
        cls.corpus, cls.report = build_teacher_corpus(
            contexts=cls.contexts,
            configurations=cls.configurations,
            seed=3,
            gold_split="test",
            include_profile_language=False,
        )
        cls.preparation_rows = [
            row
            for row in cls.corpus
            if row["context"]["pot_status"] == "cooking"
            and row["referenced_subgoal"] in COOKING_PREPARATION_SUBGOALS
        ]

    def test_default_context_source_reenumerates_empty_hand_cooking(self) -> None:
        contexts = load_generation_contexts(None, load_gold_weights())
        cooking = [
            context
            for context in contexts
            if context["context"]["pot_status"] == "cooking"
            and context["context"]["agent_holding"] is None
            and set(context["feasible_subgoals"])
            & COOKING_PREPARATION_SUBGOALS
        ]
        self.assertGreater(len(cooking), 0)

    def test_every_context_is_augmented_across_every_reward_configuration(self) -> None:
        audit = self.report["context_reward_augmentation"]
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["observed_edges"], audit["expected_edges"])

    def test_corpus_has_positive_and_negative_cooking_feedback(self) -> None:
        counts = self.report["counts"]["cooking_preparation_feedback"]
        context_counts = self.report["counts"]["empty_hand_cooking_contexts"]
        self.assertEqual(counts["examples"], len(self.preparation_rows))
        self.assertGreater(context_counts["total"], 0)
        self.assertGreater(counts["positive"], 0)
        self.assertGreater(counts["negative"], 0)

    def test_corpus_exposes_style_and_explicit_preparation_language(self) -> None:
        self.assertTrue(self.preparation_rows)
        self.assertTrue(
            all(row["preparation_style"] in PROACTIVITY_LABELS.values()
                for row in self.preparation_rows)
        )
        self.assertTrue(
            any("next round" in row["local_text"].lower()
                for row in self.preparation_rows
                if row["attributed_sentiment_score"] > 0)
        )
        self.assertTrue(
            any(
                any(
                    phrase in row["local_text"].lower()
                    for phrase in ("early", "too soon", "only when an item is needed")
                )
                for row in self.preparation_rows
                if row["attributed_sentiment_score"] < 0)
        )
        self.assertTrue(
            all(row["route2_trajectory_features"].get("pot_cooking") == 1.0
                for row in self.preparation_rows)
        )


if __name__ == "__main__":
    unittest.main()
