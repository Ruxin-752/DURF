from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baseline_b_pipeline import (  # noqa: E402
    run_pipeline,
)
from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from scripts.evaluate_learning_curve import (  # noqa: E402
    random_baseline_accuracy,
    run_learning_curve,
)
from src.belief_model import GaussianBelief  # noqa: E402
from src.feature_schema import (  # noqa: E402
    collect_action_feature_library,
    empty_weights,
    load_features,
    load_weights,
    read_json,
    validate_feedback_examples,
)
from src.feedback_observations import (  # noqa: E402
    build_feedback_observations,
    valence_with_safety_gate,
)
from src.evaluation_splits import (  # noqa: E402
    make_split_manifest,
    make_train_dev_test_split,
    validate_split_manifest,
)
from src.neural_inference import (  # noqa: E402
    TrajectoryFeedbackRewardPredictor,
    build_vocab,
    collate_batch,
    make_folds,
)
from src.observations import build_observations, reference_vector  # noqa: E402
from src.overcooked_grounding import (  # noqa: E402
    build_grounding_rows,
    ground_feedback,
    predict_grounded_features,
    resolve_opposite_features,
    train_phrase_grounding,
)
from src.probe_evaluator import evaluate_probes, load_probe_states  # noqa: E402
from src.reward_weight_model import BayesianRewardLearner  # noqa: E402
from src.route1_online import OnlineRoute1Learner  # noqa: E402
from src.phrase_reference_classifier import (  # noqa: E402
    fallback_prediction,
    predict_reference_type,
)
from scripts.train_phrase_reference_classifier import (  # noqa: E402
    build_phrase_rows,
    reproduce_original_paper_benchmark,
    train_classifier,
)
from src.sentiment_extractor import modified_vader_observation  # noqa: E402
from src.session_bridge import build_session_feedback_examples  # noqa: E402
from src.subgoal_featurizer import (  # noqa: E402
    LIVE_DECISION_FEATURES,
    SubgoalContext,
    audit_live_feature_coverage,
    featurize_subgoal,
)
from src.subgoal_reranker import (  # noqa: E402
    choose_subgoal,
    evaluate_subgoal_probes,
)
from src.subgoal_planner import (  # noqa: E402
    enumerate_feasible_subgoals,
    plan_subgoal,
    rank_subgoals,
)
from src.human_intent import (  # noqa: E402
    infer_human_intent,
    with_inferred_intent,
)
from src import text_analysis  # noqa: E402
from src.text_analysis import (  # noqa: E402
    limited_punc_tokenization,
    nn_tokenize,
    normalize_reference_vector,
)
from src.subgoal_teacher import (  # noqa: E402
    feedback_intents,
    label_context,
    load_gold_weights,
)
from src.feedback_templates import (  # noqa: E402
    context_description,
    render_templates,
    theme_for_intent,
)
from src.neural_inference import (  # noqa: E402
    predict_reward_vector,
    save_checkpoint,
)
from scripts.enumerate_subgoal_contexts import enumerate_contexts  # noqa: E402
from scripts.generate_synthetic_feedback import (  # noqa: E402
    _parse_json_strings,
    build_corpus,
)
from scripts.generate_route2_teacher_corpus import (  # noqa: E402
    REWARD_CONDITIONING_PHRASES,
    build_teacher_corpus,
)
from src.reward_configurations import (  # noqa: E402
    LEVELS,
    PREFERENCE_GROUPS,
    sample_reward_configurations,
    validate_reward_configurations,
)
from scripts.train_route2 import train as train_route2  # noqa: E402


class FeedbackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probes = load_probe_states()
        cls.feedback = read_json(ROOT / "data" / "feedback_examples.json")

    def test_all_feedback_examples_validate(self) -> None:
        self.assertEqual(len(self.feedback), 100)
        validate_feedback_examples(self.feedback, probe_states=self.probes)

    def test_unknown_feature_is_rejected_with_feedback_id(self) -> None:
        invalid = [copy.deepcopy(self.feedback[0])]
        invalid[0]["trajectory_features"]["not_in_schema"] = 1
        with self.assertRaisesRegex(ValueError, invalid[0]["feedback_id"]):
            validate_feedback_examples(invalid, probe_states=self.probes)

    def test_every_example_grounds_to_nonempty_features(self) -> None:
        action_library = {
            action["action_id"]: action["features"]
            for probe in self.probes
            for action in probe["available_actions"]
        }
        for feedback in self.feedback:
            result = ground_feedback(
                feedback,
                feedback_type=feedback["expected_feedback_type"],
                action_feature_library=action_library,
            )
            self.assertTrue(result["target_features"], feedback["feedback_id"])


class ProbeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probes = load_probe_states()

    def test_action_order_does_not_change_result(self) -> None:
        weights = load_weights()
        original = evaluate_probes(self.probes, weights)
        reversed_probes = copy.deepcopy(self.probes)
        for probe in reversed_probes:
            probe["available_actions"].reverse()
        reversed_result = evaluate_probes(reversed_probes, weights)
        self.assertEqual(
            [item["chosen_action"] for item in original["results"]],
            [item["chosen_action"] for item in reversed_result["results"]],
        )
        self.assertEqual(original["correct"], reversed_result["correct"])

    def test_zero_weight_ties_are_failures(self) -> None:
        evaluation = evaluate_probes(self.probes, empty_weights())
        self.assertEqual(evaluation["correct"], 0)
        self.assertEqual(evaluation["tie_count"], len(self.probes))
        self.assertTrue(all(item["chosen_action"] is None for item in evaluation["results"]))

    def test_leave_one_probe_out_excludes_derived_examples(self) -> None:
        feedback = read_json(ROOT / "data" / "feedback_examples.json")
        result = run_pipeline(
            feedback_examples=feedback,
            probe_states=self.probes,
            initial_weights=load_weights(),
            learning_rate=1.0,
        )
        holdout = result["probe_evaluations"]["leave_one_probe_out"]
        for item in holdout["results"]:
            expected_excluded = sum(
                example.get("probe_id") == item["probe_id"]
                for example in feedback
            )
            self.assertEqual(item["excluded_examples"], expected_excluded)
            self.assertEqual(
                item["training_examples"],
                len(feedback) - expected_excluded,
            )


class ProbeBenchmarkTests(unittest.TestCase):
    """Invariants for the expanded action-probe benchmark."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probes = load_probe_states()
        cls.feedback = read_json(ROOT / "data" / "feedback_examples.json")
        cls.features = set(load_features())

    def test_benchmark_has_expected_size_and_categories(self) -> None:
        self.assertEqual(len(self.probes), 14)
        categories = {probe.get("category") for probe in self.probes}
        for category in ("RecipeCorrectness", "ServingReadiness", "Safety", "Efficiency"):
            self.assertIn(category, categories)

    def test_probe_action_features_are_in_schema(self) -> None:
        for probe in self.probes:
            actions = probe.get("available_actions", [])
            self.assertGreaterEqual(len(actions), 2, probe.get("probe_id"))
            for action in actions:
                unknown = sorted(f for f in action["features"] if f not in self.features)
                self.assertEqual(unknown, [], f"{probe['probe_id']}/{action['action_id']}")

    def test_probe_acceptable_actions_exist(self) -> None:
        for probe in self.probes:
            action_ids = {a["action_id"] for a in probe["available_actions"]}
            acceptable = probe.get("acceptable_actions") or [probe.get("expected_action")]
            for accepted in acceptable:
                self.assertIn(accepted, action_ids, probe.get("probe_id"))

    def test_learned_reward_solves_full_benchmark(self) -> None:
        result = run_pipeline(
            feedback_examples=self.feedback,
            probe_states=self.probes,
            initial_weights=load_weights(),
        )
        learned = result["probe_evaluations"]["learned_weights"]
        self.assertEqual(learned["correct"], learned["total"])
        # Every hard tradeoff probe is still decided strictly (no ties).
        self.assertEqual(learned["tie_count"], 0)


class BayesianLearnerTests(unittest.TestCase):
    """Lock in the faithful reproduction of the paper's Bayesian update."""

    def test_prior_is_zero_mean_var_25(self) -> None:
        belief = GaussianBelief.prior(["a", "b"])
        self.assertEqual(belief.mean_as_dict(), {"a": 0.0, "b": 0.0})
        self.assertEqual(belief.variance_as_dict(), {"a": 25.0, "b": 25.0})

    def test_conjugate_update_shifts_mean_and_shrinks_variance(self) -> None:
        learner = BayesianRewardLearner(["blocks_human_path", "supports_serving"])
        prior_var = learner.belief.variance_as_dict()["blocks_human_path"]

        learner.update({"blocks_human_path": 1}, valence=-1.0)

        posterior = learner.as_dict()
        posterior_var = learner.belief.variance_as_dict()["blocks_human_path"]
        # Negative feedback on a mentioned feature drives its weight negative.
        self.assertLess(posterior["blocks_human_path"], 0.0)
        # An unmentioned feature stays at the prior mean in literal mode.
        self.assertAlmostEqual(posterior["supports_serving"], 0.0)
        # Observing the feature reduces its posterior variance.
        self.assertLess(posterior_var, prior_var)

    def test_update_is_deterministic(self) -> None:
        def learned():
            learner = BayesianRewardLearner(["blocks_human_path", "supports_serving"])
            learner.update({"blocks_human_path": 1}, valence=-1.0)
            learner.update({"supports_serving": 1}, valence=1.0)
            return learner.as_dict()

        self.assertEqual(learned(), learned())

    def test_pragmatic_mode_pushes_unmentioned_features_negative(self) -> None:
        features = load_features()
        literal = BayesianRewardLearner(features)
        pragmatic = BayesianRewardLearner(
            features, pragmatic_valence=-30.0, pragmatic_precision=2.0
        )
        target = {"supports_serving": 1}
        literal.update(target, valence=1.0)
        pragmatic.update(target, valence=1.0)

        # An unmentioned feature is untouched by literal but pushed down by pragmatic.
        self.assertAlmostEqual(literal.as_dict()["blocks_human_path"], 0.0)
        self.assertLess(pragmatic.as_dict()["blocks_human_path"], 0.0)


class OnlineRoute1Tests(unittest.TestCase):
    """Route 1 classification/grounding must be visible in online traces."""

    @classmethod
    def setUpClass(cls) -> None:
        probe = read_json(ROOT / "data" / "subgoal_probe_states.json")[1]
        cls.decision = {
            "chosen_subgoal": "GET_ONION",
            "ranking": [
                {
                    "subgoal": subgoal,
                    "features": featurize_subgoal(probe["context"], subgoal),
                }
                for subgoal in probe["feasible_subgoals"]
            ],
        }

    @staticmethod
    def _form_prediction(
        text: str,
        feedback_type: str,
        *,
        confidence: float = 0.95,
        abstained: bool = False,
    ) -> dict:
        return {
            "feedback_type": feedback_type,
            "label": feedback_type.title(),
            "confidence": confidence,
            "probabilities": {feedback_type: confidence},
            "classifier": "test_feedback_form",
            "confidence_threshold": 0.55,
            "abstained": abstained,
            "text": text,
        }

    def test_inferred_prohibitive_command_updates_named_action_negative(self) -> None:
        learner = OnlineRoute1Learner(
            load_features(),
            mode="route1-literal",
            minimum_reference_confidence=0.45,
        )
        trace = learner.update(
            "Stop taking the onion.",
            decision=self.decision,
            interpretation="inferred",
            feedback_form_prediction=self._form_prediction(
                "Stop taking the onion.", "imperative"
            ),
        )

        self.assertEqual(trace["status"], "updated")
        # The paper's three-way f_G owns credit assignment. The compatible
        # five-way subtype only refines action grounding.
        self.assertEqual(trace["reference_types"], ["action_spatial"])
        self.assertEqual(trace["effective_reference_types"], ["action_spatial"])
        self.assertEqual(trace["target_action"], "GET_ONION")
        self.assertTrue(all(obs["valence"] < 0 for obs in trace["observations"]))
        self.assertLess(learner.weights()["duplicate_human_task"], 0.0)
        duplicate = next(
            change
            for change in trace["top_changes"]
            if change["feature"] == "duplicate_human_task"
        )
        self.assertLess(duplicate["variance_after"], duplicate["variance_before"])

    def test_feedback_form_wins_cross_taxonomy_reference_conflict(self) -> None:
        learner = OnlineRoute1Learner(
            ["pick_onion", "blocks_human_path"],
            mode="route1-literal",
            max_update_kl=100.0,
        )
        conflicting_reference = {
            "reference_type": "feature",
            "confidence": 0.99,
            "probabilities": {"feature": 0.99},
            "classifier": "test_reference",
            "abstained": False,
            "top2_margin": 0.98,
        }
        with mock.patch(
            "src.feedback_observations.predict_reference_type",
            return_value=conflicting_reference,
        ):
            trace = learner.update(
                "Great job.",
                trajectory_features={"pick_onion": 1.0},
                feedback_form_prediction=self._form_prediction(
                    "Great job.", "evaluative"
                ),
            )

        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["reference_types"], ["feature"])
        self.assertEqual(trace["effective_reference_types"], ["trajectory"])
        self.assertTrue(trace["observations"][0]["reference_conflict"])
        self.assertEqual(
            trace["observations"][0]["grounding_source"], "trajectory_features"
        )
        self.assertGreater(learner.weights()["pick_onion"], 0.0)
        self.assertAlmostEqual(learner.weights()["blocks_human_path"], 0.0)

    def test_low_confidence_feedback_form_rejects_before_weight_update(self) -> None:
        learner = OnlineRoute1Learner(["pick_onion"], max_update_kl=100.0)
        reference = {
            "reference_type": "trajectory",
            "confidence": 0.99,
            "probabilities": {"trajectory": 0.99},
            "classifier": "test_reference",
            "abstained": False,
            "top2_margin": 0.98,
        }
        with mock.patch(
            "src.feedback_observations.predict_reference_type",
            return_value=reference,
        ):
            trace = learner.update(
                "That was useful.",
                trajectory_features={"pick_onion": 1.0},
                feedback_form_prediction=self._form_prediction(
                    "That was useful.",
                    "evaluative",
                    confidence=0.4,
                    abstained=True,
                ),
            )

        self.assertEqual(trace["status"], "rejected_low_confidence")
        self.assertIn("feedback_form", trace["trace_state"])
        self.assertAlmostEqual(learner.weights()["pick_onion"], 0.0)

    def test_mixed_feedback_uses_each_phrase_form_for_credit_assignment(self) -> None:
        learner = OnlineRoute1Learner(
            ["pick_onion", "pick_dish"],
            mode="route1-literal",
            max_update_kl=100.0,
        )
        decision = {
            "chosen_subgoal": "GET_ONION",
            "ranking": [
                {"subgoal": "GET_ONION", "features": {"pick_onion": 1.0}},
                {"subgoal": "GET_DISH", "features": {"pick_dish": 1.0}},
            ],
        }
        mixed_prediction = {
            "feedback_type": "mixed",
            "label": "Mixed",
            "phrases": [
                self._form_prediction("Great job", "evaluative"),
                self._form_prediction("Please take a dish", "imperative"),
            ],
        }

        def reference_for_phrase(phrase: str) -> dict:
            label = "trajectory" if phrase == "Great job" else "action_spatial"
            return {
                "reference_type": label,
                "confidence": 0.95,
                "probabilities": {label: 0.95},
                "classifier": "test_reference",
                "abstained": False,
                "top2_margin": 0.9,
            }

        with mock.patch(
            "src.feedback_observations.predict_reference_type",
            side_effect=reference_for_phrase,
        ):
            trace = learner.update(
                "Great job. Please take a dish.",
                decision=decision,
                trajectory_features={"pick_onion": 1.0},
                feedback_form_prediction=mixed_prediction,
            )

        self.assertEqual(trace["status"], "updated")
        self.assertEqual(trace["feedback_type"], "mixed")
        self.assertEqual(trace["feedback_types"], ["evaluative", "imperative"])
        self.assertEqual(
            [row["grounding_source"] for row in trace["observations"]],
            ["trajectory_features", "target_action:GET_DISH"],
        )
        self.assertGreater(learner.weights()["pick_onion"], 0.0)
        self.assertGreater(learner.weights()["pick_dish"], 0.0)

    def test_oracle_trace_uses_annotations(self) -> None:
        learner = OnlineRoute1Learner(load_features(), mode="route1-pseudopragmatic")
        trace = learner.update(
            "That gets in my way.",
            decision=self.decision,
            oracle_feedback={
                "expected_feedback_type": "descriptive",
                "target_features": {"blocks_human_path": 1},
                "attributed_sentiment_score": -1.0,
            },
            interpretation="oracle",
        )

        self.assertEqual(trace["feedback_type"], "descriptive")
        self.assertEqual(trace["observations"][0]["grounding_source"], "target_features")
        self.assertLess(learner.weights()["blocks_human_path"], 0.0)
        self.assertLess(learner.weights()["supports_serving"], 0.0)

    def test_rejects_other_reference_without_mutating_posterior(self) -> None:
        learner = OnlineRoute1Learner(load_features())
        before = learner.weights()
        trace = learner.update("Can you hear me?", decision=self.decision)
        self.assertEqual(trace["status"], "rejected_low_confidence")
        self.assertIn("grounding", trace["trace_state"])
        self.assertEqual(learner.weights(), before)

    def test_semantic_duplicate_window_rejects_paraphrase(self) -> None:
        learner = OnlineRoute1Learner(
            load_features(),
            semantic_dedup_threshold=0.6,
            minimum_reference_confidence=0.45,
        )
        first = learner.update(
            "Stop taking the onion.",
            decision=self.decision,
            feedback_form_prediction=self._form_prediction(
                "Stop taking the onion.", "imperative"
            ),
        )
        second = learner.update("Please stop taking onion!", decision=self.decision)
        self.assertEqual(first["status"], "updated")
        self.assertEqual(second["status"], "rejected_duplicate")

    def test_single_update_delta_is_capped(self) -> None:
        learner = OnlineRoute1Learner(
            ["blocks_human_path"], max_abs_delta=0.2, max_update_kl=100.0
        )
        trace = learner.update(
            "That blocked me.",
            oracle_feedback={
                "expected_feedback_type": "evaluative",
                "target_features": {"blocks_human_path": 1},
                "attributed_sentiment_score": -1.0,
            },
            interpretation="oracle",
        )
        self.assertEqual(trace["status"], "updated")
        self.assertTrue(trace["update_constraint"]["capped"])
        self.assertLessEqual(abs(learner.weights()["blocks_human_path"]), 0.2000001)

    def test_low_confidence_disables_pseudopragmatic_observation(self) -> None:
        learner = OnlineRoute1Learner(
            ["pick_onion", "unmentioned"],
            mode="route1-pseudopragmatic",
            max_update_kl=100.0,
            minimum_reference_confidence=0.45,
        )
        trace = learner.update(
            "Stop taking the onion.",
            decision={
                "chosen_subgoal": "GET_ONION",
                "ranking": [
                    {"subgoal": "GET_ONION", "features": {"pick_onion": 1.0}}
                ],
            },
            feedback_form_prediction=self._form_prediction(
                "Stop taking the onion.", "imperative"
            ),
        )
        self.assertEqual(trace["status"], "updated")
        self.assertAlmostEqual(learner.weights()["unmentioned"], 0.0)


class SubgoalCompatTests(unittest.TestCase):
    """The learned reward must transfer to H0's subgoals via the featurizer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.subgoal_probes = read_json(ROOT / "data" / "subgoal_probe_states.json")

    def test_featurizer_flags_serving_conflict_and_yield(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            human_intent="pick_dish_then_serve",
        )
        grab_dish = featurize_subgoal(context, "GET_DISH")
        self.assertIn("duplicate_human_task", grab_dish)
        self.assertNotIn("blocks_serving_route", grab_dish)

        wait = featurize_subgoal(context, "WAIT")
        self.assertIn("respects_human_intent", wait)
        self.assertNotIn("blocks_serving_route", wait)
        self.assertNotIn("clears_human_path", wait)

    def test_same_ingredient_is_not_duplicate_when_multiple_units_remain(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["onion"],
            pot_status="partial",
            human_holding="tomato",
            human_intent="get_tomato",
            human_committed_units=1,
        )
        tomato = featurize_subgoal(context, "GET_TOMATO")
        self.assertNotIn("duplicate_human_task", tomato)
        self.assertNotIn("steals_human_target", tomato)
        self.assertIn("complementary_to_human", tomato)

    def test_serving_chain_roles_are_not_automatically_duplicate(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            human_holding="dish",
            human_intent="pick_dish_then_serve",
            human_committed_units=1,
        )
        pickup = featurize_subgoal(context, "PICKUP_SOUP")
        self.assertNotIn("duplicate_human_task", pickup)
        self.assertIn("complementary_to_human", pickup)

    def test_two_dishes_compete_for_the_same_ready_soup(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            agent_holding="dish",
            human_holding="dish",
            human_intent="pick_dish_then_serve",
            human_committed_units=1,
            candidate_target_overlaps_human={"PICKUP_SOUP": True},
        )
        pickup = featurize_subgoal(context, "PICKUP_SOUP")
        self.assertIn("duplicate_human_task", pickup)
        self.assertIn("crowds_human_target", pickup)
        self.assertNotIn("complementary_to_human", pickup)
        self.assertNotIn("avoids_duplicate_human_task", pickup)

    def test_two_held_soups_are_independent_serving_work(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=[],
            pot_status="empty",
            agent_holding="soup",
            human_holding="soup",
            human_intent="serve_soup",
            human_committed_units=1,
        )
        serve = featurize_subgoal(context, "SERVE_SOUP")
        self.assertNotIn("duplicate_human_task", serve)
        self.assertIn("complementary_to_human", serve)

    def test_native_and_state_facts_context_agree(self) -> None:
        native = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato"],
            "pot_status": "ready",
            "human_intent": "pick_dish_then_serve",
        }
        state_facts = {
            "current_orders": [["tomato", "tomato", "onion"]],
            "pot_state": {"ingredients": ["tomato", "tomato"], "status": "ready"},
            "human_intent": "pick_dish_then_serve",
        }
        self.assertEqual(
            featurize_subgoal(native, "GET_DISH"),
            featurize_subgoal(state_facts, "GET_DISH"),
        )

    def test_learned_prior_weights_solve_subgoal_probes(self) -> None:
        evaluation = evaluate_subgoal_probes(self.subgoal_probes, load_weights())
        self.assertEqual(evaluation["correct"], evaluation["total"])
        self.assertEqual(evaluation["tie_count"], 0)

    def test_zero_weights_cannot_choose_subgoals(self) -> None:
        evaluation = evaluate_subgoal_probes(self.subgoal_probes, empty_weights())
        self.assertEqual(evaluation["correct"], 0)
        self.assertEqual(evaluation["tie_count"], len(self.subgoal_probes))

    def test_choose_subgoal_requires_feasible_options(self) -> None:
        with self.assertRaises(ValueError):
            choose_subgoal(load_weights(), {"human_intent": "get_onion"}, [])

    def test_all_53_features_are_decision_relevant_or_explicitly_masked(self) -> None:
        candidate_sets = []
        for pot in ([], ["tomato"], ["tomato", "tomato"], ["tomato", "tomato", "onion"]):
            for status in ("empty", "partial", "cooking", "ready"):
                for held in (None, "tomato", "onion", "dish", "soup"):
                    for intent in (None, "get tomato", "get onion", "pick dish", "serve soup"):
                        context = SubgoalContext(
                            pot_ingredients=list(pot),
                            pot_status=status,
                            agent_holding=held,
                            human_intent=intent,
                            candidate_path_effects={
                                "GET_TOMATO": "clears",
                                "PUT_TOMATO_IN_POT": "blocks",
                                "GET_ONION": "blocks",
                                "PUT_ONION_IN_POT": "enters",
                                "GET_DISH": "enters",
                                "PICKUP_SOUP": "clears",
                                "SERVE_SOUP": "enters",
                                "YIELD_PATH": "clears",
                                "WAIT": "neutral",
                            },
                        )
                        feasible = enumerate_feasible_subgoals(context)
                        feasible.insert(-1, "YIELD_PATH")
                        candidate_sets.append(
                            [featurize_subgoal(context, subgoal) for subgoal in feasible]
                        )
        audit = audit_live_feature_coverage(
            load_features(), candidate_sets=candidate_sets
        )
        self.assertEqual(audit["schema_size"], 53)
        self.assertEqual(audit["decision_feature_count"], 33)
        self.assertEqual(len(audit["decision_null_features"]), 20)
        self.assertEqual(audit["missing_declared_decision_features"], [])
        self.assertEqual(audit["undeclared_varying_features"], [])
        self.assertTrue(audit["complete"])


class SessionBridgeTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_attributed_window_becomes_provenance_rich_example(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session_001"
            session.mkdir()
            timestamp = "2026-07-14T10:00:00+00:00"
            self._write_jsonl(
                session / "trajectory.jsonl",
                [
                    {
                        "total_step": 5,
                        "layout": "ring_tomato_onion_10x6",
                        "ai_action_name": "interact",
                        "human_action_name": "east",
                        "environment_reward": 0,
                        "state_facts": {
                            "ai_pos": [2, 1],
                            "human_pos": [1, 1],
                            "ai_held_object": {"name": "tomato"},
                            "pot_states": {},
                        },
                        "extra": {
                            "state_before": {
                                "ai_pos": [2, 1],
                                "human_pos": [1, 1],
                                "ai_held_object": None,
                            }
                        },
                    }
                ],
            )
            self._write_jsonl(
                session / "feedback_events.jsonl",
                [
                    {
                        "source": "chat_messages.csv",
                        "timestamp_utc": timestamp,
                        "episode": 0,
                        "total_step": 5,
                        "feedback_text": "You blocked me.",
                        "role": "human_language",
                        "extra": {"layout": "ring_tomato_onion_10x6"},
                    }
                ],
            )
            self._write_jsonl(
                session / "attribution_preview.jsonl",
                [
                    {
                        "feedback_event_id": f"chat_messages.csv:5:{timestamp}",
                        "feedback_type": "evaluative",
                        "target_time_window": [5, 5],
                        "target_event": "AI_blocked_human_path",
                        "polarity": "negative",
                        "confidence": 0.9,
                        "needs_clarification": False,
                    }
                ],
            )

            examples = build_session_feedback_examples(session)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["session_id"], "session_001")
        self.assertEqual(examples[0]["attributed_sentiment_score"], -1.0)
        self.assertEqual(examples[0]["trajectory_features"]["pick_tomato"], 1.0)
        self.assertEqual(examples[0]["trajectory_features"]["blocks_human_path"], 2.0)
        validate_feedback_examples(examples, probe_states=load_probe_states())


class VaderSentimentTests(unittest.TestCase):
    """VADER valence port of the paper's modified_vader_observation."""

    def test_positive_and_negative_signs(self) -> None:
        self.assertGreater(modified_vader_observation("This step is good."), 0)
        self.assertLess(modified_vader_observation("This step is bad."), 0)

    def test_zero_markers_return_zero(self) -> None:
        self.assertEqual(modified_vader_observation("this is worthless"), 0.0)
        self.assertEqual(modified_vader_observation("that scores zero"), 0.0)

    def test_neutral_defaults_to_positive_implicature(self) -> None:
        # A phrase VADER cannot score falls back to the +0.5 default.
        self.assertEqual(modified_vader_observation("go to the pot tile"), 0.5)

    def test_hard_negative_gate_blocks_positive_default(self) -> None:
        score, confidence, source = valence_with_safety_gate(
            "That was not helpful at all.", 0.5
        )
        self.assertLess(score, 0)
        self.assertGreaterEqual(confidence, 0.8)
        self.assertEqual(source, "hard_negative")

    def test_all_feedback_texts_are_ascii_english(self) -> None:
        feedback = read_json(ROOT / "data" / "feedback_examples.json")
        for example in feedback:
            text = example["text"]
            self.assertTrue(
                all(ord(char) < 128 for char in text),
                msg=f"non-ASCII text in {example['feedback_id']}: {text}",
            )


class TokenizationTests(unittest.TestCase):
    def test_limited_punc_tokenization_splits_and_strips(self) -> None:
        phrases = limited_punc_tokenization("Do not block me, move aside please.")
        self.assertEqual(phrases, ["Do not block me", "move aside please"])

    def test_nn_tokenize_lowercases_and_never_empty(self) -> None:
        self.assertEqual(nn_tokenize("!!!"), [""])
        tokens = nn_tokenize("Move ASIDE, please.")
        self.assertIn("move", tokens)
        self.assertIn("aside", tokens)

    def test_nn_tokenize_falls_back_without_nltk_runtime(self) -> None:
        with mock.patch.object(
            text_analysis,
            "_word_tokenize",
            return_value=["plates", "tomatoes", "onions"],
        ), mock.patch.object(text_analysis, "_lemmatizer", side_effect=ImportError):
            self.assertEqual(
                nn_tokenize("ignored"),
                ["plate", "tomato", "onion"],
            )
    def test_reference_vector_normalized_to_sum_one(self) -> None:
        features = ["a", "b", "c"]
        vector = reference_vector({"a": 1, "b": 1}, features, normalize=True)
        self.assertAlmostEqual(float(vector.sum()), 1.0)
        self.assertAlmostEqual(float(vector[0]), 0.5)

    def test_normalize_zero_vector_is_unchanged(self) -> None:
        import numpy as np

        zero = np.zeros(3)
        self.assertTrue((normalize_reference_vector(zero) == zero).all())

    def test_pragmatic_observation_is_normalized(self) -> None:
        features = ["a", "b", "c", "d"]
        observations = build_observations(
            {"a": 1}, 1.0, features, pragmatic_valence=-30.0
        )
        pragmatic = [obs for obs in observations if obs.kind == "pragmatic"][0]
        self.assertAlmostEqual(float(pragmatic.reference_vector.sum()), 1.0)


class BeliefUpdateFormTests(unittest.TestCase):
    """The learner must use the paper's active multiply() update form."""

    def test_multiply_matches_gaussian_factor_product(self) -> None:
        import numpy as np

        features = ["a", "b"]
        belief = GaussianBelief.prior(features)
        ref = np.array([1.0, 0.0])
        valence, precision = 30.0, 2.0

        updated = belief.multiply_observation(ref, valence, precision)

        # Independent recomputation of N(mean=r*v, precision=p*rr^T) product.
        prior_precision = belief.precision
        obs_precision = np.outer(ref, ref) * precision
        expected_cov = np.linalg.inv(prior_precision + obs_precision)
        expected_mean = expected_cov @ (
            prior_precision @ belief.mean + obs_precision @ (ref * valence)
        )
        self.assertTrue(np.allclose(updated.mean, expected_mean))

    def test_multiply_and_blr_forms_differ(self) -> None:
        import numpy as np

        features = ["a", "b"]
        belief = GaussianBelief.prior(features)
        ref = np.array([0.5, 0.5])  # normalized reference: r.r = 0.5 != 1
        multiply = belief.multiply_observation(ref, 30.0, 2.0)
        blr = belief.update_from_observation(ref, 30.0, 2.0)
        self.assertFalse(np.allclose(multiply.mean, blr.mean))


class LearningCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probes = load_probe_states()
        cls.feedback = read_json(ROOT / "data" / "feedback_examples.json")
        cls.features = load_features()

    def test_curve_improves_and_beats_random(self) -> None:
        result = run_learning_curve(
            feedback_examples=self.feedback,
            probes=self.probes,
            features=self.features,
            mode="literal",
            seeds=[0, 1, 2],
            n_samples=50,
            # Fifteen randomly sampled comments is too noisy across the three
            # fixed seeds; thirty still tests sample efficiency while covering
            # enough reward directions for a stable regression assertion.
            max_steps=30,
        )
        mean = result["deterministic_accuracy"]["mean"]
        random_baseline = random_baseline_accuracy(self.probes)
        # Online updates improve over the prior and clearly beat the random baseline.
        self.assertLessEqual(mean[0], mean[-1])
        self.assertGreater(mean[-1], random_baseline)
        self.assertGreater(mean[-1], 0.75)


class Route2ScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probes = load_probe_states()
        cls.feedback = read_json(ROOT / "data" / "feedback_examples.json")
        cls.features = load_features()
        cls.dataset = build_dataset(
            cls.feedback, cls.probes, cls.features, min_freq=1, n_folds=5, seed=0
        )

    def test_dataset_triples_have_correct_dimensions(self) -> None:
        n_features = self.dataset["n_features"]
        self.assertEqual(n_features, len(self.features))
        self.assertEqual(len(self.dataset["examples"]), len(self.feedback))
        for example in self.dataset["examples"]:
            self.assertEqual(len(example["feature_counts"]), n_features)
            self.assertEqual(len(example["target_reward"]), n_features)
            self.assertGreaterEqual(len(example["tokens"]), 1)

    def test_folds_hold_out_whole_probe_groups(self) -> None:
        for fold in self.dataset["folds"]:
            train = set(fold["train_indices"])
            test = set(fold["test_indices"])
            self.assertEqual(train & test, set())
            self.assertEqual(len(train) + len(test), len(self.dataset["examples"]))

    def test_model_forward_output_dim_is_n_features(self) -> None:
        import torch

        model = TrajectoryFeedbackRewardPredictor(
            vocab_size=self.dataset["vocab_size"], n_features=self.dataset["n_features"]
        )
        batch = collate_batch(self.dataset["examples"][:4], self.dataset["vocab"])
        with torch.no_grad():
            predictions = model(batch["tokens"], batch["offsets"], batch["feature_counts"])
        self.assertEqual(tuple(predictions.shape), (4, self.dataset["n_features"]))

    def test_vocab_reserves_pad_and_unk(self) -> None:
        vocab = build_vocab([["move", "aside"], ["move"]], min_freq=1)
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)

    def test_make_folds_holds_out_groups(self) -> None:
        folds = make_folds(["g1", "g1", "g2", None], n_folds=2, seed=0)
        for fold in folds:
            for index in fold["test_indices"]:
                self.assertIsNotNone(["g1", "g1", "g2", None][index])


class SubgoalPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subgoal_probes = read_json(ROOT / "data" / "subgoal_probe_states.json")
        cls.features = load_features()

    def test_enumeration_is_superset_of_probe_feasible_sets(self) -> None:
        # The auto-enumerated candidate set must cover every hand-authored
        # feasible subgoal, so the live loop never drops an option the probes
        # relied on.
        for probe in self.subgoal_probes:
            enumerated = set(enumerate_feasible_subgoals(probe["context"]))
            hand_authored = set(probe["feasible_subgoals"])
            self.assertTrue(
                hand_authored.issubset(enumerated),
                msg=f"{probe['probe_id']}: {hand_authored} !<= {enumerated}",
            )

    def test_enumeration_always_offers_wait_and_valid_names(self) -> None:
        for probe in self.subgoal_probes:
            candidates = enumerate_feasible_subgoals(probe["context"])
            self.assertIn("WAIT", candidates)
            self.assertEqual(len(candidates), len(set(candidates)))

    def test_holding_ingredient_enumerates_only_potting(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato"],
            "pot_status": "partial",
            "agent_holding": "onion",
        }
        candidates = enumerate_feasible_subgoals(context)
        self.assertIn("PUT_ONION_IN_POT", candidates)
        self.assertIn("WAIT", candidates)
        self.assertNotIn("GET_ONION", candidates)

    def test_duplicate_held_ingredient_offers_stash_without_forcing_it(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato"],
            pot_status="partial",
            agent_holding="onion",
            human_holding="onion",
            human_intent="get_onion",
            human_committed_units=1,
        )

        candidates = enumerate_feasible_subgoals(context)

        self.assertEqual(
            candidates,
            ["PUT_ONION_IN_POT", "STASH_HELD_OBJECT", "WAIT"],
        )
        put_features = featurize_subgoal(context, "PUT_ONION_IN_POT")
        stash_features = featurize_subgoal(context, "STASH_HELD_OBJECT")
        self.assertIn("duplicate_human_task", put_features)
        self.assertIn("avoids_duplicate_human_task", stash_features)
        self.assertIn("respects_human_intent", stash_features)
        self.assertNotIn("complementary_to_human", stash_features)

        weights = empty_weights(self.features)
        weights["duplicate_human_task"] = -2.0
        weights["avoids_duplicate_human_task"] = 1.0
        decision = plan_subgoal(weights, context, tie_fallback="PUT_ONION_IN_POT")
        self.assertEqual(decision["chosen_subgoal"], "STASH_HELD_OBJECT")
        self.assertEqual(decision["score_formula"], "w_dot_phi")

    def test_unusable_held_ingredient_offers_stash_instead_of_wait_only(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            agent_holding="onion",
        )
        self.assertEqual(
            enumerate_feasible_subgoals(context),
            ["STASH_HELD_OBJECT", "WAIT"],
        )
        stash = featurize_subgoal(context, "STASH_HELD_OBJECT")
        self.assertEqual(stash.get("time_cost"), 1.0)
        self.assertNotIn("supports_serving", stash)
        wait = featurize_subgoal(context, "WAIT")
        self.assertEqual(wait.get("time_cost"), 1.0)
        self.assertEqual(wait.get("delays_serving"), 1.0)

    def test_stashing_unusable_object_supports_a_ready_soup(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            agent_holding="onion",
        )
        candidates = enumerate_feasible_subgoals(context)
        self.assertEqual(candidates, ["STASH_HELD_OBJECT", "WAIT"])
        stash = featurize_subgoal(context, "STASH_HELD_OBJECT")
        self.assertEqual(stash.get("supports_serving"), 1.0)
        self.assertNotIn("time_cost", stash)

    def test_ready_and_partial_pots_offer_both_serving_and_filling(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            pot_snapshots=[
                {
                    "position": (1, 0),
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "ready",
                },
                {
                    "position": (4, 0),
                    "ingredients": ["onion"],
                    "status": "partial",
                },
            ],
        )
        self.assertEqual(
            set(enumerate_feasible_subgoals(context)),
            {"GET_DISH", "GET_TOMATO", "WAIT"},
        )
        tomato = featurize_subgoal(context, "GET_TOMATO")
        self.assertIn("adds_needed_tomato", tomato)
        self.assertIn("matches_current_order", tomato)
        for invalid in ("adds_extra_tomato", "wrong_ingredient", "breaks_recipe"):
            self.assertNotIn(invalid, tomato)

    def test_cooking_and_partial_pots_do_not_hide_open_recipe_work(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            pot_snapshots=[
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "cooking",
                },
                {
                    "ingredients": ["tomato", "onion"],
                    "status": "partial",
                },
            ],
        )
        self.assertEqual(
            set(enumerate_feasible_subgoals(context)),
            {"GET_TOMATO", "WAIT"},
        )
        wait = featurize_subgoal(context, "WAIT")
        self.assertIn("time_cost", wait)
        self.assertIn("delays_serving", wait)

    def test_held_ingredient_can_fill_partial_beside_ready_pot(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            agent_holding="tomato",
            pot_snapshots=[
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "ready",
                },
                {"ingredients": ["onion"], "status": "partial"},
            ],
        )
        self.assertEqual(
            set(enumerate_feasible_subgoals(context)),
            {"PUT_TOMATO_IN_POT", "WAIT"},
        )

    def test_single_plural_snapshot_matches_legacy_context(self) -> None:
        legacy = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["onion"],
            pot_status="partial",
            human_holding="tomato",
            human_intent="get_tomato",
            human_committed_units=1,
        )
        plural = SubgoalContext(
            recipe=list(legacy.recipe),
            pot_ingredients=list(legacy.pot_ingredients),
            pot_status=legacy.pot_status,
            human_holding=legacy.human_holding,
            human_intent=legacy.human_intent,
            human_committed_units=legacy.human_committed_units,
            pot_snapshots=[
                {"ingredients": ["onion"], "status": "partial"}
            ],
        )
        self.assertEqual(
            enumerate_feasible_subgoals(legacy),
            enumerate_feasible_subgoals(plural),
        )
        for subgoal in enumerate_feasible_subgoals(legacy):
            self.assertEqual(
                featurize_subgoal(legacy, subgoal),
                featurize_subgoal(plural, subgoal),
            )

    def test_multiple_pots_make_same_ingredient_work_complementary(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["onion"],
            pot_status="partial",
            human_holding="tomato",
            human_intent="get_tomato",
            human_committed_units=1,
            pot_snapshots=[
                {"ingredients": ["onion"], "status": "partial"},
                {"ingredients": ["tomato", "onion"], "status": "partial"},
            ],
        )
        tomato = featurize_subgoal(context, "GET_TOMATO")
        self.assertNotIn("duplicate_human_task", tomato)
        self.assertNotIn("steals_human_target", tomato)
        self.assertIn("complementary_to_human", tomato)

    def test_two_ready_pots_leave_one_dish_task_for_each_agent(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            agent_holding="dish",
            human_holding="dish",
            human_intent="pick_dish_then_serve",
            human_committed_units=1,
            candidate_target_overlaps_human={"PICKUP_SOUP": True},
            pot_snapshots=[
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "ready",
                },
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "ready",
                },
            ],
        )
        pickup = featurize_subgoal(context, "PICKUP_SOUP")
        self.assertNotIn("duplicate_human_task", pickup)
        self.assertIn("complementary_to_human", pickup)

    def test_empty_pot_enumerates_missing_ingredient_pickups(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": [],
            "pot_status": "empty",
            "agent_holding": None,
        }
        candidates = enumerate_feasible_subgoals(context)
        self.assertIn("GET_TOMATO", candidates)
        self.assertIn("GET_ONION", candidates)

    def test_staged_prefetch_inventory_prevents_immediate_repickup(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="cooking",
            staged_inventory={"tomato": 1, "onion": 1, "dish": 1},
        )

        candidates = enumerate_feasible_subgoals(context)

        # One more tomato is a genuine next-order deficit. The onion and dish
        # already on counters are completed preparation, not new pickup work.
        self.assertEqual(candidates, ["GET_TOMATO", "WAIT"])
        context.staged_inventory["tomato"] = 2
        self.assertEqual(enumerate_feasible_subgoals(context), ["WAIT"])

    def test_staged_ingredient_is_pickable_again_when_a_pot_opens(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato"],
            pot_status="partial",
            staged_inventory={"onion": 1},
        )

        self.assertEqual(
            enumerate_feasible_subgoals(context),
            ["GET_ONION", "WAIT"],
        )

    def test_human_held_unit_prevents_staged_open_pot_repickup(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "onion"],
            pot_status="partial",
            human_holding="tomato",
            human_committed_units=1,
            staged_inventory={"tomato": 1},
        )

        feasible = enumerate_feasible_subgoals(context)
        self.assertNotIn("GET_TOMATO", feasible)
        self.assertEqual(feasible, ["GET_DISH", "WAIT"])

        context.staged_inventory["dish"] = 1
        self.assertEqual(enumerate_feasible_subgoals(context), ["WAIT"])

        two_units_missing = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["onion"],
            pot_status="partial",
            human_holding="tomato",
            human_committed_units=1,
            staged_inventory={"tomato": 1},
        )
        self.assertIn(
            "GET_TOMATO", enumerate_feasible_subgoals(two_units_missing)
        )

    def test_dish_prefetch_is_bounded_by_all_active_soups(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            pot_snapshots=[
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "ready",
                },
                {
                    "ingredients": ["tomato", "tomato", "onion"],
                    "status": "cooking",
                },
            ],
            staged_inventory={"dish": 1},
        )

        self.assertIn("GET_DISH", enumerate_feasible_subgoals(context))
        context.staged_inventory["dish"] = 2
        self.assertNotIn("GET_DISH", enumerate_feasible_subgoals(context))

    def test_ready_soup_does_not_duplicate_a_human_held_dish(self) -> None:
        context = SubgoalContext(
            recipe=["tomato", "tomato", "onion"],
            pot_ingredients=["tomato", "tomato", "onion"],
            pot_status="ready",
            human_holding="dish",
            human_intent="pick_dish_then_serve",
            human_committed_units=1,
        )

        self.assertEqual(enumerate_feasible_subgoals(context), ["WAIT"])

    def test_cooking_state_offers_neutral_preparation_candidates(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
            "human_holding": "dish",
            "human_intent": "pick_dish_then_serve",
        }
        candidates = enumerate_feasible_subgoals(context)
        self.assertEqual(
            set(candidates),
            {"GET_TOMATO", "GET_ONION", "WAIT"},
        )
        wait_features = featurize_subgoal(context, "WAIT")
        self.assertNotIn("time_cost", wait_features)
        prefetch = featurize_subgoal(context, "GET_ONION")
        self.assertEqual(prefetch["pot_cooking"], 1.0)
        self.assertEqual(prefetch["ingredient_onion"], 1.0)
        self.assertEqual(prefetch["pick_onion"], 1.0)
        self.assertEqual(prefetch["moves_toward_needed_object"], 1.0)
        for hard_coded_judgment in (
            "adds_needed_onion",
            "recipe_needs_onion",
            "matches_current_order",
            "wrong_ingredient",
            "breaks_recipe",
        ):
            self.assertNotIn(hard_coded_judgment, prefetch)

    def test_same_cooking_state_is_selected_only_by_reward_weights(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
        }
        feasible = ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"]
        cases = (
            (
                {"pick_dish": -1.0, "pick_tomato": -1.0, "pick_onion": -1.0},
                "WAIT",
            ),
            (
                {"pick_dish": 2.0, "pick_tomato": -1.0, "pick_onion": -1.0},
                "GET_DISH",
            ),
            (
                {"pick_dish": -1.0, "pick_tomato": -1.0, "pick_onion": 2.0},
                "GET_ONION",
            ),
        )
        for changes, expected in cases:
            weights = empty_weights(self.features)
            weights.update(changes)
            decision = plan_subgoal(
                weights,
                context,
                feasible_subgoals=feasible,
                tie_fallback="WAIT",
            )
            self.assertEqual(decision["chosen_subgoal"], expected)
            self.assertEqual(decision["score_formula"], "w_dot_phi")
            for row in decision["ranking"]:
                expected_score = sum(
                    weights.get(feature, 0.0) * value
                    for feature, value in row["features"].items()
                )
                self.assertAlmostEqual(row["total_score"], expected_score)

    def test_zero_reward_tie_uses_h0_fallback_not_schema_order(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "cooking",
            "agent_holding": None,
        }
        ordered = ["GET_DISH", "GET_TOMATO", "GET_ONION", "WAIT"]
        for feasible in (ordered, list(reversed(ordered))):
            decision = plan_subgoal(
                empty_weights(self.features),
                context,
                feasible_subgoals=feasible,
                tie_fallback="WAIT",
            )
            self.assertEqual(decision["chosen_subgoal"], "WAIT")
            self.assertTrue(decision["is_tie"])
            self.assertEqual(decision["decision_source"], "h0_tie_fallback")

    def test_manual_comfort_rescaling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires lambda_pref=1.0"):
            plan_subgoal(
                empty_weights(self.features),
                self.subgoal_probes[0]["context"],
                lambda_pref=0.75,
            )
    def test_ready_soup_empty_handed_offers_dish(self) -> None:
        context = {
            "recipe": ["tomato", "tomato", "onion"],
            "pot_ingredients": ["tomato", "tomato", "onion"],
            "pot_status": "ready",
            "agent_holding": None,
        }
        candidates = enumerate_feasible_subgoals(context)
        self.assertEqual(set(candidates), {"GET_DISH", "WAIT"})

    def test_plan_subgoal_matches_choose_on_enumerated_set(self) -> None:
        weights = empty_weights(self.features)
        for probe in self.subgoal_probes:
            plan = plan_subgoal(weights, probe["context"])
            self.assertIn(plan["chosen_subgoal"], plan["feasible_subgoals"])
            self.assertEqual(
                plan["feasible_subgoals"],
                enumerate_feasible_subgoals(probe["context"]),
            )

    def test_plan_subgoal_rejects_unknown_override(self) -> None:
        weights = empty_weights(self.features)
        with self.assertRaises(ValueError):
            plan_subgoal(
                weights,
                self.subgoal_probes[0]["context"],
                feasible_subgoals=["NOT_A_SUBGOAL"],
            )

    def test_rank_subgoals_returns_full_enumerated_ranking(self) -> None:
        weights = empty_weights(self.features)
        probe = self.subgoal_probes[2]
        ranking = rank_subgoals(weights, probe["context"])
        ranked = {item["subgoal"] for item in ranking}
        self.assertEqual(ranked, set(enumerate_feasible_subgoals(probe["context"])))


class HumanIntentTests(unittest.TestCase):
    def test_held_object_wins_over_target(self) -> None:
        # Committed (holding a dish) overrides the directional heading.
        self.assertEqual(
            infer_human_intent(human_holding="dish", target_resource="onion"),
            "pick_dish_then_serve",
        )

    def test_target_resource_used_when_empty_handed(self) -> None:
        self.assertEqual(
            infer_human_intent(human_holding=None, target_resource="onion"),
            "get_onion",
        )

    def test_unknown_intent_is_none(self) -> None:
        self.assertIsNone(infer_human_intent())

    def test_inferred_intent_resolves_to_expected_resource(self) -> None:
        # The inferred intent string must round-trip through the featurizer's
        # resource resolver to the intended resource.
        intent = infer_human_intent(human_holding=None, target_resource="onion")
        context = SubgoalContext.coerce(
            {"recipe": ["onion"], "human_intent": intent}
        )
        self.assertEqual(context.human_target_resource, "onion")

    def test_with_inferred_intent_does_not_clobber_explicit(self) -> None:
        fields = {"human_intent": "get_tomato", "human_holding": "dish"}
        result = with_inferred_intent(fields, target_resource="onion")
        self.assertEqual(result["human_intent"], "get_tomato")

    def test_with_inferred_intent_fills_missing(self) -> None:
        fields = {"human_holding": None}
        result = with_inferred_intent(fields, target_resource="soup")
        self.assertEqual(result["human_intent"], "pickup_soup")


class SubgoalTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_gold_weights()
        cls.subgoal_probes = read_json(ROOT / "data" / "subgoal_probe_states.json")

    def test_gold_teacher_solves_subgoal_probes(self) -> None:
        # Only-tested anchor: the rule teacher must pick the acceptable subgoal
        # on every hand-authored probe, with no training involved.
        for probe in self.subgoal_probes:
            label = label_context(
                self.weights, probe["context"], probe["feasible_subgoals"]
            )
            self.assertIn(label["expected_subgoal"], probe["acceptable_subgoals"])

    def test_positive_intents_ground_to_positive_weight_features(self) -> None:
        for probe in self.subgoal_probes:
            intents = feedback_intents(
                self.weights, probe["context"], probe["feasible_subgoals"]
            )
            self.assertTrue(intents, msg=f"no intents for {probe['probe_id']}")
            for intent in intents:
                for feature in intent.target_features:
                    weight = self.weights.get(feature, 0.0)
                    if intent.polarity > 0:
                        self.assertGreater(weight, 0, msg=f"{feature} not positive")
                    else:
                        self.assertLess(weight, 0, msg=f"{feature} not negative")


class ContextEnumerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_gold_weights()
        cls.contexts = enumerate_contexts(cls.weights)

    def test_enumeration_is_deterministic(self) -> None:
        again = enumerate_contexts(self.weights)
        self.assertEqual(
            [c["scenario_id"] for c in self.contexts],
            [c["scenario_id"] for c in again],
        )

    def test_all_contexts_are_decision_points(self) -> None:
        self.assertGreater(len(self.contexts), 0)
        for entry in self.contexts:
            self.assertGreaterEqual(len(entry["feasible_subgoals"]), 2)
            self.assertIn(entry["expected_subgoal"], entry["acceptable_subgoals"])


class FeedbackTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_gold_weights()
        cls.contexts = enumerate_contexts(cls.weights)

    def test_every_intent_renders_nonempty_text(self) -> None:
        for entry in self.contexts:
            context = SubgoalContext.coerce(entry["context"])
            for intent in feedback_intents(
                self.weights, context, entry["feasible_subgoals"]
            ):
                variants = render_templates(intent, context)
                self.assertTrue(variants)
                for text in variants:
                    self.assertIsInstance(text, str)
                    self.assertTrue(text.strip())
                self.assertIn(theme_for_intent(intent), _KNOWN_THEMES)

    def test_context_description_is_readable(self) -> None:
        context = SubgoalContext.coerce(self.contexts[0]["context"])
        description = context_description(context)
        self.assertIn("Recipe needs", description)


_KNOWN_THEMES = {
    "split_work",
    "make_room",
    "get_order_out",
    "needed_ingredient",
    "good_move",
    "blocking",
    "stealing",
    "duplicating",
    "wrong_ingredient",
    "annoying",
    "wasting_time",
    "bad_move",
}


class SyntheticCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_gold_weights()
        cls.contexts = enumerate_contexts(cls.weights)[:8]
        cls.corpus = build_corpus(
            contexts=cls.contexts, weights=cls.weights, use_llm=False
        )
        cls.probes = load_probe_states()

    def test_corpus_validates_and_has_expected_fields(self) -> None:
        self.assertTrue(self.corpus)
        validate_feedback_examples(self.corpus, probe_states=self.probes)
        for example in self.corpus:
            self.assertIn(example["expected_feedback_type"], {"evaluative", "imperative", "descriptive"})
            self.assertIn("group_id", example)
            self.assertIn("target_features", example)
            self.assertIn(example["attributed_sentiment_score"], (1.0, -1.0))

    def test_llm_disabled_produces_template_only(self) -> None:
        self.assertTrue(all(e["source"] == "template" for e in self.corpus))

    def test_parse_json_strings_is_robust(self) -> None:
        self.assertEqual(
            _parse_json_strings('junk ["a", "b"] trailing'), ["a", "b"]
        )
        self.assertEqual(_parse_json_strings("not json at all"), [])
        self.assertEqual(_parse_json_strings('["  ", "keep"]'), ["keep"])


class PaperAlignedRoute2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = load_features()
        cls.configurations = sample_reward_configurations(n=36, features=cls.features)
        cls.contexts = enumerate_contexts(load_gold_weights())[:6]
        cls.corpus, cls.report = build_teacher_corpus(
            contexts=cls.contexts,
            configurations=cls.configurations,
            seed=3,
            gold_split="test",
        )
        cls.dataset = build_dataset(
            cls.corpus,
            load_probe_states(),
            cls.features,
        )

    def test_reward_configurations_are_complete_unique_and_sign_safe(self) -> None:
        validate_reward_configurations(self.configurations)
        self.assertEqual(len(self.configurations), 36)
        self.assertEqual(
            set(self.configurations[0]["weights"]),
            set(self.features),
        )

    def test_reward_strengths_are_balanced_and_compositional(self) -> None:
        for group in PREFERENCE_GROUPS:
            self.assertEqual(
                {config["multipliers"][group] for config in self.configurations},
                set(LEVELS),
            )

        example = self.corpus[0]
        self.assertEqual(example["text"], example["local_text"])
        self.assertNotEqual(example["text"], example["unconditioned_local_text"])
        self.assertNotIn(example["reward_profile_text"], example["text"])
        self.assertNotIn(example["reward_config_id"], example["text"])
        conditioning_tokens = set(nn_tokenize(example["text"]))
        semantic_tokens = {
            token
            for levels in REWARD_CONDITIONING_PHRASES.values()
            for phrase in levels.values()
            for token in nn_tokenize(phrase)
        }
        self.assertTrue(conditioning_tokens & semantic_tokens)

    def test_generator_does_not_prepartition_teacher_reward_or_context_axes(self) -> None:
        self.assertTrue(all("split" not in example for example in self.corpus))
        self.assertTrue(self.report["teacher_reward_bipartite"]["complete"])
        self.assertTrue(self.report["context_reward_augmentation"]["complete"])
        self.assertEqual(
            self.report["teacher_reward_bipartite"]["observed_edges"],
            12 * len(self.configurations),
        )
        self.assertIn(
            "reward_00_gold",
            {example["reward_config_id"] for example in self.corpus},
        )

    def test_teacher_identity_is_an_independent_stable_author_not_a_form_label(self) -> None:
        self.assertGreaterEqual(self.report["counts"]["synthetic_teachers"], 10)
        self.assertTrue(all(example.get("teacher_id") for example in self.corpus))
        self.assertTrue(all("cv_teacher_id" not in example for example in self.corpus))
        self.assertTrue(
            all(example.get("feedback_form_style_id") for example in self.corpus)
        )
        self.assertTrue(
            all(
                example["teacher_id"] != example["feedback_form_style_id"]
                for example in self.corpus
            )
        )
        for coverage in self.report["author_coverage"].values():
            self.assertEqual(len(coverage["reward_configs"]), len(self.configurations))
            self.assertGreaterEqual(len(coverage["reference_types"]), 2)
            self.assertGreaterEqual(len(coverage["speech_acts"]), 2)
        authors_by_base: dict[str, set[str]] = {}
        rewards_by_base: dict[str, set[str]] = {}
        for example in self.corpus:
            base_id = example["base_feedback_id"]
            authors_by_base.setdefault(base_id, set()).add(example["teacher_id"])
            rewards_by_base.setdefault(base_id, set()).add(example["reward_config_id"])
        self.assertTrue(all(len(authors) == 1 for authors in authors_by_base.values()))
        self.assertTrue(
            all(
                len(reward_ids) == len(self.configurations)
                for reward_ids in rewards_by_base.values()
            )
        )

    def test_reward_conditioning_removes_same_input_different_target_conflicts(self) -> None:
        self.assertEqual(self.report["corpus_audit"]["supervision_conflict_count"], 0)
        signatures: dict[tuple, set[str]] = {}
        for example in self.corpus:
            key = (
                example["unconditioned_local_text"],
                tuple(sorted(example["route2_trajectory_features"].items())),
            )
            signatures.setdefault(key, set()).add(example["text"])
        self.assertTrue(any(len(texts) > 1 for texts in signatures.values()))

    def test_full_target_uses_complete_reward_and_nonzero_trajectory(self) -> None:
        self.assertEqual(self.dataset["target_mode"], "full_teacher_reward")
        self.assertTrue(self.dataset["use_feature_counts"])
        example = self.dataset["examples"][0]
        self.assertEqual(len(example["target_reward"]), len(self.features))
        self.assertGreater(sum(abs(value) for value in example["feature_counts"]), 0.0)

    def test_training_automatically_consumes_trajectory_features(self) -> None:
        result = train_route2(
            self.dataset,
            epochs=2,
            patience=2,
            batch_size=64,
            seed=2,
        )
        self.assertTrue(result["use_feature_counts"])
        self.assertIsNotNone(result["untouched_test_metrics"])
        self.assertIn("mean_cosine_similarity", result["untouched_test_metrics"])

    def test_original_sgd_optimizer_uses_the_same_network(self) -> None:
        result = train_route2(
            self.dataset,
            epochs=1,
            patience=1,
            batch_size=64,
            optimizer_name="sgd",
            lr=0.005,
            weight_decay=1e-4,
            seed=2,
        )
        self.assertIsInstance(result["model"], TrajectoryFeedbackRewardPredictor)
        self.assertEqual(
            result["model"].fc2.out_features,
            len(self.features),
        )


class Route2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = load_gold_weights()
        contexts = enumerate_contexts(cls.weights)[:16]
        cls.corpus = build_corpus(contexts=contexts, weights=cls.weights, use_llm=False)
        cls.features = load_features()
        cls.probes = load_probe_states()
        cls.dataset = build_dataset(
            cls.corpus, cls.probes, cls.features, n_folds=4, seed=0
        )

    def test_training_beats_zero_baseline_on_held_out_subgoals(self) -> None:
        result = train_route2(
            self.dataset,
            val_fold=0,
            epochs=40,
            lr=0.02,
            batch_size=32,
            patience=40,
            use_feature_counts=False,
            seed=0,
            allow_local_feedback_ablation=True,
        )
        model = result["model"]
        fold = self.dataset["folds"][0]
        val_examples = [self.dataset["examples"][i] for i in fold["test_indices"]]
        correct = 0
        for example in val_examples:
            text = _corpus_text(self.corpus, example["feedback_id"])
            w_hat = predict_reward_vector(
                model, self.dataset["vocab"], self.dataset["features"], text
            )
            choice = choose_subgoal(
                w_hat, example["context"], example["feasible_subgoals"]
            )
            acceptable = example.get("acceptable_subgoals") or [example.get("expected_subgoal")]
            correct += int((not choice["is_tie"]) and choice["chosen_subgoal"] in acceptable)
        accuracy = correct / len(val_examples) if val_examples else 0.0
        self.assertGreater(accuracy, 0.5)

    def test_checkpoint_roundtrip(self) -> None:
        result = train_route2(
            self.dataset,
            epochs=3,
            patience=3,
            seed=0,
            allow_local_feedback_ablation=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(
                path, result["model"], self.dataset["vocab"], self.dataset["features"]
            )
            self.assertTrue(path.exists())
            self.assertTrue((path.parent / "vocab.json").exists())


def _corpus_text(corpus, feedback_id):
    for example in corpus:
        if example["feedback_id"] == feedback_id:
            return example["text"]
    return None


class ComfortEnvWrapperTests(unittest.TestCase):
    """Validate the PPO shaping arithmetic without ray/overcooked."""

    def test_wrapper_adds_comfort_to_both_agents(self) -> None:
        from durf.baseline.comfort_env import _wrap_step_with_comfort

        class _FakeBaseEnv:
            mdp = object()
            state = object()

        class _FakeEnv:
            def __init__(self):
                self.base_env = _FakeBaseEnv()
                self.curr_agents = ("ppo_0", "ppo_1")
                self.reward_shaping_factor = 1.0

            def step(self, action_dict):
                rewards = {"ppo_0": 1.0, "ppo_1": 2.0}
                infos = {"ppo_0": {}, "ppo_1": {}}
                return {}, rewards, {"__all__": False}, infos

        class _StubComfort:
            def shaping(self, state, mdp, ai_index=0):
                return 0.5 if ai_index == 0 else -0.25

        env = _FakeEnv()
        _wrap_step_with_comfort(env, _StubComfort())
        _obs, rewards, _dones, infos = env.step({"ppo_0": 0, "ppo_1": 0})
        self.assertAlmostEqual(rewards["ppo_0"], 1.5)
        self.assertAlmostEqual(rewards["ppo_1"], 1.75)
        self.assertAlmostEqual(infos["ppo_0"]["comfort_shaping"], 0.5)

    def test_wrapper_respects_annealing_factor(self) -> None:
        from durf.baseline.comfort_env import _wrap_step_with_comfort

        class _FakeEnv:
            def __init__(self):
                self.base_env = type("B", (), {"mdp": object(), "state": object()})()
                self.curr_agents = ("ppo_0", "ppo_1")
                self.reward_shaping_factor = 0.0  # fully annealed

            def step(self, action_dict):
                return {}, {"ppo_0": 1.0, "ppo_1": 1.0}, {"__all__": False}, {"ppo_0": {}, "ppo_1": {}}

        env = _FakeEnv()

        class _StubComfort:
            def shaping(self, state, mdp, ai_index=0):
                return 5.0

        _wrap_step_with_comfort(env, _StubComfort())
        _obs, rewards, _dones, _infos = env.step({"ppo_0": 0, "ppo_1": 0})
        self.assertAlmostEqual(rewards["ppo_0"], 1.0)  # comfort * 0 anneal


class OnlineHumanLearningTests(unittest.TestCase):
    def test_source_precision_matches_repeated_observations(self) -> None:
        features = ["a", "b"]
        high = BayesianRewardLearner(features)
        repeated = BayesianRewardLearner(features)
        high.update({"a": 1}, -1.0, precision_multiplier=4.0)
        for _ in range(4):
            repeated.update({"a": 1}, -1.0)
        self.assertTrue(np.allclose(high.belief.mean, repeated.belief.mean))
        self.assertTrue(np.allclose(high.belief.covariance, repeated.belief.covariance))

    def test_route1_posterior_roundtrip(self) -> None:
        features = load_features()
        learner = OnlineRoute1Learner(features, mode="route1-literal")
        learner.update(
            "Stop taking the onion.",
            decision={
                "chosen_subgoal": "GET_ONION",
                "ranking": [
                    {"subgoal": "GET_ONION", "features": {"pick_onion": 1.0}}
                ],
            },
            source="human_live",
        )
        restored = OnlineRoute1Learner(features, mode="route1-literal")
        restored.load_state_dict(learner.state_dict())
        self.assertEqual(restored.update_count, learner.update_count)
        self.assertTrue(
            np.allclose(restored.learner.belief.mean, learner.learner.belief.mean)
        )
        self.assertTrue(
            np.allclose(
                restored.learner.belief.covariance,
                learner.learner.belief.covariance,
            )
        )

    def test_route1_rejected_checkpoint_is_atomic(self) -> None:
        features = load_features()
        learner = OnlineRoute1Learner(features, mode="route1-literal")
        learner.update_count = 2
        learner.feedback_count = 3
        learner._recent_feedback = [(3, "keep the lane clear")]
        before = learner.state_dict()

        damaged = copy.deepcopy(before)
        damaged["mean"] = [value + 5.0 for value in damaged["mean"]]
        damaged["covariance"] = (
            np.asarray(damaged["covariance"], dtype=float) * 2.0
        ).tolist()
        # This field is validated late, after the posterior and counters have
        # been parsed.  A failure must not partially install the candidate.
        damaged["source_precision"]["human_live"] = float("nan")

        with self.assertRaisesRegex(ValueError, "source precision"):
            learner.load_state_dict(damaged)
        after = learner.state_dict()
        self.assertEqual(after, before)

    def test_route1_resume_rejects_different_update_hyperparameters(self) -> None:
        features = load_features()
        source = OnlineRoute1Learner(features, mode="route1-literal")
        incompatible = OnlineRoute1Learner(
            features,
            mode="route1-literal",
            valence_scale=source.learner.valence_scale + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "hyperparameters do not match"):
            incompatible.load_state_dict(source.state_dict())

    def test_grouped_fixed_split_is_disjoint(self) -> None:
        groups = ["a", "a", "b", "c", "d", "e", None]
        split = make_train_dev_test_split(groups, seed=3)
        train = set(split["train_indices"])
        dev = set(split["dev_indices"])
        test = set(split["test_indices"])
        self.assertFalse(train & dev)
        self.assertFalse(train & test)
        self.assertFalse(dev & test)
        self.assertIn(6, train)

    def test_scenario_holdout_automatically_expands_test_scenarios(self) -> None:
        rows = [
            {
                "text": f"unique wording {scenario} variant {variant}",
                "group_id": f"scenario-{scenario}",
                "template_family": f"shared-template-{variant}",
            }
            for scenario in range(30)
            for variant in range(2)
        ]
        manifest = make_split_manifest(
            rows,
            seed=3,
            grouping_policy="scenario",
            near_duplicate_threshold=1.0,
        )
        validate_split_manifest(manifest, rows)

        self.assertEqual(len(manifest["test_groups"]), 6)
        self.assertEqual(len(manifest["dev_groups"]), 6)
        self.assertEqual(len(manifest["train_groups"]), 18)
        self.assertFalse(set(manifest["train_groups"]) & set(manifest["test_groups"]))
        # Reused language templates are reported, not allowed to collapse the
        # scenario holdout into one giant joint component.
        self.assertTrue(
            set(manifest["splits"]["train"]["template_families"])
            & set(manifest["splits"]["test"]["template_families"])
        )

    def test_cross_split_near_duplicate_is_pruned_from_lower_priority_split(self) -> None:
        rows = [
            {"text": f"feedback {index}", "group_id": f"scenario-{index}"}
            for index in range(8)
        ]
        manifest = None
        for seed in range(20):
            candidate = make_split_manifest(
                rows,
                seed=seed,
                dev_fraction=0.25,
                test_fraction=0.25,
                grouping_policy="scenario",
                near_duplicate_pairs=[(0, 1, 0.95)],
            )
            if candidate["excluded_indices"]:
                manifest = candidate
                break
        self.assertIsNotNone(manifest)

class ReferenceClassifierTests(unittest.TestCase):
    def test_classifier_reproduces_original_paper_accuracy(self) -> None:
        benchmark = reproduce_original_paper_benchmark()
        self.assertEqual(benchmark["rows"], 982)
        self.assertEqual(benchmark["test_rows"], 148)
        self.assertGreaterEqual(benchmark["reproduced"]["accuracy"], 0.87)
        self.assertGreaterEqual(benchmark["reproduced"]["macro_f1"], 0.75)

    def test_fallback_distinguishes_reference_types(self) -> None:
        self.assertEqual(
            fallback_prediction("Great job.")["reference_type"], "trajectory"
        )
        self.assertEqual(
            fallback_prediction("Grab the onion.")["reference_type"],
            "action_spatial",
        )
        self.assertEqual(
            fallback_prediction("You keep taking my onion.")["reference_type"],
            "action_behavioral",
        )
        self.assertEqual(
            fallback_prediction("Can you hear me?")["reference_type"], "other"
        )

    def test_phrase_annotations_take_precedence_without_floor_rows(self) -> None:
        rows = build_phrase_rows(
            [
                {
                    "text": "Great, move left.",
                    "reference_type": "trajectory",
                    "phrase_annotations": [
                        {"phrase": "Great", "reference_type": "trajectory"},
                        {"phrase": "move left", "reference_type": "action_spatial"},
                    ],
                }
            ]
        )
        self.assertEqual([row["label"] for row in rows], ["trajectory", "action_spatial"])
        self.assertTrue(all(row["annotation_source"] == "phrase_annotations" for row in rows))
        self.assertFalse(any("floor" in str(row.get("source")) for row in rows))

    def test_calibrated_classifier_reports_abstention_metadata(self) -> None:
        examples = []
        phrases = {
            "trajectory": ["great job", "bad move"],
            "feature": ["the onion feature", "the serving feature"],
            "action_spatial": ["move to the left", "go near the pot"],
            "action_behavioral": ["you keep repeating", "you always wait"],
            "other": ["can you hear me", "what is the score"],
        }
        for label, values in phrases.items():
            for group in range(8):
                examples.append(
                    {
                        "text": values[group % len(values)] + f" sample {group}",
                        "reference_type": label,
                        "group_id": f"{label}_{group}",
                    }
                )
        artifact, report = train_classifier(examples, min_df=1, seed=3)
        self.assertIn("temperature", artifact)
        self.assertIn("class_thresholds", artifact)
        self.assertEqual(artifact["input_mode"], "raw_phrase")
        self.assertEqual(
            report["feature_config"]["version"],
            "raw_word_char_tfidf_v3",
        )
        self.assertEqual(
            report["selection_protocol"]["test_examples_evaluated_during_selection"],
            0,
        )
        self.assertIn(
            "paper_style_source_weight",
            report["selected_hyperparameters"],
        )
        self.assertIn(
            "hard_contrastive_source_weight",
            report["selected_hyperparameters"],
        )
        transformers = dict(artifact["vectorizer"].transformer_list)
        self.assertEqual(set(transformers), {"word", "char"})
        # "you" is deliberately retained: it is a discourse cue that the
        # paper preprocessing removed as an English stop word.
        self.assertIn("you", transformers["word"].vocabulary_)
        self.assertIn("ece_10_bin", report["dev"]["calibration"])
        with tempfile.TemporaryDirectory() as directory:
            from joblib import dump

            path = Path(directory) / "model.joblib"
            dump(artifact, path)
            prediction = predict_reference_type("great job sample", model_path=path)
        self.assertIn("top2_margin", prediction)
        self.assertIn("abstained", prediction)
        self.assertEqual(prediction["input_mode"], "raw_phrase")


class PhraseGroundingTests(unittest.TestCase):
    def test_phrase_annotations_precede_utterance_features(self) -> None:
        rows = build_grounding_rows(
            [
                {
                    "text": "Move aside and get a dish.",
                    "target_features": {"clears_human_path": 1, "pick_dish": 1},
                    "phrase_annotations": [
                        {
                            "phrase": "Move aside",
                            "target_features": {"clears_human_path": 1},
                        },
                        {"phrase": "get a dish", "features": {"pick_dish": 1}},
                    ],
                }
            ]
        )
        self.assertEqual([row["labels"] for row in rows], [["clears_human_path"], ["pick_dish"]])

    def test_ovr_model_loads_and_masks_infeasible_features(self) -> None:
        examples = [
            {"text": "clear the path", "target_features": {"clears_human_path": 1}},
            {"text": "step aside now", "target_features": {"clears_human_path": 1}},
            {"text": "get the dish", "target_features": {"pick_dish": 1}},
            {"text": "grab a plate", "target_features": {"pick_dish": 1}},
        ]
        artifact = train_phrase_grounding(examples, min_df=1)
        with tempfile.TemporaryDirectory() as directory:
            from joblib import dump

            path = Path(directory) / "grounding.joblib"
            dump(artifact, path)
            result = predict_grounded_features(
                "please get the dish",
                model_path=path,
                feasible_features={"pick_dish"},
            )
        self.assertEqual(set(result["target_features"]), {"pick_dish"})
        self.assertFalse(result["abstained"])

    def test_opposite_features_never_coactivate(self) -> None:
        resolved = resolve_opposite_features(
            {"blocks_human_path": 1, "clears_human_path": 1}
        )
        self.assertNotIn("blocks_human_path", resolved)
        self.assertNotIn("clears_human_path", resolved)

    def test_single_action_uses_nearest_causal_event(self) -> None:
        result = ground_feedback(
            {
                "text": "That was bad.",
                "total_step": 10,
                "trajectory_features": {"pick_onion": 1},
                "recent_events": [
                    {"total_step": 4, "features": {"pick_tomato": 1}},
                    {"total_step": 9, "features": {"blocks_human_path": 1}},
                    {"total_step": 11, "features": {"pick_onion": 1}},
                ],
            },
            reference_type="action_spatial",
        )
        self.assertEqual(result["target_features"], {"blocks_human_path": 1.0})
        self.assertEqual(result["grounding_source"], "recent_event_features")
        self.assertEqual(result["target_event_step"], 9)
        self.assertEqual(result["event_distance"], 1)
        self.assertEqual(result["temporal_credit_mode"], "single_recent_event")

    def test_behavioral_reference_requires_repetition(self) -> None:
        result = ground_feedback(
            {
                "text": "You keep blocking my way.",
                "total_step": 8,
                "recent_events": [
                    {"total_step": 2, "features": {"blocks_human_path": 1}},
                    {"total_step": 5, "features": {"pick_onion": 1}},
                    {"total_step": 7, "features": {"blocks_human_path": 1}},
                ],
            },
            reference_type="action_behavioral",
        )
        self.assertEqual(result["target_features"], {"blocks_human_path": 2.0})
        self.assertEqual(result["target_event_steps"], [2, 7])
        self.assertEqual(result["temporal_credit_mode"], "behavior_repetition")

    def test_live_mask_rejects_decision_null_feature(self) -> None:
        result = ground_feedback(
            {"text": "The pot is cooking.", "target_features": {"pot_cooking": 1}},
            reference_type="feature",
            live_feature_mask=LIVE_DECISION_FEATURES,
        )
        self.assertTrue(result["abstained"])
        self.assertEqual(result["target_features"], {})
        self.assertEqual(result["masked_target_features"], ["pot_cooking"])


def _overcooked_available() -> bool:
    repo_root = ROOT.parents[2]
    for path in (str(repo_root), str(repo_root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import overcooked_ai_py  # noqa: F401
        import durf.baseline.comfort_reward  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_overcooked_available(), "overcooked_ai_py / durf not importable")
class ComfortBridgeTests(unittest.TestCase):
    def _mdp_state(self):
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        mdp = OvercookedGridworld.from_layout_name("ring_tomato_onion_10x6")
        return mdp, mdp.get_standard_start_state()

    def test_shaping_is_finite(self) -> None:
        import math
        from durf.baseline.comfort_reward import ComfortReward

        mdp, state = self._mdp_state()
        comfort = ComfortReward(weights=load_gold_weights(), coeff=0.5)
        value = comfort.shaping(state, mdp, ai_index=0)
        self.assertTrue(math.isfinite(value))

    def test_context_from_state_reads_recipe(self) -> None:
        from durf.baseline.comfort_reward import context_from_state

        mdp, state = self._mdp_state()
        context = context_from_state(state, mdp, ai_index=0)
        self.assertEqual(context.recipe, ["tomato", "tomato", "onion"])


if __name__ == "__main__":
    unittest.main()
