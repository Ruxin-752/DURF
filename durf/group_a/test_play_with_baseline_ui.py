"""Regression tests for the human-feedback Pygame UI helpers."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from durf.group_a.play_with_baseline import (
    CHAT_ANALYSIS_RECT,
    CHAT_BUTTON,
    CHAT_HISTORY_RECT,
    CHAT_INPUT_RECT,
    CHAT_PANEL_RECT,
    CHAT_STATUS_RECT,
    GAME_VIEW_RECT,
    GAME_CONTROLS_RECT,
    GAME_INFO_RECT,
    GAME_STATUS_RECT,
    GAME_TIMING_RECT,
    PAUSE_BUTTON,
    SIDEBAR_RECT,
    WINDOW_SIZE,
    apply_live_feedback_update,
    classify_feedback_form_for_ui,
    clipped_wrapped_lines,
    configured_learner_state_path,
    default_learner_state_for_teacher,
    draw_wrapped_text,
    feedback_mode_banner,
    feedback_supervision_metadata,
    feedback_update_metadata,
    feedback_update_record,
    insert_chat_text,
    load_ui_font,
    no_learning_reply,
    normalize_feedback_probabilities,
    online_learning_enabled,
    parse_args,
    render_chat_panel,
    render_feedback_analysis,
    resolve_executed_ai_action,
    split_feedback_form_phrases,
    validate_feedback_configuration,
    weight_update_reply,
    wrap_text,
)


class HumanFeedbackUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.font.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.font.quit()

    def test_chinese_ime_text_is_inserted_at_cursor(self) -> None:
        value, cursor = insert_chat_text("AI不好", 2, "刚才")
        self.assertEqual(value, "AI刚才不好")
        self.assertEqual(cursor, 4)

    def test_input_limit_applies_to_committed_or_pasted_text(self) -> None:
        value, cursor = insert_chat_text("1234", 2, "中文反馈", limit=6)
        self.assertEqual(value, "12中文34")
        self.assertEqual(cursor, 4)

    def test_chinese_text_wraps_inside_narrow_input(self) -> None:
        font = load_ui_font(17)
        text = "你刚才挡住了我的路应该主动让开"
        lines = wrap_text(text, font, max_width=80)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), text)

    def test_sidebar_never_overlaps_game_view(self) -> None:
        self.assertFalse(GAME_VIEW_RECT.colliderect(SIDEBAR_RECT))
        self.assertFalse(GAME_INFO_RECT.colliderect(SIDEBAR_RECT))
        self.assertFalse(GAME_VIEW_RECT.colliderect(GAME_INFO_RECT))
        self.assertTrue(GAME_INFO_RECT.contains(GAME_STATUS_RECT))
        self.assertTrue(GAME_INFO_RECT.contains(GAME_TIMING_RECT))
        self.assertTrue(GAME_INFO_RECT.contains(GAME_CONTROLS_RECT))
        self.assertFalse(GAME_STATUS_RECT.colliderect(GAME_TIMING_RECT))
        self.assertFalse(GAME_TIMING_RECT.colliderect(GAME_CONTROLS_RECT))
        self.assertTrue(SIDEBAR_RECT.contains(CHAT_PANEL_RECT))
        self.assertTrue(CHAT_PANEL_RECT.contains(CHAT_ANALYSIS_RECT))
        self.assertTrue(CHAT_PANEL_RECT.contains(CHAT_INPUT_RECT))
        self.assertTrue(SIDEBAR_RECT.contains(CHAT_BUTTON))
        self.assertTrue(SIDEBAR_RECT.contains(PAUSE_BUTTON))
        self.assertFalse(CHAT_PANEL_RECT.colliderect(CHAT_BUTTON))
        self.assertFalse(CHAT_PANEL_RECT.colliderect(PAUSE_BUTTON))
        self.assertTrue(CHAT_PANEL_RECT.contains(CHAT_HISTORY_RECT))
        self.assertTrue(CHAT_PANEL_RECT.contains(CHAT_STATUS_RECT))
        self.assertLessEqual(CHAT_ANALYSIS_RECT.bottom, CHAT_HISTORY_RECT.top)
        self.assertLessEqual(CHAT_HISTORY_RECT.bottom, CHAT_STATUS_RECT.top)
        self.assertLessEqual(CHAT_STATUS_RECT.bottom, CHAT_INPUT_RECT.top)

    def test_human_feedback_entry_defaults_to_paper_route2(self) -> None:
        with patch("sys.argv", ["play_with_baseline.py"]):
            args = parse_args()
        self.assertEqual(args.ai_mode, "comfort_subgoal")
        self.assertEqual(args.comfort_feedback_mode, "route2")

    def test_frozen_mode_is_explicitly_control_only(self) -> None:
        self.assertFalse(online_learning_enabled("comfort_subgoal", "frozen"))
        self.assertIn(
            "RECORD ONLY",
            feedback_mode_banner("comfort_subgoal", "frozen"),
        )
        reply = no_learning_reply("comfort_subgoal", "frozen")
        self.assertIn("were not updated", reply)
        self.assertIn("route2", reply)
        supervision = feedback_supervision_metadata("comfort_subgoal", "frozen")
        self.assertEqual(supervision["kind"], "record_only_no_learning")
        self.assertIn("never updates", supervision["note"])
        self.assertIsNone(
            configured_learner_state_path(
                "frozen",
                None,
                "control-1",
                session_id="session-1",
            )
        )
        with self.assertRaisesRegex(ValueError, "not supported in frozen"):
            validate_feedback_configuration("comfort_subgoal", "frozen", True)

    def test_frozen_feedback_record_reports_no_learning(self) -> None:
        record = feedback_update_record(
            {"status": "ignored", "mode": "frozen", "text": "do not block"},
            timestamp_utc="2026-08-12T00:00:00.000+00:00",
            episode=1,
            episode_step=4,
            total_step=4,
            session_id="session-1",
            teacher_id="control-1",
            feedback_event_id="feedback-1",
        )
        self.assertEqual(record["source"], "human_live")
        self.assertFalse(record["learning_applied"])
        self.assertFalse(record["online_learning_enabled"])
        self.assertEqual(record["rejection_reason"], "frozen_control_mode")

    def test_live_learning_reply_reports_weights_without_promising_an_action(self) -> None:
        reply = weight_update_reply(
            {
                "status": "updated",
                "before_subgoal": "GET_ONION",
                "after_subgoal": "GET_DISH",
            }
        )
        self.assertIn("Reward weights", reply)
        self.assertIn("GET_ONION", reply)
        self.assertIn("GET_DISH", reply)
        self.assertIn("w·φ", reply)
        self.assertIn("no rule was hard-coded", reply)
        self.assertNotIn("I will", reply)

        rejected = weight_update_reply({"status": "error"})
        self.assertIn("weights were not changed", rejected)
        self.assertNotIn("I will", rejected)

    def test_three_class_metadata_reports_source_confidence_and_actual_delta(self) -> None:
        classification = classify_feedback_form_for_ui(
            "Please take a dish instead."
        )
        metadata = feedback_update_metadata(
            {
                "status": "updated",
                "mode": "route2",
                "update_rule": "paper_gaussian_precision",
                "before_subgoal": "GET_ONION",
                "after_subgoal": "GET_DISH",
                "top_changes": [
                    {
                        "feature": "pick_dish",
                        "before": -0.1,
                        "after": 0.35,
                        "delta": 0.45,
                    }
                ],
            },
            classification,
        )
        self.assertEqual(classification["label"], "Imperative")
        self.assertIn("Strategy: Imperative", metadata)
        self.assertIn("P1 Imperative", metadata)
        self.assertIn("model score", metadata)
        self.assertIn("uncalibrated", metadata)
        self.assertIn("not learning gate", metadata)
        self.assertIn("Gaussian precision update", metadata)
        self.assertIn("Weight delta: pick_dish +0.45", metadata)
        self.assertIn("GET_ONION -> GET_DISH", metadata)

    def test_route1_metadata_identifies_shared_feedback_form_gate(self) -> None:
        metadata = feedback_update_metadata(
            {"status": "rejected_low_confidence", "mode": "route1-literal"},
            {
                "label": "Descriptive",
                "confidence": 0.51,
                "classifier": "tfidf_logistic_regression_low_confidence",
                "abstained": True,
            },
        )
        self.assertIn("Descriptive", metadata)
        self.assertIn("Route 1 f_G learning gate", metadata)
        self.assertIn("Prediction Descriptive", metadata)
        self.assertIn("numeric score unavailable", metadata)
        self.assertIn("Weights: unchanged", metadata)

    def test_live_update_forwards_exact_ui_prediction_to_route1(self) -> None:
        prediction = {
            "feedback_type": "imperative",
            "label": "Imperative",
            "confidence": 0.91,
            "phrases": [
                {
                    "text": "Please take a dish",
                    "feedback_type": "imperative",
                    "confidence": 0.91,
                    "confidence_threshold": 0.55,
                    "abstained": False,
                }
            ],
        }

        class RecordingAgent:
            def __init__(self) -> None:
                self.kwargs = None

            def update_from_feedback(self, text, **kwargs):
                self.kwargs = kwargs
                return {"status": "updated", "mode": "route1-literal", "text": text}

        agent = RecordingAgent()
        trace = apply_live_feedback_update(
            agent,
            "Please take a dish",
            "feedback-1",
            prediction,
        )

        self.assertEqual(trace["status"], "updated")
        self.assertIs(agent.kwargs["feedback_form_prediction"], prediction)
        self.assertEqual(agent.kwargs["confidence"], 1.0)

    def test_mixed_feedback_is_split_and_classified_per_clause(self) -> None:
        text = (
            "When the pot has nothing, you should put the food in the pot. "
            "That previous delay was bad."
        )
        phrases = split_feedback_form_phrases(text)
        self.assertEqual(len(phrases), 3)
        result = classify_feedback_form_for_ui(text)
        self.assertEqual(result["label"], "Mixed")
        self.assertEqual(len(result["phrases"]), 3)
        self.assertEqual(result["phrases"][-1]["label"], "Evaluative")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(
            result["has_uncertain_phrase"],
            any(row["abstained"] for row in result["phrases"]),
        )

    def test_three_way_scores_are_actual_model_distribution(self) -> None:
        result = classify_feedback_form_for_ui("Please take a dish instead.")
        self.assertEqual(result["label"], "Imperative")
        self.assertEqual(
            set(result["probabilities"]),
            {"evaluative", "imperative", "descriptive"},
        )
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertAlmostEqual(
            result["confidence"], result["probabilities"]["imperative"]
        )
        self.assertEqual(
            result["score_kind"],
            (
                "model_probability_calibrated"
                if result["calibrated"]
                else "model_probability_uncalibrated"
            ),
        )
        self.assertEqual(
            result["classification_target"],
            "paper_feedback_strategy/reference_collapsed",
        )

    def test_low_model_score_keeps_top_type_and_marks_uncertain(self) -> None:
        prediction = {
            "feedback_type": "descriptive",
            "classification_label": "Descriptive",
            "confidence": 0.40,
            "probabilities": {
                "evaluative": 0.30,
                "imperative": 0.30,
                "descriptive": 0.40,
            },
            "classifier": "test_feedback_form",
            "confidence_threshold": 0.55,
            "abstained": True,
            "calibrated": True,
            "calibration_version": "test_v1",
        }
        with patch(
            "src.feedback_form_classifier.predict_feedback_form",
            return_value=prediction,
        ):
            result = classify_feedback_form_for_ui(
                "There is a spare dish near the stove."
            )
        self.assertEqual(result["label"], "Descriptive")
        self.assertTrue(result["abstained"])
        self.assertLess(result["confidence"], 0.55)

    def test_questions_no_longer_receive_hard_coded_91_percent(self) -> None:
        for text in ("What is the score?", "Why is the pot empty?"):
            with self.subTest(text=text):
                result = classify_feedback_form_for_ui(text)
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
                self.assertNotAlmostEqual(max(result["probabilities"].values()), 0.91)

    def test_chat_panel_renders_agent_answer_and_small_metadata(self) -> None:
        screen = pygame.Surface(WINDOW_SIZE)
        font = load_ui_font(21, bold=True)
        small_font = load_ui_font(17)
        meta_font = load_ui_font(13)
        render_chat_panel(
            screen,
            font,
            small_font,
            meta_font,
            [
                {"role": "user", "content": "Please stop blocking my path."},
                {
                    "role": "assistant",
                    "content": "Reward weights updated. Actions still use live-state w·φ.",
                    "meta": (
                        "Speech form: Imperative | phrase-level UI audit\n"
                        "P1 Imperative 97% | syntax classifier\n"
                        "Update: UPDATED | route2 | paper_gaussian_precision\n"
                        "Weights: blocks_human_path +0.000 -> -0.400 (delta -0.400)"
                    ),
                },
            ],
            "",
            0,
            "",
            False,
            "Feedback processed.",
            "ROUTE2 | ONLINE LEARNING",
            True,
        )
        # Message history, status, and input occupy distinct clipped bands.
        self.assertLessEqual(CHAT_HISTORY_RECT.bottom, CHAT_STATUS_RECT.top)
        self.assertLessEqual(CHAT_STATUS_RECT.bottom, CHAT_INPUT_RECT.top)
        self.assertEqual(screen.get_clip(), screen.get_rect())

    def test_long_chat_cards_render_at_multiple_scroll_offsets(self) -> None:
        font = load_ui_font(21, bold=True)
        body_font = load_ui_font(17)
        meta_font = load_ui_font(13)
        long_answer = (
            "I updated the reward model from your detailed feedback. "
            "The policy will rescore every feasible subgoal from the live state "
            "and will not hard-code the action you mentioned. "
        ) * 3
        long_meta = (
            "Speech form: Descriptive | phrase-level UI audit\n"
            "P1 Descriptive 88% | syntax classifier\n"
            "Update: UPDATED | route2 | Gaussian precision update | "
            "top GET_TOMATO -> GET_DISH\n"
            "Weights: blocks_human_path +0.10->-0.42 (d -0.52); "
            "pick_dish -0.04->+0.31 (d +0.35)"
        )
        messages = []
        for index in range(5):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Feedback {index}: please coordinate with me and "
                            "do not block the narrow path while I carry soup."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": long_answer,
                        "meta": long_meta,
                    },
                ]
            )

        images: list[bytes] = []
        for scroll in (0, 2, 6):
            screen = pygame.Surface(WINDOW_SIZE)
            screen.fill((24, 27, 32))
            render_chat_panel(
                screen,
                font,
                body_font,
                meta_font,
                messages,
                "A long input remains safely inside its own input box. " * 6,
                50,
                "",
                False,
                "Feedback processed; this long status is clipped to its own band.",
                "ROUTE2 | ONLINE LEARNING",
                True,
                scroll,
            )
            self.assertEqual(screen.get_clip(), screen.get_rect())
            images.append(pygame.image.tostring(screen, "RGB"))
        self.assertNotEqual(images[0], images[1])
        self.assertNotEqual(images[1], images[2])

    def test_long_card_text_is_truncated_to_its_allocated_lines(self) -> None:
        font = load_ui_font(17)
        lines = clipped_wrapped_lines("very long feedback " * 50, font, 268, 4)
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[-1].endswith("..."))
        self.assertTrue(all(font.size(line)[0] <= 268 for line in lines))

    def test_probability_normalization_rejects_invalid_values(self) -> None:
        normalized = normalize_feedback_probabilities(
            {
                "evaluative": float("nan"),
                "imperative": -4,
                "descriptive": 2,
                "unexpected": 999,
            }
        )
        self.assertEqual(
            normalized,
            {"evaluative": 0.0, "imperative": 0.0, "descriptive": 1.0},
        )

    def test_feedback_analysis_draws_three_scores_inside_its_band(self) -> None:
        screen = pygame.Surface(WINDOW_SIZE)
        screen.fill((1, 2, 3))
        before_clip = screen.get_clip()
        render_feedback_analysis(
            screen,
            load_ui_font(17),
            load_ui_font(13),
            {
                "feedback_type": "imperative",
                "label": "Imperative",
                "probabilities": {
                    "evaluative": 0.1,
                    "imperative": 0.7,
                    "descriptive": 0.2,
                },
                "abstained": False,
                "calibrated": False,
            },
        )
        self.assertEqual(screen.get_clip(), before_clip)
        self.assertNotEqual(screen.get_at(CHAT_ANALYSIS_RECT.center)[:3], (1, 2, 3))

    def test_wrapped_text_uses_real_font_height_and_clips_to_rect(self) -> None:
        screen = pygame.Surface((220, 120))
        screen.fill((0, 0, 0))
        font = load_ui_font(17)
        rect = pygame.Rect(20, 20, 75, font.get_linesize() * 2)
        draw_wrapped_text(
            screen,
            "Long text that would otherwise cross every neighboring panel " * 5,
            font,
            (255, 255, 255),
            rect,
            line_height=10,
            max_lines=20,
        )
        self.assertEqual(screen.get_clip(), screen.get_rect())
        for y in range(screen.get_height()):
            for x in range(screen.get_width()):
                if not rect.collidepoint(x, y):
                    self.assertEqual(screen.get_at((x, y))[:3], (0, 0, 0))

    def test_default_learner_state_is_namespaced_by_teacher(self) -> None:
        alice_first = default_learner_state_for_teacher(
            "route2", "participant_a", session_id="session_one"
        )
        alice_later = default_learner_state_for_teacher(
            "route2", "participant_a", session_id="session_two"
        )
        bob = default_learner_state_for_teacher(
            "route2", "participant_b", session_id="session_three"
        )
        anonymous_first = default_learner_state_for_teacher(
            "route2", "anonymous", session_id="session_one"
        )
        anonymous_later = default_learner_state_for_teacher(
            "route2", "anonymous", session_id="session_two"
        )

        self.assertEqual(alice_first, alice_later)
        self.assertNotEqual(alice_first, bob)
        self.assertNotIn("participant_a", alice_first.name)
        self.assertTrue(alice_first.name.startswith("route2_state_v5_teacher_"))
        self.assertNotEqual(anonymous_first, anonymous_later)

    def test_comfort_action_is_not_overridden_by_legacy_rules(self) -> None:
        with patch(
            "durf.group_a.play_with_baseline.cooperative_action_wrapper"
        ) as legacy_wrapper:
            action, event = resolve_executed_ai_action(
                "comfort_subgoal",
                object(),
                object(),
                2,
                3,
                "GET_ONION",
            )
        self.assertEqual((action, event), (2, ""))
        legacy_wrapper.assert_not_called()

    def test_live_feedback_event_id_is_forwarded_logged_and_idempotent(self) -> None:
        from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent
        from src.feature_schema import load_features

        features = load_features()
        prediction = {feature: 1.0 for feature in features}
        accepted_reference = {
            "phrase": "Stop blocking me",
            "reference_type": "feature",
            "confidence": 0.90,
            "threshold": 0.50,
            "abstained": False,
            "probabilities": {"feature": 0.90},
            "top2_margin": 0.80,
            "margin_threshold": 0.05,
        }
        agent = ComfortSubgoalAgent(
            SimpleNamespace(mdp=object()),
            feedback_mode="route2",
        )
        event_id = "pygame-feedback-event-1"
        with (
            patch(
                "src.phrase_reference_classifier.classify_utterance",
                return_value=[accepted_reference],
            ),
            patch.object(
                agent,
                "_ensure_route2",
                return_value=(object(), {"<unk>": 0}, features, False),
            ),
            patch(
                "src.neural_inference.predict_reward_vector",
                return_value=prediction,
            ),
        ):
            first = apply_live_feedback_update(
                agent,
                "Stop blocking me.",
                event_id,
            )
            precision_after_first = dict(agent._route2_precision)
            duplicate = apply_live_feedback_update(
                agent,
                "Stop blocking me.",
                event_id,
            )

        logged = feedback_update_record(
            duplicate,
            timestamp_utc="2026-08-10T00:00:00.000+00:00",
            episode=1,
            episode_step=4,
            total_step=4,
            session_id="session-1",
            teacher_id="teacher-1",
            feedback_event_id=event_id,
        )
        self.assertEqual(first["feedback_event_id"], event_id)
        self.assertEqual(duplicate["feedback_event_id"], event_id)
        self.assertEqual(duplicate["trace_state"], "ignored_duplicate_feedback_event")
        self.assertFalse(duplicate["posterior_changed"])
        self.assertEqual(agent._route2_precision, precision_after_first)
        self.assertEqual(logged["feedback_event_id"], event_id)


if __name__ == "__main__":
    unittest.main()
