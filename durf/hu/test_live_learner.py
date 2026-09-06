"""The live learner is what turns "Ours" from an offline-updating method into
one that changes during the session.  These tests pin what makes that safe:
the runtime keeps scoring with the SAME object; an LLM outage degrades to a
labelled stand-in instead of a crash; the checkpoint schedule counts accepted
feedback only; and everything is written where the offline tools expect it.
No network: the attributor is injected."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from durf.feedback_attribution.schemas import (
    ATTRIBUTOR_LLM,
    ATTRIBUTOR_RULE_BASELINE,
    ATTRIBUTOR_RULE_FALLBACK,
    attribution_result,
    feedback_event,
    trajectory_step,
)
from durf.hu.live_learner import LiveLearner

LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
CANDIDATES = [
    {"subgoal": "GET_TOMATO", "task_score": 70.0, "feasible": True},
    {"subgoal": "GET_ONION", "task_score": 65.0, "feasible": True},
    {"subgoal": "WAIT", "task_score": 0.0, "feasible": True},
]
CONDITION = {"pot_empty": True, "ai_empty_handed": True}


def _step(total_step: int) -> dict:
    return trajectory_step(
        source="live", timestamp_utc="t", episode=1, episode_step=total_step,
        total_step=total_step, layout=LAYOUT, ai_action=0, ai_action_name="north",
        human_action=4, human_action_name="stay", environment_reward=0.0,
        episode_reward=0.0, done=False, ai_subgoal="GET_TOMATO",
        ai_condition_features=dict(CONDITION),
        ai_subgoal_candidates=[dict(c) for c in CANDIDATES],
        task_decision={
            "decision_id": f"task:e1:t{total_step}", "decision_level": "task",
            "current_timestep": total_step, "condition_at_decision": dict(CONDITION),
            "candidate_set": [c["subgoal"] for c in CANDIDATES],
            "candidates": [dict(c) for c in CANDIDATES], "selected": "GET_TOMATO",
        },
        state_facts={}, extra={"state_before": {}},
    )


def _feedback(total_step: int, text: str) -> dict:
    return feedback_event(
        source="live", timestamp_utc="t", episode=1, episode_step=total_step,
        total_step=total_step, feedback_text=text, feedback_value=None,
        role="human_language", extra={"layout": LAYOUT}, user_id="P01",
    )


def _llm_says(preferred, rejected, *, level="task"):
    """An injected attributor that returns a fixed LLM-style answer."""
    def attribute(*, feedback, trajectory, candidate_events, baseline_attribution, lookback_steps):
        result = attribution_result(
            feedback_event_id=baseline_attribution["feedback_event_id"],
            feedback_type="preference", target_time_window=None, target_event=None,
            polarity="positive", preference="explicit", key_conditions={},
            condition_features=dict(CONDITION), preferred_subgoals=list(preferred),
            rejected_subgoals=list(rejected), confidence=0.9, needs_clarification=False,
            clarification_question=None, proposed_schema_update=[], notes="test",
            decision_level=level, preference_source="explicit_text_llm",
            attributor=ATTRIBUTOR_LLM, model_id="deepseek-test", prompt_hash="h",
        )
        return result, {"status": "ok", "attributor": ATTRIBUTOR_LLM}
    return attribute


def _llm_fails(**_kwargs):
    raise RuntimeError("DeepSeek HTTP 503: busy")


class LiveLearnerTest(unittest.TestCase):
    def _learner(self, tmp, **kw):
        learner = LiveLearner(user_id="P01", session_dir=Path(tmp), layout=LAYOUT, **kw)
        for step in range(1, 31):
            learner.observe_step(_step(step))
        return learner

    def test_the_runtime_model_object_never_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(tmp, attribute_fn=_llm_says(["GET_ONION"], ["GET_TOMATO"]))
            before = learner.model
            learner.on_feedback(_feedback(30, "onion first please"))
            self.assertIs(learner.model, before)

    def test_a_preference_changes_the_score_in_the_stated_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(tmp, attribute_fn=_llm_says(["GET_ONION"], ["GET_TOMATO"]))
            model = learner.model
            def score(sg):
                return model.score("task", CONDITION, sg)["final_score"]

            gap_before = score("GET_ONION") - score("GET_TOMATO")
            update = learner.on_feedback(_feedback(30, "get the onion first, I'll do tomatoes"))
            gap_after = score("GET_ONION") - score("GET_TOMATO")
            self.assertEqual(update.attributor, ATTRIBUTOR_LLM)
            self.assertEqual(update.labels_added, 1)
            self.assertGreater(gap_after, gap_before)
            self.assertGreater(gap_after, 0.0)
            self.assertTrue(model.has_support("task", "GET_ONION"))

    def test_llm_failure_is_a_labelled_stand_in_that_still_updates(self) -> None:
        """The participant should not be punished for an outage: the keyword
        baseline's label still updates the model -- but it is marked, counted,
        and never mistaken for LLM output."""
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(tmp, attribute_fn=_llm_fails)
            update = learner.on_feedback(
                _feedback(30, "get the onion first, I'll take care of the tomatoes")
            )
            self.assertEqual(update.attributor, ATTRIBUTOR_RULE_FALLBACK)
            self.assertIn("503", update.error)
            self.assertEqual(update.labels_added, 1)  # keyword fallback read it
            self.assertEqual(learner.summary()["llm_fallbacks"], 1)
            labels = [
                json.loads(l) for l in (Path(tmp) / "hu_subgoal_preferences.jsonl").read_text().splitlines()
            ]
            self.assertEqual(labels[0]["attributor"], ATTRIBUTOR_RULE_FALLBACK)

    def test_checkpoints_follow_accepted_feedback_not_chatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(tmp, attribute_fn=_llm_says(["GET_ONION"], ["GET_TOMATO"]))
            self.assertTrue((Path(tmp) / "checkpoints" / "F0.json").exists())
            for i in range(4):
                learner.on_feedback(_feedback(30, f"onion first #{i}"))
            self.assertFalse((Path(tmp) / "checkpoints" / "F5.json").exists())
            # chatter that yields no label must not advance the count
            learner.attribute_fn = _llm_says([], [])
            update = learner.on_feedback(_feedback(30, "nice weather"))
            self.assertEqual(update.labels_added, 0)
            self.assertEqual(learner.accepted_feedback_count, 4)
            self.assertFalse((Path(tmp) / "checkpoints" / "F5.json").exists())
            learner.attribute_fn = _llm_says(["GET_ONION"], ["GET_TOMATO"])
            update = learner.on_feedback(_feedback(30, "onion first #5"))
            self.assertEqual(learner.accepted_feedback_count, 5)
            self.assertTrue(update.checkpoint_written.endswith("F5.json"))

    def test_cross_domain_llm_output_yields_no_label_and_is_flagged(self) -> None:
        """First live run paired SERVE_SOUP with YIELD.  That must produce no
        training label -- and must be visible, not silently dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(
                tmp, attribute_fn=_llm_says(["SERVE_SOUP"], ["YIELD"], level="coordination")
            )
            update = learner.on_feedback(_feedback(30, "don't yield when close"))
            self.assertEqual(update.labels_added, 0)
            self.assertIn("cross_domain_attribution", update.attribution_flags)

    def test_no_llm_mode_labels_everything_rule_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            learner = self._learner(tmp, use_llm=False)
            update = learner.on_feedback(
                _feedback(30, "get the onion first, I'll take care of the tomatoes")
            )
            self.assertEqual(update.attributor, ATTRIBUTOR_RULE_BASELINE)
            self.assertEqual(update.labels_added, 1)

    def test_retraining_is_a_function_of_the_label_set_not_the_order(self) -> None:
        """reset()+train(all) after every feedback: two learners fed the same
        labels in different orders end with identical weights."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            la = self._learner(a); lb = self._learner(b)
            la.attribute_fn = _llm_says(["GET_ONION"], ["GET_TOMATO"]); la.on_feedback(_feedback(30, "x"))
            la.attribute_fn = _llm_says(["GET_DISH"], ["GET_TOMATO"]);  la.on_feedback(_feedback(30, "y"))
            lb.attribute_fn = _llm_says(["GET_DISH"], ["GET_TOMATO"]);  lb.on_feedback(_feedback(30, "y"))
            lb.attribute_fn = _llm_says(["GET_ONION"], ["GET_TOMATO"]); lb.on_feedback(_feedback(30, "x"))
            for sg in ("GET_ONION", "GET_TOMATO", "GET_DISH"):
                self.assertAlmostEqual(
                    la.model.score("task", CONDITION, sg)["final_score"],
                    lb.model.score("task", CONDITION, sg)["final_score"], places=5,
                )

    def test_latency_and_session_files_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ticks = iter([0.0, 0.25, 1.0, 1.4])
            learner = self._learner(
                tmp, attribute_fn=_llm_says(["GET_ONION"], ["GET_TOMATO"]), clock=lambda: next(ticks)
            )
            update = learner.on_feedback(_feedback(30, "onion first"))
            self.assertEqual(update.update_latency_ms, 250.0)
            for name in (
                "attribution_preview.jsonl", "hu_attribution_provenance.jsonl",
                "hu_subgoal_preferences.jsonl", "llm_attribution_audit.jsonl",
                "live_feedback_updates.jsonl",
            ):
                self.assertTrue((Path(tmp) / name).exists(), name)
            self.assertEqual(learner.summary()["median_update_latency_ms"], 250.0)


if __name__ == "__main__":
    unittest.main()
