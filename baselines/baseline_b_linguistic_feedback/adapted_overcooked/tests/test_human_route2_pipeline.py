from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch


ADAPTED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ADAPTED_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ADAPTED_ROOT))

from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent  # noqa: E402
from durf.feedback_attribution.session_converter import convert_session  # noqa: E402
from scripts.export_human_route2_corpus import export_session  # noqa: E402
from scripts.prepare_route2_dataset import build_dataset  # noqa: E402
from scripts.recover_route2_session_feedback import recover_session  # noqa: E402
from scripts.train_route2 import train  # noqa: E402
from src.feature_schema import load_features  # noqa: E402
from src.neural_inference import (  # noqa: E402
    TrajectoryFeedbackRewardPredictor,
    save_checkpoint,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "human_route2_session"
TEST_TMP = ADAPTED_ROOT / "outputs" / "_human_route2_test_tmp"


@contextmanager
def workspace_temp():
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP / uuid.uuid4().hex
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class HumanRoute2PipelineTests(unittest.TestCase):
    def copy_fixture(self, root: Path) -> Path:
        session = root / "session"
        shutil.copytree(FIXTURE, session)
        return session

    def make_ensemble(self, root: Path, *, members: int = 2) -> Path:
        features = sorted(load_features())
        vocab = {"<pad>": 0, "<unk>": 1, "please": 2, "stop": 3, "waiting": 4}
        manifest_members = []
        for fold in range(members):
            model = TrajectoryFeedbackRewardPredictor(
                vocab_size=len(vocab), n_features=len(features)
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.fc2.bias.fill_(1.0 + 2.0 * fold)
            checkpoint = root / f"fold_{fold:02d}" / "model.pt"
            save_checkpoint(
                checkpoint,
                model,
                vocab,
                features,
                use_feature_counts=True,
            )
            manifest_members.append(
                {"fold": fold, "checkpoint": str(checkpoint.relative_to(root))}
            )
        manifest = root / "ensemble_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "route2-ensemble-v1",
                    "features": features,
                    "use_feature_counts": True,
                    "members": manifest_members,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest

    def test_converter_consumes_duplicate_updates_and_preserves_reference_types(self) -> None:
        with workspace_temp() as tmp:
            session = self.copy_fixture(Path(tmp))
            counts = convert_session(session)
            self.assertEqual(counts["feedback"], 2)
            events = read_jsonl(session / "feedback_events.jsonl")
            self.assertEqual([row["feedback_event_id"] for row in events], ["event-one", "event-two"])
            self.assertEqual(
                events[0]["extra"]["reference_types"], ["past_action", "trajectory"]
            )
            self.assertEqual(events[1]["extra"]["reference_types"], ["future_action"])
            self.assertEqual(events[0]["extra"]["update_id"], "update-one")
            self.assertEqual(events[1]["extra"]["update_id"], "update-two")
            audit = json.loads(
                (session / "session_conversion_audit.json").read_text(encoding="utf-8")
            )["feedback_matching"]
            self.assertEqual(audit["match_counts"]["feedback_event_id"], 2)
            self.assertEqual(audit["matched_update_count"], 2)
            self.assertEqual(audit["unmatched_update_count"], 0)
            self.assertEqual(audit["duplicate_legacy_update_keys"], 1)

    def test_converter_backfills_stable_unique_ids_for_legacy_duplicates(self) -> None:
        with workspace_temp() as tmp:
            session = self.copy_fixture(Path(tmp))
            chat_path = session / "chat_messages.csv"
            with chat_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            for row in rows:
                row["feedback_event_id"] = ""
            with chat_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            update_path = session / "feedback_updates.jsonl"
            updates = read_jsonl(update_path)
            for update in updates:
                update.pop("feedback_event_id", None)
            update_path.write_text(
                "".join(json.dumps(update) + "\n" for update in updates), encoding="utf-8"
            )

            convert_session(session)
            first = read_jsonl(session / "feedback_events.jsonl")
            convert_session(session)
            second = read_jsonl(session / "feedback_events.jsonl")
            first_ids = [row["feedback_event_id"] for row in first]
            self.assertEqual(first_ids, [row["feedback_event_id"] for row in second])
            self.assertEqual(len(set(first_ids)), 2)
            self.assertTrue(all(value.startswith("feedback_") for value in first_ids))
            self.assertEqual([row["extra"]["update_id"] for row in first], ["update-one", "update-two"])

    def test_language_only_export_is_unlabeled_and_training_fails_closed(self) -> None:
        with workspace_temp() as tmp:
            session = self.copy_fixture(Path(tmp))
            convert_session(session)
            report = export_session(session)
            self.assertEqual(report["unlabeled_count"], 2)
            self.assertEqual(report["paper_supervised_count"], 0)
            rows = json.loads(
                (session / "human_route2_unlabeled.json").read_text(encoding="utf-8")
            )
            self.assertTrue(rows)
            self.assertTrue(
                all(row["eligible_for_route2_supervised_training"] is False for row in rows)
            )
            self.assertTrue(all("teacher_reward_weights" not in row for row in rows))
            self.assertTrue(all("prediction" not in row for row in rows))
            self.assertTrue(all("reference_types" not in row for row in rows))
            with self.assertRaisesRegex(ValueError, "unlabeled human language"):
                build_dataset(rows, [], load_features())
            with self.assertRaisesRegex(ValueError, "independent complete"):
                train({"target_mode": "local_feedback"})

    def test_only_independent_complete_reward_config_exports_supervision(self) -> None:
        with workspace_temp() as tmp:
            root = Path(tmp)
            session = self.copy_fixture(root)
            convert_session(session)
            features = sorted(load_features())
            assignment = root / "assignment.json"
            assignment.write_text(
                json.dumps(
                    {
                        "target_source": "experiment_assigned_reward_config",
                        "teacher_id": "teacher-independent",
                        "reward_config_id": "reward-independent",
                        "teacher_reward_weights": {
                            feature: float(index % 3 - 1)
                            for index, feature in enumerate(features)
                        },
                        "split": "train",
                    }
                ),
                encoding="utf-8",
            )
            report = export_session(session, reward_assignment_path=assignment)
            self.assertEqual(report["unlabeled_count"], 0)
            self.assertEqual(report["paper_supervised_count"], 2)
            rows = json.loads(
                (session / "human_route2_paper_supervised.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(rows[0]["teacher_reward_weights"]), set(features))
            self.assertEqual(
                rows[0]["target_provenance"], "experiment_assigned_reward_config"
            )
            dataset = build_dataset(rows, [], features)
            self.assertEqual(dataset["target_mode"], "full_teacher_reward")
            self.assertEqual(dataset["examples"][0]["target_reward"], [
                rows[0]["teacher_reward_weights"][feature] for feature in features
            ])

            bad = json.loads(assignment.read_text(encoding="utf-8"))
            bad["target_source"] = "route2_model_prediction"
            assignment.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be reward gold"):
                export_session(
                    session,
                    reward_assignment_path=assignment,
                    overwrite=True,
                )

    def test_recovery_matches_live_gaussian_update_and_emits_resumable_v3(self) -> None:
        with workspace_temp() as tmp:
            root = Path(tmp)
            session = self.copy_fixture(root)
            convert_session(session)
            # Simulate a legacy normalized session written before event IDs
            # were part of the schema. Recovery must deterministically backfill
            # them so resume + replay is idempotent.
            feedback_path = session / "feedback_events.jsonl"
            legacy_feedback = read_jsonl(feedback_path)
            for event in legacy_feedback:
                event.pop("feedback_event_id", None)
            feedback_path.write_text(
                "".join(json.dumps(event) + "\n" for event in legacy_feedback),
                encoding="utf-8",
            )
            manifest = self.make_ensemble(root / "models", members=2)
            legacy_state = session / "route2_recovered_state.json"
            legacy_state.write_text('{"version": 1}\n', encoding="utf-8")
            report = recover_session(
                session,
                model_path=manifest,
                expected_ensemble_size=2,
            )
            self.assertEqual(report["recovered_update_count"], 2)
            self.assertEqual(
                json.loads(legacy_state.read_text(encoding="utf-8"))["version"], 1
            )
            self.assertEqual(report["observation_precision"], 2.0)
            state = json.loads(
                (session / "route2_recovered_v3_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["version"], 3)
            self.assertEqual(state["score_formula"], "w_dot_phi")
            self.assertEqual(
                state["route2_posterior"]["update_rule"],
                "paper_independent_gaussian",
            )
            self.assertTrue(
                all(
                    abs(value - 4.04) < 1e-12
                    for value in state["route2_posterior"]["precision"].values()
                )
            )
            recovered_traces = read_jsonl(
                session / "route2_recovered_v3_updates.jsonl"
            )
            recovered_event_ids = [
                trace["feedback_event_id"] for trace in recovered_traces
            ]
            self.assertTrue(
                all(event_id.startswith("feedback_") for event_id in recovered_event_ids)
            )
            self.assertEqual(
                state["route2_posterior"]["processed_feedback_event_ids"],
                sorted(recovered_event_ids),
            )

            trajectories = read_jsonl(session / "trajectory.jsonl")
            events = read_jsonl(session / "feedback_events.jsonl")
            direct = ComfortSubgoalAgent(
                SimpleNamespace(mdp=object()),
                model_path=manifest,
                feedback_mode="route2",
                online_blend=None,
                route2_observation_precision=2.0,
                route1_lookback=25,
            )
            for event, recovered_trace in zip(events, recovered_traces):
                direct.trajectory_history = [
                    row
                    for row in trajectories
                    if row["episode"] == event["episode"]
                    and row["total_step"] <= event["total_step"]
                ][-25:]
                direct.update_from_feedback(
                    event["feedback_text"],
                    feedback_event_id=recovered_trace["feedback_event_id"],
                    source="human_live_recovered",
                )
            self.assertEqual(
                state["route2_posterior"], direct.learner_state_dict()["route2_posterior"]
            )

            resumed = ComfortSubgoalAgent(
                SimpleNamespace(mdp=object()),
                model_path=manifest,
                feedback_mode="route2",
                route2_observation_precision=2.0,
                learner_state_path=session / "route2_recovered_v3_state.json",
                resume_learner_state=True,
            )
            self.assertEqual(resumed.weights, state["route2_posterior"]["mean"])

            replay_report = recover_session(
                session,
                model_path=manifest,
                initial_state_path=session / "route2_recovered_v3_state.json",
                expected_ensemble_size=2,
                output_prefix="route2_replay_v3",
            )
            self.assertEqual(replay_report["recovered_update_count"], 0)
            self.assertEqual(replay_report["ignored_feedback_count"], 2)
            replay_traces = read_jsonl(session / "route2_replay_v3_updates.jsonl")
            self.assertEqual(
                [trace["feedback_event_id"] for trace in replay_traces],
                recovered_event_ids,
            )
            self.assertTrue(
                all(
                    trace["trace_state"] == "ignored_duplicate_feedback_event"
                    for trace in replay_traces
                )
            )
            replay_state = json.loads(
                (session / "route2_replay_v3_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                replay_state["route2_posterior"], state["route2_posterior"]
            )


if __name__ == "__main__":
    unittest.main()
