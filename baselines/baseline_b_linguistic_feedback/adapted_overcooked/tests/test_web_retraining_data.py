from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_web_retraining_data as pipeline  # noqa: E402
from scripts.prepare_web_retraining_data import (  # noqa: E402
    finalize_classifier_annotations,
    prepare_export,
)


def export_rows() -> list[dict]:
    rows: list[dict] = []
    for session_number, participant in ((1, "user-a"), (2, "user-a"), (3, "user-b")):
        session_id = f"session-{session_number}"
        event_id = f"event-{session_number}"
        feedback_id = f"feedback-{session_number}"
        phrase = "Please get the onion." if session_number < 3 else "That route was bad."
        label = "Imperative" if session_number < 3 else "Evaluative"
        rows.extend(
            [
                {
                    "recordType": "session",
                    "session_id": session_id,
                    "anonymous_user_id": participant,
                    "consent_version": "consent-v1",
                    "client_version": "client-v1",
                },
                {
                    "recordType": "event",
                    "event_id": event_id,
                    "session_id": session_id,
                    "sequence_number": 1,
                    "event_type": "feedback",
                    "schema_version": "event-v3",
                    "model_hash": "a" * 64,
                    "payload": {
                        "gameSnapshot": {
                            "features": {"pick_onion": session_number}
                        }
                    },
                },
                {
                    "recordType": "feedback",
                    "feedback_id": feedback_id,
                    "event_id": event_id,
                    "session_id": session_id,
                    "utterance": phrase,
                    "route": "route1",
                    "top_label": label,
                    "probabilities": {
                        "Evaluative": 0.1,
                        "Imperative": 0.8,
                        "Descriptive": 0.1,
                    },
                    "phrases": [
                        {
                            "phrase": phrase,
                            "label": label,
                            "confidence": 0.8,
                            "probabilities": {
                                "Evaluative": 0.1,
                                "Imperative": 0.8,
                                "Descriptive": 0.1,
                            },
                            "abstained": False,
                        }
                    ],
                    "model_hash": "a" * 64,
                    "schema_version": "event-v3",
                    "route_trace": "paper-route1",
                },
            ]
        )
    return rows


def prepare_without_filesystem():
    jsonl_writes: list[tuple[Path, list[dict]]] = []
    json_writes: list[tuple[Path, object]] = []

    def capture_jsonl(path: Path, rows):
        jsonl_writes.append((path, list(rows)))

    def capture_json(path: Path, value: object):
        json_writes.append((path, value))

    with (
        patch.object(pipeline, "_read_jsonl", return_value=export_rows()),
        patch.object(pipeline, "_write_jsonl", side_effect=capture_jsonl),
        patch.object(pipeline, "_write_json", side_effect=capture_json),
        patch.object(pipeline, "_sha256_file", return_value="f" * 64),
        patch.object(Path, "mkdir"),
    ):
        manifest = prepare_export(
            Path("export.jsonl"),
            Path("prepared"),
            initialize_split_registry=True,
            split_seed="test-seed",
        )
    tasks = next(rows for path, rows in jsonl_writes if path.name == "feedback_annotation_tasks.jsonl")
    joins = next(rows for path, rows in jsonl_writes if path.name == "feedback_annotation_join.jsonl")
    route2 = next(value for path, value in json_writes if path.name == "route2_unlabeled.json")
    return manifest, tasks, joins, route2


def sha256_jsonl(rows: list[dict]) -> str:
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare_manifest(tasks_hash: str = "f" * 64, join_hash: str = "f" * 64) -> dict:
    return {
        "schema_version": pipeline.MANIFEST_SCHEMA,
        "stage": "prepared_for_blind_human_annotation",
        "outputs": {
            "annotation_tasks": {"sha256": tasks_hash},
            "annotation_join": {"sha256": join_hash},
        },
    }


class WebRetrainingDataTests(unittest.TestCase):
    def test_prepare_blinds_predictions_and_keeps_participants_disjoint(self):
        manifest, tasks, joins, route2 = prepare_without_filesystem()
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(row["classification_label"] is None for row in tasks))
        self.assertTrue(all("online_prediction" not in row for row in tasks))
        self.assertTrue(all("session_id" not in row for row in tasks))
        self.assertTrue(all(row["online_prediction_is_gold"] is False for row in joins))
        sessions_one_and_two = {
            row["split"] for row in joins if row["session_id"] in {"session-1", "session-2"}
        }
        self.assertEqual(len(sessions_one_and_two), 1)
        self.assertEqual(joins[0]["split"], joins[1]["split"])
        self.assertTrue(
            all(row["eligible_for_route2_supervised_training"] is False for row in route2)
        )
        self.assertEqual(route2[0]["route2_trajectory_features"], {"pick_onion": 1.0})
        self.assertFalse(manifest["route2"]["model_predictions_used_as_reward_targets"])

    def test_registry_keeps_later_same_text_participant_in_old_frozen_split(self):
        new_rows = json.loads(json.dumps(export_rows()[:3]))
        new_rows[0]["session_id"] = "session-new"
        new_rows[0]["anonymous_user_id"] = "user-new"
        new_rows[1]["event_id"] = "event-new"
        new_rows[1]["session_id"] = "session-new"
        new_rows[2]["feedback_id"] = "feedback-new"
        new_rows[2]["event_id"] = "event-new"
        new_rows[2]["session_id"] = "session-new"
        normalized = pipeline._normalized_text("Please get the onion.")
        old_participant = pipeline._stable_id("participant", "user-a")
        registry = {
            "schema_version": pipeline.REGISTRY_SCHEMA,
            "split_seed": "test-seed",
            "participants": {old_participant: "frozen_test"},
            "sessions": {"session-1": "frozen_test"},
            "normalized_text_sha256": {
                pipeline._normalized_text_sha256(normalized): "frozen_test"
            },
        }
        jsonl_writes: list[tuple[Path, list[dict]]] = []
        json_writes: list[tuple[Path, object]] = []
        with (
            patch.object(pipeline, "_read_jsonl", return_value=new_rows),
            patch.object(pipeline, "_read_json_object", return_value=registry),
            patch.object(
                pipeline,
                "_write_jsonl",
                side_effect=lambda path, rows: jsonl_writes.append((path, list(rows))),
            ),
            patch.object(
                pipeline,
                "_write_json",
                side_effect=lambda path, value: json_writes.append((path, value)),
            ),
            patch.object(pipeline, "_sha256_file", return_value="f" * 64),
            patch.object(Path, "mkdir"),
        ):
            manifest = prepare_export(
                Path("new-export.jsonl"),
                Path("next-run"),
                initialize_split_registry=False,
                split_seed="test-seed",
                split_registry_path=Path("previous-split-registry.json"),
            )
        joins = next(
            rows
            for path, rows in jsonl_writes
            if path.name == "feedback_annotation_join.jsonl"
        )
        updated_registry = next(
            value
            for path, value in json_writes
            if path.name == "split_registry.json"
        )
        new_participant = pipeline._stable_id("participant", "user-new")
        self.assertEqual({row["split"] for row in joins}, {"frozen_test"})
        self.assertEqual(updated_registry["participants"][old_participant], "frozen_test")
        self.assertEqual(updated_registry["participants"][new_participant], "frozen_test")
        self.assertFalse(manifest["split_policy"]["existing_assignments_migrated"])

    def test_registry_rejects_component_joining_two_fixed_splits(self):
        registry = pipeline._empty_split_registry("test-seed")
        registry["normalized_text_sha256"].update(
            {
                pipeline._normalized_text_sha256("alpha"): "train",
                pipeline._normalized_text_sha256("beta"): "frozen_test",
            }
        )
        rows = [
            {"participant_key": "participant-new", "session_id": "session-new", "normalized_text": text}
            for text in ("alpha", "beta")
        ]
        with self.assertRaisesRegex(ValueError, "connects fixed splits"):
            pipeline._assign_disjoint_splits(rows, "test-seed", registry)

    def test_prepare_rejects_input_output_path_collision(self):
        with self.assertRaisesRegex(ValueError, "input/output path collision"):
            prepare_export(
                Path("prepared/feedback_annotation_tasks.jsonl"),
                Path("prepared"),
                initialize_split_registry=True,
            )

    def test_prepare_requires_exactly_one_registry_mode(self):
        with self.subTest("neither"):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_export(
                    Path("export.jsonl"),
                    Path("prepared"),
                    initialize_split_registry=False,
                )
        with self.subTest("both"):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_export(
                    Path("export.jsonl"),
                    Path("prepared"),
                    initialize_split_registry=True,
                    split_registry_path=Path("previous-split-registry.json"),
                )

    def test_finalize_rejects_join_split_tampered_after_prepare_manifest(self):
        _, tasks, joins, _ = prepare_without_filesystem()
        tampered_joins = [dict(row) for row in joins]
        tampered_joins[0]["split"] = (
            "frozen_test"
            if tampered_joins[0]["split"] != "frozen_test"
            else "train"
        )
        task_hash = sha256_jsonl(tasks)
        original_join_hash = sha256_jsonl(joins)
        tampered_join_hash = sha256_jsonl(tampered_joins)
        manifest = prepare_manifest(task_hash, original_join_hash)

        def current_file_hash(path: Path) -> str:
            if path.name == "tasks.jsonl":
                return task_hash
            if path.name == "join.jsonl":
                return tampered_join_hash
            return "f" * 64

        with (
            patch.object(pipeline, "_read_json_object", return_value=manifest),
            patch.object(pipeline, "_sha256_file", side_effect=current_file_hash),
            patch.object(pipeline, "_read_jsonl") as read_jsonl,
        ):
            with self.assertRaisesRegex(ValueError, "annotation join hash"):
                finalize_classifier_annotations(
                    Path("tasks.jsonl"),
                    Path("join.jsonl"),
                    Path("annotations.jsonl"),
                    Path("prepare_manifest.json"),
                    Path("finalized"),
                )
        read_jsonl.assert_not_called()

    def test_finalize_uses_reviewed_human_labels_not_online_predictions(self):
        _, tasks, joins, _ = prepare_without_filesystem()
        annotations = []
        for index, task in enumerate(tasks):
            annotations.append(
                {
                    "annotation_id": task["annotation_id"],
                    # Deliberately disagree with the online prediction.
                    "classification_label": "Descriptive",
                    "review_status": "approved",
                    "annotator_id": "reviewer-1",
                    "annotation_revision": index + 1,
                }
            )
        json_writes: list[tuple[Path, object]] = []
        jsonl_writes: list[tuple[Path, list[dict]]] = []
        with (
            patch.object(pipeline, "_read_json_object", return_value=prepare_manifest()),
            patch.object(pipeline, "_read_jsonl", side_effect=[tasks, joins, annotations]),
            patch.object(pipeline, "_write_json", side_effect=lambda path, value: json_writes.append((path, value))),
            patch.object(pipeline, "_write_jsonl", side_effect=lambda path, rows: jsonl_writes.append((path, list(rows)))),
            patch.object(pipeline, "_sha256_file", return_value="f" * 64),
            patch.object(Path, "mkdir"),
        ):
            manifest = finalize_classifier_annotations(
                Path("tasks.jsonl"),
                Path("join.jsonl"),
                Path("annotations.jsonl"),
                Path("prepare_manifest.json"),
                Path("finalized"),
            )
        emitted = []
        for path, value in json_writes:
            if path.name in {"human_feedback_form_web_train.json", "human_feedback_form_web_dev.json"}:
                emitted.extend(value)
        emitted.extend(
            next(rows for path, rows in jsonl_writes if path.name == "human_feedback_form_web_frozen_test.jsonl")
        )
        self.assertEqual(len(emitted), 3)
        self.assertTrue(
            all(row["classification_label"] == "Descriptive" for row in emitted)
        )
        self.assertFalse(manifest["online_predictions_used_as_labels"])

    def test_finalize_rejects_unreviewed_annotation(self):
        _, tasks, joins, _ = prepare_without_filesystem()
        annotations = [
            {
                "annotation_id": task["annotation_id"],
                "classification_label": "Evaluative",
                "review_status": "pending",
                "annotator_id": "reviewer-1",
                "annotation_revision": 1,
            }
            for task in tasks
        ]
        with (
            patch.object(pipeline, "_read_json_object", return_value=prepare_manifest()),
            patch.object(pipeline, "_read_jsonl", side_effect=[tasks, joins, annotations]),
            patch.object(pipeline, "_sha256_file", return_value="f" * 64),
            patch.object(Path, "mkdir"),
        ):
            with self.assertRaisesRegex(ValueError, "not approved"):
                finalize_classifier_annotations(
                    Path("tasks.jsonl"),
                    Path("join.jsonl"),
                    Path("annotations.jsonl"),
                    Path("prepare_manifest.json"),
                    Path("finalized"),
                )


if __name__ == "__main__":
    unittest.main()
