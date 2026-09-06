"""Prepare auditable offline retraining inputs from the Web D1 JSONL export.

This script deliberately separates blind annotation tasks from online model
predictions. Predictions remain in a private join file for error analysis and
are never copied into the supervised target field.

Two explicit stages are provided:

``prepare``
    Validate and join D1 session/event/feedback records, create a blind phrase
    annotation queue, preserve prediction provenance separately, assign stable
    participant-disjoint splits, and create an explicitly *unlabeled* Route 2
    pool.

``finalize-classifier``
    Join a reviewed human annotation file back to the private provenance file
    and emit the existing classifier trainer's train/dev format plus a strict
    two-field frozen-test input. This command never trains or deploys a model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable


TASK_SCHEMA = "durf-web-feedback-annotation-task-v1"
JOIN_SCHEMA = "durf-web-feedback-annotation-join-v1"
MANIFEST_SCHEMA = "durf-web-retraining-manifest-v1"
REGISTRY_SCHEMA = "durf-web-split-registry-v1"
CANONICAL_LABELS = ("Evaluative", "Imperative", "Descriptive")
INTERNAL_LABELS = {label: label.lower() for label in CANONICAL_LABELS}
SPLIT_THRESHOLDS = (("train", 8000), ("dev", 9000), ("frozen_test", 10000))
VALID_SPLITS = frozenset(split for split, _ in SPLIT_THRESHOLDS)
SAFE_VERSION = re.compile(r"^[A-Za-z0-9._:+-]{1,128}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _guard_no_input_output_collision(
    inputs: Iterable[Path | None],
    outputs: Iterable[Path],
) -> None:
    output_paths = {path.resolve() for path in outputs}
    for input_path in inputs:
        if input_path is not None and input_path.resolve() in output_paths:
            raise ValueError(f"input/output path collision: {input_path.resolve()}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(encoded, encoding="utf-8")


def _require_text(row: dict, key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _require_version(row: dict, key: str, *, context: str) -> str:
    value = _require_text(row, key, context=context)
    if not SAFE_VERSION.fullmatch(value):
        raise ValueError(f"{context}: invalid {key}")
    return value


def _index_unique(rows: Iterable[dict], key: str, *, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for position, row in enumerate(rows, start=1):
        identity = _require_text(row, key, context=f"{label}[{position}]")
        if identity in result:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


class _DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _split_for_component(component: tuple[str, ...], seed: str) -> str:
    digest = hashlib.sha256(
        f"{seed}\0{'|'.join(component)}".encode("utf-8")
    ).hexdigest()
    bucket = int(digest[:16], 16) % 10000
    for split, upper in SPLIT_THRESHOLDS:
        if bucket < upper:
            return split
    raise AssertionError("unreachable split bucket")


def _empty_split_registry(seed: str) -> dict:
    return {
        "schema_version": REGISTRY_SCHEMA,
        "split_seed": seed,
        "participants": {},
        "sessions": {},
        "normalized_text_sha256": {},
    }


def _validate_registry_mapping(payload: dict, key: str) -> dict[str, str]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"split registry {key} must be an object")
    result: dict[str, str] = {}
    for identity, split in value.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"split registry {key} has an invalid identity")
        if key == "normalized_text_sha256" and not SHA256_PATTERN.fullmatch(identity):
            raise ValueError("split registry has an invalid normalized-text hash")
        if split not in VALID_SPLITS:
            raise ValueError(f"split registry {key} has an invalid split")
        result[identity] = str(split)
    return result


def _load_split_registry(path: Path | None, seed: str) -> dict:
    if path is None:
        return _empty_split_registry(seed)
    payload = _read_json_object(path)
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unsupported split registry schema")
    if payload.get("split_seed") != seed:
        raise ValueError("split registry seed does not match --split-seed")
    return {
        "schema_version": REGISTRY_SCHEMA,
        "split_seed": seed,
        "participants": _validate_registry_mapping(payload, "participants"),
        "sessions": _validate_registry_mapping(payload, "sessions"),
        "normalized_text_sha256": _validate_registry_mapping(
            payload, "normalized_text_sha256"
        ),
    }


def _normalized_text_sha256(normalized: str) -> str:
    return _sha256_bytes(normalized.encode("utf-8"))


def _assign_disjoint_splits(
    join_rows: list[dict],
    seed: str,
    registry: dict,
) -> tuple[dict[str, str], dict]:
    participants = sorted({str(row["participant_key"]) for row in join_rows})
    disjoint = _DisjointSet(participants)
    owners: dict[str, str] = {}
    for row in join_rows:
        participant = str(row["participant_key"])
        normalized = str(row["normalized_text"])
        previous = owners.setdefault(normalized, participant)
        disjoint.union(previous, participant)

    components: dict[str, list[str]] = defaultdict(list)
    for participant in participants:
        components[disjoint.find(participant)].append(participant)
    rows_by_component: dict[str, list[dict]] = defaultdict(list)
    for row in join_rows:
        rows_by_component[disjoint.find(str(row["participant_key"]))].append(row)

    participant_splits: dict[str, str] = {}
    for root, members in components.items():
        component = tuple(sorted(members))
        fixed_splits: set[str] = set()
        for row in rows_by_component[root]:
            participant = str(row["participant_key"])
            session = str(row["session_id"])
            text_hash = _normalized_text_sha256(str(row["normalized_text"]))
            for mapping, identity in (
                (registry["participants"], participant),
                (registry["sessions"], session),
                (registry["normalized_text_sha256"], text_hash),
            ):
                registered = mapping.get(identity)
                if registered is not None:
                    fixed_splits.add(str(registered))
        if len(fixed_splits) > 1:
            raise ValueError(
                "split registry conflict: a current participant/text component "
                f"connects fixed splits {sorted(fixed_splits)}; manual resolution required"
            )
        split = (
            next(iter(fixed_splits))
            if fixed_splits
            else _split_for_component(component, seed)
        )
        participant_splits.update({member: split for member in component})

    updated = {
        "schema_version": REGISTRY_SCHEMA,
        "split_seed": seed,
        "participants": dict(registry["participants"]),
        "sessions": dict(registry["sessions"]),
        "normalized_text_sha256": dict(registry["normalized_text_sha256"]),
    }
    for row in join_rows:
        split = participant_splits[str(row["participant_key"])]
        registrations = (
            ("participants", str(row["participant_key"])),
            ("sessions", str(row["session_id"])),
            (
                "normalized_text_sha256",
                _normalized_text_sha256(str(row["normalized_text"])),
            ),
        )
        for mapping_name, identity in registrations:
            previous = updated[mapping_name].get(identity)
            if previous is not None and previous != split:
                raise ValueError(
                    "split registry conflict: an existing assignment would migrate; "
                    "manual resolution required"
                )
            updated[mapping_name][identity] = split
    updated["counts"] = {
        "participants": len(updated["participants"]),
        "sessions": len(updated["sessions"]),
        "normalized_text_sha256": len(updated["normalized_text_sha256"]),
    }
    return participant_splits, updated


def _feedback_model_provenance(feedback: dict, event: dict) -> dict:
    context = f"feedback {feedback['feedback_id']}"
    classifier_hash = _require_version(feedback, "model_hash", context=context)
    event_hash = event.get("model_hash")
    payload = event.get("payload")
    if isinstance(payload, dict) and "selectedRoute" in payload and payload["selectedRoute"] != feedback.get("route"):
        raise ValueError(f"{context}: selectedRoute/feedback route mismatch")
    if isinstance(payload, dict) and (
        "classifierModelHash" in payload or "updaterModelHash" in payload
    ):
        # New clients attach the classifier hash to feedback and the updater
        # hash to its event. Route 2 legitimately uses two different models.
        declared_classifier = _require_version(payload, "classifierModelHash", context=context)
        declared_updater = _require_version(payload, "updaterModelHash", context=context)
        if not SHA256_PATTERN.fullmatch(declared_classifier) or not SHA256_PATTERN.fullmatch(declared_updater):
            raise ValueError(f"{context}: explicit model hashes must be lowercase SHA-256")
        if declared_classifier != classifier_hash:
            raise ValueError(f"{context}: classifier model hash mismatch")
        if declared_updater != event_hash:
            raise ValueError(f"{context}: updater model hash mismatch")
        separate_grounding = (
            feedback.get("route") == "route1"
            and payload.get("feedbackSemanticsVersion") == "speech-act-grounding-v1"
            and payload.get("groundingModelHash") == declared_updater
        )
        if feedback.get("route") != "route2" and not separate_grounding and declared_updater != declared_classifier:
            raise ValueError(f"{context}: distinct updater model hash requires route2")
        return {
            "classifier_model_hash": classifier_hash,
            "updater_model_hash": declared_updater,
            "model_provenance": "explicit_classifier_and_updater",
            **({"feedback_semantics_version": "speech-act-grounding-v1",
                "grounding_model_hash": declared_updater} if separate_grounding else {}),
        }
    if event_hash not in {None, classifier_hash}:
        raise ValueError(f"{context}: model hash mismatch without explicit provenance")
    return {
        "classifier_model_hash": classifier_hash,
        "updater_model_hash": event_hash,
        "model_provenance": "legacy_shared_model_hash",
    }


def _numeric_features(value: object, source: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: features must be an object")
    features: dict[str, float] = {}
    for key, number in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError(f"{source}: features must contain finite numeric values")
        features[key] = float(number)
    return features


def _latest_trajectory_features(
    event: dict,
    events_by_session: dict[str, list[dict]],
) -> tuple[dict[str, float], str]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        snapshot = payload.get("gameSnapshot")
        if isinstance(snapshot, dict) and "recentTrajectoryFeatures" in snapshot:
            # Preserve the actual vector used for inference, including an empty
            # trajectory; a single-state snapshot is not an equivalent input.
            return (
                _numeric_features(snapshot["recentTrajectoryFeatures"], "feedback_recent_trajectory"),
                "feedback_recent_trajectory",
            )
        if isinstance(snapshot, dict) and isinstance(snapshot.get("features"), dict):
            return (
                _numeric_features(snapshot["features"], "feedback_game_snapshot"),
                "feedback_game_snapshot",
            )
    session_id = str(event["session_id"])
    sequence = int(event.get("sequence_number", -1))
    candidates = [
        candidate
        for candidate in events_by_session.get(session_id, ())
        if candidate.get("event_type") == "tick_summary"
        and int(candidate.get("sequence_number", -1)) < sequence
    ]
    if candidates:
        latest = max(candidates, key=lambda row: int(row.get("sequence_number", -1)))
        latest_payload = latest.get("payload")
        if isinstance(latest_payload, dict) and isinstance(
            latest_payload.get("featureCounts"), dict
        ):
            return (
                _numeric_features(latest_payload["featureCounts"], "preceding_tick_summary"),
                "preceding_tick_summary",
            )
    return {}, "missing"


def prepare_export(
    export_path: Path,
    output_dir: Path,
    *,
    initialize_split_registry: bool,
    split_seed: str = "durf-web-session-split-v1",
    consent_version: str | None = None,
    split_registry_path: Path | None = None,
    filter_consent_version: str | None = None,
) -> dict:
    if consent_version is not None and filter_consent_version is not None:
        raise ValueError("consent_version and filter_consent_version are mutually exclusive")
    if filter_consent_version is not None and not SAFE_VERSION.fullmatch(filter_consent_version):
        raise ValueError("invalid filter_consent_version")
    if not isinstance(initialize_split_registry, bool) or (
        initialize_split_registry == (split_registry_path is not None)
    ):
        raise ValueError(
            "exactly one of initialize_split_registry or split_registry_path is required"
        )
    task_path = output_dir / "feedback_annotation_tasks.jsonl"
    join_path = output_dir / "feedback_annotation_join.jsonl"
    route2_path = output_dir / "route2_unlabeled.json"
    registry_output_path = output_dir / "split_registry.json"
    manifest_path = output_dir / "prepare_manifest.json"
    _guard_no_input_output_collision(
        (export_path, split_registry_path),
        (task_path, join_path, route2_path, registry_output_path, manifest_path),
    )
    split_registry = _load_split_registry(split_registry_path, split_seed)
    split_registry_input_sha256 = (
        _sha256_file(split_registry_path) if split_registry_path is not None else None
    )

    rows = _read_jsonl(export_path)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for position, row in enumerate(rows, start=1):
        record_type = row.get("recordType")
        if record_type not in {"session", "event", "feedback"}:
            raise ValueError(f"row {position}: unsupported recordType {record_type!r}")
        by_type[str(record_type)].append(row)

    sessions = _index_unique(by_type["session"], "session_id", label="session")
    events = _index_unique(by_type["event"], "event_id", label="event")
    feedback_rows = _index_unique(
        by_type["feedback"], "feedback_id", label="feedback"
    )
    sequence_keys: set[tuple[str, int]] = set()
    events_by_session: dict[str, list[dict]] = defaultdict(list)
    for event_id, row in events.items():
        session_id = _require_text(row, "session_id", context=f"event {event_id}")
        if session_id not in sessions:
            raise ValueError(f"event {event_id}: missing session {session_id}")
        sequence = row.get("sequence_number")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError(f"event {event_id}: invalid sequence_number")
        sequence_key = (session_id, sequence)
        if sequence_key in sequence_keys:
            raise ValueError(f"duplicate session sequence: {sequence_key}")
        sequence_keys.add(sequence_key)
        _require_version(row, "schema_version", context=f"event {event_id}")
        events_by_session[session_id].append(row)

    source_record_counts = {
        "sessions": len(sessions), "events": len(events), "feedback": len(feedback_rows)
    }
    selected_rows = rows
    if filter_consent_version is not None:
        # Select a complete session cohort. Validate links before filtering so
        # a corrupt cross-session feedback row is not silently discarded.
        for feedback_id, feedback in feedback_rows.items():
            linked_event = events.get(feedback.get("event_id"))
            if (
                feedback.get("session_id") not in sessions
                or linked_event is None
                or linked_event.get("session_id") != feedback.get("session_id")
                or linked_event.get("event_type") != "feedback"
            ):
                raise ValueError(f"feedback {feedback_id}: invalid event/session link before consent filter")
        sessions = {
            key: row for key, row in sessions.items()
            if row.get("consent_version") == filter_consent_version
        }
        events = {key: row for key, row in events.items() if row["session_id"] in sessions}
        feedback_rows = {
            key: row for key, row in feedback_rows.items() if row["session_id"] in sessions
        }
        events_by_session = {
            key: values for key, values in events_by_session.items() if key in sessions
        }
        selected_rows = [row for row in rows if row.get("session_id") in sessions]

    task_rows: list[dict] = []
    join_rows: list[dict] = []
    route2_rows: list[dict] = []
    for feedback_id, feedback in sorted(feedback_rows.items()):
        event_id = _require_text(
            feedback, "event_id", context=f"feedback {feedback_id}"
        )
        event_row = events.get(event_id)
        if event_row is None:
            raise ValueError(f"feedback {feedback_id}: missing event {event_id}")
        if event_row.get("event_type") != "feedback":
            raise ValueError(f"feedback {feedback_id}: event is not feedback type")
        session_id = _require_text(
            feedback, "session_id", context=f"feedback {feedback_id}"
        )
        if event_row.get("session_id") != session_id:
            raise ValueError(f"feedback {feedback_id}: event/session mismatch")
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"feedback {feedback_id}: missing session {session_id}")
        if consent_version is not None and session.get("consent_version") != consent_version:
            raise ValueError(
                f"session {session_id}: consent_version does not match requested export"
            )
        anonymous_user_id = _require_text(
            session, "anonymous_user_id", context=f"session {session_id}"
        )
        participant_key = _stable_id("participant", anonymous_user_id)
        model_provenance = _feedback_model_provenance(feedback, event_row)
        model_hash = model_provenance["classifier_model_hash"]
        utterance = _require_text(
            feedback, "utterance", context=f"feedback {feedback_id}"
        )
        phrases = feedback.get("phrases")
        if not isinstance(phrases, list) or not phrases:
            raise ValueError(f"feedback {feedback_id}: phrases must be a non-empty list")
        trajectory_features, trajectory_source = _latest_trajectory_features(
            event_row, events_by_session
        )
        route2_rows.append(
            {
                "feedback_id": feedback_id,
                "feedback_event_id": event_id,
                "text": utterance,
                "source": "human_web_session",
                "session_id": session_id,
                "teacher_id": participant_key,
                "group_id": session_id,
                "route2_trajectory_features": trajectory_features,
                "trajectory_source": trajectory_source,
                "trajectory_is_exact_model_input": trajectory_source == "feedback_recent_trajectory",
                **model_provenance,
                "supervision_status": "unlabeled_no_independent_reward_target",
                "eligible_for_route2_supervised_training": False,
                "online_model_prediction_is_gold": False,
            }
        )

        for phrase_index, phrase_row in enumerate(phrases):
            if not isinstance(phrase_row, dict):
                raise ValueError(
                    f"feedback {feedback_id}: phrase {phrase_index} must be an object"
                )
            language = _require_text(
                phrase_row,
                "phrase",
                context=f"feedback {feedback_id} phrase {phrase_index}",
            )
            annotation_id = _stable_id(
                "web_annotation", feedback_id, phrase_index, language
            )
            task = {
                "schema_version": TASK_SCHEMA,
                "annotation_id": annotation_id,
                "language": language,
                "classification_label": None,
                "review_status": "pending",
            }
            task_hash = _sha256_bytes(_canonical_bytes(task))
            task_rows.append(task)
            join_rows.append(
                {
                    "schema_version": JOIN_SCHEMA,
                    "annotation_id": annotation_id,
                    "annotation_task_sha256": task_hash,
                    "feedback_id": feedback_id,
                    "event_id": event_id,
                    "session_id": session_id,
                    "participant_key": participant_key,
                    "phrase_index": phrase_index,
                    "language": language,
                    "normalized_text": _normalized_text(language),
                    "consent_version": session.get("consent_version"),
                    "client_version": session.get("client_version"),
                    "event_schema_version": event_row.get("schema_version"),
                    "feedback_schema_version": feedback.get("schema_version"),
                    "route": feedback.get("route"),
                    "route_trace": feedback.get("route_trace"),
                    "model_hash": model_hash,
                    **model_provenance,
                    "online_prediction": {
                        "label": phrase_row.get("label"),
                        "confidence": phrase_row.get("confidence"),
                        "probabilities": phrase_row.get("probabilities"),
                        "abstained": phrase_row.get("abstained"),
                    },
                    "online_prediction_is_gold": False,
                    "pii_review_required": True,
                }
            )

    participant_splits, updated_registry = _assign_disjoint_splits(
        join_rows, split_seed, split_registry
    )
    for row in join_rows:
        row["split"] = participant_splits[str(row["participant_key"])]
    for row in route2_rows:
        row["split"] = participant_splits[str(row["teacher_id"])]

    split_participants: dict[str, set[str]] = defaultdict(set)
    split_sessions: dict[str, set[str]] = defaultdict(set)
    split_texts: dict[str, set[str]] = defaultdict(set)
    for row in join_rows:
        split = str(row["split"])
        split_participants[split].add(str(row["participant_key"]))
        split_sessions[split].add(str(row["session_id"]))
        split_texts[split].add(str(row["normalized_text"]))
    for left_index, (left, _) in enumerate(SPLIT_THRESHOLDS):
        for right, _ in SPLIT_THRESHOLDS[left_index + 1 :]:
            if split_participants[left] & split_participants[right]:
                raise AssertionError("participant split overlap")
            if split_sessions[left] & split_sessions[right]:
                raise AssertionError("session split overlap")
            if split_texts[left] & split_texts[right]:
                raise AssertionError("normalized-text split overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(task_path, task_rows)
    _write_jsonl(join_path, join_rows)
    _write_json(route2_path, route2_rows)
    _write_json(registry_output_path, updated_registry)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "prepared_for_blind_human_annotation",
        "source_export": str(export_path.resolve()),
        "source_export_sha256": _sha256_file(export_path),
        "consent_selection": {
            "mode": "filter_session_cohort" if filter_consent_version is not None else (
                "strict_feedback_consent_match" if consent_version is not None else "unfiltered"
            ),
            "version": filter_consent_version if filter_consent_version is not None else consent_version,
            "source_record_counts": source_record_counts,
            "excluded_record_counts": {
                "sessions": source_record_counts["sessions"] - len(sessions),
                "events": source_record_counts["events"] - len(events),
                "feedback": source_record_counts["feedback"] - len(feedback_rows),
            },
            "selected_records_sha256": _sha256_bytes(_canonical_bytes(selected_rows)),
            "selected_hash_encoding": "canonical_json_array_in_source_order",
        },
        "record_counts": {
            "sessions": len(sessions),
            "events": len(events),
            "feedback": len(feedback_rows),
            "annotation_tasks": len(task_rows),
            "route2_unlabeled": len(route2_rows),
        },
        "split_policy": {
            "seed": split_seed,
            "unit": "participant_component_joined_by_exact_normalized_text",
            "assignment": "append_only_registry_then_stable_hash_for_new_components",
            "train_dev_frozen_test": [0.8, 0.1, 0.1],
            "participant_overlap": 0,
            "session_overlap": 0,
            "normalized_text_overlap": 0,
            "split_task_counts": dict(Counter(row["split"] for row in join_rows)),
            "registry_input": (
                str(split_registry_path.resolve())
                if split_registry_path is not None
                else None
            ),
            "registry_input_sha256": split_registry_input_sha256,
            "existing_assignments_migrated": False,
        },
        "annotation_blinding": {
            "task_contains_online_prediction": False,
            "prediction_provenance_file": str(join_path.resolve()),
            "prediction_provenance_is_gold": False,
        },
        "route2": {
            "all_rows_supervised": False,
            "model_predictions_used_as_reward_targets": False,
            "missing_trajectory_feature_rows": sum(
                row["trajectory_source"] == "missing" for row in route2_rows
            ),
        },
        "outputs": {
            "annotation_tasks": {
                "path": str(task_path.resolve()),
                "sha256": _sha256_file(task_path),
            },
            "annotation_join": {
                "path": str(join_path.resolve()),
                "sha256": _sha256_file(join_path),
            },
            "route2_unlabeled": {
                "path": str(route2_path.resolve()),
                "sha256": _sha256_file(route2_path),
            },
            "split_registry": {
                "path": str(registry_output_path.resolve()),
                "sha256": _sha256_file(registry_output_path),
            },
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _verify_prepare_manifest_bindings(
    prepare_manifest_path: Path,
    task_path: Path,
    join_path: Path,
) -> dict:
    manifest = _read_json_object(prepare_manifest_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("stage") != "prepared_for_blind_human_annotation"
    ):
        raise ValueError("unsupported prepare manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("prepare manifest has no outputs")
    for key, path, label in (
        ("annotation_tasks", task_path, "annotation tasks"),
        ("annotation_join", join_path, "annotation join"),
    ):
        entry = outputs.get(key)
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(
            expected_hash
        ):
            raise ValueError(f"prepare manifest has no valid {label} hash")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"{label} hash does not match prepare manifest")
    return manifest


def finalize_classifier_annotations(
    task_path: Path,
    join_path: Path,
    annotations_path: Path,
    prepare_manifest_path: Path,
    output_dir: Path,
) -> dict:
    train_path = output_dir / "human_feedback_form_web_train.json"
    dev_path = output_dir / "human_feedback_form_web_dev.json"
    frozen_path = output_dir / "human_feedback_form_web_frozen_test.jsonl"
    manifest_path = output_dir / "classifier_split_manifest.json"
    _guard_no_input_output_collision(
        (task_path, join_path, annotations_path, prepare_manifest_path),
        (train_path, dev_path, frozen_path, manifest_path),
    )
    _verify_prepare_manifest_bindings(
        prepare_manifest_path, task_path, join_path
    )
    tasks = _index_unique(
        _read_jsonl(task_path), "annotation_id", label="annotation task"
    )
    joins = _index_unique(
        _read_jsonl(join_path), "annotation_id", label="annotation join"
    )
    annotations = _index_unique(
        _read_jsonl(annotations_path), "annotation_id", label="annotation"
    )
    if set(tasks) != set(joins) or set(tasks) != set(annotations):
        raise ValueError("task, join, and annotation ID sets must match exactly")

    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    allowed_annotation_fields = {
        "annotation_id",
        "classification_label",
        "review_status",
        "annotator_id",
        "annotation_revision",
        "adjudication_note",
    }
    for annotation_id in sorted(tasks):
        task = tasks[annotation_id]
        join = joins[annotation_id]
        annotation = annotations[annotation_id]
        if task.get("schema_version") != TASK_SCHEMA:
            raise ValueError(f"{annotation_id}: unsupported task schema")
        if join.get("schema_version") != JOIN_SCHEMA:
            raise ValueError(f"{annotation_id}: unsupported join schema")
        if _sha256_bytes(_canonical_bytes(task)) != join.get("annotation_task_sha256"):
            raise ValueError(f"{annotation_id}: task/join hash mismatch")
        if set(annotation) - allowed_annotation_fields:
            raise ValueError(f"{annotation_id}: annotation contains unsupported fields")
        if annotation.get("review_status") != "approved":
            raise ValueError(f"{annotation_id}: annotation is not approved")
        annotator_id = annotation.get("annotator_id")
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            raise ValueError(f"{annotation_id}: missing annotator_id")
        revision = annotation.get("annotation_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError(f"{annotation_id}: invalid annotation_revision")
        label = annotation.get("classification_label")
        if label not in CANONICAL_LABELS:
            raise ValueError(f"{annotation_id}: invalid classification_label")
        if task.get("classification_label") is not None:
            raise ValueError(f"{annotation_id}: source task was not blind")
        if task.get("language") != join.get("language"):
            raise ValueError(f"{annotation_id}: task/join language mismatch")
        split = join.get("split")
        if split not in {name for name, _ in SPLIT_THRESHOLDS}:
            raise ValueError(f"{annotation_id}: invalid split")
        language = str(task["language"])
        normalized = _normalized_text(language)
        rows_by_split[str(split)].append(
            {
                "feedback_id": annotation_id,
                "text": language,
                "normalized_text": normalized,
                "classification_label": label,
                "expected_feedback_type": INTERNAL_LABELS[str(label)],
                "template_family": normalized,
                "split": split,
                "source": "human",
                "label_source": "human_explicit",
                "group_id": f"web-participant:{join['participant_key']}",
                "session_id": join["session_id"],
                "annotation_revision": revision,
            }
        )

    text_sets = {
        split: {row["normalized_text"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    participant_sets = {
        split: {row["group_id"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    for left_index, (left, _) in enumerate(SPLIT_THRESHOLDS):
        for right, _ in SPLIT_THRESHOLDS[left_index + 1 :]:
            if text_sets.get(left, set()) & text_sets.get(right, set()):
                raise ValueError("annotation split has normalized-text leakage")
            if participant_sets.get(left, set()) & participant_sets.get(right, set()):
                raise ValueError("annotation split has participant leakage")

    for rows in rows_by_split.values():
        rows.sort(key=lambda row: (row["normalized_text"], row["feedback_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(train_path, rows_by_split.get("train", []))
    _write_json(dev_path, rows_by_split.get("dev", []))
    _write_jsonl(
        frozen_path,
        (
            {
                "language": row["text"],
                "classification_label": row["classification_label"],
            }
            for row in rows_by_split.get("frozen_test", [])
        ),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "human_labels_finalized",
        "human_labels_confirmed": True,
        "online_predictions_used_as_labels": False,
        "split_counts": {
            split: len(rows_by_split.get(split, []))
            for split, _ in SPLIT_THRESHOLDS
        },
        "input_hashes": {
            "prepare_manifest": _sha256_file(prepare_manifest_path),
            "tasks": _sha256_file(task_path),
            "join": _sha256_file(join_path),
            "annotations": _sha256_file(annotations_path),
        },
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": _sha256_file(train_path)},
            "dev": {"path": str(dev_path.resolve()), "sha256": _sha256_file(dev_path)},
            "frozen_test_input": {
                "path": str(frozen_path.resolve()),
                "sha256": _sha256_file(frozen_path),
            },
        },
        "next_step": (
            "Train with train/dev only. Seal the frozen-test input with "
            "prepare_human_feedback_form_holdout.py before opening it in evaluation."
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--export", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--split-seed", default="durf-web-session-split-v1")
    registry = prepare.add_mutually_exclusive_group(required=True)
    registry.add_argument(
        "--initialize-split-registry",
        action="store_true",
        help="Explicitly initialize the first append-only split registry.",
    )
    registry.add_argument(
        "--split-registry",
        type=Path,
        help="Previous run's append-only split_registry.json.",
    )
    consent = prepare.add_mutually_exclusive_group()
    consent.add_argument("--consent-version", help="Strictly reject feedback from other consent versions; never filter.")
    consent.add_argument("--filter-consent-version", help="Explicitly retain only matching sessions and their linked events/feedback; audit exclusions.")

    finalize = subparsers.add_parser("finalize-classifier")
    finalize.add_argument("--tasks", type=Path, required=True)
    finalize.add_argument("--join", type=Path, required=True)
    finalize.add_argument("--annotations", type=Path, required=True)
    finalize.add_argument("--prepare-manifest", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        report = prepare_export(
            args.export,
            args.output_dir,
            initialize_split_registry=args.initialize_split_registry,
            split_seed=args.split_seed,
            consent_version=args.consent_version,
            split_registry_path=args.split_registry,
            filter_consent_version=args.filter_consent_version,
        )
    else:
        report = finalize_classifier_annotations(
            args.tasks,
            args.join,
            args.annotations,
            args.prepare_manifest,
            args.output_dir,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
