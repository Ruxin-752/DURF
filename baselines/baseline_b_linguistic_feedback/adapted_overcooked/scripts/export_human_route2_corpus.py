"""Export normalized human sessions into supervised and unlabeled Route 2 pools.

Natural language by itself is never a 53-dimensional reward target. A row is
paper-supervised only when the session has an independently supplied complete
reward configuration; model predictions and recovered posteriors are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from numbers import Real
from pathlib import Path


ADAPTED_ROOT = Path(__file__).resolve().parents[1]
if str(ADAPTED_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTED_ROOT))

from src.feature_schema import load_features  # noqa: E402
from src.trajectory_featurizer import featurize_trajectory_steps  # noqa: E402


ALLOWED_INDEPENDENT_TARGET_SOURCES = frozenset(
    {
        "experiment_assigned_reward_config",
        "independent_reward_annotation",
    }
)
PROHIBITED_TARGET_TERMS = (
    "model_prediction",
    "prediction",
    "pseudo",
    "recovered",
    "posterior",
    "route2_inference",
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fallback_event_id(
    session_id: str,
    event: dict,
    occurrence: int,
) -> str:
    identity = {
        "session_id": session_id,
        "episode": int(event.get("episode", 0) or 0),
        "episode_step": int(event.get("episode_step", 0) or 0),
        "total_step": int(event.get("total_step", 0) or 0),
        "timestamp_utc": str(event.get("timestamp_utc") or ""),
        "text": str(event.get("feedback_text") or "").strip(),
        "occurrence": occurrence,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "feedback_" + hashlib.sha256(encoded).hexdigest()[:24]


def _normalize_supervision(
    raw: object,
    *,
    manifest: dict,
    features: list[str],
) -> dict | None:
    """Return a checked independent target, or None for a true unlabeled session."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("session supervision must be a JSON object")
    if isinstance(raw.get("supervision"), dict):
        raw = raw["supervision"]

    weights = raw.get("teacher_reward_weights")
    reward_config_id = str(raw.get("reward_config_id") or "").strip()
    if weights is None and not reward_config_id:
        return None
    if not isinstance(weights, dict) or not reward_config_id:
        raise ValueError(
            "supervised human Route 2 export requires both reward_config_id and "
            "teacher_reward_weights"
        )

    target_source = str(raw.get("target_source") or raw.get("kind") or "").strip()
    lowered = target_source.lower()
    if any(term in lowered for term in PROHIBITED_TARGET_TERMS):
        raise ValueError("model predictions/recovered posteriors cannot be reward gold")
    if target_source not in ALLOWED_INDEPENDENT_TARGET_SOURCES:
        raise ValueError(
            "target_source must explicitly identify an independent target: "
            + ", ".join(sorted(ALLOWED_INDEPENDENT_TARGET_SOURCES))
        )

    expected = set(features)
    actual = {str(key) for key in weights}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"teacher_reward_weights must cover the exact feature schema; "
            f"missing={missing}, extra={extra}"
        )
    normalized_weights: dict[str, float] = {}
    for feature in features:
        value = weights[feature]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"teacher_reward_weights.{feature} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"teacher_reward_weights.{feature} must be finite")
        normalized_weights[feature] = numeric

    teacher_id = str(raw.get("teacher_id") or manifest.get("teacher_id") or "").strip()
    if not teacher_id:
        raise ValueError("supervised human Route 2 export requires teacher_id")
    split = raw.get("split")
    if split is not None and split not in {"train", "dev", "test"}:
        raise ValueError("supervision split must be train, dev, or test")
    return {
        "teacher_id": teacher_id,
        "reward_config_id": reward_config_id,
        "teacher_reward_weights": normalized_weights,
        "target_source": target_source,
        "split": split,
    }


def export_session(
    session: str | Path,
    *,
    reward_assignment_path: str | Path | None = None,
    lookback: int = 25,
    overwrite: bool = False,
) -> dict:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    session = Path(session).resolve()
    trajectory_path = session / "trajectory.jsonl"
    feedback_path = session / "feedback_events.jsonl"
    for required in (trajectory_path, feedback_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} (run durf.feedback_attribution.session_converter first)"
            )

    output_paths = {
        "unlabeled": session / "human_route2_unlabeled.json",
        "supervised": session / "human_route2_paper_supervised.json",
        "report": session / "human_route2_export_report.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "human Route 2 export already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    manifest_path = session / "session_manifest.json"
    manifest_raw = _read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest_raw, dict):
        raise ValueError("session_manifest.json must be an object")
    manifest = manifest_raw
    session_id = str(manifest.get("session_id") or session.name)
    features = sorted(load_features())
    if reward_assignment_path is not None:
        assignment_path = Path(reward_assignment_path).resolve()
        if not assignment_path.is_file():
            raise FileNotFoundError(assignment_path)
        raw_supervision = _read_json(assignment_path)
    else:
        assignment_path = None
        raw_supervision = manifest.get("supervision")
    supervision = _normalize_supervision(
        raw_supervision,
        manifest=manifest,
        features=features,
    )

    trajectories = _read_jsonl(trajectory_path)
    events = [
        event
        for event in _read_jsonl(feedback_path)
        if event.get("role") == "human_language"
        and str(event.get("feedback_text") or "").strip()
    ]
    events.sort(
        key=lambda event: (
            int(event.get("episode", 0) or 0),
            int(event.get("total_step", 0) or 0),
            str(event.get("timestamp_utc") or ""),
            str(event.get("feedback_event_id") or ""),
        )
    )

    occurrences: Counter[tuple] = Counter()
    unlabeled_rows: list[dict] = []
    supervised_rows: list[dict] = []
    for event in events:
        episode = int(event.get("episode", 0) or 0)
        total_step = int(event.get("total_step", 0) or 0)
        text = str(event.get("feedback_text") or "").strip()
        identity = (
            episode,
            int(event.get("episode_step", 0) or 0),
            total_step,
            str(event.get("timestamp_utc") or ""),
            text,
        )
        occurrences[identity] += 1
        event_id = str(event.get("feedback_event_id") or "").strip() or _fallback_event_id(
            session_id, event, occurrences[identity]
        )
        eligible = [
            row
            for row in trajectories
            if int(row.get("episode", -1) or -1) == episode
            and int(row.get("total_step", -1) or -1) <= total_step
        ]
        eligible.sort(
            key=lambda row: (
                int(row.get("total_step", 0) or 0),
                str(row.get("timestamp_utc") or ""),
            )
        )
        window = eligible[-lookback:]
        trajectory_features = featurize_trajectory_steps(window)
        extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
        base = {
            "feedback_id": event_id,
            "feedback_event_id": event_id,
            "text": text,
            "source": "human_session",
            "session_id": session_id,
            "group_id": f"{session_id}:episode_{episode}",
            "episode": episode,
            "episode_step": int(event.get("episode_step", 0) or 0),
            "total_step": total_step,
            "timestamp_utc": event.get("timestamp_utc"),
            "route2_trajectory_features": trajectory_features,
            "trajectory_window": len(window),
            "audit_metadata": {
                "trajectory_source": "executed_steps",
                "trajectory_first_step": (
                    int(window[0].get("total_step", 0) or 0) if window else None
                ),
                "trajectory_last_step": (
                    int(window[-1].get("total_step", 0) or 0) if window else None
                ),
                "feedback_source": event.get("source"),
                # These may be online classifier outputs, so keep them as
                # provenance only; never expose them as reference-type gold.
                "online_inferred_reference_types": list(
                    extra.get("reference_types") or []
                ),
            },
        }
        if supervision is None:
            unlabeled_rows.append(
                {
                    **base,
                    "teacher_id": manifest.get("teacher_id"),
                    "supervision_status": "unlabeled_no_independent_reward_target",
                    "eligible_for_route2_supervised_training": False,
                }
            )
        else:
            row = {
                **base,
                "teacher_id": supervision["teacher_id"],
                "cv_teacher_id": supervision["teacher_id"],
                "reward_config_id": supervision["reward_config_id"],
                "teacher_reward_weights": supervision["teacher_reward_weights"],
                "target_provenance": supervision["target_source"],
                "supervision_status": "paper_supervised_full_teacher_reward",
                "eligible_for_route2_supervised_training": True,
            }
            if supervision["split"] is not None:
                row["split"] = supervision["split"]
            supervised_rows.append(row)

    _write_json(output_paths["unlabeled"], unlabeled_rows)
    _write_json(output_paths["supervised"], supervised_rows)
    report = {
        "schema_version": 1,
        "session_id": session_id,
        "trajectory_count": len(trajectories),
        "human_language_count": len(events),
        "unlabeled_count": len(unlabeled_rows),
        "paper_supervised_count": len(supervised_rows),
        "lookback": lookback,
        "target_source": supervision["target_source"] if supervision else None,
        "reward_config_id": supervision["reward_config_id"] if supervision else None,
        "teacher_id": (
            supervision["teacher_id"] if supervision else manifest.get("teacher_id")
        ),
        "model_outputs_used_as_gold": False,
        "trajectory_sha256": _sha256(trajectory_path),
        "feedback_sha256": _sha256(feedback_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "reward_assignment_path": str(assignment_path) if assignment_path else None,
        "reward_assignment_sha256": _sha256(assignment_path) if assignment_path else None,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in output_paths.items()
            if name != "report"
        },
    }
    _write_json(output_paths["report"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--reward-assignment", type=Path)
    parser.add_argument("--lookback", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = export_session(
        args.session,
        reward_assignment_path=args.reward_assignment,
        lookback=args.lookback,
        overwrite=args.overwrite,
    )
    print(f"Session: {Path(args.session).resolve()}")
    print(f"Unlabeled rows: {report['unlabeled_count']}")
    print(f"Paper-supervised rows: {report['paper_supervised_count']}")
    print(f"Report: {Path(args.session).resolve() / 'human_route2_export_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
