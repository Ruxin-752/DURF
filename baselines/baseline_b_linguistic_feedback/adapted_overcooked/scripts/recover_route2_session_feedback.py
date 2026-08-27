"""Replay human language through the exact paper-style live Route 2 update.

The output is a resumable version-3 learner state. Neural predictions are
observations used to update that state; they are never exported as supervised
reward labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


ADAPTED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ADAPTED_ROOT.parents[2]
for import_root in (REPO_ROOT, ADAPTED_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent  # noqa: E402
from src.neural_inference import predictor_sha256  # noqa: E402


DEFAULT_MODEL = (
    ADAPTED_ROOT
    / "outputs"
    / "route2"
    / "paper_aligned_v5_seed137_selected"
    / "ensemble_manifest.json"
)
PAPER_OBSERVATION_PRECISION = 2.0
DEFAULT_ENSEMBLE_SIZE = 10
DEFAULT_OUTPUT_PREFIX = "route2_recovered_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lookback", type=int, default=25)
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="Optional v3 state captured before this session (auto-detected by default).",
    )
    parser.add_argument(
        "--expected-ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix; the v3 default never overwrites legacy v1 recovery files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _language_feedback(events: list[dict]) -> list[dict]:
    selected = [
        event
        for event in events
        if event.get("role") == "human_language"
        and str(event.get("feedback_text") or "").strip()
    ]
    return sorted(
        selected,
        key=lambda event: (
            int(event.get("episode", 0) or 0),
            int(event.get("total_step", 0) or 0),
            str(event.get("timestamp_utc") or ""),
            str(event.get("feedback_event_id") or ""),
        ),
    )


def _session_id(session: Path) -> str:
    manifest_path = session / "session_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and manifest.get("session_id"):
            return str(manifest["session_id"])
    return session.name


def _with_stable_feedback_event_ids(
    session: Path,
    events: list[dict],
) -> list[dict]:
    """Backfill the same deterministic IDs used by the session converter."""

    session_id = _session_id(session)
    occurrences: Counter[tuple] = Counter()
    normalized: list[dict] = []
    for event in events:
        copied = dict(event)
        identity = (
            str(event.get("timestamp_utc") or ""),
            int(event.get("episode", 0) or 0),
            int(event.get("episode_step", 0) or 0),
            int(event.get("total_step", 0) or 0),
            str(event.get("feedback_text") or "").strip(),
        )
        occurrences[identity] += 1
        event_id = str(event.get("feedback_event_id") or "").strip()
        if not event_id:
            canonical = {
                "session_id": session_id,
                "kind": "human_language",
                "timestamp_utc": identity[0],
                "episode": identity[1],
                "episode_step": identity[2],
                "total_step": identity[3],
                "content": identity[4],
                "occurrence": occurrences[identity],
            }
            encoded = json.dumps(
                canonical, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
            event_id = "feedback_" + hashlib.sha256(encoded).hexdigest()[:24]
        copied["feedback_event_id"] = event_id
        normalized.append(copied)
    return normalized


def _validate_initial_state(path: Path) -> None:
    state = json.loads(path.read_text(encoding="utf-8"))
    posterior = state.get("route2_posterior") or {}
    if int(state.get("version", -1)) != 3:
        raise ValueError("Route 2 recovery initial state must be version 3")
    if state.get("feedback_mode") != "route2":
        raise ValueError("Route 2 recovery initial state has the wrong feedback mode")
    if state.get("score_formula") != "w_dot_phi":
        raise ValueError("Route 2 recovery initial state is not w dot phi")
    if posterior.get("update_rule") != "paper_independent_gaussian":
        raise ValueError("Route 2 recovery refuses a legacy EMA initial state")
    if float(posterior.get("observation_precision", -1.0)) != PAPER_OBSERVATION_PRECISION:
        raise ValueError("Route 2 recovery requires the paper fixed precision of 2")


def recover_session(
    session: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    lookback: int = 25,
    initial_state_path: str | Path | None = None,
    expected_ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    overwrite: bool = False,
) -> dict:
    """Replay one normalized session with the same code path used online."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if expected_ensemble_size <= 0:
        raise ValueError("expected_ensemble_size must be positive")
    output_prefix = str(output_prefix).strip()
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a plain filename prefix")

    session = Path(session).resolve()
    model_path = Path(model_path).resolve()
    trajectory_path = session / "trajectory.jsonl"
    feedback_path = session / "feedback_events.jsonl"
    for required in (trajectory_path, feedback_path, model_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_paths = {
        "updates": session / f"{output_prefix}_updates.jsonl",
        "weights": session / f"{output_prefix}_weights.json",
        "state": session / f"{output_prefix}_state.json",
        "report": session / f"{output_prefix}_report.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "recovery outputs already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    if initial_state_path is None:
        auto_initial = session / "initial_learner_state.json"
        initial_state = auto_initial if auto_initial.is_file() else None
    else:
        initial_state = Path(initial_state_path).resolve()
        if not initial_state.is_file():
            raise FileNotFoundError(initial_state)
    if initial_state is not None:
        _validate_initial_state(initial_state)

    agent = ComfortSubgoalAgent(
        SimpleNamespace(mdp=object()),
        model_path=model_path,
        feedback_mode="route2",
        online_blend=None,
        route2_observation_precision=PAPER_OBSERVATION_PRECISION,
        route1_lookback=lookback,
        learner_state_path=output_paths["state"],
    )
    predictor = agent._ensure_route2()
    if predictor.get("kind") != "ensemble":
        raise ValueError("paper-aligned recovery requires a cross-validation ensemble")
    ensemble_size = len(predictor.get("members") or [])
    if ensemble_size != expected_ensemble_size:
        raise ValueError(
            f"expected {expected_ensemble_size} Route 2 folds, found {ensemble_size}"
        )
    if initial_state is not None:
        agent.load_learner_state(initial_state)

    trajectories = read_jsonl(trajectory_path)
    all_feedback = read_jsonl(feedback_path)
    language_feedback = _with_stable_feedback_event_ids(
        session, _language_feedback(all_feedback)
    )
    traces: list[dict] = []
    for index, feedback in enumerate(language_feedback, start=1):
        episode = int(feedback.get("episode", 0) or 0)
        total_step = int(feedback.get("total_step", 0) or 0)
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
        # Assign the exact rolling history that the live agent would hold.
        agent.trajectory_history = list(window)
        trace = agent.update_from_feedback(
            str(feedback.get("feedback_text") or ""),
            feedback_event_id=feedback["feedback_event_id"],
            source="human_live_recovered",
            confidence=1.0,
        )
        if (
            trace.get("status") == "updated"
            and trace.get("update_rule") != "paper_independent_gaussian"
        ):
            raise RuntimeError("recovery diverged from the paper Gaussian update")
        trace.update(
            {
                "recovery_index": index,
                "feedback_event_id": feedback.get("feedback_event_id"),
                "episode": episode,
                "episode_step": int(feedback.get("episode_step", 0) or 0),
                "total_step": total_step,
                "timestamp_utc": feedback.get("timestamp_utc"),
            }
        )
        traces.append(trace)

    # Also emit a valid neutral/initial v3 state when no utterances were found.
    agent.save_learner_state(output_paths["state"])
    state = json.loads(output_paths["state"].read_text(encoding="utf-8"))
    if int(state.get("version", -1)) != 3:
        raise RuntimeError("runtime learner did not emit a version-3 state")

    output_paths["updates"].write_text(
        "".join(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n" for trace in traces),
        encoding="utf-8",
    )
    posterior = state["route2_posterior"]
    write_json(output_paths["weights"], posterior["mean"])

    report = {
        "schema_version": 2,
        "session_id": _session_id(session),
        "trajectory_steps": len(trajectories),
        "feedback_event_count": len(all_feedback),
        "human_language_feedback_count": len(language_feedback),
        "replayed_feedback_count": len(traces),
        "recovered_update_count": sum(
            trace.get("status") == "updated" for trace in traces
        ),
        "ignored_feedback_count": sum(
            trace.get("status") == "ignored" for trace in traces
        ),
        "rejected_feedback_count": sum(
            str(trace.get("status") or "").startswith("rejected")
            for trace in traces
        ),
        "lookback": lookback,
        "model_path": str(model_path),
        "model_sha256": predictor_sha256(model_path),
        "predictor_kind": predictor["kind"],
        "ensemble_size": ensemble_size,
        "use_feature_counts": bool(predictor["use_feature_counts"]),
        "observation_precision": PAPER_OBSERVATION_PRECISION,
        "update_rule": "paper_independent_gaussian",
        "reference_classifier_mode": (
            "required_phrase_gate"
            if agent.require_reference_gate
            else "audit_only_paper_route2"
        ),
        "runtime_implementation": (
            "durf.baseline.comfort_subgoal_agent.ComfortSubgoalAgent.update_from_feedback"
        ),
        "state_version": state["version"],
        "trajectory_sha256": file_sha256(trajectory_path),
        "feedback_sha256": file_sha256(feedback_path),
        "initial_state_path": str(initial_state) if initial_state else None,
        "initial_state_sha256": file_sha256(initial_state) if initial_state else None,
        "outputs": {name: str(path) for name, path in output_paths.items() if name != "report"},
        "report_path": str(output_paths["report"]),
        "supervision_status": "personalized_inference_state_not_reward_gold",
        "training_eligibility": False,
        "note": (
            "Model outputs update the per-teacher posterior only. They must not be "
            "merged into Route 2 supervised targets."
        ),
    }
    write_json(output_paths["report"], report)
    return report


def main() -> int:
    args = parse_args()
    report = recover_session(
        args.session,
        model_path=args.model,
        lookback=args.lookback,
        initial_state_path=args.initial_state,
        expected_ensemble_size=args.expected_ensemble_size,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
    )
    print(f"Session: {Path(args.session).resolve()}")
    print(
        "Recovered updates: "
        f"{report['recovered_update_count']} / {report['human_language_feedback_count']}"
    )
    for name, path in report["outputs"].items():
        print(f"{name}: {path}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
