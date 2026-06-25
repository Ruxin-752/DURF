"""Generate candidate event records from a saved pygame session."""

from __future__ import annotations

import argparse
from pathlib import Path

from .event_detectors import DEFAULT_LOOKBACK_STEPS, detect_candidate_events
from .io_utils import read_jsonl, write_jsonl
from .session_converter import convert_session


def generate_candidate_events(
    session_dir: Path,
    *,
    window_steps: int = DEFAULT_LOOKBACK_STEPS,
    convert_csv: bool = True,
) -> dict[str, int]:
    if convert_csv:
        convert_session(session_dir)

    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    all_events = detect_candidate_events(trajectory)

    event_count = write_jsonl(session_dir / "candidate_events.jsonl", all_events)
    return {
        "trajectory": len(trajectory),
        "candidate_events": event_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--window-steps", type=int, default=DEFAULT_LOOKBACK_STEPS)
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Use existing trajectory.jsonl instead of converting CSV first.",
    )
    args = parser.parse_args()

    counts = generate_candidate_events(
        args.session,
        window_steps=args.window_steps,
        convert_csv=not args.no_convert,
    )
    print(f"Session: {args.session}")
    print(f"trajectory.jsonl records: {counts['trajectory']}")
    print(f"candidate_events.jsonl records: {counts['candidate_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
