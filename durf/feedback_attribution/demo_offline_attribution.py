"""Run the first offline attribution pipeline on a saved pygame session.

This script is the minimum viable research-data pipeline:

1. Convert current CSV logs to JSONL.
2. Build recent windows around feedback events.
3. Produce preview attribution records.

It does not call an LLM yet. The output is deliberately conservative and marks
missing state facts so the next engineering step is obvious.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generate_candidate_events import generate_candidate_events
from .io_utils import read_jsonl, write_jsonl
from .sample_builder import build_preview_attribution
from .session_converter import convert_session


def run_demo(session_dir: Path, lookback_steps: int) -> dict[str, int]:
    convert_counts = convert_session(session_dir)
    event_counts = generate_candidate_events(
        session_dir,
        window_steps=lookback_steps,
        convert_csv=False,
    )
    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    feedback_events = read_jsonl(session_dir / "feedback_events.jsonl")
    attributions = [
        build_preview_attribution(
            feedback=feedback,
            trajectory=trajectory,
            lookback_steps=lookback_steps,
        )
        for feedback in feedback_events
    ]
    attribution_count = write_jsonl(
        session_dir / "attribution_preview.jsonl",
        attributions,
    )
    return {
        **convert_counts,
        "candidate_events": event_counts["candidate_events"],
        "attributions": attribution_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--lookback-steps", type=int, default=8)
    args = parser.parse_args()

    counts = run_demo(args.session, args.lookback_steps)
    print(f"Session: {args.session}")
    print(f"trajectory.jsonl records: {counts['trajectory']}")
    print(f"feedback_events.jsonl records: {counts['feedback']}")
    print(f"candidate_events.jsonl records: {counts['candidate_events']}")
    print(f"attribution_preview.jsonl records: {counts['attributions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
