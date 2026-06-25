"""Run the first offline attribution pipeline on a saved pygame session.

This script is the minimum viable research-data pipeline:

1. Convert current CSV logs to JSONL.
2. Generate candidate collaboration events.
3. Align feedback events to nearby candidate events.
4. Produce preview attribution records.

By default this uses the deterministic rule baseline. Pass `--use-llm` to call
DeepSeek for semantic attribution; the script falls back to the rule baseline if
the LLM is unavailable or returns invalid output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .llm_attributor import run_llm_attribution
from .event_detectors import DEFAULT_LOOKBACK_STEPS
from .generate_candidate_events import generate_candidate_events
from .io_utils import read_jsonl, write_jsonl
from .sample_builder import build_preview_attribution
from .session_converter import convert_session


def run_demo(session_dir: Path, lookback_steps: int, use_llm: bool = False) -> dict[str, int]:
    convert_counts = convert_session(session_dir)
    event_counts = generate_candidate_events(
        session_dir,
        window_steps=lookback_steps,
        convert_csv=False,
    )
    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    feedback_events = read_jsonl(session_dir / "feedback_events.jsonl")
    candidate_events = read_jsonl(session_dir / "candidate_events.jsonl")
    attributions = []
    llm_audits = []
    for feedback in feedback_events:
        baseline = build_preview_attribution(
            feedback=feedback,
            trajectory=trajectory,
            candidate_events=candidate_events,
            lookback_steps=lookback_steps,
        )
        if not use_llm:
            attributions.append(baseline)
            continue
        try:
            llm_attribution, audit = run_llm_attribution(
                feedback=feedback,
                trajectory=trajectory,
                candidate_events=candidate_events,
                baseline_attribution=baseline,
                lookback_steps=lookback_steps,
            )
            attributions.append(llm_attribution)
            llm_audits.append(audit)
        except Exception as exc:
            attributions.append(baseline)
            llm_audits.append(
                {
                    "feedback_event_id": baseline.get("feedback_event_id"),
                    "status": "fallback_to_baseline",
                    "error": str(exc),
                }
            )
    attribution_count = write_jsonl(
        session_dir / "attribution_preview.jsonl",
        attributions,
    )
    audit_count = 0
    if use_llm:
        audit_count = write_jsonl(
            session_dir / "llm_attribution_audit.jsonl",
            llm_audits,
        )
    return {
        **convert_counts,
        "candidate_events": event_counts["candidate_events"],
        "attributions": attribution_count,
        "llm_audits": audit_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--lookback-steps", type=int, default=DEFAULT_LOOKBACK_STEPS)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Call DeepSeek for semantic attribution. Falls back to rule baseline on error.",
    )
    args = parser.parse_args()

    counts = run_demo(args.session, args.lookback_steps, use_llm=args.use_llm)
    print(f"Session: {args.session}")
    print(f"trajectory.jsonl records: {counts['trajectory']}")
    print(f"feedback_events.jsonl records: {counts['feedback']}")
    print(f"candidate_events.jsonl records: {counts['candidate_events']}")
    print(f"attribution_preview.jsonl records: {counts['attributions']}")
    if args.use_llm:
        print(f"llm_attribution_audit.jsonl records: {counts['llm_audits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
