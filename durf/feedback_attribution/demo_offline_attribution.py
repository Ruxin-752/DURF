"""Run the first offline attribution pipeline on a saved pygame session.

This script is the minimum viable research-data pipeline:

1. Convert current CSV logs to JSONL.
2. Generate candidate collaboration events.
3. Align feedback events to nearby candidate events.
4. Produce preview attribution records.
5. Build Hu provenance records and pairwise subgoal labels.
6. Detect reusable probe states for Hu-before/Hu-after evaluation.

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
from .hu_dataset_builder import build_hu_dataset
from .schemas import ATTRIBUTOR_RULE_FALLBACK
from .io_utils import read_jsonl, write_jsonl
from .probe_state_detector import generate_probe_hits
from .sample_builder import build_preview_attribution
from .session_converter import convert_session


def run_demo(
    session_dir: Path,
    lookback_steps: int,
    *,
    user_id: str,
    use_llm: bool = False,
) -> dict[str, int]:
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
    llm_fallbacks = 0
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
                candidate_events=baseline.get("candidate_events") or [],
                baseline_attribution=baseline,
                lookback_steps=lookback_steps,
            )
            attributions.append(llm_attribution)
            llm_audits.append(audit)
        except Exception as exc:
            # The LLM failed; the rule baseline stands in -- but LABELLED as a
            # stand-in.  A silent substitution here is how an outage would
            # score a healthy feedback-conversion rate.
            fallback = dict(baseline)
            fallback["attributor"] = ATTRIBUTOR_RULE_FALLBACK
            fallback["attribution_error"] = str(exc)
            attributions.append(fallback)
            llm_fallbacks += 1
            llm_audits.append(
                {
                    "feedback_event_id": baseline.get("feedback_event_id"),
                    "status": "fallback_to_baseline",
                    "attributor": ATTRIBUTOR_RULE_FALLBACK,
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
    hu_counts = build_hu_dataset(session_dir, user_id=user_id)
    probe_counts = generate_probe_hits(session_dir, convert_csv=False)
    return {
        **convert_counts,
        "candidate_events": event_counts["candidate_events"],
        "attributions": attribution_count,
        "llm_audits": audit_count,
        "hu_provenance": hu_counts["provenance_records"],
        "hu_training_samples": hu_counts["hu_training_samples"],
        "schema_updates": hu_counts["schema_updates"],
        "probe_hits": probe_counts["probe_hits"],
        "review_decisions_consumed": hu_counts["review_decisions_consumed"],
        "llm_fallbacks": llm_fallbacks,
        "reviewed_training_records": hu_counts["reviewed_training_records"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--user-id", default="PILOT01")
    parser.add_argument("--lookback-steps", type=int, default=DEFAULT_LOOKBACK_STEPS)
    llm = parser.add_mutually_exclusive_group()
    llm.add_argument(
        "--use-llm",
        dest="use_llm",
        action="store_true",
        default=True,
        help="Call DeepSeek for semantic attribution (DEFAULT since 2026-09-05). "
             "On error the rule baseline stands in, labelled attributor="
             "rule_fallback_after_llm_error.",
    )
    llm.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Deterministic keyword baseline only. Labels carry attributor="
             "rule_baseline and cannot support any claim about the LLM.",
    )
    args = parser.parse_args()

    counts = run_demo(
        args.session,
        args.lookback_steps,
        user_id=args.user_id,
        use_llm=args.use_llm,
    )
    print(f"Session: {args.session}")
    print(f"trajectory.jsonl records: {counts['trajectory']}")
    print(f"feedback_events.jsonl records: {counts['feedback']}")
    print(f"candidate_events.jsonl records: {counts['candidate_events']}")
    print(f"attribution_preview.jsonl records: {counts['attributions']}")
    print(f"hu_attribution_provenance.jsonl records: {counts['hu_provenance']}")
    print(f"hu_subgoal_preferences.jsonl records: {counts['hu_training_samples']}")
    print(f"schema_updates.jsonl records: {counts['schema_updates']}")
    print(f"probe_hits.jsonl records: {counts['probe_hits']}")
    print(
        "review decisions consumed: "
        f"{counts['review_decisions_consumed']} "
        f"(training={counts['reviewed_training_records']})"
    )
    summary_path = args.session / "hu_dataset_summary.json"
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            "training label split: "
            f"event-grounded={summary.get('event_grounded_training_records', 0)}, "
            f"direct-preference={summary.get('direct_preference_training_records', 0)}"
        )
    if args.use_llm:
        print(f"llm_attribution_audit.jsonl records: {counts['llm_audits']}")
        if counts["llm_fallbacks"]:
            print(
                f"WARNING: LLM failed on {counts['llm_fallbacks']}/{counts['feedback']} "
                "feedback events; those labels carry attributor="
                "rule_fallback_after_llm_error and must not be counted as LLM output."
            )
    else:
        print("NOTE: --no-llm: every label carries attributor=rule_baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
