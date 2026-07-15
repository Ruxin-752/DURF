"""Convert an attributed human-AI session into Baseline B feedback examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import validate_feedback_examples, write_json  # noqa: E402
from src.probe_evaluator import load_probe_states  # noqa: E402
from src.session_bridge import build_session_feedback_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Include attribution records that still require clarification.",
    )
    args = parser.parse_args()

    examples = build_session_feedback_examples(
        args.session,
        include_ambiguous=args.include_ambiguous,
    )
    validate_feedback_examples(examples, probe_states=load_probe_states())
    output = args.output or args.session / "baseline_b_feedback_examples.json"
    write_json(output, examples)
    print(f"Wrote {len(examples)} Baseline B feedback examples")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
