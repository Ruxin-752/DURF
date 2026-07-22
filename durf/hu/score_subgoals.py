"""Score candidate subgoals with a trained Hu-v0 reranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from durf.hu.subgoal_reranker import LinearSubgoalReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--user-id", default="PILOT01")
    condition_group = parser.add_mutually_exclusive_group(required=True)
    condition_group.add_argument(
        "--condition-json",
        help="JSON object containing condition features.",
    )
    condition_group.add_argument(
        "--condition-file",
        type=Path,
        help="Path to a JSON file containing condition features.",
    )
    parser.add_argument(
        "--candidate-subgoals",
        nargs="*",
        help="Optional candidate subgoals to rank. Defaults to all Hu subgoals.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = LinearSubgoalReranker.load(args.model)
    if args.condition_file:
        condition_features = json.loads(args.condition_file.read_text(encoding="utf-8-sig"))
    else:
        condition_features = json.loads(args.condition_json)
    ranked = model.rank_subgoals(
        user_id=args.user_id,
        condition_features=condition_features,
        candidate_subgoals=args.candidate_subgoals,
    )
    print(json.dumps(ranked, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
