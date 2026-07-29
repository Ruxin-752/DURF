"""Merge validated feedback corpora without duplicating scenario utterances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import read_json, write_json  # noqa: E402
from src.evaluation_splits import deduplicate_corpus  # noqa: E402


def merge(
    paths: list[Path],
    *,
    near_duplicate_threshold: float = 0.92,
    return_report: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    records = []
    for path in paths:
        records.extend(read_json(path))
    unique, report = deduplicate_corpus(
        records, near_duplicate_threshold=near_duplicate_threshold
    )
    return (unique, report) if return_report else unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    args = parser.parse_args()
    rows, report = merge(
        args.inputs,
        near_duplicate_threshold=args.near_duplicate_threshold,
        return_report=True,
    )
    write_json(args.output, rows)
    if args.audit_report:
        write_json(args.audit_report, report)
    print(f"Merged {len(rows)} unique examples into {args.output}")
    print(
        f"  removed exact duplicates: {report['exact_duplicates_removed']}; "
        f"supervision conflicts: {report['supervision_conflict_count']}; "
        f"near-duplicate pairs: {report['near_duplicates']['count']}"
    )
    if args.audit_report:
        print(f"  audit report: {args.audit_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
