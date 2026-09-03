"""Build Hu_general training dataset from invariant sim samples (protocol-v2 P2).

Reads hu_general_invariant.jsonl produced by audit_sim_personas.py, validates
sample quality, and writes the final training file consumable by
train_subgoal_reranker.py.

Protocol-v2 rules:
  - Only invariant samples (cross-persona direction-consistent) enter Hu_general.
  - Task and Coordination heads are trained separately.
  - Persona-conflicting samples are excluded.
  - label_status must be "automatic" (sim has no human review).
  - source must be "synthetic_sim_human".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from durf.feedback_attribution.io_utils import read_jsonl, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Required fields for each pairwise sample to be valid.
_REQUIRED_SAMPLE_FIELDS = (
    "record_type",
    "user_id",
    "decision_level",
    "preferred_subgoal",
    "rejected_subgoal",
    "condition_features",
)

# Allowed decision levels.
_VALID_DECISION_LEVELS = {"task", "coordination"}


def _validate_sample(sample: dict[str, Any]) -> list[str]:
    """Return list of validation errors. Empty list means valid."""
    errors: list[str] = []
    for field in _REQUIRED_SAMPLE_FIELDS:
        if field not in sample:
            errors.append(f"missing field: {field}")

    if sample.get("record_type") != "hu_pairwise_subgoal_preference":
        errors.append(
            f"wrong record_type: {sample.get('record_type')}"
        )

    if sample.get("decision_level") not in _VALID_DECISION_LEVELS:
        errors.append(
            f"invalid decision_level: {sample.get('decision_level')}"
        )

    if sample.get("preferred_subgoal") == sample.get("rejected_subgoal"):
        errors.append("preferred_subgoal == rejected_subgoal")

    if not sample.get("condition_features"):
        errors.append("empty condition_features")

    # Protocol-v2: only synthetic sim data, only automatic labels.
    # Accept None/empty for backward compatibility with pre-schema sessions.
    sample_source = sample.get("source") or "synthetic_sim_human"
    if sample_source != "synthetic_sim_human":
        errors.append(
            f"wrong source for Hu_general: {sample.get('source')}"
        )

    sample_label = sample.get("label_status") or "automatic"
    if sample_label != "automatic":
        errors.append(
            f"wrong label_status for Hu_general: {sample.get('label_status')}"
        )

    return errors


def build_general_prior_dataset(
    invariant_path: Path,
    *,
    output_dir: Path | None = None,
    min_samples_per_domain: int = 2,
) -> dict[str, Any]:
    """Validate and split invariant samples into task/coordination training files.

    Args:
        invariant_path: Path to hu_general_invariant.jsonl from audit step.
        output_dir: Directory to write hu_general_task.jsonl and
                    hu_general_coord.jsonl.
        min_samples_per_domain: Minimum samples required per decision level.

    Returns:
        Summary dict with sample counts, validation errors, and file paths.
    """
    if not invariant_path.exists():
        raise FileNotFoundError(f"Invariant file not found: {invariant_path}")

    all_samples = read_jsonl(invariant_path)
    valid_samples: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()

    for i, sample in enumerate(all_samples):
        errors = _validate_sample(sample)
        if errors:
            validation_errors.append({
                "index": i,
                "sample_id": sample.get("sample_id", f"index_{i}"),
                "errors": errors,
            })
        else:
            valid_samples.append(sample)
            domain_counts[sample["decision_level"]] += 1

    # Split by decision level
    task_samples = [
        s for s in valid_samples if s["decision_level"] == "task"
    ]
    coord_samples = [
        s for s in valid_samples if s["decision_level"] == "coordination"
    ]

    summary = {
        "total_input_samples": len(all_samples),
        "valid_samples": len(valid_samples),
        "validation_errors": len(validation_errors),
        "task_samples": len(task_samples),
        "coordination_samples": len(coord_samples),
        "task_sufficient": len(task_samples) >= min_samples_per_domain,
        "coordination_sufficient": len(coord_samples) >= min_samples_per_domain,
        "error_details": validation_errors[:20],  # first 20 for review
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "hu_general_task.jsonl", task_samples)
        write_jsonl(output_dir / "hu_general_coordination.jsonl", coord_samples)
        (output_dir / "hu_general_build_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["output_dir"] = str(output_dir)
        summary["output_files"] = {
            "task": str(output_dir / "hu_general_task.jsonl"),
            "coordination": str(output_dir / "hu_general_coordination.jsonl"),
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--invariant-file",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "hu_general" / "filtered_training"
            / "hu_general_invariant.jsonl"
        ),
        help="Path to hu_general_invariant.jsonl from audit step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "hu_general" / "filtered_training",
        help="Directory for hu_general_task.jsonl and hu_general_coordination.jsonl.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples required per decision level.",
    )
    args = parser.parse_args()

    summary = build_general_prior_dataset(
        args.invariant_file,
        output_dir=args.output_dir,
        min_samples_per_domain=args.min_samples,
    )

    print(f"=== Hu_general Dataset Build ===")
    print(f"Input samples:  {summary['total_input_samples']}")
    print(f"Valid samples:  {summary['valid_samples']}")
    print(f"  Task:          {summary['task_samples']} "
          f"({'sufficient' if summary['task_sufficient'] else 'INSUFFICIENT'})")
    print(f"  Coordination: {summary['coordination_samples']} "
          f"({'sufficient' if summary['coordination_sufficient'] else 'INSUFFICIENT'})")
    print(f"Validation errors: {summary['validation_errors']}")

    if summary["validation_errors"] > 0:
        print(f"\nFirst 5 errors:")
        for err in summary["error_details"][:5]:
            print(f"  Sample {err['sample_id']}: {err['errors']}")

    if not summary["task_sufficient"]:
        print("\nWARNING: Insufficient task samples for Hu_general training!")
    if not summary["coordination_sufficient"]:
        print("WARNING: Insufficient coordination samples for Hu_general training!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
