"""Audit sim sessions for cross-persona preference consistency (protocol-v2 P1).

For each unique combination of (decision_level, active_condition_signature,
preferred_subgoal, rejected_subgoal), this script checks whether different
sim personas agree on the pairwise direction.

Categories:
  invariant           -- all personas agree (same direction)
  conflicting         -- at least two personas disagree (opposite direction)
  unsupported         -- only one persona observed this pair
  unsupported_persona -- observation exists but only in a single persona

Outputs:
  hu_general_invariant.jsonl     -- samples safe for Hu_general training
  sim_persona_conflicts.jsonl     -- samples with direction conflicts
  sim_general_audit.json          -- audit summary (counts, distributions)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from durf.feedback_attribution.io_utils import read_jsonl, write_jsonl
from durf.feedback_attribution.condition_features import MODEL_CONDITION_KEYS


CONDITION_KEYS = list(MODEL_CONDITION_KEYS)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _session_metadata(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "session_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _persona_from_session(session_dir: Path) -> str | None:
    meta = _session_metadata(session_dir)
    if meta is None:
        return None
    sim_human = meta.get("sim_human") or {}
    return sim_human.get("persona")


def _is_sim_session(session_dir: Path) -> bool:
    meta = _session_metadata(session_dir)
    if meta is None:
        return False
    return meta.get("data_source") == "synthetic_sim_human"


def condition_signature(
    condition_features: dict[str, Any],
) -> frozenset[tuple[str, int]]:
    """Return a hashable encoding of active (non-zero) condition values.

    Only conditions that are True (+1) or False (-1) contribute to the
    signature.  None and non-numeric values are excluded.
    """
    items: list[tuple[str, int]] = []
    for key, val in condition_features.items():
        if val is None:
            continue
        try:
            v = int(float(val))
        except (ValueError, TypeError):
            continue
        if v != 0:
            items.append((key, v))
    return frozenset(items)


def sample_key(
    sample: dict[str, Any],
) -> tuple:
    """Composite key for grouping pairwise samples across personas.

    The subgoal pair is **order-invariant** (frozenset) so that
    A > B  and  B > A  land in the same comparison group.
    This is essential for cross-persona conflict detection.

    Returns None if the sample does not have the required fields.
    """
    cf = sample.get("condition_features") or {}
    sig = condition_signature(cf)
    preferred = sample.get("preferred_subgoal")
    rejected = sample.get("rejected_subgoal")
    if not preferred or not rejected:
        return None
    return (
        sample.get("decision_level", "task"),
        sig,
        frozenset((preferred, rejected)),
    )


def _pair_direction(sample: dict[str, Any]) -> str:
    """Normalize direction string for comparison: 'A>B'."""
    return (
        f"{sample.get('preferred_subgoal')}>{sample.get('rejected_subgoal')}"
    )


def _invert_direction(direction: str) -> str:
    parts = direction.split(">")
    if len(parts) != 2:
        return direction
    return f"{parts[1]}>{parts[0]}"


def audit_sim_sessions(
    session_dirs: list[Path],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit pairwise samples across all sim persona sessions.

    Returns a dict with keys:
      invariant_samples, conflicting_samples, audit_summary
    """
    # group by sample_key -> {persona: direction}
    groups: dict[tuple, dict[str, str]] = defaultdict(dict)
    all_samples: list[dict[str, Any]] = []
    persona_counts: dict[str, int] = defaultdict(int)

    for session_dir in sorted(session_dirs):
        if not session_dir.is_dir():
            continue
        if not _is_sim_session(session_dir):
            continue

        persona = _persona_from_session(session_dir)
        if persona is None:
            print(f"  WARNING: {session_dir.name} has no persona in metadata, skipping")
            continue

        prefs_path = session_dir / "hu_subgoal_preferences.jsonl"
        if not prefs_path.exists():
            print(f"  WARNING: {session_dir.name} has no hu_subgoal_preferences.jsonl")
            continue

        samples = read_jsonl(prefs_path)
        for sample in samples:
            if sample.get("record_type") != "hu_pairwise_subgoal_preference":
                continue
            # Only consider automatically attributed sim samples
            # (not reviewed, since sim has no human review)
            key = sample_key(sample)
            if key is None:
                continue
            direction = _pair_direction(sample)
            groups[key][persona] = direction
            all_samples.append(sample)
            persona_counts[persona] += 1

    # classify each group
    invariant_samples: list[dict[str, Any]] = []
    conflicting_samples: list[dict[str, Any]] = []
    stats = {
        "total_samples": len(all_samples),
        "total_groups": len(groups),
        "invariant_groups": 0,
        "conflicting_groups": 0,
        "unsupported_groups": 0,
        "invariant_samples": 0,
        "conflicting_samples": 0,
        "unsupported_samples": 0,
        "by_decision_level": {
            "task": {"invariant": 0, "conflicting": 0, "unsupported": 0},
            "coordination": {"invariant": 0, "conflicting": 0, "unsupported": 0},
        },
        "persona_sample_counts": dict(persona_counts),
        "conflict_details": [],
    }

    # Build a sample index to quickly find samples matching a key
    key_to_samples: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        key = sample_key(sample)
        if key is not None:
            key_to_samples[key].append(sample)

    for key, persona_dirs in sorted(groups.items()):
        decision_level = key[0]
        directions = set(persona_dirs.values())
        # Also check for inverted directions (should be same group if inverted)
        inverted_keys = set()
        for d in directions:
            inverted_keys.add(_invert_direction(d))

        if len(persona_dirs) <= 1:
            # Only one persona observed this pair
            category = "unsupported"
        elif len(directions) == 1:
            # All personas agree on the same direction
            category = "invariant"
        elif len(directions) > 1:
            # Personas disagree
            category = "conflicting"

        if category == "invariant":
            stats["invariant_groups"] += 1
            stats["invariant_samples"] += len(key_to_samples[key])
            stats["by_decision_level"][decision_level]["invariant"] += 1
            invariant_samples.extend(key_to_samples[key])
        elif category == "conflicting":
            stats["conflicting_groups"] += 1
            stats["conflicting_samples"] += len(key_to_samples[key])
            stats["by_decision_level"][decision_level]["conflicting"] += 1
            conflicting_samples.extend(key_to_samples[key])
            stats["conflict_details"].append({
                "decision_level": decision_level,
                "subgoal_pair": sorted(list(key[2])),
                "persona_directions": dict(persona_dirs),
                "sample_count": len(key_to_samples[key]),
            })
        else:
            stats["unsupported_groups"] += 1
            stats["unsupported_samples"] += len(key_to_samples[key])
            stats["by_decision_level"][decision_level]["unsupported"] += 1

    # Write output files if output_dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            output_dir / "hu_general_invariant.jsonl", invariant_samples
        )
        write_jsonl(
            output_dir / "sim_persona_conflicts.jsonl", conflicting_samples
        )
        (output_dir / "sim_general_audit.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "invariant_samples": invariant_samples,
        "conflicting_samples": conflicting_samples,
        "audit_summary": stats,
    }


def find_sim_sessions(base_dir: Path) -> list[Path]:
    """Find all sim session directories under base_dir."""
    sessions: list[Path] = []
    if not base_dir.is_dir():
        return sessions
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and _session_metadata(child) is not None:
            sessions.append(child)
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "human_ai_sessions",
        help="Directory containing session subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "hu_general" / "filtered_training",
        help="Output directory for invariant/conflict files.",
    )
    parser.add_argument(
        "--session-ids",
        nargs="*",
        help="Specific session IDs to audit (default: all sim sessions).",
    )
    args = parser.parse_args()

    if args.session_ids:
        session_dirs = [
            args.sessions_dir / sid for sid in args.session_ids
        ]
    else:
        session_dirs = find_sim_sessions(args.sessions_dir)

    sim_dirs = [d for d in session_dirs if _is_sim_session(d)]
    if not sim_dirs:
        print("No sim sessions found.")
        return 1

    print(f"Auditing {len(sim_dirs)} sim sessions ...")
    result = audit_sim_sessions(sim_dirs, output_dir=args.output_dir)
    summary = result["audit_summary"]

    print(f"\n=== Audit Summary ===")
    print(f"Total pairwise samples: {summary['total_samples']}")
    print(f"Unique (condition, pair) groups: {summary['total_groups']}")
    print(f"  Invariant groups: {summary['invariant_groups']} "
          f"({summary['invariant_samples']} samples)")
    print(f"  Conflicting groups: {summary['conflicting_groups']} "
          f"({summary['conflicting_samples']} samples)")
    print(f"  Unsupported groups: {summary['unsupported_groups']} "
          f"({summary['unsupported_samples']} samples)")

    print(f"\nBy decision level:")
    for domain in ("task", "coordination"):
        dl = summary["by_decision_level"][domain]
        print(f"  {domain}: invariant={dl['invariant']}, "
              f"conflicting={dl['conflicting']}, unsupported={dl['unsupported']}")

    print(f"\nPersona sample counts:")
    for persona, count in sorted(summary["persona_sample_counts"].items()):
        print(f"  {persona}: {count}")

    if summary["conflict_details"]:
        print(f"\nConflict details (first 10):")
        for detail in summary["conflict_details"][:10]:
            print(f"  {detail['decision_level']}: "
                  f"{' vs '.join(detail['subgoal_pair'])}")
            for persona, direction in sorted(detail["persona_directions"].items()):
                print(f"    {persona}: {direction}")

    if args.output_dir:
        print(f"\nOutput written to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
