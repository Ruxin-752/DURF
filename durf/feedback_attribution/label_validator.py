"""Schema gate for pairwise Hu training labels.

Protocol v2 S13 forbids two shapes of label, and the corpus contained both
before this gate existed (audit 2026-09-05):

* cross-domain pairs -- a task subgoal against a coordination option, e.g.
  ``(CONTINUE_CURRENT_SUBGOAL, GET_TOMATO)``: 8 records;
* subgoals the runtime can never offer -- ``GET_USEFUL_INGREDIENT`` is an
  attribution-level name that must be resolved to GET_TOMATO / GET_ONION
  before a pair is formed: 199 records trained a dimension nothing can select.

Neither is caught by the trainer, which will happily fit whatever it is given.
So the gate sits in front of it: ``filter_training_samples`` drops the invalid
records and returns WHY, and the counts go into ``hu_dataset_summary.json`` so
a silent 5% loss is visible.  ``python -m durf.feedback_attribution.label_validator``
audits an existing corpus without touching it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Any, Iterable

from .subgoal_preferences import (
    ATTRIBUTION_ONLY_TASK_SUBGOALS,
    COORDINATION_SUBGOALS,
    TASK_HU_SUBGOALS,
)

TASK_VOCAB = frozenset(TASK_HU_SUBGOALS) | {"WAIT"}
COORD_VOCAB = frozenset(COORDINATION_SUBGOALS)
UNREACHABLE = frozenset(ATTRIBUTION_ONLY_TASK_SUBGOALS)
KNOWN_ATTRIBUTORS = frozenset({"rule_baseline", "llm", "rule_fallback_after_llm_error"})


def validate_training_sample(sample: dict[str, Any]) -> list[str]:
    """Return the list of violations; empty means the sample is admissible."""
    problems: list[str] = []
    preferred = sample.get("preferred_subgoal")
    rejected = sample.get("rejected_subgoal")
    level = sample.get("decision_level")

    if not preferred or not rejected:
        problems.append("missing_side")
        return problems
    if preferred == rejected:
        problems.append("preferred_equals_rejected")

    for side, name in (("preferred", preferred), ("rejected", rejected)):
        if name in UNREACHABLE:
            problems.append(f"{side}_unreachable_at_runtime:{name}")
        elif name not in TASK_VOCAB and name not in COORD_VOCAB:
            problems.append(f"{side}_unknown_subgoal:{name}")

    domains = set()
    for name in (preferred, rejected):
        if name in COORD_VOCAB:
            domains.add("coordination")
        elif name in TASK_VOCAB:
            domains.add("task")
    if len(domains) > 1:
        problems.append("cross_domain_pair")
    elif domains and level and level not in domains:
        problems.append(f"decision_level_mismatch:{level}")

    attributor = sample.get("attributor")
    if attributor is None:
        problems.append("missing_attributor")
    elif attributor not in KNOWN_ATTRIBUTORS:
        problems.append(f"unknown_attributor:{attributor}")
    return problems


def filter_training_samples(
    samples: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[str]]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[tuple[dict[str, Any], list[str]]] = []
    for sample in samples:
        problems = validate_training_sample(sample)
        if problems:
            rejected.append((sample, problems))
        else:
            kept.append(sample)
    return kept, rejected


def rejection_summary(rejected: list[tuple[dict[str, Any], list[str]]]) -> dict[str, int]:
    counter: collections.Counter = collections.Counter()
    for _, problems in rejected:
        for problem in problems:
            counter[problem.split(":")[0]] += 1
    return dict(sorted(counter.items()))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit pairwise Hu labels against protocol v2 S13.")
    parser.add_argument(
        "--corpus", default="outputs/human_ai_sessions/*/hu_subgoal_preferences.jsonl",
        help="Glob of hu_subgoal_preferences.jsonl files (read-only).",
    )
    args = parser.parse_args(argv)
    total = 0
    all_rejected: list[tuple[dict, list[str]]] = []
    by_attributor: collections.Counter = collections.Counter()
    files = sorted(glob.glob(args.corpus))
    for path in files:
        rows = _read_jsonl(Path(path))
        total += len(rows)
        for row in rows:
            by_attributor[row.get("attributor", "<missing>")] += 1
        _, rejected = filter_training_samples(rows)
        all_rejected.extend(rejected)
    print(f"files: {len(files)}   labels: {total}   rejected: {len(all_rejected)}")
    print("by attributor:", dict(by_attributor))
    print("rejection reasons:", rejection_summary(all_rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
