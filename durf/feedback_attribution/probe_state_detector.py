"""Detect reusable probe states from normalized trajectory logs.

Probe states are evaluation triggers, not training labels.  Human feedback may
inspire a probe, but once a probe is defined it should be detected directly from
state/condition features so we can compare Hu-before and Hu-after behavior even
when the user does not provide feedback at that exact moment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .condition_features import extract_condition_features
from .io_utils import read_jsonl, write_jsonl
from .session_converter import convert_session


PROBE_SCHEMA_VERSION = "probe-state-v0"


TASK_PROBE_DOMAIN = "task"
COORDINATION_PROBE_DOMAIN = "coordination"
PROBE_DOMAINS = (TASK_PROBE_DOMAIN, COORDINATION_PROBE_DOMAIN)


@dataclass(frozen=True)
class ProbeDefinition:
    name: str
    description: str
    required_conditions: dict[str, bool]
    preferred_subgoals: tuple[str, ...]
    rejected_subgoals: tuple[str, ...]
    source_events: tuple[str, ...]
    rationale: str
    # Which decision level produced the candidate pool this probe evaluates.
    # Task probes read ai_subgoal_candidates; coordination probes read
    # coordination_decision.candidates, whose options are bound to concrete
    # low-level actions at construction time.
    domain: str = TASK_PROBE_DOMAIN


DEFAULT_PROBES: tuple[ProbeDefinition, ...] = (
    ProbeDefinition(
        name="pot_ready_ai_should_get_dish",
        description="Soup is cooking/ready and AI is empty-handed, so plate work should be high priority.",
        required_conditions={
            "pot_cooking_or_ready": True,
            "ai_empty_handed": True,
        },
        preferred_subgoals=("GET_DISH",),
        rejected_subgoals=("WAIT", "GET_TOMATO", "GET_ONION"),
        source_events=(
            "AI_ignored_ready_or_nearly_ready_pot",
            "AI_missed_plate_pickup_opportunity",
        ),
        rationale="Evaluates whether Hu raises dish/soup subgoals in ready-pot contexts.",
    ),
    ProbeDefinition(
        name="human_path_conflict_ai_should_yield",
        description="Human is trying to move through a tile occupied or immediately blocked by AI.",
        required_conditions={
            "human_trying_to_pass": True,
            "ai_on_human_path": True,
        },
        preferred_subgoals=("YIELD",),
        rejected_subgoals=("CONTINUE_CURRENT_SUBGOAL",),
        source_events=("AI_blocked_human_path", "AI_failed_to_yield_or_clear_path"),
        rationale="Evaluates whether Hu can prefer cooperative yielding when the human path is blocked.",
        domain=COORDINATION_PROBE_DOMAIN,
    ),
    ProbeDefinition(
        name="ai_holding_unneeded_onion_should_put_down",
        description="AI is holding onion although the current recipe no longer needs onion.",
        required_conditions={
            "ai_has_onion": True,
            "recipe_needs_onion": False,
        },
        preferred_subgoals=("PUT_DOWN_OBJECT",),
        rejected_subgoals=("WAIT",),
        source_events=("AI_held_unneeded_object_too_long",),
        rationale="Evaluates whether Hu discourages carrying an ingredient that is no longer useful.",
    ),
    ProbeDefinition(
        name="ai_holding_unneeded_tomato_should_put_down",
        description="AI is holding tomato although the current recipe no longer needs tomato.",
        required_conditions={
            "ai_has_tomato": True,
            "recipe_needs_tomato": False,
        },
        preferred_subgoals=("PUT_DOWN_OBJECT",),
        rejected_subgoals=("WAIT",),
        source_events=("AI_held_unneeded_object_too_long",),
        rationale="Evaluates whether Hu discourages carrying an ingredient that is no longer useful.",
    ),
    ProbeDefinition(
        name="pot_needs_tomato_ai_should_get_tomato",
        description="Recipe still needs tomato and AI is empty-handed.",
        required_conditions={
            "recipe_needs_tomato": True,
            "ai_empty_handed": True,
            "pot_cooking_or_ready": False,
        },
        preferred_subgoals=("GET_TOMATO",),
        rejected_subgoals=("WAIT",),
        source_events=("AI_missed_useful_ingredient_pickup",),
        rationale="Sanity-check probe for task-preserving ingredient progress.",
    ),
    ProbeDefinition(
        name="pot_needs_onion_ai_should_get_onion",
        description="Recipe still needs onion and AI is empty-handed.",
        required_conditions={
            "recipe_needs_onion": True,
            "ai_empty_handed": True,
            "pot_cooking_or_ready": False,
        },
        preferred_subgoals=("GET_ONION",),
        rejected_subgoals=("WAIT",),
        source_events=("AI_missed_useful_ingredient_pickup",),
        rationale="Sanity-check probe for task-preserving ingredient progress.",
    ),
    ProbeDefinition(
        name="human_waiting_ai_should_prepare_next",
        description="Human is waiting near pot with dish while AI has free capacity.",
        required_conditions={
            "human_waiting_near_pot": True,
            "ai_empty_handed": True,
        },
        preferred_subgoals=("GET_TOMATO", "GET_ONION"),
        rejected_subgoals=("WAIT",),
        source_events=("AI_failed_to_prepare_ingredient_while_waiting",),
        rationale="Evaluates the preference that AI should use waiting time to prepare useful work.",
    ),
    ProbeDefinition(
        name="useful_object_adjacent_ai_should_consider_pickup",
        description="A useful object or dispenser is adjacent to an empty-handed AI.",
        required_conditions={
            "useful_object_adjacent": True,
            "ai_empty_handed": True,
        },
        preferred_subgoals=("GET_TOMATO", "GET_ONION", "GET_DISH"),
        rejected_subgoals=("WAIT",),
        source_events=("AI_missed_useful_ingredient_pickup",),
        rationale="Evaluates whether nearby useful objects can affect subgoal preference.",
    ),
)


def step_total(step: dict[str, Any]) -> int:
    return int(step.get("total_step") or 0)


def matches_required_conditions(
    condition_features: dict[str, Any],
    required_conditions: dict[str, bool],
) -> bool:
    for key, expected in required_conditions.items():
        if condition_features.get(key) is not expected:
            return False
    return True


def candidate_score(candidate: dict[str, Any]) -> float:
    final_score = candidate.get("final_score")
    if final_score is not None:
        return float(final_score)
    task_score = candidate.get("task_score")
    if task_score is not None:
        return float(task_score)
    return 0.0


def candidate_name(candidate: dict[str, Any]) -> str | None:
    """Name a candidate from either candidate-pool schema.

    Task candidates use the ``subgoal`` field; coordination candidates use
    ``option``.  Both names are matched directly against probe vocabularies.
    """
    raw = candidate.get("option")
    if raw is not None:
        return str(raw)
    raw = candidate.get("subgoal")
    if raw is not None:
        return str(raw)
    return None


def rank_candidate_subgoals(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=candidate_score, reverse=True)
    rows = []
    for index, candidate in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": index,
                "subgoal": candidate_name(candidate),
                "task_score": candidate.get("task_score"),
                "hu_score": candidate.get("hu_score"),
                "final_score": candidate.get("final_score"),
                "reason": candidate.get("reason"),
                "feasible": candidate.get("feasible", True),
            }
        )
    return rows


def best_rank_for_subgoals(
    ranked_candidates: list[dict[str, Any]],
    subgoals: tuple[str, ...],
) -> int | None:
    subgoal_set = set(subgoals)
    ranks = [
        int(candidate["rank"])
        for candidate in ranked_candidates
        if candidate.get("subgoal") in subgoal_set
    ]
    return min(ranks) if ranks else None


def probe_candidate_pool(
    step: dict[str, Any],
    probe: ProbeDefinition,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return (candidates, chosen) for the probe's decision domain.

    Task probes evaluate the subgoal pool produced by the task head
    (``ai_subgoal_candidates``) against the executed task subgoal.  Coordination
    probes evaluate the coordination option pool (``coordination_decision``)
    against the option actually executed, because coord candidates are bound to
    concrete low-level actions at construction time and never appear in the
    task pool.  When a coordination probe matches but the step has no
    coordination decision record, the hit is emitted with an
    ``evaluation_unavailable`` marker instead of silently evaluating an empty
    pool.
    """
    if probe.domain == COORDINATION_PROBE_DOMAIN:
        decision = step.get("coordination_decision") or {}
        candidates = decision.get("candidates") or []
        chosen = decision.get("selected")
        return candidates, chosen
    return step.get("ai_subgoal_candidates") or [], step.get("ai_subgoal")


def build_probe_hit(step: dict[str, Any], probe: ProbeDefinition) -> dict[str, Any]:
    condition_features = extract_condition_features(step)
    candidates, chosen_subgoal = probe_candidate_pool(step, probe)
    evaluation_unavailable = (
        probe.domain == COORDINATION_PROBE_DOMAIN and not candidates
    )
    ranked_candidates = rank_candidate_subgoals(candidates)
    top_candidate = ranked_candidates[0] if ranked_candidates else None
    preferred_rank = best_rank_for_subgoals(ranked_candidates, probe.preferred_subgoals)
    rejected_rank = best_rank_for_subgoals(ranked_candidates, probe.rejected_subgoals)
    return {
        "record_type": "probe_hit",
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "probe_name": probe.name,
        "description": probe.description,
        "source_events": list(probe.source_events),
        "rationale": probe.rationale,
        "domain": probe.domain,
        "episode": step.get("episode"),
        "episode_step": step.get("episode_step"),
        "total_step": step.get("total_step"),
        "layout": step.get("layout"),
        "required_conditions": dict(probe.required_conditions),
        "condition_features": condition_features,
        "preferred_subgoals": list(probe.preferred_subgoals),
        "rejected_subgoals": list(probe.rejected_subgoals),
        "chosen_subgoal": chosen_subgoal,
        "candidate_ranking": ranked_candidates,
        "evaluation": {
            "evaluation_unavailable": evaluation_unavailable,
            "top_candidate": top_candidate.get("subgoal") if top_candidate else None,
            "preferred_available": preferred_rank is not None,
            "best_preferred_rank": preferred_rank,
            "best_rejected_rank": rejected_rank,
            "chosen_is_preferred": chosen_subgoal in probe.preferred_subgoals,
            "chosen_is_rejected": chosen_subgoal in probe.rejected_subgoals,
        },
    }


def detect_probe_hits(
    trajectory: list[dict[str, Any]],
    *,
    probes: tuple[ProbeDefinition, ...] = DEFAULT_PROBES,
    min_gap_steps: int = 3,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    last_hit_step_by_key: dict[tuple[str, int], int] = {}
    for step in trajectory:
        conditions = extract_condition_features(step)
        episode = int(step.get("episode") or 0)
        total_step = step_total(step)
        for probe in probes:
            if not matches_required_conditions(conditions, probe.required_conditions):
                continue
            key = (probe.name, episode)
            last_hit_step = last_hit_step_by_key.get(key)
            if last_hit_step is not None and total_step - last_hit_step < min_gap_steps:
                continue
            hits.append(build_probe_hit(step, probe))
            last_hit_step_by_key[key] = total_step
    return hits


def generate_probe_hits(
    session_dir: Path,
    *,
    convert_csv: bool = True,
    min_gap_steps: int = 3,
) -> dict[str, int]:
    if convert_csv:
        convert_session(session_dir)

    trajectory = read_jsonl(session_dir / "trajectory.jsonl")
    hits = detect_probe_hits(trajectory, min_gap_steps=min_gap_steps)
    hit_count = write_jsonl(session_dir / "probe_hits.jsonl", hits)
    return {
        "trajectory": len(trajectory),
        "probe_hits": hit_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Use existing trajectory.jsonl instead of converting CSV first.",
    )
    parser.add_argument(
        "--min-gap-steps",
        type=int,
        default=3,
        help="Minimum gap before emitting the same probe again in one episode.",
    )
    args = parser.parse_args()

    counts = generate_probe_hits(
        args.session,
        convert_csv=not args.no_convert,
        min_gap_steps=args.min_gap_steps,
    )
    print(f"Session: {args.session}")
    print(f"trajectory.jsonl records: {counts['trajectory']}")
    print(f"probe_hits.jsonl records: {counts['probe_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
