"""Small-sample simulated Hu training and protocol-v2 matrix evaluation.

This is a pilot-feasibility check, not evidence about real participants. It
uses one simulated persona, a small stratified training subset, and disjoint
held-out simulation seeds. The frozen Hu_general artifact is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from durf.feedback_attribution.io_utils import read_jsonl
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    COORDINATION_SUBGOALS,
    DECISION_LEVELS,
    HierarchicalHu,
    PairwiseSample,
    PerUserAdapter,
    TASK_DECISION_LEVEL,
    TASK_HU_SUBGOALS,
    load_pairwise_samples,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def session_meta(session_dir: Path) -> dict[str, Any]:
    return json.loads(
        (session_dir / "session_metadata.json").read_text(encoding="utf-8")
    )


def find_sessions(
    sessions_dir: Path,
    *,
    persona: str,
    train_seed: int,
    test_seeds: set[int],
) -> tuple[Path, list[Path]]:
    train_candidates: list[Path] = []
    test_dirs: list[Path] = []
    for child in sorted(sessions_dir.iterdir()):
        meta_path = child / "session_metadata.json"
        prefs_path = child / "hu_subgoal_preferences.jsonl"
        if not meta_path.exists() or not prefs_path.exists():
            continue
        meta = session_meta(child)
        if meta.get("data_source") != "synthetic_sim_human":
            continue
        sim = meta.get("sim_human") or {}
        if sim.get("persona") != persona:
            continue
        sim_seed = int(sim.get("seed", -1))
        top_seed = int(meta.get("seed", -1))
        # The current complete batch stores the same seed at both levels. This
        # excludes older diagnostic sessions whose environment seed was 42.
        if sim_seed != top_seed:
            continue
        if sim_seed == train_seed:
            train_candidates.append(child)
        elif sim_seed in test_seeds:
            test_dirs.append(child)
    if not train_candidates:
        raise RuntimeError(
            f"No complete {persona} sim session found for train seed {train_seed}."
        )
    # Prefer the candidate with the largest usable dataset if duplicates exist.
    train_dir = max(
        train_candidates,
        key=lambda p: len(read_jsonl(p / "hu_subgoal_preferences.jsonl")),
    )
    if not test_dirs:
        raise RuntimeError(
            f"No complete {persona} held-out sessions found for seeds {test_seeds}."
        )
    return train_dir, sorted(test_dirs)


def stratified_subset(
    samples: list[PairwiseSample], *, n: int, seed: int
) -> list[PairwiseSample]:
    if n >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    by_level = {
        level: [s for s in samples if s.decision_level == level]
        for level in DECISION_LEVELS
    }
    # Guarantee both heads receive examples; preserve the source distribution
    # as closely as possible for the remaining slots.
    selected: list[PairwiseSample] = []
    remaining = n
    for level in DECISION_LEVELS:
        pool = by_level[level]
        quota = min(len(pool), max(1, round(n * len(pool) / len(samples))))
        indices = rng.choice(len(pool), size=quota, replace=False)
        selected.extend(pool[int(i)] for i in indices)
        remaining -= quota
    if remaining > 0:
        chosen_ids = {id(s) for s in selected}
        pool = [s for s in samples if id(s) not in chosen_ids]
        indices = rng.choice(len(pool), size=remaining, replace=False)
        selected.extend(pool[int(i)] for i in indices)
    elif remaining < 0:
        indices = rng.choice(len(selected), size=n, replace=False)
        selected = [selected[int(i)] for i in indices]
    rng.shuffle(selected)
    return selected


def evaluate_bias_only(
    adapter: PerUserAdapter, samples: list[PairwiseSample]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in DECISION_LEVELS:
        domain = [s for s in samples if s.decision_level == level]
        margins = []
        for sample in domain:
            preferred = adapter.score(
                level, sample.condition_features, sample.preferred_subgoal
            )
            rejected = adapter.score(
                level, sample.condition_features, sample.rejected_subgoal
            )
            margins.append(
                preferred["general_score"]
                + preferred["user_bias_score"]
                - rejected["general_score"]
                - rejected["user_bias_score"]
            )
        result[level] = margin_metrics(margins)
    return result


def margin_metrics(margins: list[float]) -> dict[str, Any]:
    if not margins:
        return {"samples": 0, "pairwise_accuracy": None, "mean_margin": None}
    values = np.asarray(margins, dtype=np.float32)
    return {
        "samples": len(margins),
        "pairwise_accuracy": float((values > 0).mean()),
        "mean_margin": float(values.mean()),
        "min_margin": float(values.min()),
        "max_margin": float(values.max()),
    }


def template_expected_events(text: str) -> set[str]:
    lowered = text.strip().lower()
    if lowered in {
        "thanks, i'll squeeze past now",
        "good, you held still so i could get by",
        "nice, that let me through",
    }:
        return {"AI_successfully_yielded"}
    if lowered in {
        "you didn't need to move, you were one step from finishing",
        "you should have stayed, you were almost there",
    }:
        return {"AI_successfully_yielded"}
    if lowered in {
        "right, finish what you were doing",
        "good, keep going, don't mind me",
        "yes, you were closer to your task, stick with it",
    }:
        return {"AI_maintained_current_subgoal_during_conflict"}
    if lowered in {
        "you're blocking me, move",
        "you're in my way, step aside",
        "step aside, i can't pass",
    }:
        # Both detectors are valid observed representations of the same
        # obstruction, depending on exact one-step geometry.
        return {"AI_blocked_human_path", "AI_failed_to_yield_or_clear_path"}
    if lowered in {
        "the pot is ready, go get a dish",
        "soup is ready, you should get a dish",
    }:
        return {"AI_ignored_ready_or_nearly_ready_pot"}
    if lowered in {
        "you're waiting empty handed, go get the next ingredient",
        "prepare the next ingredient while the pot cooks",
    }:
        return {"AI_failed_to_prepare_ingredient_while_waiting"}
    return set()


def attribution_metrics(session_dirs: list[Path]) -> dict[str, Any]:
    total = covered = clarified = gold_known = gold_correct = 0
    with_conditions = 0
    for session_dir in session_dirs:
        feedback = read_jsonl(session_dir / "feedback_events.jsonl")
        attributions = read_jsonl(session_dir / "attribution_preview.jsonl")
        by_id = {a.get("feedback_event_id"): a for a in attributions}
        for item in feedback:
            event_id = (
                f"{item.get('source')}:{item.get('total_step')}:"
                f"{item.get('timestamp_utc')}"
            )
            result = by_id.get(event_id, {})
            total += 1
            covered += int(bool(result.get("target_event")))
            clarified += int(bool(result.get("needs_clarification")))
            with_conditions += int(bool(result.get("condition_features")))
            expected = template_expected_events(str(item.get("feedback_text") or ""))
            if expected:
                gold_known += 1
                gold_correct += int(result.get("target_event") in expected)
    return {
        "feedback_total": total,
        "event_coverage": covered / total if total else None,
        "condition_attachment_rate": with_conditions / total if total else None,
        "clarification_rate": clarified / total if total else None,
        "template_gold_known": gold_known,
        "template_event_accuracy": gold_correct / gold_known if gold_known else None,
        "warning": (
            "Template gold is an internal pipeline check because sim language "
            "was authored to align with the event vocabulary."
        ),
    }


def dataset_metrics(session_dirs: list[Path]) -> dict[str, Any]:
    feedback_ids: set[str] = set()
    represented_feedback_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        for item in read_jsonl(session_dir / "feedback_events.jsonl"):
            feedback_ids.add(
                f"{item.get('source')}:{item.get('total_step')}:"
                f"{item.get('timestamp_utc')}"
            )
        for sample in read_jsonl(session_dir / "hu_subgoal_preferences.jsonl"):
            if sample.get("record_type") != "hu_pairwise_subgoal_preference":
                continue
            samples.append(sample)
            source_id = sample.get("source_feedback_id")
            if source_id:
                represented_feedback_ids.add(str(source_id))
    invalid_cross_level = 0
    empty_conditions = 0
    for sample in samples:
        pair = {sample.get("preferred_subgoal"), sample.get("rejected_subgoal")}
        level = sample.get("decision_level")
        if level == TASK_DECISION_LEVEL and not pair.issubset(set(TASK_HU_SUBGOALS)):
            invalid_cross_level += 1
        if level == COORDINATION_DECISION_LEVEL and not pair.issubset(
            set(COORDINATION_SUBGOALS)
        ):
            invalid_cross_level += 1
        if not sample.get("condition_features"):
            empty_conditions += 1
    represented = len(feedback_ids & represented_feedback_ids)
    return {
        "feedback_total": len(feedback_ids),
        "feedback_with_at_least_one_pairwise_label": represented,
        "feedback_conversion_rate": (
            represented / len(feedback_ids) if feedback_ids else None
        ),
        "pairwise_samples_total": len(samples),
        "pairwise_samples_by_level": {
            level: sum(sample.get("decision_level") == level for sample in samples)
            for level in DECISION_LEVELS
        },
        "invalid_cross_level_pairs": invalid_cross_level,
        "samples_without_condition_snapshot": empty_conditions,
    }


def score_probe_pair(
    *,
    level: str,
    conditions: dict[str, Any],
    preferred: list[str],
    rejected: list[str],
    general: HierarchicalHu,
    adapter: PerUserAdapter,
) -> tuple[float, float] | None:
    allowed = (
        set(TASK_HU_SUBGOALS)
        if level == TASK_DECISION_LEVEL
        else set(COORDINATION_SUBGOALS)
    )
    preferred = [value for value in preferred if value in allowed]
    rejected = [value for value in rejected if value in allowed]
    if not preferred or not rejected:
        return None
    general_pref = max(
        general.score(level, "PILOT01", conditions, value) for value in preferred
    )
    general_rej = max(
        general.score(level, "PILOT01", conditions, value) for value in rejected
    )
    user_pref = max(
        adapter.score(level, conditions, value)["final_score"]
        for value in preferred
    )
    user_rej = max(
        adapter.score(level, conditions, value)["final_score"]
        for value in rejected
    )
    return general_pref - general_rej, user_pref - user_rej


def probe_metrics(
    general: HierarchicalHu,
    adapter: PerUserAdapter,
    session_dirs: list[Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in DECISION_LEVELS:
        general_margins: list[float] = []
        user_margins: list[float] = []
        changed = improved = 0
        backbone_available = backbone_correct = 0
        for session_dir in session_dirs:
            for hit in read_jsonl(session_dir / "probe_hits.jsonl"):
                if hit.get("domain") != level:
                    continue
                pair = score_probe_pair(
                    level=level,
                    conditions=hit.get("condition_features") or {},
                    preferred=hit.get("preferred_subgoals") or [],
                    rejected=hit.get("rejected_subgoals") or [],
                    general=general,
                    adapter=adapter,
                )
                if pair is None:
                    continue
                general_margin, user_margin = pair
                general_margins.append(general_margin)
                user_margins.append(user_margin)
                changed += int((general_margin > 0) != (user_margin > 0))
                improved += int(user_margin > general_margin)
                evaluation = hit.get("evaluation") or {}
                best_pref = evaluation.get("best_preferred_rank")
                best_rej = evaluation.get("best_rejected_rank")
                if isinstance(best_pref, int) and isinstance(best_rej, int):
                    backbone_available += 1
                    backbone_correct += int(best_pref < best_rej)
        result[level] = {
            "scorable_hits": len(general_margins),
            "backbone_probe_accuracy": (
                backbone_correct / backbone_available
                if backbone_available
                else None
            ),
            "backbone_scorable_hits": backbone_available,
            "hu_general": margin_metrics(general_margins),
            "hu_user": margin_metrics(user_margins),
            "sign_changed": changed,
            "margin_improved": improved,
        }
    return result


def task_log_metrics(session_dirs: list[Path]) -> dict[str, Any]:
    episode_rewards: list[float] = []
    delivered_steps = 0
    total_steps = 0
    for session_dir in session_dirs:
        trajectory = read_jsonl(session_dir / "trajectory.jsonl")
        total_steps += len(trajectory)
        delivered_steps += sum(
            float(step.get("environment_reward") or 0) > 0 for step in trajectory
        )
        if trajectory:
            episode_rewards.append(float(trajectory[-1].get("episode_reward") or 0))
    return {
        "sessions": len(session_dirs),
        "steps": total_steps,
        "mean_final_episode_reward": (
            float(np.mean(episode_rewards)) if episode_rewards else None
        ),
        "positive_reward_steps": delivered_steps,
        "hu_online_comparison": "not_run",
        "reason": (
            "Held-out sessions were logged without applying this newly trained "
            "adapter; task preservation needs a separate matched rollout."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="selfish")
    parser.add_argument("--train-samples", type=int, default=30)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--test-seeds", default="20,21,22,23")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--general",
        type=Path,
        default=REPO_ROOT / "outputs" / "hu_general" / "hierarchical_hu.json",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "human_ai_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "hu_evaluation" / "small_sim_selfish_30",
    )
    args = parser.parse_args()

    test_seeds = {int(value) for value in args.test_seeds.split(",")}
    train_dir, test_dirs = find_sessions(
        args.sessions_dir,
        persona=args.persona,
        train_seed=args.train_seed,
        test_seeds=test_seeds,
    )
    general = HierarchicalHu.load(args.general)
    all_train = load_pairwise_samples([train_dir])
    train = stratified_subset(
        all_train, n=args.train_samples, seed=args.train_seed + 4200
    )
    test = load_pairwise_samples(test_dirs)

    adapter = PerUserAdapter(
        hu_general=general,
        user_id=f"SIM_{args.persona.upper()}_SMALL",
    )
    training = adapter.train(
        train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=42,
    )
    general_eval = general.evaluate(test)
    bias_eval = evaluate_bias_only(adapter, test)
    user_eval = adapter.evaluate(test)

    report = {
        "status": "pilot_sim_feasibility_only",
        "persona": args.persona,
        "data_split": {
            "train_session": train_dir.name,
            "train_sim_seed": args.train_seed,
            "train_samples_available": len(all_train),
            "train_samples_used": len(train),
            "train_by_level": {
                level: sum(s.decision_level == level for s in train)
                for level in DECISION_LEVELS
            },
            "test_sessions": [path.name for path in test_dirs],
            "test_sim_seeds": sorted(test_seeds),
            "test_samples": len(test),
            "test_by_level": {
                level: sum(s.decision_level == level for s in test)
                for level in DECISION_LEVELS
            },
            "seed_overlap": bool({args.train_seed} & test_seeds),
        },
        "attribution_quality": attribution_metrics([train_dir, *test_dirs]),
        "dataset_quality": dataset_metrics([train_dir, *test_dirs]),
        "preference_learning": {
            "hu_general": general_eval,
            "hu_general_plus_user_bias": bias_eval,
            "hu_general_plus_user_bias_plus_condition_delta": user_eval,
            "training_final_loss": training["history"]["loss"][-1],
            "training_final_accuracy": training["history"]["accuracy"][-1],
            "observed_conditions": training["observed_conditions"],
        },
        "probe_evaluation": probe_metrics(general, adapter, test_dirs),
        "task_and_collaboration": task_log_metrics(test_dirs),
        "limitations": [
            "Sim feedback uses deterministic language templates aligned with the event schema.",
            "Automatic sim labels are used as synthetic gold; no human review is involved.",
            "No matched online rollout applied the newly trained adapter, so task non-degradation is not established.",
            "This result cannot be used as evidence of real-human semantic understanding.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter.save(args.output_dir / "hu_user.json")
    (args.output_dir / "evaluation_matrix.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
