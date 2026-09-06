"""P5: Single-participant closed loop with PerUserAdapter (protocol-v2).

For a selected sim persona (virtual participant):
  1. Load frozen Hu_general
  2. Merge train session hu_subgoal_preferences.jsonl into one dataset
  3. Train PerUserAdapter (user_bias + condition_delta)
  4. Evaluate on held-out test sessions
  5. Compare Hu_general vs Hu_user on probe states

Usage:
  python -m durf.hu.run_p5_closed_loop --persona selfish --general outputs/hu_general/hierarchical_hu.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from durf.feedback_attribution.io_utils import read_jsonl, write_jsonl
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    CONDITION_KEYS,
    COORDINATION_CONDITION_KEYS,
    HierarchicalHu,
    PerUserAdapter,
    TASK_DECISION_LEVEL,
    TASK_HU_SUBGOALS,
    PairwiseSample,
    load_pairwise_samples,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_sessions(persona: str, sessions_dir: Path) -> dict[str, list[Path]]:
    """Return {train: [...], test: [...]} session dirs for a given persona."""
    train_dirs: list[Path] = []
    test_dirs: list[Path] = []
    for child in sorted(sessions_dir.iterdir()):
        meta_path = child / "session_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("data_source") != "synthetic_sim_human":
            continue
        if (meta.get("sim_human") or {}).get("persona") != persona:
            continue
        seed = meta.get("seed", -1)
        if seed in range(0, 12):
            train_dirs.append(child)
        elif seed in range(20, 24):
            test_dirs.append(child)
    return {"train": sorted(train_dirs), "test": sorted(test_dirs)}


def merge_preferences(
    session_dirs: list[Path], user_id: str
) -> list[dict[str, Any]]:
    """Merge hu_subgoal_preferences.jsonl from multiple sessions."""
    all_samples: list[dict[str, Any]] = []
    for sd in session_dirs:
        prefs_path = sd / "hu_subgoal_preferences.jsonl"
        if not prefs_path.exists():
            continue
        for sample in read_jsonl(prefs_path):
            if sample.get("record_type") != "hu_pairwise_subgoal_preference":
                continue
            sample["user_id"] = user_id
            all_samples.append(sample)
    return all_samples


def evaluate_probes(
    adapter: PerUserAdapter,
    session_dirs: list[Path],
) -> dict[str, Any]:
    """Compare Hu_general vs Hu_user on probe hits from test sessions."""
    results: dict[str, Any] = {
        "task": {"total_hits": 0, "changed_ranking": 0, "details": []},
        "coordination": {"total_hits": 0, "changed_ranking": 0, "details": []},
    }

    for sd in session_dirs:
        probe_path = sd / "probe_hits.jsonl"
        if not probe_path.exists():
            continue
        for hit in read_jsonl(probe_path):
            if hit.get("record_type") != "probe_hit":
                continue
            probe_name = hit.get("probe_name", "")
            domain = hit.get("domain", "task")
            cond = hit.get("condition_features") or {}
            chosen = hit.get("chosen_subgoal", "")

            # Hu_general ranking
            if domain == "task":
                general_scores = {
                    sg: adapter.hu_general.score(
                        TASK_DECISION_LEVEL, "PILOT01", cond, sg
                    )
                    for sg in TASK_HU_SUBGOALS
                }
            else:
                general_scores = {
                    sg: adapter.hu_general.score(
                        COORDINATION_DECISION_LEVEL, "PILOT01", cond, sg
                    )
                    for sg in ("CONTINUE_CURRENT_SUBGOAL", "YIELD")
                }

            general_top = max(general_scores, key=general_scores.get)

            # Hu_user ranking
            user_scores = {}
            for sg in general_scores:
                s = adapter.score(domain, cond, sg)
                user_scores[sg] = s["final_score"]
            user_top = max(user_scores, key=user_scores.get)

            changed = general_top != user_top
            detail = {
                "probe_name": probe_name,
                "probe_step": hit.get("total_step"),
                "chosen_subgoal": chosen,
                "general_top": general_top,
                "user_top": user_top,
                "changed": changed,
                "general_scores": {
                    sg: round(float(v), 4)
                    for sg, v in sorted(
                        general_scores.items(), key=lambda x: -x[1]
                    )[:3]
                },
                "user_scores": {
                    sg: round(float(v), 4)
                    for sg, v in sorted(
                        user_scores.items(), key=lambda x: -x[1]
                    )[:3]
                },
            }
            results[domain]["total_hits"] += 1
            if changed:
                results[domain]["changed_ranking"] += 1
                results[domain]["details"].append(detail)
            # Only keep first 20 changed details per domain
            if len(results[domain]["details"]) > 20:
                results[domain]["details"] = results[domain]["details"][:20]

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona", default="selfish",
        choices=("cooperative", "selfish", "polite", "lenient"),
    )
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
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--user-id", default="SELFISH_VIRTUAL")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT / "outputs" / "hu_users" / args.persona
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Hu_general
    print("Loading Hu_general ...")
    hu_general = HierarchicalHu.load(args.general)

    # 2. Find sessions
    sessions = find_sessions(args.persona, args.sessions_dir)
    print(
        f"Sessions: {len(sessions['train'])} train, "
        f"{len(sessions['test'])} test"
    )

    # 3. Merge training data
    train_prefs = merge_preferences(sessions["train"], args.user_id)
    train_samples = load_pairwise_samples(
        [sd / "hu_subgoal_preferences.jsonl" for sd in sessions["train"]]
    )
    task_train = [s for s in train_samples if s.decision_level == TASK_DECISION_LEVEL]
    coord_train = [
        s for s in train_samples
        if s.decision_level == COORDINATION_DECISION_LEVEL
    ]
    print(
        f"Training samples: {len(train_samples)} "
        f"(task={len(task_train)}, coordination={len(coord_train)})"
    )

    # 4. Train PerUserAdapter
    print("Training PerUserAdapter ...")
    adapter = PerUserAdapter(hu_general=hu_general, user_id=args.user_id)
    result = adapter.train(
        train_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=42,
    )

    print(
        f"Training complete. "
        f"Final loss={result['history']['loss'][-1]:.6f}, "
        f"acc={result['history']['accuracy'][-1]:.3f}"
    )
    print(f"Observed conditions (task): {result['observed_conditions']['task']}")
    print(
        f"Observed conditions (coord): "
        f"{result['observed_conditions']['coordination']}"
    )

    # 5. Evaluate on test sessions
    test_samples = load_pairwise_samples(
        [sd / "hu_subgoal_preferences.jsonl" for sd in sessions["test"]]
    )
    if test_samples:
        # Hu_general alone
        general_eval = hu_general.evaluate(test_samples)
        # Hu_user
        user_eval = adapter.evaluate(test_samples)

        print("\n=== Test Evaluation ===")
        for domain in (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL):
            gm = general_eval.get(domain, {})
            um = user_eval.get(domain, {})
            print(f"\n{domain}:")
            print(f"  Hu_general:  acc={gm.get('pairwise_accuracy', 'N/A')}, "
                  f"margin={gm.get('mean_margin', 'N/A')}")
            print(f"  Hu_user:     acc={um.get('pairwise_accuracy', 'N/A')}, "
                  f"margin={um.get('mean_margin', 'N/A')}")
    else:
        general_eval = {}
        user_eval = {}
        print("\nNo test samples available.")

    # 6. Probe evaluation
    print("\n=== Probe Evaluation ===")
    probe_results = evaluate_probes(adapter, sessions["test"])
    for domain in (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL):
        pr = probe_results[domain]
        print(
            f"{domain}: {pr['total_hits']} hits, "
            f"{pr['changed_ranking']} ranking changes"
        )
        if pr["details"]:
            print(f"  First 3 changes:")
            for d in pr["details"][:3]:
                print(
                    f"    {d['probe_name']}: "
                    f"{d['general_top']} -> {d['user_top']}"
                )

    # 7. Save results
    adapter.save(args.output_dir / "hu_user.json")
    summary = {
        "persona": args.persona,
        "user_id": args.user_id,
        "hu_general_hash": args.general.read_text(encoding="utf-8")[:50],
        "training": {
            "samples": len(train_samples),
            "task": len(task_train),
            "coordination": len(coord_train),
            "epochs": args.epochs,
            "final_loss": result["history"]["loss"][-1],
            "final_accuracy": result["history"]["accuracy"][-1],
            "observed_conditions": result["observed_conditions"],
        },
        "evaluation": {
            "test_samples": len(test_samples),
            "hu_general": general_eval,
            "hu_user": user_eval,
        },
        "probe_evaluation": {
            domain: {
                "total_hits": pr["total_hits"],
                "changed_ranking": pr["changed_ranking"],
            }
            for domain, pr in probe_results.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
