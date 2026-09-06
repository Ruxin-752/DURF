"""P8: Learning curve analysis -- how much data does delta_user need?

For selfish persona (most divergent, most interesting):
  1. Subsample training data at {5%, 10%, 20%, 35%, 50%, 75%, 100%}
  2. Train full adapter (user_bias + condition_delta) at each level
  3. Also train user_bias-only for ablation
  4. Evaluate on held-out test set
  5. Report convergence curves

Key metrics:
  - Pairwise accuracy on test set (split by task/coordination)
  - Mean margin
  - Number of observed conditions at each data level
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from durf.hu.subgoal_reranker import (
    HierarchicalHu,
    PerUserAdapter,
    PairwiseSample,
    load_pairwise_samples,
    TASK_DECISION_LEVEL,
    COORDINATION_DECISION_LEVEL,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_sessions(
    persona: str, sessions_dir: Path
) -> tuple[list[Path], list[Path]]:
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
    return sorted(train_dirs), sorted(test_dirs)


def subsample(
    samples: list[PairwiseSample], fraction: float, seed: int = 42
) -> list[PairwiseSample]:
    if fraction >= 1.0:
        return list(samples)
    rng = np.random.RandomState(seed)
    n = max(1, int(len(samples) * fraction))
    indices = rng.choice(len(samples), size=n, replace=False)
    return [samples[i] for i in sorted(indices)]


def _evaluate_bias_only(
    adapter: PerUserAdapter, samples: list[PairwiseSample]
) -> dict[str, Any]:
    """Evaluate using only hu_general + user_bias (ignore condition_delta)."""
    metrics: dict[str, Any] = {}
    for dl in (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL):
        domain_samples = [s for s in samples if s.decision_level == dl]
        if not domain_samples:
            metrics[dl] = {"samples": 0, "pairwise_accuracy": None,
                           "mean_margin": None}
            continue
        margins = []
        for s in domain_samples:
            pref = adapter.score(dl, s.condition_features, s.preferred_subgoal)
            rej = adapter.score(dl, s.condition_features, s.rejected_subgoal)
            margin = (
                (pref["general_score"] + pref["user_bias_score"])
                - (rej["general_score"] + rej["user_bias_score"])
            )
            margins.append(margin)
        margins_np = np.asarray(margins, dtype=np.float32)
        metrics[dl] = {
            "samples": len(domain_samples),
            "pairwise_accuracy": float((margins_np > 0.0).mean()),
            "mean_margin": float(margins_np.mean()),
            "min_margin": float(margins_np.min()),
            "max_margin": float(margins_np.max()),
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="selfish")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Random subsample trials per fraction")
    args = parser.parse_args()

    fractions = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
    ablate_user_bias = True  # Compare full vs user_bias-only

    print(f"=== P8: Learning Curve Analysis for {args.persona} ===\n")

    hu_general = HierarchicalHu.load(args.general)
    train_dirs, test_dirs = find_sessions(args.persona, args.sessions_dir)
    print(f"Train sessions: {len(train_dirs)}, Test sessions: {len(test_dirs)}")

    # Load all training samples
    all_train = load_pairwise_samples(
        [d / "hu_subgoal_preferences.jsonl" for d in train_dirs]
    )
    test_samples = load_pairwise_samples(
        [d / "hu_subgoal_preferences.jsonl" for d in test_dirs]
    )
    print(
        f"Total train: {len(all_train)}, "
        f"Test: {len(test_samples)} "
        f"(task={sum(1 for s in test_samples if s.decision_level == TASK_DECISION_LEVEL)}, "
        f"coord={sum(1 for s in test_samples if s.decision_level == COORDINATION_DECISION_LEVEL)})"
    )

    # Evaluate Hu_general baseline
    general_eval = hu_general.evaluate(test_samples)
    print(f"\nHu_general baseline:")
    for dl in (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL):
        m = general_eval.get(dl, {})
        print(f"  {dl}: acc={m.get('pairwise_accuracy', 'N/A'):.3f}, "
              f"margin={m.get('mean_margin', 'N/A'):.3f}")

    # Run learning curve
    results: list[dict[str, Any]] = []

    for frac in fractions:
        n_samples = max(1, int(len(all_train) * frac))
        print(f"\n--- Fraction {frac:.0%} ({n_samples} samples) ---")

        full_results: list[dict] = []
        bias_results: list[dict] = []

        for trial in range(args.n_trials):
            train_subset = subsample(
                all_train, frac, seed=42 + trial * 100
            )
            user_id = f"CURVE_TRIAL_{trial}"

            # Train full adapter once
            adapter = PerUserAdapter(
                hu_general=hu_general, user_id=user_id
            )
            adapter.train(
                train_subset,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=42 + trial,
            )

            # Evaluate full (hu_general + user_bias + condition_delta)
            full_eval = adapter.evaluate(test_samples)
            full_results.append(full_eval)

            # Evaluate bias-only (hu_general + user_bias, ignore condition_delta)
            if ablate_user_bias:
                bias_eval = _evaluate_bias_only(adapter, test_samples)
                bias_results.append(bias_eval)

        # Aggregate
        agg = {"fraction": frac, "n_samples": n_samples, "n_trials": args.n_trials}
        for label, evals in [("full", full_results), ("bias_only", bias_results)]:
            if not evals:
                continue
            for dl in (TASK_DECISION_LEVEL, COORDINATION_DECISION_LEVEL):
                accs = [e.get(dl, {}).get("pairwise_accuracy", 0) or 0
                        for e in evals]
                margins = [e.get(dl, {}).get("mean_margin", 0) or 0
                           for e in evals]
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs))
                mean_margin = float(np.mean(margins))
                std_margin = float(np.std(margins))
                key_prefix = f"{dl}_{label}"
                agg[f"{key_prefix}_mean_acc"] = mean_acc
                agg[f"{key_prefix}_std_acc"] = std_acc
                agg[f"{key_prefix}_mean_margin"] = mean_margin
                agg[f"{key_prefix}_std_margin"] = std_margin

                acc_str = f"{mean_acc:.3f} +/- {std_acc:.3f}"
                margin_str = f"{mean_margin:.2f} +/- {std_margin:.2f}"
                print(
                    f"  [{label:>9}] {dl:>12}: acc={acc_str}, margin={margin_str}"
                )
        results.append(agg)

    # Summary table
    print("\n\n=== Learning Curve Summary ===")
    print(
        f"{'Frac':>6} {'N':>5} "
        f"{'Task_full':>10} {'Task_bias':>10} {'Task_delta':>10} "
        f"{'Coord_full':>11} {'Coord_bias':>11} {'Coord_delta':>11}"
    )
    print("-" * 76)
    for r in results:
        tf = r.get("task_full_mean_acc", 0)
        tb = r.get("task_bias_only_mean_acc", 0)
        cf = r.get("coordination_full_mean_acc", 0)
        cb = r.get("coordination_bias_only_mean_acc", 0)
        print(
            f"{r['fraction']:>6.0%} {r['n_samples']:>5} "
            f"{tf:>10.3f} {tb:>10.3f} {tf-tb:>+10.3f} "
            f"{cf:>11.3f} {cb:>11.3f} {cf-cb:>+11.3f}"
        )

    # Determine minimum data needed
    print("\n=== Data Efficiency Analysis ===")
    for dl, label in [(TASK_DECISION_LEVEL, "Task"), (COORDINATION_DECISION_LEVEL, "Coordination")]:
        general_acc = general_eval.get(dl, {}).get("pairwise_accuracy", 0)
        print(f"\n{label} (Hu_general baseline = {general_acc:.3f}):")
        for r in results:
            full_acc = r.get(f"{dl}_full_mean_acc", 0)
            bias_acc = r.get(f"{dl}_bias_only_mean_acc", 0)
            full_std = r.get(f"{dl}_full_std_acc", 0)
            bias_std = r.get(f"{dl}_bias_only_std_acc", 0) 
            n = r["n_samples"]
            # Full vs general
            full_delta = full_acc - general_acc
            bias_delta = bias_acc - general_acc
            print(
                f"  {r['fraction']:.0%} ({n:>4} samples): "
                f"full={full_acc:.3f} +/- {full_std:.3f} ({full_delta:+.3f} vs general), "
                f"bias_only={bias_acc:.3f} +/- {bias_std:.3f} ({bias_delta:+.3f} vs general)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
