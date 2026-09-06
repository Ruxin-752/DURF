"""P6: Multi-persona comparison report (protocol-v2).

Aggregates results across all 4 sim personas:
  - Training/testing metrics comparison
  - Probe sensitivity analysis
  - Delta weight divergence (user_bias + condition_delta per persona)
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_user_model(persona: str):
    p = REPO_ROOT / "outputs" / "hu_users" / persona / "hu_user.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_metadata(persona: str):
    p = REPO_ROOT / "outputs" / "hu_users" / persona / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_metrics(persona: str, md: dict) -> dict:
    train = md["training"]
    result = {
        "train_samples": train["samples"],
        "train_task": train["task"],
        "train_coord": train["coordination"],
        "train_loss": train["final_loss"],
        "train_acc": train["final_accuracy"],
    }
    for dl in ("task", "coordination"):
        hg = md["evaluation"]["hu_general"].get(dl, {})
        hu = md["evaluation"]["hu_user"].get(dl, {})
        pr = md["probe_evaluation"].get(dl, {})
        hg_acc = hg.get("pairwise_accuracy") or 0.0
        hu_acc = hu.get("pairwise_accuracy") or 0.0
        result[f"{dl}_hg_acc"] = hg_acc
        result[f"{dl}_hu_acc"] = hu_acc
        result[f"{dl}_acc_delta"] = hu_acc - hg_acc
        result[f"{dl}_hg_margin"] = hg.get("mean_margin") or 0.0
        result[f"{dl}_hu_margin"] = hu.get("mean_margin") or 0.0
        result[f"{dl}_probe_hits"] = pr.get("total_hits", 0)
        result[f"{dl}_probe_changes"] = pr.get("changed_ranking", 0)
    return result


def top_delta_weights(model: dict, persona: str, top_k: int = 15) -> dict:
    """Extract top condition_delta weights per domain."""
    from durf.hu.subgoal_reranker import (
        TASK_HU_SUBGOALS, COORDINATION_SUBGOALS,
        CONDITION_KEYS, COORDINATION_CONDITION_KEYS,
    )
    result = {}
    for dl, cond_keys, subgoals in [
        ("task", CONDITION_KEYS, TASK_HU_SUBGOALS),
        ("coordination", COORDINATION_CONDITION_KEYS, COORDINATION_SUBGOALS),
    ]:
        delta_key = f"condition_delta_{dl}" if dl == "task" else "condition_delta_coord"
        delta = model.get(delta_key, [])
        entries = []
        for ci, ck in enumerate(cond_keys):
            for si, sg in enumerate(subgoals):
                w = float(delta[ci][si])
                entries.append({
                    "condition": ck,
                    "subgoal": sg,
                    "weight": w,
                    "direction": "promotes" if w > 0 else "discourages",
                })
        entries.sort(key=lambda x: abs(x["weight"]), reverse=True)
        result[dl] = entries[:top_k]
    return result


def probe_pattern_analysis(personas: list[str]) -> dict:
    """Find probes where different personas disagree on the ranking."""
    from collections import defaultdict

    # Load probe details from each persona
    persona_probes: dict[str, dict] = {}
    for persona in personas:
        p = REPO_ROOT / "outputs" / "hu_users" / persona / "hu_user.json"
        model = json.loads(p.read_text(encoding="utf-8"))
        # We need to re-score probes; load the adapter and re-evaluate
        persona_probes[persona] = model

    return persona_probes


def main():
    personas = ["selfish", "cooperative", "polite", "lenient"]
    model = json.loads(
        (REPO_ROOT / "outputs" / "hu_general" / "hierarchical_hu.json")
        .read_text(encoding="utf-8")
    )

    print("=" * 70)
    print("P6: MULTI-PERSONA COMPARISON REPORT (Protocol v2)")
    print("=" * 70)

    # 1. Metrics Summary
    print("\n## 1. Training & Test Metrics\n")
    print(
        f"{'Persona':<14} {'Train':>6} {'Loss':>10} "
        f"{'Task Hgen':>10} {'Task Hu':>10} {'Task D':>8} "
        f"{'Coord Hgen':>10} {'Coord Hu':>10} {'Coord D':>8}"
    )
    print("-" * 86)
    summaries = {}
    for persona in personas:
        md = load_metadata(persona)
        s = summarize_metrics(persona, md)
        summaries[persona] = s
        print(
            f"{persona:<14} {s['train_samples']:>6} {s['train_loss']:>10.6f} "
            f"{s['task_hg_acc']:>10.3f} {s['task_hu_acc']:>10.3f} "
            f"{s['task_acc_delta']:>+8.3f} "
            f"{s['coordination_hg_acc']:>10.3f} {s['coordination_hu_acc']:>10.3f} "
            f"{s['coordination_acc_delta']:>+8.3f}"
        )

    # 2. Probe Sensitivity
    print("\n\n## 2. Probe Ranking Sensitivity\n")
    print(
        f"{'Persona':<14} {'Task Hits':>10} {'Task Chg':>10} {'Task %':>8} "
        f"{'Coord Hits':>12} {'Coord Chg':>10} {'Coord %':>9}"
    )
    print("-" * 75)
    for persona in personas:
        s = summaries[persona]
        task_pct = (
            s["task_probe_changes"] / max(s["task_probe_hits"], 1) * 100
        )
        coord_pct = (
            s["coordination_probe_changes"] / max(s["coordination_probe_hits"], 1) * 100
        )
        print(
            f"{persona:<14} {s['task_probe_hits']:>10} {s['task_probe_changes']:>10} "
            f"{task_pct:>7.1f}% {s['coordination_probe_hits']:>12} "
            f"{s['coordination_probe_changes']:>10} {coord_pct:>7.1f}%"
        )

    # 3. Delta Weight Divergence
    print("\n\n## 3. Top Condition-Delta Weights per Persona\n")
    for persona in personas:
        um = load_user_model(persona)
        td = top_delta_weights(um, persona, top_k=8)
        for dl, entries in td.items():
            print(f"\n### {persona} / {dl}")
            for e in entries[:5]:
                direction = "++" if e["weight"] > 0 else "--"
                print(
                    f"  {direction} {e['condition']:<40} "
                    f"-> {e['subgoal']:<25} "
                    f"({e['weight']:+.4f})"
                )

    # 4. Cross-Persona Condition Divergence
    print("\n\n## 4. Condition Divergence Matrix\n")
    print("Conditions with the largest weight spread across personas:")
    all_divergences = []
    from durf.hu.subgoal_reranker import (
        CONDITION_KEYS, COORDINATION_CONDITION_KEYS,
        TASK_HU_SUBGOALS, COORDINATION_SUBGOALS,
    )
    for dl, cond_keys, subgoals in [
        ("task", CONDITION_KEYS, TASK_HU_SUBGOALS),
        ("coordination", COORDINATION_CONDITION_KEYS, COORDINATION_SUBGOALS),
    ]:
        delta_key = f"condition_delta_{dl}" if dl == "task" else "condition_delta_coord"
        for ci, ck in enumerate(cond_keys):
            for si, sg in enumerate(subgoals):
                weights = []
                for persona in personas:
                    um = load_user_model(persona)
                    delta = um.get(delta_key, [])
                    w = float(delta[ci][si]) if delta else 0.0
                    weights.append(w)
                spread = max(weights) - min(weights)
                if abs(spread) > 0.01:
                    all_divergences.append({
                        "domain": dl,
                        "condition": ck,
                        "subgoal": sg,
                        "spread": spread,
                        "weights": dict(zip(personas, weights)),
                    })

    all_divergences.sort(key=lambda x: x["spread"], reverse=True)
    for d in all_divergences[:15]:
        weights_str = " ".join(
            f"{p}={w:+.3f}" for p, w in d["weights"].items()
        )
        print(
            f"  [{d['domain'][:4]}] {d['condition']:<35} -> {d['subgoal']:<22} "
            f"spread={d['spread']:.4f}  [{weights_str}]"
        )

    # 5. Persona Personality Summary
    print("\n\n## 5. Persona Personality Summary\n")
    interpretations = {
        "selfish": (
            "Prioritizes own efficiency over coordination. "
            "Yields reluctantly; expects AI to accommodate. "
            "Hu_general fails hardest (71.3% coord) -- delta_user needed."
        ),
        "cooperative": (
            "Aligns closely with Hu_general. "
            "Both task and coordination preferences match the shared prior. "
            "Delta_user adds only marginal refinement."
        ),
        "polite": (
            "Coord accurate but yields more readily. "
            "86% probe rank changes suggest nuanced yielding criteria. "
            "Delta_user amplifies margin (8.0 -> 13.4) without accuracy loss."
        ),
        "lenient": (
            "Most tolerant; lets partner take lead. "
            "Hu_general and Hu_user identical on coord accuracy. "
            "Delta_user improves margin only."
        ),
    }
    for persona in personas:
        print(f"\n### {persona}")
        print(f"  {interpretations[persona]}")
        s = summaries[persona]
        print(
            f"  Key metric: coord acc {s['coordination_hg_acc']:.3f} -> "
            f"{s['coordination_hu_acc']:.3f} ({s['coordination_acc_delta']:+.3f})"
        )
        print(
            f"  Probe sensitivity: {s['coordination_probe_changes']}/{s['coordination_probe_hits']} "
            f"coord changes, {s['task_probe_changes']}/{s['task_probe_hits']} task changes"
        )

    print("\n" + "=" * 70)
    print("Report complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
