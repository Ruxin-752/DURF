"""RETIRED 2026-09-06 (protocol v2.2): Hu_general is dropped; the per-user adapter
starts from zero (== H0).  Kept only to re-run attribution on the sim corpus
(--reattribute --skip-train) when checking the attribution prompt.

Rebuild the Hu_general prior from a fresh sim corpus, end to end.

    sim sessions (4 personas x N seeds, H0, no preference)
      -> attribution per session (LLM by default; every label carries its attributor)
      -> cross-persona audit: keep only labels ALL personas agree on   (protocol v2 S3.2)
      -> dataset build + validation
      -> 70/15/15 split
      -> train the two-head HierarchicalHu
      -> summary

Why a fresh corpus: the frozen Hu_general (outputs/hu_general) was trained on
labels produced by the keyword baseline (4 538 of 4 554), on the pre-fix
candidate generator, with 59% of labels coming from one preference every
persona shared.  None of that describes the mechanism as it now stands.

Why the invariant filter will keep far LESS than before: the new persona table
puts personas on opposite sides of every task dimension on purpose, so those
labels are persona-specific by construction and belong to per-user adapters,
not to the shared prior.  What survives is genuine consensus: corrective
feedback everyone gives (ready pot -> get a dish; idle while cooking -> prep)
and the coordination stances personas share.  A thin task head is the honest
outcome, not a failure.

Runs on the experiment laptop (needs DEEPSEEK_API_KEY; sim + attribution both
import the game runtime, so use the durf310 env).  Every stage is skipped if
its output already exists, so an interrupted run resumes.

    python scripts/rebuild_hu_general.py --seeds 15 --out outputs/hu_general_v4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PERSONAS = ("cooperative", "selfish", "polite", "lenient")
LAYOUT = "ring_tomato_onion_10x6_h0_full_task"
USER_ID = "SIM_GENERAL"


def run(cmd: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{REPO / 'src'}" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        cmd, cwd=REPO, env=env, text=True, encoding="utf-8", errors="replace",
        capture_output=quiet, check=False,
    )


def stage_sessions(out: Path, seeds: int, episodes: int, horizon: int) -> list[Path]:
    sessions_dir = out / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "sessions_manifest.json"
    manifest: dict[str, str] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    total = len(PERSONAS) * seeds
    done = 0
    for seed in range(seeds):
        for persona in PERSONAS:
            key = f"{persona}_s{seed}"
            done += 1
            if key in manifest and (sessions_dir / manifest[key] / "trajectory.csv").exists():
                continue
            before = {p.name for p in sessions_dir.iterdir() if p.is_dir()}
            cmd = [
                sys.executable, "-m", "durf.group_a.sim_session",
                "--layout", LAYOUT, "--seed", str(seed), "--sim-seed", str(seed),
                "--horizon", str(horizon), "--episodes", str(episodes),
                "--sim-human", persona, "--hu-user-id", USER_ID,
                "--output-dir", str(sessions_dir),
            ]
            if horizon != 800:
                cmd.append("--allow-nonstandard-setup")
            t0 = time.time()
            result = run(cmd)
            new = sorted({p.name for p in sessions_dir.iterdir() if p.is_dir()} - before)
            if result.returncode != 0 or not new:
                print(f"[{done}/{total}] {key}: FAILED\n{result.stderr[-800:]}")
                continue
            manifest[key] = new[-1]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[{done}/{total}] {key}: {new[-1]}  ({time.time() - t0:.0f}s)")
    return [sessions_dir / name for name in manifest.values()]


def stage_attribution(sessions: list[Path], *, use_llm: bool, redo: bool) -> None:
    total = len(sessions)
    for index, session in enumerate(sessions, 1):
        if (session / "hu_subgoal_preferences.jsonl").exists() and not redo:
            continue
        cmd = [
            sys.executable, "-m", "durf.feedback_attribution.demo_offline_attribution",
            "--session", str(session), "--user-id", USER_ID,
        ]
        if not use_llm:
            cmd.append("--no-llm")
        t0 = time.time()
        result = run(cmd)
        tail = [line for line in result.stdout.splitlines() if "WARNING" in line or "records:" in line][-3:]
        status = "FAILED" if result.returncode != 0 else "ok"
        print(f"[{index}/{total}] attribution {session.name}: {status} ({time.time() - t0:.0f}s) "
              + " | ".join(tail))
        if result.returncode != 0:
            print(result.stderr[-600:])


def stage_audit_build_split_train(out: Path, *, epochs: int, seed: int) -> Path:
    filtered = out / "filtered_training"
    filtered.mkdir(parents=True, exist_ok=True)
    steps = [
        ("audit", [sys.executable, "-m", "durf.hu.audit_sim_personas",
                   "--sessions-dir", str(out / "sessions"), "--output-dir", str(filtered)]),
        ("build", [sys.executable, "-m", "durf.hu.build_general_prior_dataset",
                   "--invariant-file", str(filtered / "hu_general_invariant.jsonl"),
                   "--output-dir", str(filtered)]),
        ("split", [sys.executable, str(REPO / "scripts" / "split_hu_general_data.py"),
                   "--task-file", str(filtered / "hu_general_task.jsonl"),
                   "--coord-file", str(filtered / "hu_general_coordination.jsonl"),
                   "--output-dir", str(filtered), "--seed", str(seed)]),
        ("train", [sys.executable, "-m", "durf.hu.train_subgoal_reranker",
                   "--dataset", str(filtered / "hu_general_train.jsonl"),
                   "--test-dataset", str(filtered / "hu_general_test.jsonl"),
                   "--output-dir", str(out / "model"),
                   "--epochs", str(epochs), "--seed", str(seed)]),
    ]
    for name, cmd in steps:
        result = run(cmd, quiet=False)
        if result.returncode != 0:
            raise SystemExit(f"stage {name} failed")
    return out / "model" / "hierarchical_hu.json"


def summarize(out: Path) -> None:
    import collections

    by_attr: collections.Counter = collections.Counter()
    rejected = 0
    labels = 0
    for path in (out / "sessions").glob("*/hu_subgoal_preferences.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels += 1
                by_attr[json.loads(line).get("attributor", "<missing>")] += 1
    for path in (out / "sessions").glob("*/hu_subgoal_preferences.rejected.jsonl"):
        rejected += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    audit = json.loads((out / "filtered_training" / "sim_general_audit.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "model" / "metadata.json").read_text(encoding="utf-8"))
    print("\n================ Hu_general rebuild summary ================")
    print(f"raw labels: {labels}   by attributor: {dict(by_attr)}   schema-rejected: {rejected}")
    print(f"cross-persona audit: invariant {audit.get('invariant_samples')} | "
          f"conflicting {audit.get('conflicting_samples')} | groups inv/conf "
          f"{audit.get('invariant_groups')}/{audit.get('conflicting_groups')}")
    print(f"by level: {audit.get('by_decision_level')}")
    print(f"train/val/test: {meta.get('samples_train')}/{meta.get('samples_validation')}/{meta.get('samples_test')}")
    for level in ("task", "coordination"):
        tr = (meta.get("train_metrics_by_level") or {}).get(level) or {}
        te = (meta.get("test_metrics_by_level") or meta.get("validation_metrics_by_level") or {}).get(level) or {}
        print(f"  {level:13} train acc {tr.get('pairwise_accuracy')}  n={tr.get('samples')}   "
              f"held-out acc {te.get('pairwise_accuracy')}  n={te.get('samples')}")
    print(f"model: {out / 'model' / 'hierarchical_hu.json'}")
    fallbacks = by_attr.get("rule_fallback_after_llm_error", 0)
    if fallbacks:
        print(f"WARNING: {fallbacks} labels came from the keyword stand-in after an LLM failure; "
              f"they are in the corpus but must not be counted as LLM output.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "hu_general_v4")
    parser.add_argument("--seeds", type=int, default=15, help="seeds per persona")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true", help="keyword baseline only (tests)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--reattribute", action="store_true",
                        help="re-run attribution on existing sessions (after a prompt change)")
    args = parser.parse_args()

    sessions = stage_sessions(args.out, args.seeds, args.episodes, args.horizon)
    stage_attribution(sessions, use_llm=not args.no_llm, redo=args.reattribute)
    if not args.skip_train:
        stage_audit_build_split_train(args.out, epochs=args.epochs, seed=args.seed)
        summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
