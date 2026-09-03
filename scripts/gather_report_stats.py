"""Gather all artifact statistics for the protocol-v2 final report."""
import json
from pathlib import Path

BASE = Path("outputs")

print("=== FILE INVENTORY ===")
for d in [
    "hu_general",
    "hu_general/filtered_training",
    "hu_users/selfish",
    "hu_users/cooperative",
    "hu_users/polite",
    "hu_users/lenient",
    "human_ai_sessions",
]:
    p = BASE / d
    if not p.exists():
        continue
    files = [f for f in p.iterdir() if f.is_file()]
    total_kb = sum(f.stat().st_size for f in files) // 1024
    print(f"\n{d}/ : {len(files)} files, {total_kb} KB")
    for f in sorted(f.name for f in files)[:10]:
        print(f"  {f}")
    if len(files) > 10:
        print(f"  ... +{len(files) - 10} more")


print("\n=== HU_GENERAL FREEZE ===")
hg_path = BASE / "hu_general" / "hierarchical_hu.json"
hg = json.loads(hg_path.read_text(encoding="utf-8"))
for key in ("frozen", "freeze_version", "freeze_date", "source"):
    val = hg.get(key, "N/A")
    print(f"  {key}: {val}")

hash_path = BASE / "hu_general" / "frozen_hash.txt"
if hash_path.exists():
    print(f"  SHA256: {hash_path.read_text().strip()}")


print("\n=== SESSION STATS ===")
sessions_dir = BASE / "human_ai_sessions"
sessions = list(sessions_dir.iterdir())
sim_sessions = 0
pilot_sessions = 0
by_persona = {}
total_pairs = 0
total_probes = 0
for sd in sessions:
    mf = sd / "session_metadata.json"
    if not mf.exists():
        continue
    meta = json.loads(mf.read_text(encoding="utf-8"))
    if meta.get("data_source") == "synthetic_sim_human":
        sim_sessions += 1
        persona = (meta.get("sim_human") or {}).get("persona", "unknown")
        by_persona[persona] = by_persona.get(persona, 0) + 1
    else:
        pilot_sessions += 1

    pf = sd / "hu_subgoal_preferences.jsonl"
    if pf.exists():
        n = len([l for l in pf.read_text(encoding="utf-8").strip().split("\n") if l.strip()])
        total_pairs += n

    prf = sd / "probe_hits.jsonl"
    if prf.exists():
        n = len([l for l in prf.read_text(encoding="utf-8").strip().split("\n") if l.strip()])
        total_probes += n

print(f"  Sim sessions: {sim_sessions}")
print(f"  By persona: {json.dumps(by_persona, indent=2)}")
print(f"  Pilot sessions: {pilot_sessions}")
print(f"  Total pairwise samples: {total_pairs}")
print(f"  Total probe hits: {total_probes}")


print("\n=== CONDITION MANIFEST ===")
cm = Path("docs/condition_manifest_v2.md")
if cm.exists():
    text = cm.read_text(encoding="utf-8")
    n_conds = text.count("| `")
    print(f"  Documented conditions: ~{n_conds}")


print("\n=== TRAINING SUMMARY (per persona) ===")
for persona in ["selfish", "cooperative", "polite", "lenient"]:
    md_path = BASE / "hu_users" / persona / "metadata.json"
    if not md_path.exists():
        continue
    md = json.loads(md_path.read_text(encoding="utf-8"))
    t = md["training"]
    e = md["evaluation"]
    pe = md["probe_evaluation"]

    hg_t = e["hu_general"].get("task", {})
    hu_t = e["hu_user"].get("task", {})
    hg_c = e["hu_general"].get("coordination", {})
    hu_c = e["hu_user"].get("coordination", {})

    print(f"\n  {persona}:")
    print(f"    Train: {t['samples']} samples, loss={t['final_loss']:.6f}, acc={t['final_accuracy']:.3f}")
    print(f"    Task:    Hu_general={hg_t.get('pairwise_accuracy', 'N/A')} -> Hu_user={hu_t.get('pairwise_accuracy', 'N/A')}")
    print(f"    Coord:   Hu_general={hg_c.get('pairwise_accuracy', 'N/A')} -> Hu_user={hu_c.get('pairwise_accuracy', 'N/A')}")
    print(f"    Probes:  task={pe['task']['total_hits']}h/{pe['task']['changed_ranking']}chg, coord={pe['coordination']['total_hits']}h/{pe['coordination']['changed_ranking']}chg")

print("\nDone.")
