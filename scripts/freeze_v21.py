"""Freeze Hu_general v2.1 and print top weights for report verification."""
import json, hashlib, sys
from pathlib import Path

sys.path.insert(0, ".")
from durf.hu.subgoal_reranker import HierarchicalHu

model_path = Path("outputs/hu_general/hierarchical_hu.json")
model = HierarchicalHu.load(model_path)

# Compute weight table from actual model object
weights = model.top_condition_weights(limit=30)

def print_weights(entries, top_n=8):
    for e in entries[:top_n]:
        d = "promotes" if e["weight"] > 0 else "discourages"
        print(f"  {d:>12}  {e['condition']:<42} -> {e['subgoal']:<25} ({e['weight']:+.4f})")

print("=== Top Task Weights ===")
print_weights(weights["task"])

print("\n=== Top Coord Weights ===")
print_weights(weights["coordination"])

# Now freeze
data = json.loads(model_path.read_text(encoding="utf-8"))
data["frozen"] = True
data["freeze_version"] = "v2.1"
data["freeze_date"] = "2026-08-11"
data["source"] = "simulated_prior"
data["data_summary"] = {
    "audit_total_samples": 4515,
    "invariant_groups": 148,
    "invariant_samples": 3806,
    "conflicting_groups": 5,
    "conflicting_samples": 29,
    "train_samples": 2195,
    "val_samples": 469,
    "test_samples": 305,
}

model_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
h = hashlib.sha256(model_path.read_bytes()).hexdigest()
Path("outputs/hu_general/frozen_hash.txt").write_text(h)
print(f"\nFrozen v2.1 SHA256: {h}")
