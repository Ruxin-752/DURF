"""Pre-inference contract correction requested by root; retains v1 unchanged."""
from pathlib import Path
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent
OLD_HASH = "4efe488f045c845dbf176ec7223d689c0d2a743cfb2914624f84fdfb89c75888"
old = ROOT / "frozen.json"
assert hashlib.sha256(old.read_bytes()).hexdigest() == OLD_HASH
data = json.loads(old.read_text(encoding="utf-8"))
data["dataset_id"] += "-grounding-contract-v2"
data["revision"] = {
    "previous_dataset_sha256": OLD_HASH,
    "reason": "Before any candidate inference, root clarified the paper grounding contract: an action that actually occurred, whether described or evaluated, belongs to trajectory. Action grounding denotes suggested/counterfactual action alternatives. v1 instead treated individual past moves as action.",
    "speech_act_gold_changed": False,
    "texts_changed": False,
    "candidate_inference_before_revision": False,
    "candidate_training_text_read": False,
}
data["contract"]["grounding"] = {
    "action": "A requested next/immediate action, an action alternative, or an explicit counterfactual about what should have been done instead.",
    "feature": "A resource/object/spatial/current state, general rule, preference, or value associated with it.",
    "trajectory": "An action or sequence that actually occurred, whether reported factually or evaluated, including whole runs.",
}
changes = []
for row in data["singles"]:
    if row["grounding"] != "action":
        continue
    if row["speech_act"] == "descriptive" or (
        row["speech_act"] == "evaluative"
        and row["stratum"] not in {"retrospective_should", "retrospective_regret", "retrospective_counterfactual"}
    ):
        row["grounding"] = "trajectory"
        changes.append(row["id"])
for row in data["mixed"]:
    for n, clause in enumerate(row["clauses"], 1):
        if clause["grounding"] == "action" and clause["speech_act"] in {"evaluative", "descriptive"}:
            if "should have" not in clause["text"].lower() and "ought to have" not in clause["text"].lower():
                clause["grounding"] = "trajectory"
                changes.append(f"{row['id']}:C{n}")
data["revision"]["grounding_changed_ids"] = changes
target = ROOT / "frozen-v2.json"
assert not target.exists(), "Refusing to overwrite a sealed v2 dataset"
target.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
sha = hashlib.sha256(target.read_bytes()).hexdigest()
summary = {"dataset_id": data["dataset_id"], "sha256": sha, "previous_sha256": OLD_HASH,
           "single_count": len(data["singles"]),
           "speech_act_counts": dict(Counter(r["speech_act"] for r in data["singles"])),
           "grounding_counts": dict(Counter(r["grounding"] for r in data["singles"])),
           "cross_label_counts": dict(Counter(r["speech_act"]+"/"+r["grounding"] for r in data["singles"])),
           "mixed_count": len(data["mixed"]), "uncertain_count": len(data["uncertain"]),
           "grounding_revision_count": len(changes), "speech_act_gold_changed": False,
           "texts_changed": False, "candidate_inference_before_revision": False,
           "contract": data["contract"]["grounding"]}
(ROOT / "seal-v2.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
(ROOT / "frozen-v2.sha256").write_text(sha+"  frozen-v2.json\n", encoding="ascii")
print(json.dumps(summary, indent=2))
