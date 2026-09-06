import json
with open("outputs/hu_general/filtered_training/sim_persona_conflicts.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        s = json.loads(line)
        if s.get("record_type") != "hu_pairwise_subgoal_preference":
            continue
        cf = s.get("condition_features", {})
        active = {k: v for k, v in cf.items() if v is not None and v != 0}
        print(s.get("source_feedback_id", ""), "::", active)
