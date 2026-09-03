"""Save P6 comparison report as structured JSON."""
import json
from pathlib import Path

personas = ["selfish", "cooperative", "polite", "lenient"]

report = {
    "report_type": "simulation_comparison",
    "hu_general_frozen_hash": Path("outputs/hu_general/frozen_hash.txt")
    .read_text()
    .strip(),
    "hu_general_train": {"task": 1.0, "coordination": 1.0},
    "hu_general_val": {"task": 1.0, "coordination": 0.988},
    "hu_general_test": {"task": 1.0, "coordination": 0.990},
    "persona_results": {},
}

for persona in personas:
    md = json.loads(
        Path(f"outputs/hu_users/{persona}/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    e = md["evaluation"]
    pe = md["probe_evaluation"]
    report["persona_results"][persona] = {
        "train_samples": md["training"]["samples"],
        "train_loss": md["training"]["final_loss"],
        "train_accuracy": md["training"]["final_accuracy"],
        "hu_general": {
            "task": e["hu_general"]["task"],
            "coordination": e["hu_general"]["coordination"],
        },
        "hu_user": {
            "task": e["hu_user"]["task"],
            "coordination": e["hu_user"]["coordination"],
        },
        "probe_evaluation": {
            "task": pe["task"],
            "coordination": pe["coordination"],
        },
    }

Path("outputs/hu_evaluation").mkdir(parents=True, exist_ok=True)
Path("outputs/hu_evaluation/comparison_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("P6 comparison_report.json saved")
print(json.dumps(report, indent=2, ensure_ascii=False))
