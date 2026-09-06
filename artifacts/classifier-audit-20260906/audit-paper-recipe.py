"""One-shot audit of the fixed paper recipe candidate. Never fits or exports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1] / "baselines/baseline_b_linguistic_feedback/adapted_overcooked"
MODEL = ROOT / "outputs/feedback_form_classifier_candidates/paper_recipe_20260906/model.joblib"
REPORT = OUT / "paper-recipe-semantic-report.json"
EXPECTED_HASH = "647eb77dc20c4d0a5ad06cc0bd9ed894365ac4d11a7ccbb0b49f7835612d2119"
LABELS = ["evaluative", "imperative", "descriptive"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


if REPORT.exists():
    raise SystemExit("One-shot receipt already exists; evaluation not repeated.")
if digest(MODEL) != EXPECTED_HASH:
    raise SystemExit("Frozen candidate hash mismatch; evaluation not started.")
artifact = joblib.load(MODEL)
probes = read(OUT / "probes.frozen.json")
browser = read(OUT / "browser-semantic-report.json")
rows = probes["rows"]
mixed = browser["models"]["production"]["mixed"]["predictions"]
all_text = [r["text"] for r in rows] + [phrase for row in mixed for phrase in row["phrases"]]
raw = np.asarray(artifact["classifier"].predict_proba(artifact["vectorizer"].transform(all_text)))
classes = [str(x) for x in artifact["classifier"].classes_]
predicted = [classes[i] for i in raw.argmax(axis=1)]
expected = [r["expected"] for r in rows]
scores = classification_report(expected, predicted[:len(rows)], labels=LABELS, output_dict=True, zero_division=0)
offset = len(rows)
mixed_results = []
for row in mixed:
    end = offset + len(row["phrases"])
    predictions = predicted[offset:end]
    mixed_results.append({"text": row["text"], "expected": row["expected"], "phrases": row["phrases"],
        "labels": predictions, "exact_label_sequence": predictions == row["expected"]})
    offset = end
train_path = ROOT / "data/paper_feedback_form_human_train.v1.json"
train = {normalize(r["text"]) for r in read(train_path)}
report = {
    "schema": "durf-paper-recipe-one-shot-semantic-audit-v1", "date": "2026-09-06",
    "model_sha256_before": EXPECTED_HASH, "model_sha256_after": digest(MODEL),
    "model_path": str(MODEL), "probes_file_sha256": digest(OUT / "probes.frozen.json"),
    "successful_full_batch_evaluations": 1, "training_or_parameter_updates": 0,
    "runtime": {"python": sys.version, "sklearn": sklearn.__version__},
    "scope": "AI-authored literal speech-act diagnostic, not independent human gold or current-player accuracy. Candidate was trained for paper reference-collapse functionality, a different label task. No test examples were added to training.",
    "score_policy": artifact.get("probability_calibration"),
    "rows": len(rows), "correct": sum(a == b for a, b in zip(expected, predicted)),
    "accuracy": scores["accuracy"], "macro_f1": scores["macro avg"]["f1-score"],
    "per_class": {label: scores[label] for label in LABELS}, "label_order": LABELS,
    "confusion_matrix": confusion_matrix(expected, predicted[:len(rows)], labels=LABELS).tolist(),
    "train_file_sha256": digest(train_path),
    "diagnostic_exact_train_overlaps": [r["id"] for r in rows if normalize(r["text"]) in train],
    "mixed": {"rows": len(mixed_results), "correct_label_sequences": sum(r["exact_label_sequence"] for r in mixed_results), "results": mixed_results},
    "predictions": [{**r, "label": predicted[i], "correct": predicted[i] == r["expected"],
        "raw_scores": {label: float(value) for label, value in zip(classes, raw[i])}} for i, r in enumerate(rows)],
    "web_exported": False,
    "web_export_blocker": "The candidate uses scikit-learn English stop words. Current JS word analyzer does not implement that stop-word preprocessing; export without matching preprocessing and numerical parity would be invalid.",
}
if report["model_sha256_after"] != EXPECTED_HASH:
    raise RuntimeError("Candidate bytes changed during inference.")
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("model_sha256_before", "rows", "correct", "accuracy", "macro_f1", "per_class", "diagnostic_exact_train_overlaps")}, indent=2))
print("mixed_correct", report["mixed"]["correct_label_sequences"], "/", report["mixed"]["rows"])
