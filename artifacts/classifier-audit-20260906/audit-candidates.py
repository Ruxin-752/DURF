"""Read-only frozen-model diagnostic and corpus overlap audit. Never trains."""
from __future__ import annotations

from collections import Counter, defaultdict
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
REPO = OUT.parents[1]
ROOT = REPO / "baselines/baseline_b_linguistic_feedback/adapted_overcooked"
CANDIDATES = ROOT / "outputs/feedback_form_classifier_candidates"
LABELS = ["evaluative", "imperative", "descriptive"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def corpus_audit(path, probes):
    rows = read(path)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.get("split", "train")].append(row)
    sets = {name: {normalize(row["text"]) for row in group} for name, group in by_split.items()}
    overlaps = {f"{a}:{b}": len(sets[a] & sets[b]) for i, a in enumerate(sets) for b in list(sets)[i + 1:]}
    train = sets.get("train", set())
    return {
        "path": str(path.relative_to(REPO)), "sha256": digest(path),
        "rows": len(rows), "split_counts": {k: len(v) for k, v in by_split.items()},
        "normalized_exact_overlap_across_partitions": overlaps,
        "diagnostic_exact_train_overlaps": [p["id"] for p in probes if normalize(p["text"]) in train],
        "known_metadata_overlap": {
            key: {f"{a}:{b}": len({str(r[key]) for r in by_split[a] if r.get(key)} & {str(r[key]) for r in by_split[b] if r.get(key)})
                  for i, a in enumerate(by_split) for b in list(by_split)[i + 1:]}
            for key in ("group_id", "component_id", "surface_family", "split_family", "scenario_group")
            if any(key in row for row in rows)
        },
    }


def evaluate(name, model_path, probes):
    artifact = joblib.load(model_path)
    matrix = artifact["vectorizer"].transform([p["text"] for p in probes])
    raw = np.asarray(artifact["classifier"].predict_proba(matrix))
    classes = [str(v) for v in artifact["classifier"].classes_]
    predicted = [classes[i] for i in raw.argmax(axis=1)]
    expected = [p["expected"] for p in probes]
    scores = classification_report(expected, predicted, labels=LABELS, output_dict=True, zero_division=0)
    calibration = artifact.get("probability_calibration", {})
    return {
        "model_path": str(model_path.relative_to(REPO)), "model_sha256": digest(model_path),
        "score_semantics": "Raw predict_proba for comparison; historical temperature is recorded but not applied. Argmax is unchanged by positive temperature.",
        "historical_calibration": calibration or {"temperature": artifact.get("temperature")},
        "rows": len(probes), "correct": sum(a == b for a, b in zip(expected, predicted)),
        "accuracy": scores["accuracy"], "macro_f1": scores["macro avg"]["f1-score"],
        "per_class": {k: scores[k] for k in LABELS}, "label_order": LABELS,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
        "prediction_rows": [{**p, "label": pred, "correct": p["expected"] == pred,
            "raw_scores": {label: float(value) for label, value in zip(classes, values)}}
            for p, pred, values in zip(probes, predicted, raw)],
    }


probes = read(OUT / "probes.frozen.json")["rows"]
report = {
    "schema": "durf-frozen-candidate-audit-v1", "date": "2026-09-06",
    "runtime": {"python": sys.version, "sklearn": sklearn.__version__},
    "scope": "Agent-authored diagnostic; not human gold or current-player accuracy. Frozen human test files are not opened by this script. Models and training rows are never written.",
    "models": {}, "corpora": {},
}
for name in ("direct_fg_v2", "direct_fg_clean_v3", "direct_fg_clean_v4", "direct_fg_boundary_shadow_v1_model"):
    model_path = CANDIDATES / name / "model.joblib"
    report["models"][name] = evaluate(name, model_path, probes)
    corpus = ROOT / "data/direct_fg_classifier_corpus.v2.json" if name == "direct_fg_v2" else CANDIDATES / name / ("train.locked.json" if "shadow" in name else "corpus.json")
    report["corpora"][name] = corpus_audit(corpus, probes)
    diagnostic_overlap = set(report["corpora"][name]["diagnostic_exact_train_overlaps"])
    report["models"][name]["diagnostic_without_exact_train_overlap"] = evaluate(name, model_path, [p for p in probes if p["id"] not in diagnostic_overlap])
    del report["models"][name]["diagnostic_without_exact_train_overlap"]["prediction_rows"]
    manifest = CANDIDATES / name / "manifest.json"
    data = read(manifest)
    report["models"][name]["manifest_sha256"] = digest(manifest)
    report["models"][name]["saved_release_claims"] = {k: data[k] for k in (
        "status", "quality_status", "promotion_status", "promotion_eligible", "production_promotion_eligible",
        "test_and_frozen_access", "provenance", "confidence_ready_for_player_ui") if k in data}

(OUT / "candidate-semantic-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({name: {k: model[k] for k in ("model_sha256", "rows", "correct", "accuracy", "macro_f1", "per_class")} for name, model in report["models"].items()}, indent=2))
print(json.dumps(report["corpora"], indent=2))
