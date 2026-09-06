"""Score one frozen inference dump; never trains or selects models.

Predictions JSON: {"provenance": {...}, "records": [{"id": "E001",
"speech_act": "evaluative", "grounding": "action", "abstained": false,
"grounding_abstained": false, "confidence": .8, "clauses": [...]}]}.
Also predict each oracle mixed clause using ids M001:C1, M001:C2, etc.
For a whole mixed message, clauses contains {speech_act, grounding, abstained}.
Each uncertain item may include learning_applied and an explicit abstention.
This evaluator does not alter any sealed text or gold label after inference.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SEALED_DATASETS = {
    "frozen-v2.json": "4473f6647e428777f92f03e54a4c6c561976775bf64cb0fe567dfd5c50105ff1",
    "frozen-next-v2.json": "23c4272190395efa3e9b9791f68b75fda502a9019c85ab2ae5e120d8fc999fea",
    "frozen-round3.json": "ddbdc0ed38517865427c01d8ae24b72400fc163034acb857945045a064f258de",
}
SPEECH = ["evaluative", "imperative", "descriptive"]
GROUNDING = ["action", "feature", "trajectory"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(golds, predictions, labels):
    assert len(golds) == len(predictions)
    matrix = {label: Counter() for label in labels}
    per_class = {}
    for gold, pred in zip(golds, predictions):
        matrix[gold][pred or "missing"] += 1
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, predictions))
        support = golds.count(label)
        predicted = predictions.count(label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "predicted": predicted,
                            "precision": precision, "recall": recall, "f1": f1}
    correct = sum(g == p for g, p in zip(golds, predictions))
    return {"n": len(golds), "correct": correct,
            "accuracy": correct / len(golds) if golds else None,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels),
            "per_class": per_class, "confusion_matrix": {k: dict(v) for k, v in matrix.items()}}


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def extract_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            yield value
        else:
            for child in value.values():
                yield from extract_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from extract_records(child)


def audit_training(paths, dataset):
    """Automated text comparison after candidate freeze, without displaying train text."""
    if not paths:
        return {"status": "not_run", "sources": [], "normalized_exact_overlap_count": None,
                "note": "No training data was supplied to this acceptance agent; a separate audit is required."}
    texts = defaultdict(list)
    sources = []
    for path in paths:
        data = Path(path).read_text(encoding="utf-8-sig")
        if str(path).endswith(".jsonl"):
            objects = [json.loads(line) for line in data.splitlines() if line.strip()]
        else:
            objects = json.loads(data)
        records = list(extract_records(objects))
        origins = Counter()
        for row in records:
            texts[normalized(row["text"])].append(str(path))
            origins[str(row.get("origin", row.get("source", row.get("data_origin", "unspecified"))))] += 1
        sources.append({"path": str(path), "sha256": digest(path), "records": len(records),
                        "declared_record_origins": dict(origins)})
    matches = []
    for split in ("singles", "mixed", "uncertain"):
        for row in dataset[split]:
            if normalized(row["text"]) in texts:
                matches.append({"acceptance_id": row["id"], "partition": split,
                                "training_sources": sorted(set(texts[normalized(row["text"])]))})
            for n, clause in enumerate(row.get("clauses", []), 1):
                if normalized(clause["text"]) in texts:
                    matches.append({"acceptance_id": f"{row['id']}:C{n}", "partition": "mixed_oracle_clause",
                                    "training_sources": sorted(set(texts[normalized(clause["text"])]))})
    return {"status": "completed", "sources": sources, "normalized_exact_overlap_count": len(matches), "overlaps": matches,
            "limitation": "Zero exact overlap cannot establish full template independence; this set was independently authored without reading the candidate generator or training/development texts."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, action="append", default=[])
    parser.add_argument("--dataset", choices=list(SEALED_DATASETS), default="frozen-v2.json")
    args = parser.parse_args()
    expected_hash = SEALED_DATASETS[args.dataset]
    assert digest(ROOT / args.dataset) == expected_hash, "Acceptance content changed after sealing"
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    dump = json.loads(args.predictions.read_text(encoding="utf-8"))
    assert dump.get("provenance", {}).get("candidate_frozen_before_inference") is True, "Missing freeze assertion"
    rows = dump["records"]
    predictions = {row["id"]: row for row in rows}
    assert len(predictions) == len(rows), "Duplicate inference IDs"
    expected_ids = {r["id"] for group in ("singles", "mixed", "uncertain") for r in dataset[group]}
    expected_ids.update(f"{r['id']}:C{i}" for r in dataset["mixed"] for i in range(1, len(r["clauses"])+1))
    assert expected_ids == set(predictions), ("Missing or unexpected inference IDs", sorted(expected_ids - set(predictions)), sorted(set(predictions) - expected_ids))
    singles = dataset["singles"]
    speech = metrics([r["speech_act"] for r in singles], [predictions[r["id"]].get("speech_act") for r in singles], SPEECH)
    grounding = metrics([r["grounding"] for r in singles], [predictions[r["id"]].get("grounding") for r in singles], GROUNDING)
    accepted = [r for r in singles if not predictions[r["id"]].get("abstained", False)]
    accepted_grounding = [r for r in singles if not predictions[r["id"]].get("grounding_abstained", False)]
    strata = defaultdict(list)
    for r in singles:
        strata[r["stratum"]].append(r)
    strata_results = {name: {"n": len(group),
                            "speech_correct": sum(predictions[r["id"]].get("speech_act") == r["speech_act"] for r in group),
                            "grounding_correct": sum(predictions[r["id"]].get("grounding") == r["grounding"] for r in group)}
                      for name, group in sorted(strata.items())}
    oracle = [{**clause, "id": f"{r['id']}:C{i}"} for r in dataset["mixed"] for i, clause in enumerate(r["clauses"], 1)]
    mixed_results = []
    for r in dataset["mixed"]:
        pred = predictions[r["id"]]
        clauses = pred.get("clauses", [])
        got = sorted({c.get("speech_act") for c in clauses if c.get("speech_act") in SPEECH})
        accepted_got = sorted({c.get("speech_act") for c in clauses if c.get("speech_act") in SPEECH and not c.get("abstained", False)})
        expected = r["expected_speech_acts"]
        mixed_results.append({"id": r["id"], "expected": expected, "predicted": got, "accepted_predicted": accepted_got,
                              "component_set_correct": got == expected, "accepted_component_set_correct": accepted_got == expected,
                              "expected_clause_count": len(r["clauses"]), "predicted_clause_count": len(clauses)})
    uncertain_results = [{"id": r["id"], "kind": r["kind"],
                          "speech_act": predictions[r["id"]].get("speech_act"),
                          "confidence": predictions[r["id"]].get("confidence"),
                          "abstained": predictions[r["id"]].get("abstained"),
                          "grounding_abstained": predictions[r["id"]].get("grounding_abstained"),
                          "learning_applied": predictions[r["id"]].get("learning_applied"),
                          "uncertainty_reason": predictions[r["id"]].get("uncertainty_reason")}
                         for r in dataset["uncertain"]]
    failures = [{"id": r["id"], "text": r["text"], "gold_speech_act": r["speech_act"], "gold_grounding": r["grounding"],
                 "predicted_speech_act": predictions[r["id"]].get("speech_act"), "predicted_grounding": predictions[r["id"]].get("grounding"),
                 "confidence": predictions[r["id"]].get("confidence"), "abstained": predictions[r["id"]].get("abstained")}
                for r in singles if predictions[r["id"]].get("speech_act") != r["speech_act"] or predictions[r["id"]].get("grounding") != r["grounding"]]
    mixed_accuracy = sum(x["component_set_correct"] for x in mixed_results) / len(mixed_results)
    checks = dataset["predeclared_checks"]
    gates = {"speech_act": speech["macro_f1"] >= checks["speech_act_macro_f1_min"] and all(v["recall"] >= checks["speech_act_each_class_recall_min"] for v in speech["per_class"].values()),
             "grounding": grounding["macro_f1"] >= checks["grounding_macro_f1_min"] and all(v["recall"] >= checks["grounding_each_class_recall_min"] for v in grounding["per_class"].values()),
             "mixed_components": mixed_accuracy >= checks["mixed_message_component_set_accuracy_min"]}
    report = {"dataset_sha256": expected_hash, "prediction_sha256": digest(args.predictions),
              "provenance": dump["provenance"], "claim_scope": dataset["independence"]["limitation"],
              "source": dataset["source"], "speech_act": speech, "grounding": grounding,
              "speech_accepted_only": {"coverage": len(accepted)/len(singles), **metrics([r["speech_act"] for r in accepted], [predictions[r["id"]].get("speech_act") for r in accepted], SPEECH)},
              "grounding_accepted_only": {"coverage": len(accepted_grounding)/len(singles), **metrics([r["grounding"] for r in accepted_grounding], [predictions[r["id"]].get("grounding") for r in accepted_grounding], GROUNDING)},
              "strata": strata_results,
              "mixed_oracle_clause_speech": metrics([r["speech_act"] for r in oracle], [predictions[r["id"]].get("speech_act") for r in oracle], SPEECH),
              "mixed_oracle_clause_grounding": metrics([r["grounding"] for r in oracle], [predictions[r["id"]].get("grounding") for r in oracle], GROUNDING),
              "mixed_component_accuracy": mixed_accuracy, "mixed": mixed_results,
              "uncertain": {"n": len(uncertain_results), "abstained": sum(x["abstained"] is True for x in uncertain_results),
                            "automatic_learning_observed": sum(x["learning_applied"] is True for x in uncertain_results),
                            "learning_gate_unknown": sum(x["learning_applied"] is None for x in uncertain_results), "details": uncertain_results},
              "failures": failures, "predeclared_gates": gates, "aggregate_predeclared_pass": all(gates.values()),
              "training_overlap_audit": audit_training(args.training_data, dataset)}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"speech_act": speech, "grounding": grounding, "mixed_component_accuracy": mixed_accuracy,
                      "uncertain_abstained": report["uncertain"]["abstained"], "gates": gates,
                      "exact_overlap_count": report["training_overlap_audit"]["normalized_exact_overlap_count"]}, indent=2))


if __name__ == "__main__":
    main()
