"""Run an isolated word unigram/bigram + logistic regression experiment.

This uses the paper's classifier family with raw text for the kitchen domain.
It is not an exact replay of the author's lemmatized, random-row notebook.
Only explicit train/dev labels are read; no frozen test or production manifest
is opened. A label ontology is mandatory and recorded in the artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


LABELS = ("descriptive", "evaluative", "imperative")
PAPER_LABEL_SOURCE = "original_paper_human_reference_main_experiment_mapping"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def load_rows(path: Path, split: str, semantics: str) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("input must be a nonempty list of explicitly labeled rows")
    rows = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or row.get("split") != split:
            raise ValueError(f"{split} row {index}: unexpected split")
        text = row.get("text")
        canonical = row.get("classification_label")
        label = str(canonical).lower()
        if not isinstance(text, str) or not normalized(text) or label not in LABELS:
            raise ValueError(f"{split} row {index}: missing explicit text/label")
        if canonical != label.capitalize():
            raise ValueError(f"{split} row {index}: noncanonical label")
        expected = row.get("expected_feedback_type", row.get("label", label))
        if expected != label:
            raise ValueError(f"{split} row {index}: conflicting explicit labels")
        source = row.get("label_source")
        if semantics == "paper_grounding" and source != PAPER_LABEL_SOURCE:
            raise ValueError("paper experiment requires explicitly annotated paper rows")
        if semantics == "speech_act" and source != "human_explicit":
            raise ValueError("speech-act experiment requires reviewed human labels")
        group = row.get("group_id") or row.get("template_family")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{split} row {index}: missing group identity")
        rows.append({"text": text, "label": label, "group": group})
    if set(row["label"] for row in rows) != set(LABELS):
        raise ValueError(f"{split} must contain all three classes")
    return rows


def audit_split(train: list[dict], dev: list[dict]) -> dict:
    groups = {row["group"] for row in train} & {row["group"] for row in dev}
    texts = {normalized(row["text"]) for row in train} & {
        normalized(row["text"]) for row in dev
    }
    if groups or texts:
        raise ValueError(
            f"train/dev leakage: {len(groups)} groups, {len(texts)} normalized texts"
        )
    return {"group_overlap": 0, "normalized_text_overlap": 0,
            "near_duplicate_audit": "not_performed", "frozen_test_opened": False}


def run(train_path: Path, dev_path: Path, output: Path, semantics: str) -> dict:
    import joblib
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import FeatureUnion

    if output.exists():
        raise ValueError("output directory already exists; use a new run directory")
    train = load_rows(train_path, "train", semantics)
    dev = load_rows(dev_path, "dev", semantics)
    audit = audit_split(train, dev)
    vectorizer = FeatureUnion([("word", TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=5, stop_words="english",
    ))])
    matrix = vectorizer.fit_transform([row["text"] for row in train])
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=1)
    classifier.fit(matrix, [row["label"] for row in train])
    expected = [row["label"] for row in dev]
    prediction = classifier.predict(vectorizer.transform([row["text"] for row in dev]))
    metrics = classification_report(
        expected, prediction, labels=list(LABELS), output_dict=True, zero_division=0,
    )
    recipe = {
        "family": "paper_word_unigram_bigram_tfidf_logistic_regression",
        "ngram_range": [1, 2], "min_df": 5, "sublinear_tf": True,
        "stop_words": "english", "C": 1.0, "class_weight": None,
        "raw_text_instead_of_author_domain_preprocessing": True,
        "vectorizer_fit_on_train_only": True,
        "group_disjoint_instead_of_author_random_row_split": True,
        "model_selection": "one_prespecified_recipe_no_dev_search",
    }
    artifact = {
        "model_type": "feedback_form_tfidf_logistic_regression", "model_version": 3,
        "input_mode": "raw_text", "labels": list(LABELS),
        "canonical_labels": {label: label.capitalize() for label in LABELS},
        "target_semantics": semantics, "vectorizer": vectorizer,
        "classifier": classifier, "selected_hyperparameters": recipe,
        "minimum_model_confidence": 0.55,
        "probability_calibration": {
            "method": "none", "temperature": 1.0,
            "independently_validated": False,
        },
        "production_promotion_eligible": False,
    }
    output.mkdir(parents=True)
    model_path = output / "model.joblib"
    joblib.dump(artifact, model_path)
    report = {
        "target_semantics": semantics, "recipe": recipe,
        "source": "https://arxiv.org/pdf/2009.14715",
        "sklearn_version": sklearn.__version__,
        "train_sha256": digest(train_path), "dev_sha256": digest(dev_path),
        "model_sha256": digest(model_path), "split_audit": audit,
        "train_rows": len(train), "dev_rows": len(dev),
        "train_labels": dict(Counter(row["label"] for row in train)),
        "dev_labels": dict(Counter(expected)),
        "dev_accuracy": accuracy_score(expected, prediction),
        "dev_classification": metrics,
        "dev_confusion_matrix": confusion_matrix(expected, prediction, labels=list(LABELS)).tolist(),
        "confusion_matrix_label_order": list(LABELS),
        "production_promotion_eligible": False,
        "reason": "Development experiment only; no independent kitchen human test or release approval.",
        "scores_are_probability_of_correctness": False,
        "routing_threshold_validated": False,
    }
    (output / "model.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-semantics", choices=("paper_grounding", "speech_act"), required=True)
    args = parser.parse_args()
    report = run(args.train, args.dev, args.output_dir, args.target_semantics)
    print(json.dumps({key: report[key] for key in (
        "target_semantics", "train_rows", "dev_rows", "dev_accuracy",
        "dev_classification", "model_sha256", "production_promotion_eligible",
    )}, indent=2))


if __name__ == "__main__":
    main()
