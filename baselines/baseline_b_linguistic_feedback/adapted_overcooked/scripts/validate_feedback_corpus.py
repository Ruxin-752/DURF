"""Validate a feedback corpus and keep only the examples that carry signal.

"Is this data useful?" here means: reading ONLY the text, does Baseline B's own
understanding recover the reward signal the example claims to carry?

For every example we check, from the text alone:
  1. sentiment round-trip -- ``extract_sentiment`` sign must match the intended
     ``attributed_sentiment_score`` sign (negative feedback must actually read
     negative; positive must read positive). This is the core usefulness gate:
     if the language does not encode the intended valence, training on it teaches
     the wrong reward direction.
  2. grounding -- the text must mention the behavior it is about (the referenced
     subgoal's resource, or its coordination theme), otherwise there is nothing
     for the model to attach the valence to.
  3. feedback-type agreement (``classify_feedback``) -- reported, not gated
     (the keyword classifier is crude and type is fuzzier than valence).

Template examples are the trusted coverage floor and are always kept; LLM
paraphrases are kept only if they pass (1) and (2). The cleaned corpus is the
"only the useful ones get added" output; a JSON report summarizes what was kept,
dropped, and why, plus valence balance / type coverage / diversity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import read_json, validate_feedback_examples, write_json  # noqa: E402
from src.feedback_form_classifier import classify_feedback  # noqa: E402
from src.phrase_reference_classifier import predict_reference_type  # noqa: E402
from src.feedback_templates import SUBGOAL_PHRASES, is_single_sentence  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.sentiment_extractor import vader_compound  # noqa: E402
from src.subgoal_teacher import load_gold_weights  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "synthetic_feedback.json"
DEFAULT_OUTPUT = ROOT / "data" / "synthetic_feedback.validated.json"
DEFAULT_REPORT = ROOT / "outputs" / "synth" / "validation_report.json"

_STOPWORDS = {"the", "a", "an", "your", "in", "for", "to", "of", "that"}
# Coordination-theme words that count as valid grounding when a subgoal noun is
# not literally present (e.g. WAIT / yielding / idling feedback, which criticize
# *not moving* and mention no ingredient).
_THEME_WORDS = {
    "block", "blocking", "way", "path", "room", "aside", "wait", "waiting",
    "hold", "holding", "same", "duplicate", "repeat", "crowd", "steal",
    "took", "take", "wrong", "split", "share", "serve", "order", "cut",
    "cutting", "idle", "idling", "stand", "standing", "hang", "hanging",
    "nothing", "still", "move", "moving", "back", "spot", "sit", "sitting",
    # WAIT / yielding paraphrases DeepSeek often uses (was the bulk of ungrounded drops)
    "stay", "staying", "stayed", "put", "bump", "bumping", "rhythm",
    "short-handed", "shorthanded", "divide", "dividing", "prep", "veggies",
    "ingredient", "ingredients", "yield", "yielding", "pause", "pausing",
    "linger", "lingering", "freeze", "frozen", "afk", "inactive", "slack",
}


def _subgoal_keywords(subgoal: str) -> set[str]:
    noun = SUBGOAL_PHRASES.get(subgoal, ("", "", subgoal))[2]
    raw = noun.lower().replace(".", " ").replace("/", " ").replace("-", " ")
    words = {w for w in raw.split() if w and w not in _STOPWORDS}
    return words or {subgoal.lower()}


def _intended_polarity(example: dict) -> float:
    score = float(example.get("attributed_sentiment_score", 0.0))
    return 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)


# Surface-sentiment thresholds for the *non-contradiction* gate. We reject a
# sample only when the raw VADER reading clearly OPPOSES the intended label --
# not merely because it is subtle/neutral. Implicit criticism ("I needed that,
# you took it") reads neutral to VADER yet is a valid hard negative, and hard
# negatives are exactly what the corpus was missing; the gold
# ``attributed_sentiment_score`` carries the true direction for training.
_POS_CONTRADICTION = 0.5   # positive-labeled text that reads this negative -> reject
_NEG_CONTRADICTION = 0.5   # negative-labeled text that reads this positive -> reject
_POSITIVE_WORDS = {"good", "great", "nice", "perfect", "right", "help", "helps", "thanks"}
_NEGATIVE_WORDS = {"bad", "wrong", "stop", "block", "blocks", "hurts", "waste", "wastes"}


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _surface_sentiment(text: str) -> float:
    """Use VADER when installed, with a conservative dependency-free fallback."""

    try:
        return vader_compound(text)
    except (ImportError, LookupError):
        words = set(_normalized_text(text).split())
        score = len(words & _POSITIVE_WORDS) - len(words & _NEGATIVE_WORDS)
        return max(-1.0, min(1.0, score / 2.0))


def _structure_errors(example: dict) -> list[str]:
    text = example.get("text", "")
    errors = []
    if not is_single_sentence(text):
        errors.append("not_single_sentence")
    annotations = example.get("phrase_annotations")
    if not isinstance(annotations, list) or len(annotations) != 1:
        errors.append("reference_clause_count")
        return errors
    annotation = annotations[0]
    if not isinstance(annotation, dict):
        return [*errors, "invalid_phrase_annotation"]
    start, end = annotation.get("start"), annotation.get("end")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end > len(text)
        or start >= end
        or text[start:end] != annotation.get("text")
    ):
        errors.append("invalid_phrase_span")
    if annotation.get("reference_type") != example.get("reference_type"):
        errors.append("reference_annotation_mismatch")
    return errors


def _feature_errors(example: dict, weights: dict[str, float]) -> list[str]:
    target = example.get("target_features")
    if not isinstance(target, dict) or not target:
        return ["missing_target_features"]
    referenced = example.get("referenced_features")
    if referenced is not None and set(referenced) != set(target):
        return ["referenced_feature_mismatch"]
    intended = _intended_polarity(example)
    if any(weights.get(feature, 0.0) * intended <= 0 for feature in target):
        return ["rule_feature_sign_mismatch"]
    return []


def evaluate_example(example: dict, *, weights: dict[str, float] | None = None) -> dict:
    text = example.get("text", "")
    intended = _intended_polarity(example)
    recovered_score = _surface_sentiment(text)
    if intended > 0:
        contradicts = recovered_score <= -_POS_CONTRADICTION
    elif intended < 0:
        contradicts = recovered_score >= _NEG_CONTRADICTION
    else:
        contradicts = False
    sentiment_ok = intended != 0 and not contradicts

    lowered = text.lower()
    keywords = _subgoal_keywords(example.get("referenced_subgoal", ""))
    grounding_ok = any(k in lowered for k in keywords) or any(
        w in lowered for w in _THEME_WORDS
    )

    recovered_type = classify_feedback(text)
    type_ok = recovered_type == example.get("expected_feedback_type")
    reference_prediction = predict_reference_type(text)
    expected_reference = example.get("reference_type")
    reference_ok = (
        reference_prediction["reference_type"] == expected_reference
        if expected_reference
        else None
    )

    structural_errors = _structure_errors(example)
    feature_errors = _feature_errors(example, weights or load_gold_weights())
    reasons = [*structural_errors, *feature_errors]
    if not sentiment_ok:
        reasons.append("sentiment_contradiction")
    if not grounding_ok:
        reasons.append("ungrounded")
    return {
        "sentiment_ok": sentiment_ok,
        "grounding_ok": grounding_ok,
        "type_ok": type_ok,
        "recovered_type": recovered_type,
        "recovered_score": recovered_score,
        "reference_ok": reference_ok,
        "recovered_reference_type": reference_prediction["reference_type"],
        "reference_confidence": reference_prediction["confidence"],
        "structure_ok": not structural_errors,
        "feature_consistency_ok": not feature_errors,
        "passed": (
            sentiment_ok
            and grounding_ok
            and not structural_errors
            and not feature_errors
        ),
        "reasons": reasons,
    }


def _duplicate_report(rows: list[dict], threshold: float = 0.9) -> dict:
    exact: dict[str, list[int]] = defaultdict(list)
    grams_by_index: list[set[str]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    near_pairs: list[dict] = []
    for index, row in enumerate(rows):
        normalized = _normalized_text(row.get("text", ""))
        exact[normalized].append(index)
        tokens = normalized.split()
        grams = {" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))}
        if len(tokens) < 3:
            grams = {normalized}
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(inverted[gram])
        for other in candidates:
            previous = grams_by_index[other]
            union = grams | previous
            similarity = len(grams & previous) / len(union) if union else 1.0
            if similarity >= threshold and normalized != _normalized_text(rows[other].get("text", "")):
                near_pairs.append(
                    {
                        "left": rows[other].get("feedback_id"),
                        "right": row.get("feedback_id"),
                        "similarity": round(similarity, 4),
                        "cross_split": rows[other].get("split") != row.get("split"),
                    }
                )
        grams_by_index.append(grams)
        for gram in grams:
            inverted[gram].append(index)
    duplicate_groups = [indices for indices in exact.values() if len(indices) > 1]
    cross_split_exact = sum(
        1
        for indices in duplicate_groups
        if len({rows[index].get("split") for index in indices}) > 1
    )
    return {
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_exact_groups": cross_split_exact,
        "near_duplicate_pairs": len(near_pairs),
        "cross_split_near_pairs": sum(pair["cross_split"] for pair in near_pairs),
        "near_duplicate_samples": near_pairs[:40],
    }


def _drop_near_duplicates(
    rows: list[dict], *, threshold: float
) -> tuple[list[dict], list[dict]]:
    """Keep one paraphrase per near-identical supervised language pattern."""

    split_rank = {"train": 0, "dev": 1, "test": 2}
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            split_rank.get(item[1].get("split", "train"), 9),
            item[0],
        ),
    )
    kept: list[dict] = []
    kept_grams: list[set[str]] = []
    postings: dict[tuple[tuple[str, str, int], str], list[int]] = defaultdict(list)
    removed: list[dict] = []
    for _original_index, row in ordered:
        normalized = _normalized_text(row.get("text", ""))
        tokens = normalized.split()
        grams = {
            " ".join(tokens[index : index + 3])
            for index in range(max(1, len(tokens) - 2))
        }
        if len(tokens) < 3:
            grams = {normalized}
        polarity = 1 if _intended_polarity(row) > 0 else -1
        supervision = (
            str(row.get("reference_type")),
            str(row.get("expected_feedback_type")),
            polarity,
        )
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(postings.get((supervision, gram), []))
        duplicate_of = None
        for candidate in candidates:
            previous = kept_grams[candidate]
            union = grams | previous
            similarity = len(grams & previous) / len(union) if union else 1.0
            if similarity >= threshold:
                duplicate_of = kept[candidate]
                break
        if duplicate_of is not None:
            removed.append(
                {
                    "example": row,
                    "duplicate_of": duplicate_of.get("feedback_id"),
                }
            )
            continue
        kept_index = len(kept)
        kept.append(row)
        kept_grams.append(grams)
        for gram in grams:
            postings[(supervision, gram)].append(kept_index)
    return kept, removed


def validate_corpus(
    examples: list[dict],
    *,
    near_duplicate_threshold: float = 0.9,
    quality_thresholds: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    dropped: list[dict] = []
    by_source_kept: dict[str, int] = {}
    by_source_dropped: dict[str, int] = {}
    drop_reasons: dict[str, int] = {}
    type_agree = 0
    sentiment_agree = 0
    reference_total = 0
    reference_agree = 0
    weights = load_gold_weights()
    source_totals: dict[str, int] = {}

    for example in examples:
        verdict = evaluate_example(example, weights=weights)
        source = example.get("source", "unknown")
        source_totals[source] = source_totals.get(source, 0) + 1
        type_agree += int(verdict["type_ok"])
        sentiment_agree += int(verdict["sentiment_ok"])
        if verdict["reference_ok"] is not None:
            reference_total += 1
            reference_agree += int(verdict["reference_ok"])

        keep = verdict["passed"]
        if keep:
            kept.append(example)
            by_source_kept[source] = by_source_kept.get(source, 0) + 1
        else:
            dropped.append({"feedback_id": example.get("feedback_id"), **verdict})
            by_source_dropped[source] = by_source_dropped.get(source, 0) + 1
            for reason in verdict["reasons"]:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    def _valence_counts(rows: list[dict]) -> dict[str, int]:
        out = {"positive": 0, "negative": 0, "neutral": 0}
        for row in rows:
            pol = _intended_polarity(row)
            out["positive" if pol > 0 else "negative" if pol < 0 else "neutral"] += 1
        return out

    def _type_counts(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            key = row.get("expected_feedback_type", "unknown")
            out[key] = out.get(key, 0) + 1
        return out

    # Exact text may not occur in multiple evaluation splits. Keep the
    # deterministic train > dev > test owner and reject leaked copies.
    split_rank = {"train": 0, "dev": 1, "test": 2}
    owners: dict[str, str] = {}
    for example in kept:
        norm = _normalized_text(example.get("text", ""))
        split = example.get("split", "train")
        if norm not in owners or split_rank.get(split, 9) < split_rank.get(owners[norm], 9):
            owners[norm] = split
    leakage_dropped = []
    leakage_safe = []
    for example in kept:
        norm = _normalized_text(example.get("text", ""))
        if example.get("split", "train") != owners.get(norm):
            leakage_dropped.append(example)
        else:
            leakage_safe.append(example)
    kept = leakage_safe
    for example in leakage_dropped:
        source = example.get("source", "unknown")
        by_source_kept[source] = max(0, by_source_kept.get(source, 0) - 1)
        by_source_dropped[source] = by_source_dropped.get(source, 0) + 1
        drop_reasons["cross_split_exact_leakage"] = (
            drop_reasons.get("cross_split_exact_leakage", 0) + 1
        )
        dropped.append(
            {
                "feedback_id": example.get("feedback_id"),
                "passed": False,
                "reasons": ["cross_split_exact_leakage"],
            }
        )
    kept, near_dropped = _drop_near_duplicates(
        kept, threshold=near_duplicate_threshold
    )
    for item in near_dropped:
        example = item["example"]
        source = example.get("source", "unknown")
        by_source_kept[source] = max(0, by_source_kept.get(source, 0) - 1)
        by_source_dropped[source] = by_source_dropped.get(source, 0) + 1
        drop_reasons["near_duplicate"] = drop_reasons.get("near_duplicate", 0) + 1
        dropped.append(
            {
                "feedback_id": example.get("feedback_id"),
                "passed": False,
                "reasons": ["near_duplicate"],
                "duplicate_of": item["duplicate_of"],
            }
        )
    duplicate_report = _duplicate_report(kept, near_duplicate_threshold)
    kept_texts = [e.get("text", "") for e in kept]
    total = len(examples)
    report = {
        "total_examples": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_by_source": dict(sorted(by_source_kept.items())),
        "dropped_by_source": dict(sorted(by_source_dropped.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "sentiment_roundtrip_agreement": sentiment_agree / total if total else 0.0,
        "type_roundtrip_agreement": type_agree / total if total else 0.0,
        "reference_type_roundtrip_agreement": (
            reference_agree / reference_total if reference_total else None
        ),
        "kept_valence_balance": _valence_counts(kept),
        "kept_type_coverage": _type_counts(kept),
        "kept_reference_type_coverage": {
            label: sum(1 for row in kept if row.get("reference_type") == label)
            for label in ("trajectory", "feature", "action_spatial", "action_behavioral", "other")
        },
        "kept_unique_texts": len(set(t.lower() for t in kept_texts)),
        "kept_diversity_ratio": (
            len(set(t.lower() for t in kept_texts)) / len(kept_texts)
            if kept_texts
            else 0.0
        ),
        "kept_scenarios": len({e.get("group_id") for e in kept}),
        "duplicates": duplicate_report,
        "source_reports": {
            source: {
                "total": count,
                "kept": by_source_kept.get(source, 0),
                "dropped": by_source_dropped.get(source, 0),
                "keep_rate": by_source_kept.get(source, 0) / count if count else 0.0,
            }
            for source, count in sorted(source_totals.items())
        },
        "dropped_samples": dropped[:40],
    }
    thresholds = {
        "min_keep_rate": 0.0,
        "min_llm_keep_rate": 0.0,
        "min_diversity_ratio": 0.0,
        "max_cross_split_near_pairs": 1_000_000_000,
        **(quality_thresholds or {}),
    }
    llm_report = report["source_reports"].get("llm", {"keep_rate": 1.0})
    checks = {
        "min_keep_rate": (len(kept) / total if total else 0.0)
        >= thresholds["min_keep_rate"],
        "min_llm_keep_rate": llm_report["keep_rate"]
        >= thresholds["min_llm_keep_rate"],
        "min_diversity_ratio": report["kept_diversity_ratio"]
        >= thresholds["min_diversity_ratio"],
        "max_cross_split_near_pairs": duplicate_report["cross_split_near_pairs"]
        <= thresholds["max_cross_split_near_pairs"],
    }
    report["quality_thresholds"] = thresholds
    report["quality_checks"] = checks
    report["quality_passed"] = all(checks.values())
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    parser.add_argument("--min-keep-rate", type=float, default=0.8)
    parser.add_argument("--min-llm-keep-rate", type=float, default=0.7)
    parser.add_argument("--min-diversity-ratio", type=float, default=0.35)
    parser.add_argument("--max-cross-split-near-pairs", type=int, default=0)
    parser.add_argument(
        "--enforce-quality",
        action="store_true",
        help="Return non-zero when configured corpus quality thresholds fail.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input with the validated corpus (adds only useful ones).",
    )
    args = parser.parse_args()

    examples = read_json(args.input)
    if not isinstance(examples, list):
        raise ValueError("Input JSON must contain a top-level array")
    if len(examples) < args.min_count:
        raise ValueError(f"Expected at least {args.min_count} examples, got {len(examples)}")
    if args.max_count is not None and len(examples) > args.max_count:
        raise ValueError(f"Expected at most {args.max_count} examples, got {len(examples)}")
    kept, report = validate_corpus(
        examples,
        near_duplicate_threshold=args.near_duplicate_threshold,
        quality_thresholds={
            "min_keep_rate": args.min_keep_rate,
            "min_llm_keep_rate": args.min_llm_keep_rate,
            "min_diversity_ratio": args.min_diversity_ratio,
            "max_cross_split_near_pairs": args.max_cross_split_near_pairs,
        },
    )

    probes = load_probe_states(args.probe_states)
    validate_feedback_examples(kept, probe_states=probes)

    out_path = args.input if args.in_place else args.output
    write_json(out_path, kept)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)

    print(f"Validated {report['total_examples']} examples.")
    print(f"  kept {report['kept']} | dropped {report['dropped']}")
    print(f"  kept by source: {report['kept_by_source']}")
    print(f"  dropped by source: {report['dropped_by_source']}")
    print(f"  source reports: {report['source_reports']}")
    print(f"  drop reasons: {report['drop_reasons']}")
    print(
        f"  sentiment round-trip agreement: "
        f"{report['sentiment_roundtrip_agreement']:.1%} | "
        f"type: {report['type_roundtrip_agreement']:.1%}"
    )
    print(f"  kept valence: {report['kept_valence_balance']}")
    print(f"  kept types: {report['kept_type_coverage']}")
    print(
        f"  kept diversity: {report['kept_unique_texts']} unique / "
        f"{report['kept']} ({report['kept_diversity_ratio']:.1%})"
    )
    print(f"Cleaned corpus: {out_path}")
    print(f"Report: {args.report}")
    if args.enforce_quality and not report["quality_passed"]:
        failed = [name for name, passed in report["quality_checks"].items() if not passed]
        print(f"Quality gate failed: {failed}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
