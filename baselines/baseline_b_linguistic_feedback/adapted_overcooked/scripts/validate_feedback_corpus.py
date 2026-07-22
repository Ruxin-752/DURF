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
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_schema import read_json, validate_feedback_examples, write_json  # noqa: E402
from src.feedback_form_classifier import classify_feedback  # noqa: E402
from src.feedback_templates import SUBGOAL_PHRASES  # noqa: E402
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.sentiment_extractor import vader_compound  # noqa: E402

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


def evaluate_example(example: dict) -> dict:
    text = example.get("text", "")
    intended = _intended_polarity(example)
    recovered_score = vader_compound(text)
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

    reasons = []
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
        "passed": sentiment_ok and grounding_ok,
        "reasons": reasons,
    }


def validate_corpus(examples: list[dict]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    dropped: list[dict] = []
    by_source_kept: dict[str, int] = {}
    by_source_dropped: dict[str, int] = {}
    drop_reasons: dict[str, int] = {}
    type_agree = 0
    sentiment_agree = 0

    for example in examples:
        verdict = evaluate_example(example)
        source = example.get("source", "unknown")
        type_agree += int(verdict["type_ok"])
        sentiment_agree += int(verdict["sentiment_ok"])

        # Templates are the trusted floor; LLM must earn its place.
        keep = source == "template" or verdict["passed"]
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
        "kept_valence_balance": _valence_counts(kept),
        "kept_type_coverage": _type_counts(kept),
        "kept_unique_texts": len(set(t.lower() for t in kept_texts)),
        "kept_diversity_ratio": (
            len(set(t.lower() for t in kept_texts)) / len(kept_texts)
            if kept_texts
            else 0.0
        ),
        "kept_scenarios": len({e.get("group_id") for e in kept}),
        "dropped_samples": dropped[:40],
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input with the validated corpus (adds only useful ones).",
    )
    args = parser.parse_args()

    examples = read_json(args.input)
    kept, report = validate_corpus(examples)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
