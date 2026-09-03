#!/usr/bin/env python3
"""Portable validator for LLM-generated reward-feedback corpora.

Standalone (no repo imports). Gates each example by the two rules from the
skill: sentiment *non-contradiction* and *grounding*. Keeps templates as a
trusted floor; LLM samples must pass. Prints health metrics and writes a
cleaned corpus.

Input: a JSON array or JSONL file; each example needs at least
    - "text": the utterance
    - "attributed_sentiment_score" (or "polarity"): gold reward direction (>0 / <0)
  optional:
    - "source": "template" is always kept; anything else must pass the gate
    - "keywords": list of words that count as grounded (overrides the map)
    - "referenced_subgoal": key into a built-in keyword map
    - "group_id"/"scenario_id": for diversity/scenario counts

Sentiment uses NLTK VADER if available; otherwise a small built-in lexicon.

    python validate_feedback.py corpus.json --output corpus.validated.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reject only when surface tone CLEARLY contradicts the gold label (see skill).
CONTRADICTION_THRESHOLD = 0.5

# Coordination-theme words that count as grounding when no resource is named
# (yielding / idling feedback mentions no ingredient).
THEME_WORDS = {
    "block", "blocking", "way", "path", "room", "aside", "wait", "waiting",
    "hold", "holding", "same", "duplicate", "repeat", "crowd", "steal", "took",
    "take", "wrong", "split", "share", "serve", "order", "cut", "cutting",
    "idle", "idling", "stand", "standing", "hang", "hanging", "nothing",
    "still", "move", "moving", "back", "spot", "sit", "sitting",
}

# Minimal subgoal -> grounding keywords map (extend for your task).
SUBGOAL_KEYWORDS = {
    "GET_TOMATO": {"tomato"}, "PUT_TOMATO_IN_POT": {"tomato", "pot"},
    "GET_ONION": {"onion"}, "PUT_ONION_IN_POT": {"onion", "pot"},
    "GET_DISH": {"dish", "plate"}, "PICKUP_SOUP": {"soup"},
    "SERVE_SOUP": {"soup", "serve", "order"}, "WAIT": {"wait", "idle", "stand"},
}

_POS_LEX = {"good", "nice", "great", "perfect", "thanks", "right", "well", "yes", "love"}
_NEG_LEX = {"bad", "wrong", "stop", "don't", "dont", "avoid", "no", "not", "worse", "hate"}


def _vader_scorer():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return lambda text: float(analyzer.polarity_scores(text or "")["compound"])
    except Exception:
        def fallback(text: str) -> float:
            words = set((text or "").lower().replace(".", " ").replace(",", " ").split())
            score = len(words & _POS_LEX) - len(words & _NEG_LEX)
            return max(-1.0, min(1.0, score / 2.0))

        return fallback


def _load(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _polarity(example: dict) -> float:
    score = float(example.get("attributed_sentiment_score", example.get("polarity", 0.0)))
    return 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)


def evaluate(example: dict, score_sentiment) -> dict:
    text = example.get("text", "")
    lowered = text.lower()
    intended = _polarity(example)
    compound = score_sentiment(text)
    if intended > 0:
        contradicts = compound <= -CONTRADICTION_THRESHOLD
    elif intended < 0:
        contradicts = compound >= CONTRADICTION_THRESHOLD
    else:
        contradicts = True  # unlabeled: cannot use for reward learning
    sentiment_ok = intended != 0 and not contradicts

    keywords = set(example.get("keywords") or [])
    if not keywords:
        keywords = SUBGOAL_KEYWORDS.get(example.get("referenced_subgoal", ""), set())
    grounding_ok = (
        any(k in lowered for k in keywords) if keywords else False
    ) or any(w in lowered for w in THEME_WORDS)

    reasons = []
    if not sentiment_ok:
        reasons.append("sentiment_contradiction")
    if not grounding_ok:
        reasons.append("ungrounded")
    return {"passed": sentiment_ok and grounding_ok, "reasons": reasons, "compound": compound}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    examples = _load(args.input)
    score_sentiment = _vader_scorer()

    kept, dropped, drop_reasons = [], [], {}
    non_contra = 0
    for e in examples:
        v = evaluate(e, score_sentiment)
        non_contra += int("sentiment_contradiction" not in v["reasons"])
        if e.get("source") == "template" or v["passed"]:
            kept.append(e)
        else:
            dropped.append({"text": e.get("text"), **v})
            for r in v["reasons"]:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1

    pos = sum(1 for e in kept if _polarity(e) > 0)
    neg = sum(1 for e in kept if _polarity(e) < 0)
    texts = [e.get("text", "").lower() for e in kept]
    uniq = len(set(texts))
    groups = {e.get("group_id", e.get("scenario_id")) for e in kept}
    report = {
        "total": len(examples), "kept": len(kept), "dropped": len(dropped),
        "drop_reasons": drop_reasons,
        "non_contradiction_rate": round(non_contra / len(examples), 4) if examples else 0.0,
        "valence": {"positive": pos, "negative": neg},
        "diversity_ratio": round(uniq / len(texts), 4) if texts else 0.0,
        "unique_texts": uniq, "scenarios": len(groups - {None}),
    }

    out = args.output or args.input.with_suffix(".validated.json")
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"cleaned corpus -> {out}")
    if report["valence"]["negative"] == 0 or report["valence"]["positive"] == 0:
        print("WARNING: corpus lacks one polarity; it cannot teach that direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
