"""Generate the synthetic feedback corpus (templates + optional LLM augment).

Pipeline (all offline except the optional, cached LLM step):

    contexts (enumerate_subgoal_contexts)
    -> teacher feedback intents (subgoal_teacher)
    -> template sentences (feedback_templates)      [coverage floor]
    -> optional DeepSeek paraphrases (cached)        [diversity]
    -> validated data/synthetic_feedback.json

Each example carries an explicit ``target_features`` reference vector and an
``attributed_sentiment_score`` (the teacher polarity), so learning does not
depend on VADER's sign for synthetic text. Grouping is by ``group_id`` (the
scenario id) so Route 2's grouped CV holds out whole scenarios.

The LLM step is skipped automatically when ``DEEPSEEK_API_KEY`` is unset or
``--no-llm`` is passed; results are cached to ``outputs/synth/llm_cache.jsonl``
keyed by a prompt hash so reruns are cheap and reproducible offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.enumerate_subgoal_contexts import enumerate_contexts  # noqa: E402
from src.feature_schema import read_json, validate_feedback_examples, write_json  # noqa: E402
from src.feedback_templates import (  # noqa: E402
    NEGATIVE_THEME_PHRASES,
    POSITIVE_THEME_PHRASES,
    SUBGOAL_PHRASES,
    context_description,
    render_templates,
    theme_for_intent,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_featurizer import SubgoalContext  # noqa: E402
from src.subgoal_teacher import feedback_intents, load_gold_weights  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "synthetic_feedback.json"
DEFAULT_CONTEXTS = ROOT / "outputs" / "synth" / "contexts.json"
DEFAULT_LLM_CACHE = ROOT / "outputs" / "synth" / "llm_cache.jsonl"

_POLARITY_LABEL = {1.0: "positive", -1.0: "negative"}


# --------------------------------------------------------------------------- #
# LLM augmentation (optional, cached)
# --------------------------------------------------------------------------- #
def _load_llm_cache(path: Path) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["key"]] = record["paraphrases"]
    return cache


def _append_llm_cache(path: Path, key: str, paraphrases: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"key": key, "paraphrases": paraphrases}, ensure_ascii=False)
            + "\n"
        )


def _prompt_key(context_desc: str, intent, n: int) -> str:
    payload = (
        f"v2||{context_desc}||{intent.role}||{intent.subgoal}||"
        f"{theme_for_intent(intent)}||{intent.feedback_type}||{intent.polarity}||{n}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _type_instruction(feedback_type: str, positive: bool) -> str:
    """Plain-language definition of the requested feedback speech-act."""

    if feedback_type == "imperative":
        return (
            "Give a COMMAND telling the AI to do that now."
            if positive
            else "Give a COMMAND telling the AI to stop / not do that."
        )
    if feedback_type == "descriptive":
        return (
            "Describe, as a plain fact, what that move does for us."
            if positive
            else "Describe, as a plain fact, what that move does TO YOU "
            "(how it gets in your way)."
        )
    # evaluative
    return (
        "Judge that move as good / the right call."
        if positive
        else "Judge that move as bad / the wrong call."
    )


def _llm_paraphrases(
    context_desc: str,
    intent,
    *,
    n: int,
    cache: dict[str, list[str]],
    cache_path: Path,
) -> list[str]:
    key = _prompt_key(context_desc, intent, n)
    if key in cache:
        return cache[key]

    from durf.group_a.deepseek_chat import chat_once

    positive = intent.polarity > 0
    stance = "GOOD" if positive else "BAD"
    opposite = "negative/critical" if positive else "positive/approving"
    _verb, gerund, noun = SUBGOAL_PHRASES.get(
        intent.subgoal, ("do that", "doing that", "that")
    )
    theme = theme_for_intent(intent)
    theme_phrase = (
        POSITIVE_THEME_PHRASES.get(theme)
        if positive
        else NEGATIVE_THEME_PHRASES.get(theme)
    ) or ("that's the right move" if positive else "that's not helping")

    system = (
        "You are a person playing the cooperative game Overcooked with an AI "
        "teammate. You are giving the AI quick, natural, spoken feedback about "
        "the move it is making right now. Always first person, casual, one "
        "short sentence each, no quotes, no emojis."
    )
    user = (
        f"Situation: {context_desc}\n"
        f"The AI is {gerund} ({noun}).\n"
        f"Your honest view: this is {stance} because {theme_phrase}.\n"
        f"{_type_instruction(intent.feedback_type, positive)}\n"
        f"Write {n} DIFFERENT natural one-sentence reactions you'd say out loud "
        f"to the AI. Hard rules:\n"
        f"- Every sentence must stay clearly {stance}; never sound {opposite}.\n"
        f"- Every sentence must clearly be about {noun} / {gerund}.\n"
        f"- Vary the wording; do not just reorder the same words.\n"
        f"Respond with a JSON array of {n} strings and nothing else."
    )
    try:
        raw = chat_once(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
    except Exception as exc:  # network hiccup / service error: skip this intent
        # Do NOT cache the empty result, so a later rerun retries this intent
        # instead of permanently losing its paraphrases.
        print(f"  [llm] skipped (transient error, will retry on rerun): {exc}")
        return []

    paraphrases = _parse_json_strings(raw)
    cache[key] = paraphrases
    _append_llm_cache(cache_path, key, paraphrases)
    return paraphrases


def _parse_json_strings(raw: str) -> list[str]:
    text = raw.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in parsed if isinstance(item, (str,)) and str(item).strip()]


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #
def build_corpus(
    *,
    contexts: list[dict],
    weights: dict[str, float],
    lambda_pref: float = 1.0,
    use_llm: bool = False,
    llm_per_intent: int = 3,
    llm_cache_path: Path = DEFAULT_LLM_CACHE,
) -> list[dict]:
    cache = _load_llm_cache(llm_cache_path) if use_llm else {}
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for entry in contexts:
        scenario_id = entry["scenario_id"]
        context = SubgoalContext.coerce(entry["context"])
        feasible = entry["feasible_subgoals"]
        intents = feedback_intents(weights, context, feasible, lambda_pref=lambda_pref)
        context_desc = context_description(context)

        for intent_idx, intent in enumerate(intents):
            texts: list[tuple[str, str]] = [
                (text, "template") for text in render_templates(intent, context)
            ]
            if use_llm and texts:
                for para in _llm_paraphrases(
                    context_desc,
                    intent,
                    n=llm_per_intent,
                    cache=cache,
                    cache_path=llm_cache_path,
                ):
                    texts.append((para, "llm"))

            for text_idx, (text, source) in enumerate(texts):
                dedup_key = (scenario_id, text.lower())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                feedback_id = f"{scenario_id}#{intent_idx}_{intent.role}_{source}{text_idx}"
                examples.append(
                    {
                        "feedback_id": feedback_id,
                        "text": text,
                        "expected_feedback_type": intent.feedback_type,
                        "target_features": {
                            feature: float(value)
                            for feature, value in intent.target_features.items()
                        },
                        "attributed_sentiment_score": float(intent.polarity),
                        "group_id": scenario_id,
                        "referenced_subgoal": intent.subgoal,
                        "role": intent.role,
                        "source": source,
                        "context": entry["context"],
                        "feasible_subgoals": feasible,
                        "expected_subgoal": entry["expected_subgoal"],
                        "acceptable_subgoals": entry["acceptable_subgoals"],
                    }
                )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument("--llm-cache", type=Path, default=DEFAULT_LLM_CACHE)
    parser.add_argument("--llm-per-intent", type=int, default=3)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force template-only generation even if DEEPSEEK_API_KEY is set.",
    )
    args = parser.parse_args()

    weights = load_gold_weights()
    if args.contexts.exists():
        contexts = read_json(args.contexts)
    else:
        print(f"Contexts file not found ({args.contexts}); enumerating on the fly.")
        contexts = enumerate_contexts(weights, lambda_pref=args.lambda_pref)

    use_llm = (not args.no_llm) and bool(os.getenv("DEEPSEEK_API_KEY"))
    if not use_llm:
        reason = "--no-llm" if args.no_llm else "DEEPSEEK_API_KEY not set"
        print(f"LLM augmentation disabled ({reason}); using templates only.")

    examples = build_corpus(
        contexts=contexts,
        weights=weights,
        lambda_pref=args.lambda_pref,
        use_llm=use_llm,
        llm_per_intent=args.llm_per_intent,
        llm_cache_path=args.llm_cache,
    )

    probes = load_probe_states(args.probe_states)
    validate_feedback_examples(examples, probe_states=probes)
    write_json(args.output, examples)

    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for example in examples:
        by_type[example["expected_feedback_type"]] = (
            by_type.get(example["expected_feedback_type"], 0) + 1
        )
        by_source[example["source"]] = by_source.get(example["source"], 0) + 1
    print(f"Generated {len(examples)} synthetic feedback examples (validated).")
    print(f"  by feedback type: {dict(sorted(by_type.items()))}")
    print(f"  by source: {dict(sorted(by_source.items()))}")
    print(f"  scenarios (groups): {len({e['group_id'] for e in examples})}")
    print(f"Corpus JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
