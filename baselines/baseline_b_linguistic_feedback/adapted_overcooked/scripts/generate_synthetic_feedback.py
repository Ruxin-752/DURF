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
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[2]))

from scripts.enumerate_subgoal_contexts import enumerate_contexts  # noqa: E402
from src.feature_schema import read_json, validate_feedback_examples, write_json  # noqa: E402
from src.feedback_templates import (  # noqa: E402
    NEGATIVE_THEME_PHRASES,
    POSITIVE_THEME_PHRASES,
    SUBGOAL_PHRASES,
    context_description,
    is_single_sentence,
    phrase_annotations,
    render_templates,
    theme_for_intent,
)
from src.probe_evaluator import DEFAULT_PROBE_STATES_PATH, load_probe_states  # noqa: E402
from src.subgoal_featurizer import SubgoalContext  # noqa: E402
from src.subgoal_teacher import feedback_intents, load_gold_weights  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "synthetic_feedback.json"
DEFAULT_CONTEXTS = ROOT / "outputs" / "synth" / "contexts.json"
DEFAULT_LLM_CACHE = ROOT / "outputs" / "synth" / "llm_cache.jsonl"
GENERATOR_VERSION = "deepseek-observer-v7"
DEFAULT_MODEL = "deepseek-chat"
PROMPT_VARIANTS = ("direct", "contrastive", "colloquial")

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
            paraphrases = [
                str(p).strip()
                for p in (record.get("paraphrases") or [])
                if str(p).strip()
            ]
            # Prefer a non-empty entry if the same key appears more than once.
            if record["key"] not in cache or (paraphrases and not cache[record["key"]]):
                cache[record["key"]] = paraphrases
    # Drop empties so missing intents are retried.
    return {key: paras for key, paras in cache.items() if paras}


def _append_llm_cache(
    path: Path, key: str, paraphrases: list[str], metadata: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"key": key, "paraphrases": paraphrases, "metadata": metadata or {}},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _prompt_key(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    top_p: float | None,
    split: str,
    prompt_variant: str,
    n: int,
) -> str:
    """Hash every semantic/reproducibility input; never include credentials."""

    payload = {
        "version": GENERATOR_VERSION,
        "messages": messages,
        "model": model,
        "sampling": {"temperature": temperature, "top_p": top_p, "n": n},
        "split": split,
        "prompt_variant": prompt_variant,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "dev" if bucket == 8 else "test"


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


def _reference_instruction(reference_type: str, feedback_type: str) -> str:
    if reference_type == "action_behavioral":
        if feedback_type == "imperative":
            return (
                "Command the AI to continue or stop a REPEATED BEHAVIOR PATTERN. "
                "Name the pattern with a history marker such as 'keep', 'always', "
                "'again', or 'every time'; do not reduce it to one isolated move."
            )
        return (
            "Describe a REPEATED BEHAVIOR PATTERN, not only this one move. "
            "Use a natural history marker such as 'keep', 'always', 'again', "
            "'every time', or 'usually'. Do not phrase it as a command."
        )
    if reference_type == "feature":
        if feedback_type == "imperative":
            return (
                "Command the AI to change or preserve the named behavior/property "
                "and explicitly name its effect (for example blocking, duplication, "
                "or supplying a needed ingredient). Focus on that property, not a "
                "location or whole sequence, and do not describe a repeated history."
            )
        return (
            "Explicitly name the behavior/property and its effect (for example "
            "blocking, duplication, or supplying a needed ingredient). Avoid "
            "location cues and whole-sequence language. Do not describe a repeated history."
        )
    if reference_type == "action_spatial":
        if feedback_type == "imperative":
            return (
                "Command one isolated action at the current place/time. Include a "
                "natural deictic cue such as 'right now', 'right there', 'at the pot', "
                "or 'from that spot'; do not describe a habit or whole sequence."
            )
        return (
            "Refer to one isolated action at the current place/time and include a "
            "natural deictic cue such as 'right now', 'right there', 'at the pot', "
            "or 'from that spot'. Do not describe a habit, property, or whole trajectory."
        )
    if feedback_type == "imperative":
        return (
            "Command the AI to repeat or avoid the just-completed trajectory as a whole. "
            "Include a whole-trajectory cue such as 'the whole move', 'that sequence', "
            "or 'what you just did from start to finish'."
        )
    return (
        "Refer to the just-completed trajectory as a whole. Include a natural "
        "whole-trajectory cue such as 'overall', 'the whole move', 'that sequence', "
        "or 'what you just did from start to finish'; do not mention only one action."
    )


def _llm_paraphrases(
    context_desc: str,
    intent,
    *,
    n: int,
    cache: dict[str, list[str]],
    cache_path: Path,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    split: str = "train",
    prompt_variant: str = "direct",
) -> list[str]:
    """Ask DeepSeek to verbalize a rule-teacher label as an observer/critic."""

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

    variant_hint = {
        "direct": "Use direct, concrete wording.",
        "contrastive": "Make the reaction distinct from stock praise or criticism.",
        "colloquial": "Use casual spoken wording and natural contractions.",
    }.get(prompt_variant, "Use direct, concrete wording.")
    system = (
        "You are a person playing the cooperative game Overcooked with an AI "
        "teammate. Act only as an OBSERVER/CRITIC of the AI's behavior. Never "
        "narrate what you intend to do and never choose the label: the rule "
        "teacher below is authoritative. Return short spoken feedback."
    )
    user = (
        f"Rule-teacher label: polarity={stance}; speech_act={intent.feedback_type}; "
        f"reference_type={intent.reference_type}; features="
        f"{','.join(intent.referenced_features)}.\n"
        f"Situation: {context_desc}\n"
        f"The AI is {gerund} ({noun}).\n"
        f"Your honest view: this is {stance} because {theme_phrase}.\n"
        f"{_type_instruction(intent.feedback_type, positive)}\n"
        f"Reference style: {intent.reference_type}. "
        f"{_reference_instruction(intent.reference_type, intent.feedback_type)}\n"
        f"{variant_hint}\nWrite {n} DIFFERENT natural one-sentence reactions "
        f"you'd say to the AI. Hard rules:\n"
        f"- Every sentence must stay clearly {stance}; never sound {opposite}.\n"
        f"- Every sentence must clearly be about {noun} / {gerund}.\n"
        f"- Use exactly ONE reference clause and no semicolon or newline.\n"
        f"- You are the critic; never say what you will do.\n"
        f"- Vary the wording; do not just reorder the same words.\n"
        f"Respond with a JSON array of {n} strings and nothing else."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    key = _prompt_key(
        messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        split=split,
        prompt_variant=prompt_variant,
        n=n,
    )
    if key in cache:
        return cache[key]
    try:
        raw = chat_once(
            messages,
            temperature=temperature,
            top_p=top_p,
            model=model,
        )
    except Exception as exc:  # network hiccup / service error: skip this intent
        # Do NOT cache the empty result, so a later rerun retries this intent
        # instead of permanently losing its paraphrases.
        print(f"  [llm] skipped (transient error, will retry on rerun): {exc}")
        return []

    paraphrases = [
        text for text in _parse_json_strings(raw)
        if is_single_sentence(text) and ";" not in text and "\n" not in text
    ]
    if not paraphrases:
        # Successful HTTP but empty/unparseable payload: still do not cache.
        print("  [llm] skipped (empty/unparseable response, will retry on rerun)")
        return []

    cache[key] = paraphrases
    _append_llm_cache(
        cache_path,
        key,
        paraphrases,
        {
            "version": GENERATOR_VERSION,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "split": split,
            "prompt_variant": prompt_variant,
            "prompt": messages,
        },
    )
    return paraphrases


def _parse_json_strings(raw: str) -> list[str]:
    def clean_item(item: str) -> str:
        # DeepSeek occasionally emits a valid JSON string whose content itself
        # has one unmatched quote at the boundary. Feedback never needs outer
        # quotation marks, so remove those artifacts before caching.
        return str(item).strip().strip("\"'").strip()

    text = raw.strip()
    if "```" in text:
        fenced = [
            block.strip()
            for index, block in enumerate(text.split("```"))
            if index % 2 == 1 and block.strip()
        ]
        if fenced:
            text = fenced[0]
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("paraphrases", "reactions", "sentences", "responses", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if isinstance(parsed, list):
        return [
            clean_item(item)
            for item in parsed
            if isinstance(item, str) and clean_item(item)
        ]

    # Some otherwise useful responses ignore the JSON-only instruction and
    # return bullets/numbered lines. Recover those strings; corpus validation
    # still enforces valence and grounding before they can enter training.
    recovered = []
    for line in raw.splitlines():
        candidate = line.strip()
        structured_line = bool(
            re.match(r"^(?:[-*]\s+|\d+[.)]\s+|[\"'])", candidate)
        )
        if not structured_line:
            continue
        candidate = candidate.removeprefix("```json").removeprefix("```")
        candidate = candidate.rstrip("`").strip()
        candidate = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", candidate)
        candidate = clean_item(candidate.rstrip(","))
        if (
            candidate
            and len(candidate.split()) >= 3
            and not candidate.lower().startswith(("here are", "sure", "note:"))
        ):
            recovered.append(candidate)
    return recovered


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
    llm_reference_types: set[str] | None = None,
    llm_max_intents: int | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float | None = 0.9,
    prompt_variants: tuple[str, ...] = PROMPT_VARIANTS,
    resample_rounds: int = 1,
    target_size: int | None = None,
    bucket_target: int | None = None,
    seed: int = 0,
) -> list[dict]:
    cache = _load_llm_cache(llm_cache_path) if use_llm else {}
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()
    llm_selected_keys: set[tuple[str, int, str]] | None = None
    if use_llm and llm_max_intents is not None:
        candidates: list[tuple[str, tuple[str, int, str]]] = []
        for entry in contexts:
            scenario_id = entry["scenario_id"]
            context = SubgoalContext.coerce(entry["context"])
            intents = feedback_intents(
                weights,
                context,
                entry["feasible_subgoals"],
                lambda_pref=lambda_pref,
            )
            for intent_idx, intent in enumerate(intents):
                if (
                    llm_reference_types is not None
                    and intent.reference_type not in llm_reference_types
                ):
                    continue
                key = (scenario_id, intent_idx, intent.role)
                priority = hashlib.sha1(
                    f"{seed}|{scenario_id}|{intent_idx}|{intent.role}".encode("utf-8")
                ).hexdigest()
                candidates.append((priority, key))
        candidates.sort()
        llm_selected_keys = {
            key for _priority, key in candidates[: max(0, llm_max_intents)]
        }

    for entry in contexts:
        scenario_id = entry["scenario_id"]
        split = _split_for_group(scenario_id)
        context = SubgoalContext.coerce(entry["context"])
        feasible = entry["feasible_subgoals"]
        intents = feedback_intents(weights, context, feasible, lambda_pref=lambda_pref)
        context_desc = context_description(context)

        for intent_idx, intent in enumerate(intents):
            texts: list[tuple[str, str]] = [
                (text, "template") for text in render_templates(intent, context)
            ]
            reference_selected = (
                llm_reference_types is None
                or intent.reference_type in llm_reference_types
            )
            intent_key = (scenario_id, intent_idx, intent.role)
            under_budget = (
                llm_selected_keys is None or intent_key in llm_selected_keys
            )
            if use_llm and texts and reference_selected and under_budget:
                variants = prompt_variants or ("direct",)
                variant_offset = int(
                    hashlib.sha1(intent_key.__repr__().encode("utf-8")).hexdigest()[:8],
                    16,
                )
                for round_index in range(max(1, resample_rounds)):
                    variant = variants[(variant_offset + round_index) % len(variants)]
                    for para in _llm_paraphrases(
                        context_desc,
                        intent,
                        n=llm_per_intent,
                        cache=cache,
                        cache_path=llm_cache_path,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        split=split,
                        prompt_variant=variant,
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
                        "reference_type": intent.reference_type,
                        "phrase_annotations": phrase_annotations(
                            text, intent.reference_type
                        ),
                        "paraphrase_family": (
                            f"{scenario_id}#{intent_idx}_{intent.role}"
                        ),
                        "target_features": {
                            feature: float(value)
                            for feature, value in intent.target_features.items()
                        },
                        "referenced_features": list(intent.referenced_features),
                        "attributed_sentiment_score": float(intent.polarity),
                        "group_id": scenario_id,
                        "referenced_subgoal": intent.subgoal,
                        "role": intent.role,
                        "source": source,
                        "label_source": intent.label_source,
                        "generator_version": GENERATOR_VERSION,
                        "split": split,
                        "context": entry["context"],
                        "feasible_subgoals": feasible,
                        "expected_subgoal": entry["expected_subgoal"],
                        "acceptable_subgoals": entry["acceptable_subgoals"],
                    }
                )
    if bucket_target is None and target_size is None:
        return examples

    buckets: dict[tuple[str, str, str], list[dict]] = {}
    for example in examples:
        polarity = "positive" if example["attributed_sentiment_score"] > 0 else "negative"
        key = (
            example["reference_type"],
            example["expected_feedback_type"],
            polarity,
        )
        buckets.setdefault(key, []).append(example)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    if bucket_target is None:
        bucket_target = max(1, (target_size or len(examples)) // max(1, len(buckets)))
    selected = [row for key in sorted(buckets) for row in buckets[key][:bucket_target]]
    if target_size is not None and len(selected) < target_size:
        used = {row["feedback_id"] for row in selected}
        remainder = [row for row in examples if row["feedback_id"] not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: target_size - len(selected)])
    return selected[:target_size] if target_size is not None else selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-pref", type=float, default=1.0)
    parser.add_argument("--llm-cache", type=Path, default=DEFAULT_LLM_CACHE)
    parser.add_argument("--llm-per-intent", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        choices=PROMPT_VARIANTS,
        default=list(PROMPT_VARIANTS),
    )
    parser.add_argument(
        "--resample-rounds",
        type=int,
        default=1,
        help="Retry underfilled intent buckets with successive prompt variants.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Deterministically balance and cap the corpus (for example 10000).",
    )
    parser.add_argument(
        "--bucket-target",
        type=int,
        default=None,
        help="Maximum examples per reference/speech-act/polarity bucket.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm-reference-types",
        nargs="+",
        choices=(
            "trajectory",
            "feature",
            "action_spatial",
            "action_behavioral",
        ),
        default=None,
        help="Spend API calls only on selected weak reference classes.",
    )
    parser.add_argument(
        "--llm-max-intents",
        type=int,
        default=None,
        help="Maximum selected intents allowed to trigger an API/cache lookup.",
    )
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
        llm_reference_types=(
            set(args.llm_reference_types) if args.llm_reference_types else None
        ),
        llm_max_intents=args.llm_max_intents,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_variants=tuple(args.prompt_variants),
        resample_rounds=args.resample_rounds,
        target_size=args.target_size,
        bucket_target=args.bucket_target,
        seed=args.seed,
    )

    probes = load_probe_states(args.probe_states)
    validate_feedback_examples(examples, probe_states=probes)
    write_json(args.output, examples)

    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    for example in examples:
        by_type[example["expected_feedback_type"]] = (
            by_type.get(example["expected_feedback_type"], 0) + 1
        )
        by_source[example["source"]] = by_source.get(example["source"], 0) + 1
        polarity = "positive" if example["attributed_sentiment_score"] > 0 else "negative"
        bucket = (
            f"{example['reference_type']}|"
            f"{example['expected_feedback_type']}|{polarity}"
        )
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
    print(f"Generated {len(examples)} synthetic feedback examples (validated).")
    print(f"  by feedback type: {dict(sorted(by_type.items()))}")
    print(f"  by source: {dict(sorted(by_source.items()))}")
    print(f"  by orthogonal bucket: {dict(sorted(by_bucket.items()))}")
    print(f"  scenarios (groups): {len({e['group_id'] for e in examples})}")
    print(f"Corpus JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
