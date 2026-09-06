"""Expand the persona utterance seeds into a paraphrase bank with DeepSeek.

Why offline and once: with the LLM attributor on, sim personas no longer have
to speak in keyword-matcher-friendly templates -- but calling an LLM at every
sim step would make the simulator slow and non-reproducible.  So the variety is
generated HERE, once, checked in as data, and sampled at sim time.

Output: data/sim_personas/utterances_v1.json
  {"generated_at", "model", "n_per_seed", "utterances": {dimension: {preference: [..]}}}

sim_session.py loads that file at import if it exists and otherwise falls back
to the inline seeds, so the simulator works either way.

Every generated sentence is passed back through the deterministic fallback
parser; sentences the fallback cannot read are KEPT (the LLM attributor is the
primary reader now) but counted and listed, so you can see how much of the bank
depends on the LLM being up.

Usage (needs DEEPSEEK_API_KEY):
    python scripts/generate_persona_utterances.py --n 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from durf.feedback_attribution.llm_attributor import extract_json_object  # noqa: E402
from durf.feedback_attribution.subgoal_preferences import (  # noqa: E402
    infer_explicit_preference_from_text,
)
from durf.group_a.deepseek_chat import chat_once_detailed  # noqa: E402
from durf.group_a.persona_utterances import UTTERANCE_SEEDS as _UTTERANCE_SEEDS  # noqa: E402

SITUATION = {
    "ingredient_order": "The pot still needs both a tomato and an onion; the AI teammate is empty-handed and about to fetch one of them.",
    "division_of_labour": "A soup is cooking. The AI teammate could either fetch a plate for it or start fetching ingredients for the next soup.",
    "backup_dish": "A soup is cooking and the human already holds a plate. The AI teammate could fetch a second plate or go fetch ingredients instead.",
    "plate_wait": "The AI teammate is holding an empty plate, but the soup is not ready yet and no pot is cooking.",
}
MEANING = {
    ("ingredient_order", "onion_first"): "the human wants the AI to fetch the ONION before the tomato",
    ("ingredient_order", "tomato_first"): "the human wants the AI to fetch the TOMATO before the onion",
    ("division_of_labour", "prep_first"): "the human will fetch the plate themself and wants the AI to fetch ingredients for the next soup instead of the plate",
    ("division_of_labour", "dish_first"): "the human wants the AI to fetch the plate while the human handles the ingredients",
    ("backup_dish", "take_backup"): "the human wants the AI to fetch a second plate even though the human already has one",
    ("backup_dish", "no_backup"): "the human does NOT want the AI to fetch another plate since the human already has one; the AI should fetch ingredients instead",
    ("plate_wait", "hold_plate"): "the human wants the AI to keep holding the plate and wait by the pot",
    ("plate_wait", "free_hands"): "the human wants the AI to put the plate down for now and go do something useful, picking it up later",
}


def prompt(dimension: str, preference: str, seeds: list[str], n: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You write short, natural things a person might say out loud to an AI teammate "
                "while playing the cooperative cooking game Overcooked. Casual spoken English, "
                "one sentence each, 5-20 words, no numbering, varied phrasing and tone "
                "(some polite, some blunt, some joking). " + ALLOWED_HINT + " Output strict JSON: "
                '{"sentences": ["...", "..."]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Situation: {SITUATION[dimension]}\n"
                f"What the person means: {MEANING[(dimension, preference)]}.\n"
                f"Examples of the intended meaning:\n- " + "\n- ".join(seeds) + "\n\n"
                f"Write {n} NEW sentences with exactly that meaning. Each must make clear "
                f"which option the person wants; do not contradict the meaning; do not "
                f"mention rules or scores."
            ),
        },
    ]


# The game has exactly these things.  The first live run produced "I'm on the
# garlic", "grab the flour", "start peeling" -- sentences the attributor cannot
# map to any subgoal and that would only add noise to the sim corpus.
BANNED_WORDS = (
    "carrot", "garlic", "flour", "pepper", "salt", "rice", "noodle", "meat", "chicken",
    "fish", "cheese", "mushroom", "lettuce", "potato", "chop", "peel", "slice", "cut",
    "knife", "board", "veggie", "vegetable", "recipe", "oven", "stove", "fry", "bake",
    "sauce", "spice", "season", "wash", "sink",
)
ALLOWED_HINT = (
    "The kitchen has ONLY: tomatoes, onions, plates (also called dishes), one pot, "
    "and soup. There is no chopping, no other ingredient, no other equipment. "
    "Never mention anything else."
)


def repair_json(text: str) -> str:
    """Close a JSON object the model forgot to close (seen live: 20 sentences
    and then end-of-output with no ']}')."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if "{" in stripped and stripped.rfind("}") < stripped.find("{"):
        # drop a trailing partial string, then close array + object
        last_quote = stripped.rfind('"')
        cut = stripped[: last_quote + 1] if last_quote > 0 else stripped
        cut = cut.rstrip().rstrip(",")
        if cut.count("[") > cut.count("]"):
            cut += "]"
        cut += "}"
        return cut
    return stripped


def ask(messages: list[dict], attempts: int = 3):
    last = None
    for _ in range(attempts):
        result = chat_once_detailed(messages, temperature=0.7, max_tokens=1200, use_cache=False)
        try:
            parsed = extract_json_object(repair_json(result.text))
            return parsed, result
        except (ValueError, json.JSONDecodeError) as exc:  # noqa: PERF203
            last = exc
    raise RuntimeError(f"could not parse a JSON answer after {attempts} attempts: {last}")


def acceptable(sentence: str) -> bool:
    low = sentence.lower()
    return not any(word in low for word in BANNED_WORDS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=20, help="new sentences per preference")
    parser.add_argument("--batch", type=int, default=10, help="sentences per API call")
    parser.add_argument("--output", type=Path, default=REPO / "data/sim_personas/utterances_v1.json")
    parser.add_argument("--resume", action="store_true",
                        help="keep preferences already present in --output and only fill the missing ones")
    args = parser.parse_args(argv)

    existing: dict = {}
    if args.resume and args.output.exists():
        existing = (json.loads(args.output.read_text(encoding="utf-8")).get("utterances") or {})

    bank: dict[str, dict[str, list[str]]] = {}
    unreadable: list[tuple[str, str, str]] = []
    banned_dropped = 0
    model_seen = None
    for dimension, prefs in _UTTERANCE_SEEDS.items():
        bank[dimension] = {}
        for preference, seeds in prefs.items():
            prior = (existing.get(dimension) or {}).get(preference)
            if prior and len(prior) >= len(seeds) + args.n:
                bank[dimension][preference] = prior
                print(f"{dimension}/{preference}: kept {len(prior)} from previous run")
                continue
            new: list[str] = []
            while len(new) < args.n:
                want = min(args.batch, args.n - len(new))
                parsed, result = ask(prompt(dimension, preference, seeds + new, want))
                model_seen = result.model_returned or result.model_requested
                batch = [str(s).strip() for s in parsed.get("sentences") or [] if str(s).strip()]
                if not batch:
                    break
                for sentence in batch:
                    if not acceptable(sentence):
                        banned_dropped += 1
                        continue
                    if sentence not in seeds and sentence not in new:
                        new.append(sentence)
            merged = list(dict.fromkeys(seeds + new))
            for sentence in new:
                if not infer_explicit_preference_from_text(sentence)[0]:
                    unreadable.append((dimension, preference, sentence))
            bank[dimension][preference] = merged
            print(f"{dimension}/{preference}: {len(seeds)} seeds + {len(new)} new")
            # checkpoint after every preference so a crash keeps what was done
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "model": model_seen, "n_per_seed": args.n, "partial": True,
                            "utterances": {**existing, **bank}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    args.output.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model": model_seen,
                "n_per_seed": args.n,
                "partial": False,
                "utterances": bank,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(len(v) for d in bank.values() for v in d.values())
    print(f"\nwrote {total} sentences -> {args.output}")
    print(f"{banned_dropped} generated sentences dropped for naming things the game does not have")
    print(
        f"{len(unreadable)} kept sentences are NOT readable by the keyword fallback "
        f"(fine: the LLM attributor is the primary reader). If DeepSeek is down during a "
        f"session, feedback drawn from these will be labelled rule_fallback_after_llm_error "
        f"with no preference extracted."
    )
    for dimension, preference, sentence in unreadable[:15]:
        print(f"   [{dimension}/{preference}] {sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
