"""Generate the classifier's ``other`` rejection class with DeepSeek language.

Labels and topic groups are deterministic; DeepSeek only supplies varied text.
These examples train the reference classifier and never enter reward learning.
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
sys.path.insert(0, str(ROOT.parents[2]))

from scripts.generate_synthetic_feedback import _parse_json_strings  # noqa: E402
from src.feedback_templates import is_single_sentence, phrase_annotations  # noqa: E402
from src.feature_schema import write_json  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "synth" / "deepseek_other_reference.json"
DEFAULT_CACHE = ROOT / "outputs" / "synth" / "other_llm_cache.jsonl"
GENERATOR_VERSION = "deepseek-hard-other-v2"
TOPICS = {
    "greeting": "greetings and checking whether the teammate can hear you",
    "controls": "learning controls, key bindings, or asking how to interact",
    "technical": "lag, frame rate, audio, connection, or display problems",
    "break": "asking for a pause, break, restart, or another round",
    "score": "asking about score, timer, map, recipe, or game rules",
    "social": "brief casual small talk unrelated to any move in the kitchen",
    "kitchen_question": (
        "questions containing kitchen words (onion, tomato, soup, pot, dish, "
        "counter, serve) that ask about game state or rules, not the AI's behavior"
    ),
    "self_intent": (
        "first-person plans using kitchen words, such as what I will fetch, cook, "
        "plate, or serve; they must describe the human speaker's own intent"
    ),
    "state_statement": (
        "neutral kitchen-state observations about ingredients, pots, dishes, "
        "orders, counters, or timers, with no judgment of the AI"
    ),
}
FALLBACK = {
    "greeting": ["Hello teammate.", "Can you hear me?", "Are you ready?"],
    "controls": ["Which key interacts?", "I am learning the controls.", "How do I move?"],
    "technical": ["The game is lagging.", "My audio is broken.", "The screen froze."],
    "break": ["I need a short break.", "Can we restart?", "One moment please."],
    "score": ["What is the score?", "How much time is left?", "Which map is this?"],
    "social": ["This is fun.", "That kitchen looks different.", "Ready for another round?"],
    "kitchen_question": [
        "How many onions does this soup need?",
        "Is that pot ready yet?",
        "Where do clean dishes spawn?",
    ],
    "self_intent": [
        "I'll grab the next tomato.",
        "I'm going to plate the soup.",
        "Let me handle the onions.",
    ],
    "state_statement": [
        "The left pot has two onions.",
        "A clean dish is on the counter.",
        "The tomato order has thirty seconds left.",
    ],
}


def _load_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key") and row.get("texts"):
                cache[row["key"]] = row["texts"]
    return cache


def generate(
    per_topic: int,
    *,
    cache_path: Path = DEFAULT_CACHE,
    model: str = "deepseek-chat",
    temperature: float = 0.8,
    top_p: float | None = 0.95,
    prompt_variant: str = "hard-negative",
    use_llm: bool = True,
) -> list[dict]:
    cache = _load_cache(cache_path)
    rows = []
    for topic, description in TOPICS.items():
        split_digit = int(hashlib.sha1(topic.encode()).hexdigest()[:4], 16) % 10
        split = "train" if split_digit < 8 else "dev" if split_digit == 8 else "test"
        prompt = (
            f"Write {per_topic} diverse one-sentence things a human player might say "
            f"during Overcooked about {description}. These must NOT praise, criticize, "
            "command, or describe the AI teammate's current or past action. Kitchen "
            "words are encouraged because these are hard negatives. Self-intent must "
            "use I/me/my, questions must remain questions, and state statements must "
            "stay neutral. Respond with a JSON array of strings."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate hard classifier rejection examples as an Overcooked "
                    "player. Never turn them into feedback about the AI's behavior."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        cache_payload = {
            "version": GENERATOR_VERSION,
            "prompt": messages,
            "model": model,
            "sampling": {"temperature": temperature, "top_p": top_p, "n": per_topic},
            "split": split,
            "prompt_variant": prompt_variant,
        }
        key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        generated = cache.get(key, [])
        if use_llm and not generated:
            try:
                from durf.group_a.deepseek_chat import chat_once

                generated = [
                    text
                    for text in _parse_json_strings(
                        chat_once(
                            messages,
                            model=model,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    )
                    if is_single_sentence(text)
                ]
            except Exception as exc:
                print(f"[other:{topic}] DeepSeek failed; using floor: {exc}")
                generated = []
            if generated:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"key": key, "texts": generated, "metadata": cache_payload},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        texts = list(dict.fromkeys([*FALLBACK[topic], *generated]))
        for index, text in enumerate(texts):
            digest = hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:10]
            rows.append(
                {
                    "feedback_id": f"other_{topic}_{digest}",
                    "text": text,
                    "reference_type": "other",
                    "role": "other",
                    "group_id": f"other_topic_{topic}",
                    "paraphrase_family": f"other_topic_{topic}",
                    "source": "llm" if index >= len(FALLBACK[topic]) else "template",
                    "phrase_annotations": phrase_annotations(text, "other"),
                    "split": split,
                    "generator_version": GENERATOR_VERSION,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-topic", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-variant", default="hard-negative")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    use_llm = not args.no_llm and bool(os.getenv("DEEPSEEK_API_KEY"))
    rows = generate(
        args.per_topic,
        cache_path=args.cache,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_variant=args.prompt_variant,
        use_llm=use_llm,
    )
    write_json(args.output, rows)
    print(
        json.dumps(
            {
                "examples": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
