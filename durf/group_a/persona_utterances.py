"""Persona utterance seeds and the paraphrase bank -- stdlib only.

Kept out of sim_session.py on purpose: that module imports the whole game
runtime (ray, tensorflow), which is both slow and fragile across environments
(a numpy-2 / tensorflow mismatch on the experiment laptop made it unimportable).
Anything that only needs the WORDS -- the paraphrase generator, tests, the
attributor -- imports this module instead.
"""

from __future__ import annotations

import json
from pathlib import Path

# What each preference sounds like.  These are SEEDS: with the LLM attributor
# on there is no keyword matcher to appease, so the wording can be natural and
# varied.  scripts/generate_persona_utterances.py expands each list to a few
# dozen paraphrases into data/sim_personas/utterances_v1.json; if that file
# exists it replaces these seeds at import time.
UTTERANCE_SEEDS: dict[str, dict[str, list[str]]] = {
    "ingredient_order": {
        "onion_first": [
            "get the onion first, I'll take care of the tomatoes",
            "you handle the onion, don't worry about tomatoes",
            "stop grabbing the tomato every time, start with the onion",
        ],
        "tomato_first": [
            "tomatoes first, the onion goes in last",
            "you just do tomatoes, I've got the onion",
            "start with a tomato, onion can wait",
        ],
    },
    "division_of_labour": {
        "prep_first": [
            "i'll get the plate, you start the next ingredient",
            "leave the plate to me, keep prepping the next batch",
            "leave the plate to me and go start the next batch, nobody is fetching food",
        ],
        "dish_first": [
            "you get the plate, i'll take care of the ingredients",
            "grab the dish now, i'll handle the next ingredient",
            "plates are yours, ingredients are mine",
        ],
    },
    "backup_dish": {
        "take_backup": [
            "I've got a plate but grab another one too, for the next pot",
            "take a spare dish so the next soup goes straight out",
            "get a second plate ready, we'll need it",
        ],
        "no_backup": [
            "I already have the plate, don't go get another one",
            "why would two of us carry plates -- go do something useful",
            "one plate is enough, leave the dish and fetch ingredients",
        ],
    },
    "plate_wait": {
        "hold_plate": [
            "don't put the plate down, wait by the pot with it",
            "keep hold of the plate and stand by the pot",
            "hang on to the dish and stay by the pot, the soup is almost there",
        ],
        "free_hands": [
            "soup isn't ready, put the plate down and go do something",
            "don't stand there holding a plate, drop it and get ingredients",
            "put the dish down for now, pick it up when the soup is done",
        ],
    },
}
UTTERANCE_BANK_PATH = Path("data/sim_personas/utterances_v1.json")


def load_utterance_bank() -> dict[str, dict[str, list[str]]]:
    bank = {dim: {k: list(v) for k, v in vals.items()} for dim, vals in UTTERANCE_SEEDS.items()}
    if UTTERANCE_BANK_PATH.exists():
        try:
            loaded = json.loads(UTTERANCE_BANK_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return bank
        for dim, vals in (loaded.get("utterances") or {}).items():
            for pref, sentences in (vals or {}).items():
                if isinstance(sentences, list) and sentences:
                    bank.setdefault(dim, {})[pref] = [str(x) for x in sentences]
    return bank




UTTERANCE_BANK = load_utterance_bank()
