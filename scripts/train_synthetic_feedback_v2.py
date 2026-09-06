"""Second synthetic speech-act candidate after the first acceptance set failed.

The exposed first acceptance set is now development data, never training data.
No round-two/frozen-next source is accessed. The v1 grounding model is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import shutil

import train_synthetic_feedback_v1 as v1

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026090602
EXPOSED_PATH = ROOT / "artifacts" / "synthetic-acceptance-20260906" / "frozen-v2.json"
EXPOSED_SHA = "4473f6647e428777f92f03e54a4c6c561976775bf64cb0fe567dfd5c50105ff1"

QUALITY = ["good", "bad", "great", "poor", "excellent", "terrible", "awful", "nice",
    "brilliant", "smart", "foolish", "sensible", "unwise", "wise", "helpful", "unhelpful",
    "useful", "useless", "valuable", "worthless", "important", "unimportant", "efficient",
    "inefficient", "wasteful", "effective", "ineffective", "ideal", "perfect", "pointless",
    "convenient", "inconvenient", "beneficial", "detrimental", "harmful", "preferable", "worthwhile",
    "enjoyable", "frustrating", "annoying", "satisfying", "impressive", "disappointing",
    "fantastic", "wonderful", "unnecessary", "necessary", "unacceptable", "acceptable",
    "clever", "clumsy", "careless", "excellent", "wrong", "right", "a mistake", "a nuisance",
    "a success", "a failure", "a waste", "a waste of time", "a waste of effort",
    "the best", "the worst", "my favorite", "my least favorite", "no use", "no help"]
ADJECTIVES = [q for q in QUALITY if not q.startswith(("a ", "the ", "my ", "no "))]
EMOTIONS = ["pleased", "happy", "unhappy", "satisfied", "dissatisfied", "delighted",
    "disappointed", "annoyed", "impressed", "frustrated", "upset", "content"]
ITEMS = ["onion", "tomato", "pot", "bowl", "plate", "dish", "ladle", "ingredient",
    "vegetable", "carrot", "tray", "soup", "meal", "recipe", "order"]
ITEM_NPS = ["the onion", "that tomato", "this empty bowl", "a clean plate", "the dirty dish",
    "the spare ladle", "the cooked soup", "that ready meal", "another ingredient", "the last onion",
    "your plate", "my bowl", "the full pot", "this hot soup", "the remaining tomato",
    "the extra plate", "one more bowl", "the chopped vegetable", "the next order"]
STATIONS = ["counter", "shelf", "stove", "sink", "pot", "serving hatch", "delivery window",
    "worktop", "onion crate", "tomato dispenser", "dish rack", "island", "corridor", "passage",
    "doorway", "tile", "square", "cooking station", "serving station", "waiting area"]
STATION_NPS = [f"the {side} {station}" for side, station in itertools.product(
    ["left", "right", "upper", "lower", "central", "nearest", "far", "empty", "other", "middle"], STATIONS)]
ALL_NPS = ITEM_NPS + STATION_NPS
LOCATIONS = [f"{prep} {station}" for prep, station in itertools.product(
    ["beside", "near", "behind", "next to", "in front of", "to the left of", "to the right of",
     "opposite", "across from", "above", "below", "under", "to the west of", "to the east of"],
    ["the stove", "the sink", "the pot", "the island", "the serving hatch", "the window", "the crate"])]
FACT_STATES = ["empty", "full", "clean", "dirty", "ready", "cooking", "occupied", "unoccupied",
    "blocked", "open", "closed", "available", "unavailable", "hot", "cold", "on the left",
    "on the right", "at the top", "at the bottom", "covered", "uncovered", "lit", "unlit"]
ACTORS = [("You", "are", "have"), ("I", "am", "have"), ("We", "are", "have"),
    ("Your partner", "is", "has"), ("My teammate", "is", "has"), ("The cook", "is", "has"),
    ("The other chef", "is", "has"), ("Both cooks", "are", "have"), ("The two chefs", "are", "have")]


def make_actions():
    actions = list(v1.ACTION_RECORDS)
    for obj in ITEM_NPS:
        for base, ing, past in [("pick up", "picking up", "picked up"), ("fetch", "fetching", "fetched"),
            ("collect", "collecting", "collected"), ("grab", "grabbing", "grabbed"),
            ("retrieve", "retrieving", "retrieved"), ("take", "taking", "took"),
            ("get", "getting", "got"), ("hold", "holding", "held"), ("release", "releasing", "released")]:
            actions.append((f"{base} {obj}", f"{ing} {obj}", f"{past} {obj}"))
        for loc in ["on the worktop", "on the lower shelf", "beside the sink", "near the window", "by the crate"]:
            for base, ing, past in [("put", "putting", "put"), ("place", "placing", "placed"),
                ("set", "setting", "set"), ("leave", "leaving", "left"), ("drop", "dropping", "dropped")]:
                actions.append((f"{base} {obj} {loc}", f"{ing} {obj} {loc}", f"{past} {obj} {loc}"))
        for dest in ["to the hatch", "to the pot", "to the sink", "across the kitchen", "back to the crate"]:
            for base, ing, past in [("bring", "bringing", "brought"), ("carry", "carrying", "carried"),
                ("deliver", "delivering", "delivered"), ("pass", "passing", "passed")]:
                actions.append((f"{base} {obj} {dest}", f"{ing} {obj} {dest}", f"{past} {obj} {dest}"))
    for place in ["the sink", "the crate", "the stove", "the pot", "the shelf", "the worktop", "the hatch"]:
        for base, ing, past in [("face", "facing", "faced"), ("turn toward", "turning toward", "turned toward"),
            ("turn away from", "turning away from", "turned away from"),
            ("stand beside", "standing beside", "stood beside"), ("walk past", "walking past", "walked past"),
            ("step around", "stepping around", "stepped around"), ("go to", "going to", "went to")]:
            actions.append((f"{base} {place}", f"{ing} {place}", f"{past} {place}"))
    for obj, location in itertools.product(ITEM_NPS,
        ["directly in front of you", "at the back", "on the other side", "beside your partner", "to your left"]):
        actions.append((f"take {obj} {location}", f"taking {obj} {location}", f"took {obj} {location}"))
        actions.append((f"pick up {obj} {location}", f"picking up {obj} {location}", f"picked up {obj} {location}"))
        actions.append((f"let go of {obj} {location}", f"letting go of {obj} {location}", f"let go of {obj} {location}"))
    actions.extend([
        ("keep your hands free", "keeping your hands free", "kept your hands free"),
        ("leave the bowl in place", "leaving the bowl in place", "left the bowl in place"),
        ("wait until the soup is cooked", "waiting until the soup is cooked", "waited until the soup was cooked"),
        ("make sure the counter stays clear", "making sure the counter stays clear", "made sure the counter stayed clear"),
        ("let go of the plate", "letting go of the plate", "let go of the plate"),
        ("get out of the way", "getting out of the way", "got out of the way"),
        ("open up the passage", "opening up the passage", "opened up the passage"),
    ])
    return actions


def augment(train, excluded):
    rng = random.Random(SEED)
    seen = {v1.normalized(r["text"]) for r in train}
    collisions = []
    rows = list(train)
    actions = make_actions()
    def add(family, label, candidates, limit=650):
        unique = list(dict.fromkeys(candidates))
        rng.shuffle(unique)
        count = 0
        for text in unique:
            text = text[0].upper() + text[1:]
            text = " ".join(text.split())
            key = v1.normalized(text)
            if key in excluded:
                collisions.append({"family": family, "text": text})
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "speech_act": label, "grounding": None,
                "group_id": f"v2_{family}", "split": "train", "source": "new_programmatic_synthetic_v2",
                "label_source": "explicit_synthetic_semantics", "human_data": False})
            count += 1
            if count >= limit:
                break

    # New noun/verb/location combinations appear across all communicative acts.
    for family, forms in {
        "command_expanded": ["{a}.", "{a} right away.", "Please {a}.", "{a}, please."],
        "request_expanded": ["Could you {a}?", "Can you {a}?", "Would you please {a}?"],
        "advice_expanded": ["You should {a}.", "We need you to {a}.", "Try to {a}."],
        "negative_expanded": ["Don't {a}.", "Do not {a}.", "Avoid {ing}."],
        "indirect_expanded": ["I would like you to {a}.", "Would you mind {ing}?", "I suggest {ing}."],
        "continuation_expanded": ["Keep {ing}.", "Stop {ing}.", "Continue {ing}."],
        "timed_expanded": ["Before the next delivery, {a}.", "For the next meal, {a}.", "As soon as possible, {a}."],
    }.items():
        add(family, "imperative", (t.format(a=a, ing=ing) for t, (a, ing, past) in itertools.product(forms, actions)), 1200)
    add("pronoun_commands", "imperative", (f"{prefix}{verb} {pronoun}{suffix}."
        for prefix, verb, pronoun, suffix in itertools.product(["", "Please ", "Don't "],
            ["take", "grab", "serve", "deliver", "drop", "hold", "leave", "get", "fetch", "collect", "carry", "move", "put down", "pick up", "bring", "keep"],
            ["it", "that", "this", "them"], ["", " now", " over here", " beside the stove", " on the shelf"])), 900)
    add("spatial_extended_command", "imperative", (f"{verb} {np} {loc}."
        for verb, np, loc in itertools.product(["Put", "Place", "Leave", "Set", "Move", "Keep", "Drop"], ITEM_NPS, LOCATIONS)), 1200)
    add("avoid_location_command", "imperative", (f"{verb} {place} {loc}."
        for verb, place, loc in itertools.product(["Stay out of", "Avoid", "Keep out of", "Don't stand in", "Go around"],
            ["the square", "the passage", "the doorway", "the tile", "the area", "the corner"], LOCATIONS)), 650)
    add("release_hold_command", "imperative", (f"{verb} {np} {loc}."
        for verb, np, loc in itertools.product(["Let go of", "Release", "Stop holding"], ITEM_NPS,
            LOCATIONS + ["at the nearest shelf", "at the far counter", "at the top station", "at the worktop", "right here"])), 750)

    for family, forms in {
        "reported_action_expanded": ["{who} {past}.", "{who} {past} earlier.", "{who} just {past}.", "{who} {past} a moment ago."],
        "observed_action_expanded": ["{who} {be} {ing}.", "{who} {be} currently {ing}.", "{who} {be} still {ing}."],
        "negative_action_expanded": ["{who} did not {a}.", "{who} didn't {a}.", "{who} {be} not {ing}."],
    }.items():
        add(family, "descriptive", (t.format(who=who, be=be, a=a, ing=ing, past=past)
            for t, (who, be, have), (a, ing, past) in itertools.product(forms, ACTORS, actions)), 1600)

    add("evaluated_past_infinitive", "evaluative", (f"It was {q} to {a}."
        for q, (a, ing, past) in itertools.product(QUALITY, actions)), 1600)
    add("evaluated_present_infinitive", "evaluative", (f"It is {q} to {a}."
        for q, (a, ing, past) in itertools.product(QUALITY, actions)), 1000)
    add("executed_gerund_quality_expanded", "evaluative", (f"{ing} was {q}."
        for q, (a, ing, past) in itertools.product(QUALITY, actions)), 1400)
    add("long_past_quality", "evaluative", (f"The way {who.lower()} {past} was {q}."
        for q, (who, be, have), (a, ing, past) in itertools.product(QUALITY, ACTORS, actions)), 1400)
    add("past_quality_clause", "evaluative", (f"{who} {past}, which was {q}."
        for q, (who, be, have), (a, ing, past) in itertools.product(QUALITY, ACTORS, actions)), 1000)
    add("past_quality_thanks", "evaluative", (f"{q.capitalize()} job {ing}."
        for q, (a, ing, past) in itertools.product(ADJECTIVES, actions)), 1000)
    add("past_mistake_negation", "evaluative", (t.format(who=who, ing=ing, a=a, past=past)
        for t, (who, be, have), (a, ing, past) in itertools.product([
            "{who} made no mistake {ing}.", "{who} did not do well {ing}.",
            "{who} did not do a bad job {ing}.", "{who} did not make a poor choice by {ing}.",
            "{who} never should have acted that way when {ing}.",
            "I don't think {who} made a mistake {ing}.",
        ], ACTORS, actions)), 1000)
    irregular_pp = {"went": "gone", "took": "taken", "did": "done", "got": "gotten"}
    add("expanded_counterfactual", "evaluative", (t.format(pp=irregular_pp.get(past.split()[0], past.split()[0]) + " " + " ".join(past.split()[1:]))
        for t, (a, ing, past) in itertools.product([
            "You should have {pp}.", "You should not have {pp}.", "You shouldn't have {pp}.",
            "I wish you had {pp}.", "It would have been better if you had {pp}.",
        ], actions)), 1600)
    add("approval_of_behavior", "evaluative", (f"I {verb} of {ing}."
        for verb, (a, ing, past) in itertools.product(["approve", "disapprove", "do not approve", "strongly disapprove"], actions)), 650)

    # Quality versus state is controlled with the same entity vocabulary.
    add("object_quality", "evaluative", (f"{np} is {q}."
        for np, q in itertools.product(ALL_NPS, QUALITY)), 1500)
    add("object_quality_negated", "evaluative", (f"{np} is not {q}."
        for np, q in itertools.product(ALL_NPS, QUALITY)), 1000)
    add("object_quality_modified", "evaluative", (f"{np} is {modifier} {q}."
        for np, modifier, q in itertools.product(ALL_NPS, ["very", "really", "particularly", "surprisingly", "rather", "quite"], ADJECTIVES)), 1000)
    add("object_quality_relative", "evaluative", (f"{np} {loc} is {q}."
        for np, loc, q in itertools.product(ITEM_NPS, LOCATIONS, QUALITY)), 1200)
    add("feature_adverb_judgment", "evaluative", (f"{np} is {judgment}."
        for np, judgment in itertools.product(STATION_NPS,
            ["wonderfully spacious", "terribly cramped", "pleasantly wide", "awfully narrow",
             "unnecessarily distant", "annoyingly obstructed", "nicely positioned", "badly placed", "conveniently close"])), 650)
    add("object_preference_relative", "evaluative", (f"{pref} {np} {loc}."
        for pref, np, loc in itertools.product(["I like", "I prefer", "I dislike", "I value", "I don't like",
            "I do not prefer", "I would rather have", "I hate"], ITEM_NPS + ["the space", "the corner", "the counter"], LOCATIONS)), 1000)
    add("object_emotion", "evaluative", (f"I am {emotion} with {np}."
        for emotion, np in itertools.product(EMOTIONS, ALL_NPS)), 900)
    add("object_emotion_relative", "evaluative", (f"I am {emotion} with {np} {loc}."
        for emotion, np, loc in itertools.product(EMOTIONS, ITEM_NPS, LOCATIONS)), 650)
    add("overall_emotion", "evaluative", (f"I am {emotion} with {performance}."
        for emotion, performance in itertools.product(EMOTIONS, ["how we worked together", "the way the round ended",
            "how you completed those orders", "your overall effort", "your work this shift", "our whole performance"])), 300)
    add("object_possessive_quality", "evaluative", (f"{np} was {q} for {purpose}."
        for np, q, purpose in itertools.product(ALL_NPS, ADJECTIVES,
            ["our next meal", "cooking", "our team", "serving orders", "holding the ingredients"])), 650)
    add("quality_route", "evaluative", (f"{np} is a {q} {kind}."
        for np, q, kind in itertools.product(STATION_NPS, ADJECTIVES,
            ["place to wait", "spot to stand", "route", "choice", "location", "place to leave a plate"])), 650)
    add("holding_state", "descriptive", (f"{who} {be} {neg}{verb} {np}."
        for (who, be, have), neg, verb, np in itertools.product(ACTORS, ["", "not "],
            ["holding", "carrying", "facing", "looking at"], ITEM_NPS)), 900)
    add("location_state", "descriptive", (f"{np} {verb} {loc}."
        for np, verb, loc in itertools.product(ALL_NPS, ["is", "is not", "remains", "sits", "lies", "stands", "is located"], LOCATIONS)), 1500)
    add("plain_object_state", "descriptive", (f"{np} is {neg}{state}."
        for np, neg, state in itertools.product(ALL_NPS, ["", "not ", "still ", "currently "], FACT_STATES)), 1500)
    add("elliptical_state", "descriptive", (f"{np}: {state}."
        for np, state in itertools.product(ALL_NPS, FACT_STATES)), 1000)
    add("object_contents", "descriptive", (f"{np} {contents}."
        for np, contents in itertools.product(ALL_NPS, ["has soup in it", "has nothing in it", "has no ingredients on it",
            "contains one onion", "contains two tomatoes", "contains no soup", "has a chef next to it", "has a cook in front of it"])), 1000)
    add("numbered_state", "descriptive", (f"{number} {plural} {verb} {loc}."
        for number, plural, verb, loc in itertools.product(["Two", "Three", "Four", "Several", "No"],
            ["bowls", "onions", "tomatoes", "plates", "pots", "ingredients"], ["are", "remain", "are sitting"], LOCATIONS)), 650)
    add("state_remaining_time", "descriptive", (f"{np} {verb} {time} {units} {end}."
        for np, verb, time, units, end in itertools.product(["The soup", "The pot", "Cooking this meal", "This order"],
            ["needs", "requires", "will take"], ["two", "four", "six", "ten", "another five", "three more"],
            ["ticks", "seconds", "time steps"], ["to finish", "until it is ready", "before completion"])), 650)
    add("factual_negative_report", "descriptive", (f"{np} {negative} {loc}."
        for np, negative, loc in itertools.product(ALL_NPS, ["isn't", "wasn't", "is no longer", "has never been"], LOCATIONS)), 800)
    add("factual_passive_action", "descriptive", (f"{np} was {neg}{verb} {when}."
        for np, neg, verb, when in itertools.product(ITEM_NPS, ["", "not "],
            ["moved", "taken", "picked up", "fetched", "collected", "served", "delivered", "dropped", "put down", "carried"],
            ["on the previous turn", "just now", "a second ago", "during this round", "while the chef was waiting"])), 900)
    add("negative_passive_item", "descriptive", (f"No {item} was {verb} {when}."
        for item, verb, when in itertools.product(ITEMS,
            ["moved", "taken", "collected", "served", "delivered", "dropped", "carried"],
            ["at the start", "during this round", "in the previous episode", "while I was by the sink"])), 500)
    add("passive_movement_path", "descriptive", (f"{np} was moved from {source} to {destination}."
        for np, source, destination in itertools.product(ITEM_NPS,
            ["the crate", "the pot", "the shelf", "the counter", "the dispenser"],
            ["the sink", "the hatch", "the worktop", "the rack", "the table"])), 650)
    add("event_nominal_description", "descriptive", (f"{event} was {description}."
        for event, description in itertools.product(["Your last action", "The previous move", "That movement",
            "The last step", "Your partner's action", "The latest turn", "That trip"],
            ["a turn to the right", "a step toward the stove", "a walk to the window", "a delivery", "a pause",
             "picking up a plate", "moving left", "putting down a bowl", "one square to the north", "staying in place"])), 200)
    add("neutral_cooking_identity", "descriptive", (f"{np} {location} is {recipe}."
        for np, location, recipe in itertools.product(["The order", "The recipe", "The meal", "The displayed dish"],
            ["above the stove", "on the board", "in the corner", "by the counter", "shown on the screen", "for this round"],
            ["onion soup", "tomato soup", "a vegetable dish", "a bowl of soup", "the next order"])), 400)
    add("factual_simultaneous_action", "descriptive", (f"{who} {past} while {event}."
        for (who, be, have), (a, ing, past), event in itertools.product(ACTORS, actions,
            ["I was waiting", "the pot was cooking", "your partner moved", "the timer was running", "the other chef stood still"])), 900)
    add("round_numeric_account", "descriptive", (f"{episode} {verb} {number} {countable}."
        for episode, verb, number, countable in itertools.product(["The round", "This episode", "Our run", "The previous shift", "That session"],
            ["had", "included", "contained", "ended with"], ["two", "three", "six", "eight", "ten"],
            ["deliveries", "completed orders", "collisions", "pauses", "cooking steps"])), 650)

    # Desired state commands use the same nouns as neutral state facts.
    add("desired_state_command", "imperative", (f"{verb} {np} {state}."
        for verb, np, state in itertools.product(["Keep", "Leave", "Make sure you keep"], ALL_NPS,
            ["empty", "ready", "clear", "on the counter", "where it is", "in place", "away from the path"])), 900)
    add("ensure_state_command", "imperative", (f"Make sure {np} {condition}."
        for np, condition in itertools.product(ALL_NPS, ["stays on the counter", "remains clear", "is ready in time",
            "is not in the way", "stays within reach", "does not block the path"])), 650)
    add("wait_condition", "imperative", (f"{verb} until {condition}."
        for verb, condition in itertools.product(["Wait", "Hold on", "Stay here", "Keep still", "Don't move"],
            ["I have passed the stove", "the bowl is full", "the soup is ready", "your partner has left",
             "the other chef finishes", "the pot stops cooking", "the passage is free", "I have reached the sink"])), 200)
    return rows, collisions


def load_exposed():
    if v1.sha256(EXPOSED_PATH) != EXPOSED_SHA:
        raise ValueError("exposed development data changed")
    payload = json.loads(EXPOSED_PATH.read_text(encoding="utf-8-sig"))
    rows = []
    for row in payload["singles"]:
        rows.append({**row, "split": "dev", "group_id": "exposed_first_acceptance_" + row["id"],
            "source": "first_acceptance_now_exposed_development_only", "human_data": False})
    for mixed in payload["mixed"]:
        for index, row in enumerate(mixed["clauses"]):
            rows.append({**row, "split": "dev", "group_id": f'exposed_first_acceptance_{mixed["id"]}_{index}',
                "source": "first_acceptance_now_exposed_development_only", "human_data": False})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic_feedback_v1" / "final-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "synthetic_feedback_v1" / "final-v2")
    args = parser.parse_args()
    if args.data_dir.exists() or args.output_dir.exists():
        raise SystemExit("Refusing overwrite; use fresh development/output paths")
    original_train, original_dev = v1.make_corpus()
    exposed = load_exposed()
    excluded = {v1.normalized(r["text"]) for r in original_dev + exposed}
    train, collisions = augment(original_train, excluded)
    # Remove any inherited exact overlaps now that the exposed test has become
    # development material. Record actual counts rather than assuming a number.
    inherited_collisions = sum(v1.normalized(r["text"]) in excluded for r in original_train)
    train = [r for r in train if v1.normalized(r["text"]) not in excluded]
    combined_dev = original_dev + exposed
    assert not {v1.normalized(r["text"]) for r in train} & excluded
    args.data_dir.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)
    train_path = args.data_dir / "train.json"
    dev_path = args.data_dir / "dev.json"
    v1.save_json(train_path, train)
    v1.save_json(dev_path, combined_dev)
    v1.save_json(args.data_dir / "generation_audit.json", {
        "human_data": False, "previous_acceptance_now_exposed_dev": True,
        "exposed_source_sha256": EXPOSED_SHA, "excluded_train_dev_text_collisions": collisions,
        "inherited_train_rows_removed_for_dev_overlap": inherited_collisions,
        "new_training_rows": len(train) - len(original_train),
        "train_label_counts": dict(Counter(r["speech_act"] for r in train)),
        "training_group_count": len({r["group_id"] for r in train}),
        "train_dev_normalized_overlap": 0,
        "new_independent_test_read": False,
        "development_scope": "Model development and C selection; no independent accuracy claim."})
    print(f"Synthetic v2: {len(train)} train, {len(original_dev)} original dev, {len(exposed)} exposed regression clauses", flush=True)
    head = v1.fit_head(train, combined_dev, "speech_act", args.output_dir, [train_path, dev_path])
    import joblib
    from sklearn.metrics import classification_report, accuracy_score
    artifact = joblib.load(args.output_dir / "speech_act.joblib")
    regression = {}
    for name, split in [("original_synthetic_dev", original_dev), ("exposed_first_acceptance_singles", exposed[:150]),
        ("exposed_first_acceptance_clauses", exposed[150:])]:
        actual = artifact["classifier"].predict(artifact["vectorizer"].transform([r["text"] for r in split]))
        expected = [r["speech_act"] for r in split]
        regression[name] = {"rows": len(split), "accuracy": float(accuracy_score(expected, actual)),
            "classification": classification_report(expected, actual, output_dict=True, zero_division=0),
            "errors": [{"text": row["text"], "expected": row["speech_act"], "actual": str(pred)}
                for row, pred in zip(split, actual) if row["speech_act"] != pred]}
        print(name, regression[name]["accuracy"], flush=True)
    for name in ["grounding.json", "grounding.joblib"]:
        shutil.copyfile(ROOT / "outputs" / "synthetic_feedback_v1" / "final" / name, args.output_dir / name)
    browser = json.loads((args.output_dir / "speech_act.json").read_text())
    browser["source"]["training_script_sha256"] = v1.sha256(Path(__file__))
    browser["source"]["base_training_script_sha256"] = v1.sha256(Path(v1.__file__))
    browser["source"]["exposed_development_sha256"] = EXPOSED_SHA
    browser["model_card"]["development_scope"] = "First acceptance was exposed after failure and used only as dev. This candidate needs a fresh independent acceptance."
    (args.output_dir / "speech_act.json").write_text(json.dumps(browser, sort_keys=True, separators=(",", ":")) + "\n")
    head["browser_sha256"] = v1.sha256(args.output_dir / "speech_act.json")
    report = {"training_scope": "synthetic_only", "human_data_read": False,
        "prior_acceptance_exposed": True, "prior_acceptance_sha256": EXPOSED_SHA,
        "new_independent_test_read": False, "regression_not_independent_accuracy": True,
        "train_rows": len(train), "dev_rows": len(combined_dev),
        "train_sha256": v1.sha256(train_path), "dev_sha256": v1.sha256(dev_path),
        "training_script_sha256": v1.sha256(Path(__file__)),
        "development_iterations": 3,
        "new_train_labels": dict(Counter(r["speech_act"] for r in train)),
        "word_tfidf_fit_split": "train_only",
        "coarse_grounding_training": "not_retrained; copied exact frozen v1 bytes",
        "train_dev_normalized_overlap": 0,
        "speech_act": head, "regression": regression,
        "grounding": {"status": "unchanged_frozen_v1", "browser_sha256": v1.sha256(args.output_dir / "grounding.json")}}
    v1.save_json(args.output_dir / "training_report.json", report)
    # Numerical fixtures are drawn from the trainer-owned original development
    # families. Never publish the new independent test in engineering fixtures.
    parity = []
    seen_groups = set()
    for row in original_dev:
        if row["group_id"] not in seen_groups:
            seen_groups.add(row["group_id"])
            parity.append({"text": row["text"], "speech_act": row["speech_act"], "grounding": row["grounding"]})
    for head_name in ["speech_act", "grounding"]:
        fitted = joblib.load(args.output_dir / f"{head_name}.joblib")
        scores = fitted["classifier"].predict_proba(fitted["vectorizer"].transform([r["text"] for r in parity]))
        for row, probability in zip(parity, scores):
            row[f"{head_name}_scores"] = {str(label): float(value) for label, value in zip(fitted["classifier"].classes_, probability)}
    v1.save_json(args.output_dir / "browser_parity_fixtures.json", parity)
    v1.save_json(args.output_dir / "freeze_receipt.json", {"status": "frozen_before_new_independent_test",
        "training_report_sha256": v1.sha256(args.output_dir / "training_report.json"),
        "speech_browser_sha256": head["browser_sha256"],
        "speech_model_sha256": head["model_sha256"],
        "grounding_browser_sha256": report["grounding"]["browser_sha256"],
        "parity_fixtures_sha256": v1.sha256(args.output_dir / "browser_parity_fixtures.json"),
        "old_test_exposed_dev_only": True, "new_test_read": False})
    print(json.dumps({"speech_browser_sha256": head["browser_sha256"], "selected_C": head["selected_C"]}), flush=True)


if __name__ == "__main__":
    main()
