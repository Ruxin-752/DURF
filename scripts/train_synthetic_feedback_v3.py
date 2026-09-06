"""Third synthetic-only two-head development run; previous tests are now dev.

Reads only the explicitly exposed first and second sets, never a third test.
Uses word TF-IDF (1,2), train-only fitting, and dev-selected logistic regression.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random

import train_synthetic_feedback_v1 as v1
import train_synthetic_feedback_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026090603
BASE_TRAIN = ROOT / "data/synthetic_feedback_v1/final-v2/train.json"
BASE_SHA = "3e3bf9abb367f0f8a36f556ab2f9d945a21e49dd6b801e97c234da326fe96d43"
SECOND_DEV = ROOT / "artifacts/synthetic-acceptance-20260906/frozen-next-v2.json"
SECOND_SHA = "23c4272190395efa3e9b9791f68b75fda502a9019c85ab2ae5e120d8fc999fea"
EXTERNAL_SEEDS = ROOT / "data/synthetic_feedback_v1/v3_external_seeds.json"

FEATURE_GROUPS = {
    "evaluated_present_infinitive", "object_quality", "object_quality_negated", "object_quality_modified",
    "object_quality_relative", "feature_adverb_judgment", "object_preference_relative", "object_emotion",
    "object_emotion_relative", "object_possessive_quality", "quality_route", "holding_state", "location_state",
    "plain_object_state", "elliptical_state", "object_contents", "numbered_state", "state_remaining_time",
    "factual_negative_report", "neutral_cooking_identity",
}
TRAJECTORY_GROUPS = {
    "reported_action_expanded", "observed_action_expanded", "negative_action_expanded", "evaluated_past_infinitive",
    "executed_gerund_quality_expanded", "long_past_quality", "past_quality_clause", "past_quality_thanks",
    "past_mistake_negation", "approval_of_behavior", "overall_emotion", "factual_passive_action",
    "negative_passive_item", "passive_movement_path", "event_nominal_description", "factual_simultaneous_action",
    "round_numeric_account",
}

OBJECTS = ["the clean bowl", "that dirty plate", "the spare dish", "a spare ladle", "the onion", "the ripe tomato",
    "my ingredient", "your soup", "this cooked meal", "the full saucepan", "the empty pot", "the ready stew",
    "the chopped vegetable", "that portion", "the last order", "the serving tray", "the soup bowl", "the utensil"]
STATIONS = ["burner", "cooker", "stove", "saucepan", "pot", "bench", "worktop", "counter", "island",
    "shelf", "ledge", "rack", "dish rack", "cupboard", "cabinet", "storage area", "pantry", "supply bin",
    "onion crate", "ingredient dispenser", "hatch", "serving point", "delivery window", "serving area",
    "passage", "aisle", "corridor", "lane", "path", "nook", "corner", "bottleneck", "door", "wall", "tile"]
NPS = OBJECTS + [f"the {modifier} {station}" for modifier, station in itertools.product(
    ["", "left", "right", "upper", "lower", "central", "nearby", "nearest", "far", "eastern", "western", "other"], STATIONS)]
NPS = [" ".join(x.split()) for x in NPS]
PLACES = [f"{prep} the {station}" for prep, station in itertools.product(
    ["near", "beside", "beyond", "beneath", "above", "behind", "opposite", "between me and", "in front of",
     "to the left of", "to the right of", "on the far side of", "just past", "around", "under"],
    ["burner", "cupboard", "bench", "counter", "dish rack", "serving hatch", "pantry", "island", "door"])]
ACTORS = [("You", "are", "have"), ("I", "am", "have"), ("We", "are", "have"),
    ("Your teammate", "is", "has"), ("The blue cook", "is", "has"), ("The red chef", "is", "has"),
    ("The player on the left", "is", "has"), ("My partner", "is", "has"), ("Both chefs", "are", "have")]
EPISODES = ["your performance", "your cooperation", "our teamwork", "the last round", "your last delivery",
    "that handoff", "your last ingredient pickup", "this attempt", "the completed shift", "the whole run",
    "your last move", "your most recent interaction", "that last trip", "the cooking you just did", "that effort"]
QUALITY = list(dict.fromkeys(v2.QUALITY + ["essential", "advantageous", "horrible", "handy", "ridiculous", "decent",
    "superior", "inferior", "promising", "reliable", "unreliable", "problematic", "troublesome", "awkward",
    "appalling", "disastrous", "optimal", "suboptimal", "costly", "economical", "lousy", "marvelous", "splendid",
    "mediocre", "solid", "exceptional", "amazing", "satisfactory", "unsatisfactory", "unimpressive", "fruitful",
    "unproductive", "a benefit", "an advantage", "a disadvantage", "a drawback", "a problem", "a hindrance",
    "a blessing", "a burden", "a bonus", "a relief", "a disappointment", "an improvement", "a disaster"]))
ADVERBS = ["brilliantly", "beautifully", "expertly", "skillfully", "flawlessly", "perfectly", "impeccably",
    "magnificently", "superbly", "admirably", "sensibly", "cleverly", "wisely", "effectively", "efficiently",
    "promptly", "swiftly", "smoothly", "neatly", "carefully", "thoughtfully", "correctly", "well", "badly",
    "poorly", "terribly", "awfully", "horribly", "clumsily", "carelessly", "recklessly", "foolishly",
    "wastefully", "inefficiently", "unnecessarily", "needlessly", "incorrectly", "wrongly", "awkwardly", "sloppily"]
VERBS = [
    ("retrieve", "retrieving", "retrieved"), ("collect", "collecting", "collected"),
    ("fetch", "fetching", "fetched"), ("grab", "grabbing", "grabbed"), ("take", "taking", "took"),
    ("discard", "discarding", "discarded"), ("release", "releasing", "released"), ("deposit", "depositing", "deposited"),
    ("remove", "removing", "removed"), ("transfer", "transferring", "transferred"), ("move", "moving", "moved"),
    ("pass", "passing", "passed"), ("hand over", "handing over", "handed over"),
    ("put down", "putting down", "put down"), ("set down", "setting down", "set down"),
    ("pick up", "picking up", "picked up"), ("carry", "carrying", "carried"),
    ("deliver", "delivering", "delivered"), ("serve", "serving", "served"), ("stash", "stashing", "stashed"),
    ("replace", "replacing", "replaced"), ("reposition", "repositioning", "repositioned"),
    ("shift", "shifting", "shifted"), ("use", "using", "used"), ("save", "saving", "saved"),
    ("keep", "keeping", "kept"), ("hold", "holding", "held"), ("leave", "leaving", "left"),
    ("bring", "bringing", "brought"), ("get", "getting", "got"), ("drop", "dropping", "dropped"),
]


def exposed_rows(path, expected_hash, prefix):
    if v1.sha256(path) != expected_hash:
        raise ValueError("An exposed development source changed")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = [{**r, "split": "dev", "group_id": f'{prefix}_{r["id"]}', "source": prefix,
             "human_data": False} for r in value["singles"]]
    for mixed in value["mixed"]:
        rows.extend({**r, "split": "dev", "group_id": f'{prefix}_{mixed["id"]}_{i}',
            "source": prefix, "human_data": False} for i, r in enumerate(mixed["clauses"]))
    return rows


def base_rows():
    if v1.sha256(BASE_TRAIN) != BASE_SHA:
        raise ValueError("The frozen v2 synthetic training source changed")
    rows = json.loads(BASE_TRAIN.read_text())
    for row in rows:
        if row.get("grounding") is None:
            group = row["group_id"].removeprefix("v2_")
            row["grounding"] = "feature" if group in FEATURE_GROUPS else "trajectory" if group in TRAJECTORY_GROUPS else "action"
            row["grounding_label_source"] = "explicit_generation_family_semantics_v3"
    return rows


def make_new_rows(existing, excluded):
    rng = random.Random(SEED)
    rows = list(existing)
    seen = {v1.normalized(r["text"]): (r["speech_act"], r["grounding"]) for r in rows}
    skipped = []
    def add(family, speech, grounding, candidates, limit=500):
        values = list(dict.fromkeys(candidates))
        rng.shuffle(values)
        count = 0
        for text in values:
            text = " ".join((text[0].upper() + text[1:]).split())
            key = v1.normalized(text)
            if key in excluded:
                skipped.append(text)
                continue
            if key in seen:
                continue
            seen[key] = speech, grounding
            rows.append({"text": text, "speech_act": speech, "grounding": grounding, "group_id": f"v3_{family}",
                "source": "fresh_synthetic_linguistic_contrasts_v3", "label_source": "explicit_family_semantics",
                "human_data": False, "split": "train"})
            count += 1
            if count >= limit:
                break

    # Every object/space is paired with neutral, evaluative, and directive frames.
    add("state_extended", "descriptive", "feature", (f"{np} is {place}." for np, place in itertools.product(NPS, PLACES)), 1600)
    add("judgment_extended", "evaluative", "feature", (f"{np} is {q}." for np, q in itertools.product(NPS, QUALITY)), 1600)
    add("request_extended", "imperative", "action", (f"{verb} {np} {place}." for verb, np, place in itertools.product(
        ["Use", "Move", "Go to", "Try", "Avoid", "Clear", "Check", "Leave", "Keep away from"], NPS, PLACES)), 1600)
    add("nominal_quality", "evaluative", "feature", (f"{np} is {judgment}." for np, judgment in itertools.product(NPS,
        ["in an awkward position", "in a convenient spot", "in the wrong place", "a poor choice for cooking",
         "badly designed", "well designed", "poorly arranged", "nicely laid out", "easier to use", "harder to reach",
         "a source of frustration", "worth the effort", "not worth the effort", "more trouble than it is worth"])), 1100)
    for family, templates in {
        "opinion_verbs": ["I consider {np} {q}.", "I find {np} {q}.", "I regard {np} as {q}.", "I think {np} is {q}."],
        "preference_idioms": ["I am a fan of {np}.", "I cannot stand {np}.", "I am fond of {np}.", "I have a soft spot for {np}.",
            "I am not keen on {np}.", "I don't care for {np}.", "I have no use for {np}.", "I would choose {np} over the others."],
        "rating_object": ["I would rate {np} highly.", "I give {np} a low rating.", "I rank {np} above the rest.",
            "{np} gets my approval.", "{np} earns my disapproval.", "{np} gets a thumbs up.", "{np} gets a thumbs down."],
        "object_emotional_reaction": ["{np} disappoints me.", "{np} pleases me.", "{np} annoys me.", "{np} makes me happy.",
            "{np} drives me crazy.", "I appreciate {np}.", "I admire {np}.", "I resent {np}."],
    }.items():
        add(family, "evaluative", "feature", (t.format(np=np, q=q) for t, np, q in itertools.product(templates, NPS, QUALITY[:12])), 900)
    add("feature_pros_cons", "evaluative", "feature", (f"{feature} is {q}." for feature, q in itertools.product([
        "Having an empty counter", "Having both pots full", "Keeping a delivery lane free", "The distance to the dispenser",
        "The gap between the cooks", "Sharing a small workstation", "Two chefs using the same space", "The layout around the cookers",
        "The approach to the window", "The time required to fetch ingredients", "The long walk across the room", "The lack of clean bowls",
    ], QUALITY)), 700)
    add("object_comparative", "evaluative", "feature", (f"{np} is {q} than {other}." for np, q, other in itertools.product(NPS,
        ["better", "worse", "more useful", "less helpful", "easier to use", "harder to reach", "more convenient", "less efficient"],
        ["the other one", "the alternative", "the one beside the sink"])), 650)

    # Evaluation is also conveyed by actions, nouns, idioms, and adverbs.
    add("evaluative_adverbs", "evaluative", "trajectory", (f"{who} {past} {adv}." for (who, be, have), (verb, ing, past), adv in itertools.product(
        ACTORS, [((verb + " " + obj), (ing + " " + obj), (past + " " + obj)) for (verb, ing, past), obj in itertools.product(VERBS, OBJECTS)], ADVERBS)), 1800)
    add("overall_adverbs", "evaluative", "trajectory", (f"{who} {verb} {adv}." for (who, be, have), verb, adv in itertools.product(ACTORS,
        ["handled the rush", "managed the kitchen", "completed the handoff", "worked together", "responded to the order", "cooperated", "played that round"], ADVERBS)), 950)
    for family, templates in {
        "appraisal_nouns": ["I give {p} a thumbs up.", "I give {p} a thumbs down.", "{p} deserves credit.",
            "{p} deserves praise.", "{p} deserves criticism.", "{p} gets my approval.", "{p} gets my disapproval.",
            "I give you credit for {p}.", "I blame you for {p}.", "I applaud {p}.", "I commend {p}."],
        "appraisal_idioms": ["{p} fell short.", "{p} missed the mark.", "{p} hit the mark.", "{p} was spot on.",
            "{p} was off the mark.", "{p} was a step backward.", "{p} was a step in the right direction.",
            "{p} exceeded my expectations.", "{p} let the team down.", "{p} made my day.", "{p} paid off."],
        "overall_emotional_verbs": ["{p} impressed me.", "{p} disappointed me.", "{p} delighted me.",
            "{p} frustrated me.", "{p} pleased me.", "{p} upset me.", "I appreciate {p}.", "I regret {p}."],
        "overall_emotions": ["I am pleased with {p}.", "I am disappointed in {p}.", "I am happy about {p}.",
            "I am dissatisfied with {p}.", "I am proud of {p}.", "I am ashamed of {p}.", "I am impressed by {p}."],
        "performance_comparison": ["{p} was an improvement.", "{p} was a huge improvement on the previous effort.",
            "{p} was no improvement.", "{p} was a setback.", "{p} was worse than last time.", "{p} was better than before."],
    }.items():
        add(family, "evaluative", "trajectory", (t.format(p=p) for t, p in itertools.product(templates, EPISODES)), 650)
    add("nailed_failed", "evaluative", "trajectory", (f"{who} {outcome} {event}." for (who, be, have), outcome, event in itertools.product(ACTORS,
        ["nailed", "botched", "messed up", "aced", "ruined", "wasted", "bungled", "mastered", "excelled at", "struggled with"],
        ["the last handoff", "that delivery", "the final order", "the timing", "the pickup", "that interaction", "the whole shift"])), 700)
    add("exclamatory_past_choice", "evaluative", "trajectory", (f"What a {q} decision to {a}!" for q, (a, ing, past) in itertools.product(
        ["bad", "good", "poor", "great", "foolish", "smart", "terrible", "clever", "wasteful", "brilliant"], v2.make_actions())), 800)
    add("retrospective_timing", "evaluative", "trajectory", (f"That was {rating} time to {a}." for rating, (a, ing, past) in itertools.product(
        ["the right", "the wrong", "not the right", "not a good", "a terrible", "a perfect"], v2.make_actions())), 650)
    add("retrospective_relative", "evaluative", "trajectory", (f"The {event} you just {verb} was {q}." for event, verb, q in itertools.product(
        ["turn", "move", "handoff", "delivery", "decision", "trip"], ["made", "completed", "finished"], QUALITY)), 700)

    # Long object modifiers are shared across future commands and past reports.
    actions = [(f"{verb} {obj} {place}", f"{ing} {obj} {place}", f"{past} {obj} {place}")
        for (verb, ing, past), obj, place in itertools.product(VERBS, OBJECTS,
            ["beside the cooker", "across the bench", "from the cupboard", "by the saucepan", "near the storage area"])]
    for family, templates in {
        "expanded_directives": ["{a}.", "Please {a}.", "Don't {a}.", "We need you to {a}."],
        "expanded_polite": ["Can you please {a}?", "Could you {a}?", "Would you kindly {a}?", "I'd like you to {a}."],
        "future_constraint": ["{a} before I reach the stove.", "{a} until your partner returns.", "{a} when the timer stops.",
            "For this order, {a}.", "When you have room, {a}."],
        "proposal_words": ["Why don't you {a}?", "How about {ing}?", "What about {ing}?", "You had better {a}.",
            "Would it be possible for you to {a}?", "I recommend {ing}.", "Could I get you to {a}?"],
    }.items():
        add(family, "imperative", "action", (t.format(a=a, ing=ing) for t, (a, ing, past) in itertools.product(templates, actions)), 1400)
    add("complex_referent_command", "imperative", "action", (f"{verb} {obj} {relative}." for verb, obj, relative in itertools.product(
        ["Take", "Collect", "Fetch", "Bring", "Hand over", "Keep", "Leave", "Pass", "Retrieve", "Use"],
        OBJECTS, ["that you are carrying", "that your partner put down", "which is nearest the door", "on the bench beside the cooker",
            "between the bowls", "next to the two tomatoes", "from the dispenser at the far end", "with the shortest path to the hatch"])), 1000)
    add("propose_hypothetical_action", "descriptive", "action", (f"{prefix} {a}." for prefix, (a, ing, past) in itertools.product(
        ["One alternative is to", "A possible next move is to", "I could", "We could", "An available action would be to",
         "The other option is to", "I am considering whether to"], actions)), 900)
    pp_map = {"took": "taken", "got": "gotten", "went": "gone"}
    add("explicit_counterfactual", "evaluative", "action", (f"{prefix} {pp_map.get(past.split()[0], past.split()[0])} {' '.join(past.split()[1:])}."
        for prefix, (a, ing, past) in itertools.product(["You should have", "You should not have", "You ought to have",
            "I wish you had", "It would have helped if you had", "A better choice would have been to have"], actions)), 1000)
    add("third_person_past_fact", "descriptive", "trajectory", (f"{who} {past}{when}." for (who, be, have), (a, ing, past), when in itertools.product(
        ACTORS, actions, ["", " a moment ago", " before the bell", " on the previous tick", " during that turn"])), 1700)
    add("negative_past_fact", "descriptive", "trajectory", (f"{who} did not {a}." for (who, be, have), (a, ing, past) in itertools.product(ACTORS, actions)), 1000)
    add("event_passive_fact", "descriptive", "trajectory", (f"{obj} was {neg}{past} {place}." for obj, neg, (verb, ing, past), place in itertools.product(
        OBJECTS, ["", "not "], [v for v in VERBS if v[2] not in {"took", "got"}], PLACES)), 850)
    add("event_subject_fact", "descriptive", "trajectory", (f"{event} {verb} {obj}." for event, verb, obj in itertools.product(
        ["Your latest action", "The previous interaction", "Your partner's last move", "The last step", "The recent handoff"],
        ["moved", "removed", "deposited", "transferred", "collected", "delivered", "repositioned"], OBJECTS)), 700)
    add("object_arrival_fact", "descriptive", "trajectory", (f"{obj} {event} {when}." for obj, event, when in itertools.product(OBJECTS,
        ["reached the window", "changed hands", "arrived at the counter", "moved to the other cook", "was delivered", "left the cupboard"],
        ["before the bell", "on the last turn", "a moment ago", "during the round"])), 650)
    add("episode_factual_duration", "descriptive", "trajectory", (f"{episode} {verb} {number} {units}." for episode, verb, number, units in itertools.product(
        ["The completed shift", "Our last game", "The previous episode", "That round", "The run"], ["lasted", "took", "continued for"],
        ["two hundred", "four hundred", "six hundred", "eight hundred", "a thousand"], ["steps", "ticks", "seconds"])), 450)
    add("episode_factual_outcome", "descriptive", "trajectory", (f"{episode} {outcome}." for episode, outcome in itertools.product(
        ["The previous game", "The round", "The completed shift", "That run", "Our last episode"],
        ["ended with two bowls on the bench", "finished with four deliveries", "had no deliveries", "included three handoffs",
         "ended before the order was served", "had both cooks carrying soup", "finished when time ran out"])), 200)
    add("score_factual_change", "descriptive", "trajectory", (f"The score {change} when {event}." for change, event in itertools.product(
        ["increased", "changed", "stayed the same", "rose", "went up"], ["your soup arrived", "the order was completed", "you delivered the meal",
            "your teammate served the bowl", "the round ended"])), 200)

    # Current static state, possession, quantities, and geometry.
    add("possessive_inventory", "descriptive", "feature", (f"{who} {have} {obj} {place}." for (who, be, have), obj, place in itertools.product(ACTORS,
        ["an onion", "a tomato", "a plate", "a bowl", "some soup", "no item", "nothing"], ["in hand", "in their hands", "within reach", "next to them"])), 650)
    add("object_relational_inventory", "descriptive", "feature", (f"{np} {verb} {obj}." for np, verb, obj in itertools.product(NPS,
        ["holds", "contains", "has", "does not contain", "doesn't have"], ["a plate", "two onions", "the same ingredients", "three tomatoes", "some soup", "a cook beside it"])), 850)
    add("quantified_inventory", "descriptive", "feature", (f"{subject} {verb} {object_text}." for subject, verb, object_text in itertools.product(
        ["Both pots", "Both saucepans", "The two counters", "All the bowls", "Neither of the two counters"],
        ["contain", "hold", "have"], ["the same ingredients", "no food", "some soup", "one onion", "two tomatoes", "a dish"])), 300)
    add("neutral_static_orientation", "descriptive", "feature", (f"{who} {be} {orientation} {place}." for (who, be, have), orientation, place in itertools.product(ACTORS,
        ["looking toward", "facing", "standing near", "positioned beside", "located beside"], NPS)), 800)
    add("existence_geometry", "descriptive", "feature", (f"There is {obj} {place}." for obj, place in itertools.product(
        ["a cooker", "a gap", "no gap", "no space", "a tile", "a cupboard", "one bowl", "a serving point", "an onion"], PLACES)), 700)
    add("spatial_capacity", "descriptive", "feature", (f"{np} {relation}." for np, relation in itertools.product(NPS,
        ["has room for one cook", "has two tiles on either side", "has a wall behind it", "is two steps from the sink",
         "has the hatch to its right", "leads to the pantry", "passes the dish rack", "connects the two rooms",
         "is at the end of the bench", "is a serving point", "is an ingredient dispenser", "is one tile wide"])), 850)
    add("quantified_geometry", "descriptive", "feature", (f"{who} {have} {count} {measure} between them and {np}." for (who, be, have), count, measure, np in itertools.product(ACTORS,
        ["two", "three", "four"], ["tiles", "counter squares", "steps"], NPS)), 600)
    add("state_clause_fact", "descriptive", "feature", (f"{np} {clause} is {state}." for np, clause, state in itertools.product(NPS,
        ["beside your partner", "under the timer", "nearest the doorway", "between the two cookers", "that you are facing"],
        ["empty", "full", "ready", "blocked", "on the left", "at the edge of the room"])), 900)
    # Matched examples distribute the exact same noun and spatial context across
    # all three speech acts. Distinct vocabulary alone must not determine speech.
    paired = {label: [] for label in ["descriptive", "evaluative", "imperative"]}
    for index, np in enumerate(NPS):
        for shift in range(4):
            place = PLACES[(index + shift * 13) % len(PLACES)]
            state = ["ready", "empty", "available", "clear"][shift]
            quality = ["helpful", "inconvenient", "a big help", "not worthwhile"][shift]
            paired["descriptive"].append(f"{np} is {state} {place}.")
            paired["evaluative"].append(f"{np} is {quality} {place}.")
            paired["imperative"].append(f"Keep {np} {state} {place}.")
    for label, texts in paired.items():
        add(f"matched_np_{label}", label, "action" if label == "imperative" else "feature", texts, 5000)
    add("embedded_favorite_subject", "evaluative", "feature", (f"My {rating} {kind} is {np}." for rating, kind, np in itertools.product(
        ["favorite", "least favorite", "preferred", "most disliked"], ["station", "cooking area", "storage space", "choice", "workplace"], NPS)), 650)
    add("appraisal_cleft_clause", "evaluative", "trajectory", (f"{reaction} how you {past}." for reaction, (a, ing, past) in itertools.product(
        ["I am disappointed in", "I am proud of", "I am happy about", "I dislike", "There is a lot to like about", "I am pleased with"], actions)), 800)
    add("regret_that_completed", "evaluative", "trajectory", (f"It is {reaction} that you {past}." for reaction, (a, ing, past) in itertools.product(
        ["a pity", "a shame", "a disappointment", "a relief", "wonderful", "unfortunate", "a good thing"], actions)), 650)
    add("polite_embedded_future", "imperative", "action", (f"{prefix} {a}." for prefix, (a, ing, past) in itertools.product(
        ["I would appreciate it if you could", "I would be grateful if you would", "I would be glad if you could",
         "It would be helpful if you could", "May I ask you to", "Would it trouble you to"], actions)), 900)
    add("observe_present_action", "descriptive", "trajectory", (f"{prefix} {a}." for prefix, (a, ing, past) in itertools.product(
        ["I am watching the cook", "I can see my partner", "I watched the chef", "I saw the other cook"], actions)), 650)
    add("observe_past_clause", "descriptive", "trajectory", (f"{prefix} {past}." for prefix, (a, ing, past) in itertools.product(
        ["I watched while you", "I saw that you", "I noticed you", "The log says you"], actions)), 650)
    add("feature_salience_request", "imperative", "feature", (f"{prefix} {f}." for prefix, f in itertools.product(
        ["Give more emphasis to", "Put the emphasis on", "Give more attention to", "Focus your attention on", "Think carefully about the value of"], v1.FEATURES)), 500)
    add("status_report_fragment", "descriptive", "feature", (f"Current report for {np}: {state}." for np, state in itertools.product(NPS,
        ["empty", "ready", "full", "clear", "at the top", "behind you", "near the sink"])), 650)
    add("game_factual_rules", "descriptive", "feature", (f"{subject} {rule}." for subject, rule in itertools.product(
        ["This recipe", "The current order", "The dish on the board"],
        ["requires onions rather than tomatoes", "contains three ingredients", "uses two tomatoes", "requires a clean bowl", "takes ten ticks"])), 150)
    add("physical_current_rules", "descriptive", "feature", (f"{subject} {rule}." for subject, rule in itertools.product(
        ["The hatch", "The serving point", "The delivery window"], ["accepts cooked soup", "takes bowls of soup", "receives completed orders", "is where the soup is served"])), 100)
    add("static_present_location", "descriptive", "feature", (f"{who} {be} {place}." for (who, be, have), place in itertools.product(ACTORS,
        PLACES + ["on the opposite side of the island", "in the way", "at the edge of the room", "between the cupboards"])), 800)
    add("holding_relative_location", "descriptive", "feature", (f"{np} {place} {verb} {obj}." for np, place, verb, obj in itertools.product(NPS,
        PLACES[:20], ["holds", "contains", "has", "does not hold", "occupies"], ["a plate", "a tomato", "one tile", "an onion", "two bowls"])), 650)
    add("current_heating_state", "descriptive", "feature", (f"{np} {state}." for np, state in itertools.product(NPS,
        ["has not started cooking", "has started heating", "is in use", "is not in use", "has finished heating", "is still cold", "shows six ticks remaining"])), 650)
    add("short_primitive_command", "imperative", "action", (f"{verb} {obj}{location}." for verb, obj, location in itertools.product(
        ["Give", "Fetch", "Put", "Take", "Bring", "Hand over", "Move", "Serve", "Enter", "Leave", "Back up", "Clear"],
        ["an onion", "the soup", "that gap", "one bowl", "a square", "the dish", "it", "that", "a space"],
        ["", " in it", " to me", " to the customer", " right now", " for the next order"])), 850)
    add("trajectory_strategy_request", "imperative", "trajectory", (f"{prefix} {strategy}." for prefix, strategy in itertools.product(
        ["Repeat", "Keep using", "Please follow", "Try", "Stick to", "Don't repeat"],
        ["the same overall strategy", "that approach to the whole round", "your earlier performance", "your plan from last round", "that pattern of teamwork"])), 150)
    actual_references = [("choice", "made"), ("route", "took"), ("handoff", "completed"), ("plan", "followed"),
        ("delivery", "finished"), ("interaction", "performed"), ("strategy", "used"), ("task", "attempted"),
        ("approach", "selected"), ("turn", "made"), ("pickup", "performed"), ("path", "chose")]
    add("emotional_relative_appraisal", "evaluative", "trajectory", (f"I am {emotion} {prep} the {noun} you {recent}{past}."
        for emotion, prep, (noun, past), recent in itertools.product(v2.EMOTIONS, ["with", "about", "by"], actual_references, ["", "just "])), 800)
    add("disappointment_relative_appraisal", "evaluative", "trajectory", (f"{prefix} the {noun} you {recent}{past}."
        for prefix, (noun, past), recent in itertools.product(["I am disappointed in", "I am proud of", "I disapprove of", "I approve of",
            "There was no problem with", "There was a problem with", "I see a benefit in", "I see a drawback in"], actual_references, ["", "just "])), 650)
    add("counterfactual_could_have", "evaluative", "action", (f"You could have {pp_map.get(past.split()[0], past.split()[0])} {' '.join(past.split()[1:])}."
        for a, ing, past in actions), 650)
    add("counterfactual_avoid_error", "evaluative", "action", (f"You could have avoided {mistake} by {ing}."
        for mistake, (a, ing, past) in itertools.product(["the delay", "the blockage", "the mistake", "that error", "the collision"], actions)), 700)
    add("referential_question_command", "imperative", "action", (f"{prefix} {verb} {reference}?"
        for prefix, verb, reference in itertools.product(["Can you", "Could you", "Would you", "Will you please"],
            ["collect", "take", "get", "bring", "fetch", "pick up", "retrieve", "move"],
            ["what is beside the pot", "whatever is on the counter", "what your partner is carrying", "the thing by the crate",
             "whatever remains on the shelf", "the object that is blocking the aisle"])), 700)
    add("coordinate_spatial_commands", "imperative", "action", (f"{first} and {second}." for first, second in itertools.product(
        ["Turn around", "Step back", "Turn toward the wall", "Back up a square", "Move to the side", "Walk to the bench"],
        ["face the pot", "leave the path open", "wait for me", "face the cooker", "make room for your partner", "let me pass"])), 150)
    add("nominal_essential_judgment", "evaluative", "feature", (f"I consider {thing} {q}." for thing, q in itertools.product(
        ["a free passage", "an open lane", "an empty bowl", "a clean counter", "the space between the cookers", "the gap by the hatch"],
        ["essential", "important", "necessary", "a big help", "a nuisance", "unnecessary", "useful"])), 200)
    add("plain_spatial_relational_verbs", "descriptive", "feature", (f"{subject} {relation} {obj}." for subject, relation, obj in itertools.product(
        ["The path", "The route around the kitchen", "The way to the cooker", "The approach to the hatch", "The aisle"],
        ["passes", "goes past", "runs beside", "leads past", "goes around", "ends at"], NPS)), 650)
    add("space_reservation_command", "imperative", "action", (f"{verb} {thing} {condition}." for verb, thing, condition in itertools.product(
        ["Leave", "Keep", "Make sure you leave"], ["one counter", "one bench", "a space", "the worktop", "the area by the stove"],
        ["empty for the next order", "free for the handoff", "clear for me", "available for the soup"])), 250)
    add("occupied_relative_state", "descriptive", "feature", (f"{item} occupies {place} where {other} was."
        for item, place, other in itertools.product(["A plate", "An onion", "A tomato", "A bowl"],
            ["the square", "the tile", "the space", "the spot"], ["the soup", "the other ingredient", "the dish"])), 250)
    add("stopped_location_event", "descriptive", "trajectory", (f"{who} {event} {place}." for (who, be, have), event, place in itertools.product(
        ACTORS, ["stopped", "paused", "waited", "arrived", "turned around", "walked"], PLACES)), 750)
    return rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/synthetic_feedback_v1/final-v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/synthetic_feedback_v1/final-v3")
    parser.add_argument("--allow-missing-external-seeds", action="store_true")
    parser.add_argument("--seed-weight", type=int, choices=[1, 8, 24], default=24)
    args = parser.parse_args()
    if args.data_dir.exists() or args.output_dir.exists():
        raise SystemExit("Use new output paths; frozen artifacts are immutable")
    _, old_dev = v1.make_corpus()
    first = exposed_rows(v2.EXPOSED_PATH, v2.EXPOSED_SHA, "exposed_first")
    second = exposed_rows(SECOND_DEV, SECOND_SHA, "exposed_second")
    dev = old_dev + first + second
    excluded = {v1.normalized(r["text"]) for r in dev}
    train, collisions = make_new_rows(base_rows(), excluded)
    external = []
    if EXTERNAL_SEEDS.exists():
        external = json.loads(EXTERNAL_SEEDS.read_text(encoding="utf-8-sig"))
        for row in external:
            if row.get("human_data") is not False or row.get("speech_act") not in {"descriptive", "evaluative", "imperative"} or row.get("grounding") not in {"action", "feature", "trajectory"}:
                raise ValueError("Invalid external synthetic seed")
            if v1.normalized(row["text"]) not in excluded:
                train.extend({**row, "split": "train", "seed_sampling_weight": args.seed_weight} for _ in range(args.seed_weight))
    elif not args.allow_missing_external_seeds:
        raise ValueError("Waiting for independent-style synthetic training seeds")
    # Make repetitions within one high-cardinality template less dominant.
    grouped = defaultdict(list)
    for row in train:
        if v1.normalized(row["text"]) not in excluded:
            grouped[row["group_id"]].append(row)
    rng = random.Random(SEED)
    train = []
    for family, values in grouped.items():
        rng.shuffle(values)
        train.extend(values[:1800] if family.startswith('v3_matched_np_') else values[:800])
    rng.shuffle(train)
    assert not {v1.normalized(r["text"]) for r in train} & excluded
    args.data_dir.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)
    paths = [args.data_dir / "train.json", args.data_dir / "dev.json"]
    v1.save_json(paths[0], train)
    v1.save_json(paths[1], dev)
    print(f"v3 dual head: {len(train)} synthetic train / {len(dev)} exposed development; external seeds {len(external)}", flush=True)
    report = {"training_scope": "synthetic_only", "human_data_read": False, "new_third_test_read": False,
        "first_two_acceptance_sets_now_exposed_dev": True, "exposed_dev_hashes": [v2.EXPOSED_SHA, SECOND_SHA],
        "train_rows": len(train), "dev_rows": len(dev), "train_sha256": v1.sha256(paths[0]), "dev_sha256": v1.sha256(paths[1]),
        "training_script_sha256": v1.sha256(Path(__file__)), "train_dev_normalized_overlap": 0,
        "excluded_generated_text_collisions": collisions, "external_seeds": len(external),
        "external_seed_sha256": v1.sha256(EXTERNAL_SEEDS) if external else None,
        "external_seed_sampling_weight": args.seed_weight,
        "seed_weight_development_grid": [8, 24],
        "earlier_exploratory_seed_weights": [1, 8, 24],
        "seed_weight_selection": {
            "criterion": "Mean of the two heads' macro F1 on the combined exposed development set; no third test access.",
            "final_corpus_comparison": {"8": 0.9899198180255533, "24": 0.9916380903139117},
            "selected_weight": 24,
        },
        "unique_train_texts": len({v1.normalized(r['text']) for r in train}),
        "train_joint_labels": dict(Counter(f'{r["speech_act"]}/{r["grounding"]}' for r in train)), "heads": {}, "regressions": {}}
    import joblib
    from sklearn.metrics import classification_report, accuracy_score
    for head_name in ["speech_act", "grounding"]:
        head = v1.fit_head(train, dev, head_name, args.output_dir, paths)
        artifact = joblib.load(args.output_dir / f"{head_name}.joblib")
        report["regressions"][head_name] = {}
        for split_name, split in [("original_dev", old_dev), ("first_exposed_singles", first[:150]),
            ("second_exposed_singles", second[:150]), ("exposed_mixed_clauses", first[150:] + second[150:])]:
            prediction = artifact["classifier"].predict(artifact["vectorizer"].transform([r["text"] for r in split]))
            expected = [r[head_name] for r in split]
            metrics = {"accuracy": float(accuracy_score(expected, prediction)), "rows": len(split),
                "classification": classification_report(expected, prediction, output_dict=True, zero_division=0),
                "errors": [{"text": r["text"], "expected": r[head_name], "actual": str(p)} for r, p in zip(split, prediction) if r[head_name] != p]}
            report["regressions"][head_name][split_name] = metrics
            print(head_name, split_name, metrics["accuracy"], flush=True)
        browser_path = args.output_dir / f"{head_name}.json"
        browser = json.loads(browser_path.read_text())
        browser["source"]["training_script_sha256"] = v1.sha256(Path(__file__))
        browser["source"]["exposed_dev_sha256"] = [v2.EXPOSED_SHA, SECOND_SHA]
        browser["model_card"]["development_scope"] = "First and second acceptance now exposed development only. Requires a fresh independent third evaluation."
        browser["model_card"]["external_synthetic_seed_weight"] = args.seed_weight
        browser_path.write_text(json.dumps(browser, sort_keys=True, separators=(",", ":")) + "\n")
        head["browser_sha256"] = v1.sha256(browser_path)
        report["heads"][head_name] = head
    v1.save_json(args.output_dir / "training_report.json", report)
    parity = []
    groups = set()
    for row in old_dev:
        if row["group_id"] not in groups:
            groups.add(row["group_id"])
            parity.append({"text": row["text"], "speech_act": row["speech_act"], "grounding": row["grounding"]})
    for head_name in ["speech_act", "grounding"]:
        artifact = joblib.load(args.output_dir / f"{head_name}.joblib")
        scores = artifact["classifier"].predict_proba(artifact["vectorizer"].transform([r["text"] for r in parity]))
        for row, probability in zip(parity, scores):
            row[f"{head_name}_scores"] = {str(k): float(v) for k, v in zip(artifact["classifier"].classes_, probability)}
    v1.save_json(args.output_dir / "browser_parity_fixtures.json", parity)
    v1.save_json(args.output_dir / "freeze_receipt.json", {"status": "frozen_before_third_independent_test",
        "training_report_sha256": v1.sha256(args.output_dir / "training_report.json"),
        "heads": {h: {"browser_sha256": report["heads"][h]["browser_sha256"], "model_sha256": report["heads"][h]["model_sha256"]} for h in report["heads"]},
        "new_third_test_read": False, "first_two_tests_exposed_dev": True})
    print(json.dumps({h: report["heads"][h]["browser_sha256"] for h in report["heads"]}), flush=True)


if __name__ == "__main__":
    main()
