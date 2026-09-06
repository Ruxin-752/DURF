"""Build and freeze two synthetic-only feedback heads with the paper model family.

This script never opens production data, human data, or an independent test.
Run from the repository with .venv/Scripts/python.exe scripts/train_synthetic_feedback_v1.py.
The first run creates a new corpus and immutable run directory; it refuses overwrites.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260906
C_GRID = (0.25, 1.0, 4.0, 16.0, 64.0)

ACTION_RECORDS = [
    ("move left", "moving left", "moved left"),
    ("move right", "moving right", "moved right"),
    ("walk up", "walking up", "walked up"),
    ("walk down", "walking down", "walked down"),
    ("go to the stove", "going to the stove", "went to the stove"),
    ("go around the counter", "going around the counter", "went around the counter"),
    ("head to the serving window", "heading to the serving window", "headed to the serving window"),
    ("pick up the onion", "picking up the onion", "picked up the onion"),
    ("pick up the tomato", "picking up the tomato", "picked up the tomato"),
    ("take a clean plate", "taking a clean plate", "took a clean plate"),
    ("grab a bowl", "grabbing a bowl", "grabbed a bowl"),
    ("fetch another onion", "fetching another onion", "fetched another onion"),
    ("put the onion in the pot", "putting the onion in the pot", "put the onion in the pot"),
    ("drop the plate on the counter", "dropping the plate on the counter", "dropped the plate on the counter"),
    ("place the bowl beside the stove", "placing the bowl beside the stove", "placed the bowl beside the stove"),
    ("add the last ingredient", "adding the last ingredient", "added the last ingredient"),
    ("fill the empty pot", "filling the empty pot", "filled the empty pot"),
    ("start cooking", "starting to cook", "started cooking"),
    ("cook the soup", "cooking the soup", "cooked the soup"),
    ("serve the soup", "serving the soup", "served the soup"),
    ("deliver the order", "delivering the order", "delivered the order"),
    ("bring the bowl to me", "bringing the bowl to me", "brought the bowl to me"),
    ("carry the soup to the window", "carrying the soup to the window", "carried the soup to the window"),
    ("wash the dirty dishes", "washing the dirty dishes", "washed the dirty dishes"),
    ("clear the counter", "clearing the counter", "cleared the counter"),
    ("leave the passage open", "leaving the passage open", "left the passage open"),
    ("make room for me", "making room for me", "made room for me"),
    ("wait by the pot", "waiting by the pot", "waited by the pot"),
    ("stay still", "staying still", "stayed still"),
    ("avoid the narrow passage", "avoiding the narrow passage", "avoided the narrow passage"),
    ("return to the kitchen", "returning to the kitchen", "returned to the kitchen"),
    ("turn toward the sink", "turning toward the sink", "turned toward the sink"),
    ("check the timer", "checking the timer", "checked the timer"),
    ("chop the vegetables", "chopping the vegetables", "chopped the vegetables"),
    ("help with the dishes", "helping with the dishes", "helped with the dishes"),
    ("use the other stove", "using the other stove", "used the other stove"),
    ("keep the plate", "keeping the plate", "kept the plate"),
    ("hand me the onion", "handing me the onion", "handed me the onion"),
    ("interact with the dispenser", "interacting with the dispenser", "interacted with the dispenser"),
]

SUBJECT_STATES = [
    ("the pot", "empty"), ("the soup", "ready"), ("the onion", "on the counter"),
    ("the tomato", "beside the pot"), ("the plate", "on the left"),
    ("the bowl", "on the right"), ("the serving window", "on the right"),
    ("the stove", "behind you"), ("the counter", "blocked"),
    ("the passage", "clear"), ("the sink", "occupied"),
    ("the dispenser", "at the top"), ("the order", "waiting"),
    ("the other chef", "carrying a bowl"), ("the timer", "at zero"),
    ("the second pot", "still cooking"), ("the dirty dish", "near the sink"),
    ("the kitchen", "crowded"), ("your plate", "empty"),
    ("my bowl", "full"), ("the meal", "finished"),
    ("the burner", "switched off"), ("the lower corridor", "open"),
    ("the left counter", "free"), ("the top station", "available"),
]

FEATURES = ["onions", "tomatoes", "clean plates", "empty bowls", "ready soup",
    "filled pots", "open paths", "clear counters", "washed dishes", "delivered soup",
    "completed orders", "cooking time", "walking distance", "extra ingredients",
    "the left stove", "the serving window", "the nearest pot", "the bottom counter",
    "a full pot", "an empty counter", "the onion supply", "tomato soup", "onion soup",
    "the number of deliveries", "unnecessary movement", "idle time", "collisions"]
GOOD = ["good", "great", "excellent", "nice", "helpful", "clever", "efficient",
    "smart", "useful", "impressive", "brilliant", "perfect", "fantastic", "wonderful", "worthwhile"]
BAD = ["bad", "poor", "terrible", "unhelpful", "wasteful", "slow", "careless",
    "inefficient", "wrong", "awful", "unnecessary", "disappointing", "clumsy", "unwise", "regrettable", "a shame"]
PERFORMANCE = ["that", "your work", "your performance", "our teamwork", "this round",
    "the last round", "the whole run", "your cooking", "your cooperation", "that effort"]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def feature_be(feature: str) -> str:
    return "are" if feature in {"onions", "tomatoes", "clean plates", "empty bowls",
        "filled pots", "open paths", "clear counters", "washed dishes", "completed orders",
        "extra ingredients", "collisions"} else "is"


def audit_family_cores(train, dev):
    """Slot-masked exact skeleton checks supplement explicit family/text isolation.

    Shared vocabulary and partial clauses are expected. This does not certify
    that an independent linguist would group every paraphrase the same way.
    """
    slots = set(FEATURES + GOOD + BAD + [x for x in PERFORMANCE if x != "that"])
    slots.update(itertools.chain.from_iterable(ACTION_RECORDS))
    slots.update(itertools.chain.from_iterable(SUBJECT_STATES))
    phrases = sorted({normalized(x) for x in slots}, key=lambda x: (-len(x), x))
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in phrases) + r")\b")
    def core(row):
        return pattern.sub("SLOT", normalized(row["text"]))
    train_cores = {core(r) for r in train}
    dev_cores = {core(r) for r in dev}
    overlap = train_cores & dev_cores
    if overlap:
        raise ValueError(f"slot-masked template core shared across train/dev: {sorted(overlap)}")
    return {"slot_masked_exact_core_overlap": 0,
        "train_distinct_slot_masked_cores": len(train_cores),
        "dev_distinct_slot_masked_cores": len(dev_cores),
        "partial_clause_and_lexical_overlap": "Present by design; this remains synthetic development, not an independent natural-language test.",
        "scope": "Explicit family isolation plus exact normalized text and slot-masked syntax checks; no claim of universal paraphrase disjointness."}


def make_corpus() -> tuple[list[dict], list[dict]]:
    """Each explicit family belongs to one split. Slot values are synthetic facts."""
    rows = []
    seen = {}
    def add(family, split, speech, grounding, texts):
        for text in texts:
            text = text[0].upper() + text[1:]
            text = re.sub(r" +", " ", text).strip()
            key = normalized(text)
            if key in seen:
                old = seen[key]
                if (old["speech_act"], old["grounding"]) != (speech, grounding):
                    raise ValueError(f"conflicting synthetic label: {text}")
                if old["split"] != split:
                    raise ValueError(f"cross-split normalized duplicate: {text}")
                continue
            row = {"text": text, "speech_act": speech, "grounding": grounding,
                   "group_id": family, "split": split, "source": "new_programmatic_synthetic_20260906",
                   "label_source": "explicit_synthetic_template_semantics", "human_data": False}
            seen[key] = row
            rows.append(row)

    def action_family(family, split, speech, grounding, templates):
        irregular = {"went": "gone", "took": "taken", "did": "done"}
        add(family, split, speech, grounding,
            (template.format(a=a, ing=ing, past=past,
                pp=irregular.get(past.split()[0], past.split()[0]) + " " + " ".join(past.split()[1:]))
             for template, (a, ing, past) in itertools.product(templates, ACTION_RECORDS)))

    # Direct, polite, indirect, negative, and cooperative instructions.
    for family, templates in {
        "bare_directive": ["{a}.", "{a}!", "{a} now.", "{a} next."],
        "please_directive": ["Please {a}.", "{a}, please.", "Please do {a}."],
        "can_request": ["Can you {a}?", "Can you please {a}?"],
        "could_request": ["Could you {a}?", "Could you please {a}?"],
        "would_request": ["Would you {a}?", "Would you kindly {a}?"],
        "will_request": ["Will you {a}?", "Will you please {a}?"],
        "modal_should": ["You should {a}.", "We should {a}."],
        "modal_must": ["You must {a}.", "You ought to {a}.", "You have to {a}."],
        "need_action": ["You need to {a}.", "We need to {a}.", "I need you to {a}."],
        "want_action": ["I want you to {a}.", "I would like you to {a}."],
        "let_us": ["Let's {a}.", "Let us {a}."],
        "negative_directive": ["Don't {a}.", "Do not {a}.", "Never {a}."],
        "negative_modal": ["You should not {a}.", "You shouldn't {a}.", "You must not {a}."],
        "stop_gerund": ["Stop {ing}.", "Quit {ing}.", "Avoid {ing}."],
        "start_gerund": ["Start {ing}.", "Keep {ing}.", "Continue {ing}."],
        "try_action": ["Try to {a}.", "Try {ing}.", "Remember to {a}."],
        "instead_action": ["Instead, {a}.", "Rather than waiting, {a}."],
        "conditional_request": ["If you can, {a}.", "When you are ready, {a}."],
        "time_instruction": ["Next time, {a}.", "For now, {a}.", "Your next move is to {a}."],
        "request_reason": ["{a} so we can finish the order.", "{a} because the timer is running."],
        "imperative_advice": ["It would help if you could {a}.", "It is time to {a}."],
        "request_mind": ["Would you mind {ing}?", "Do you mind {ing}?"],
        "request_suggest": ["I suggest you {a}.", "I recommend that you {a}."],
        "request_consider": ["Think about {ing}.", "Consider {ing}.", "Consider whether to {a}."],
        "no_need_action": ["There is no need to {a}.", "You don't need to {a}."],
    }.items():
        action_family(family, "train", "imperative", "action", templates)

    # Counterfactual criticism targets an action; it is not a new command.
    for family, templates in {
        "should_have": ["You should have {pp}.", "You ought to have {pp}."],
        "should_not_have": ["You should not have {pp}.", "You shouldn't have {pp}."],
        "past_action_judgment": ["{ing} was a good move.", "{ing} was a bad idea.",
                                  "{ing} was the wrong choice.", "{ing} was the right choice."],
        "gerund_praise": ["Good choice {ing}.", "Nice job {ing}.", "Well done {ing}."],
        "gerund_criticism": ["You wasted time {ing}.", "You made a mistake by {ing}."],
        "retrospective_thanks": ["Thanks for {ing}.", "Thank you for {ing}."],
        "wish_past": ["I wish you had {pp}.", "I wish you had not {pp}."],
        "counterfactual_better": ["It would have been better if you had {pp}.",
                                  "It would have been worse if you had {pp}."],
        "past_action_reaction": ["I'm glad you {past}.", "I'm disappointed that you {past}."],
        "past_specific_rating": ["You {past} too slowly.", "You {past} very well.",
                                  "You {past} at the wrong time.", "You {past} perfectly."],
        "past_action_dislike": ["I liked that you {past}.", "I hated how you {past}."],
    }.items():
        action_family(family, "train", "evaluative", "action" if family in {
            "should_have", "should_not_have", "wish_past", "counterfactual_better"
        } else "trajectory", templates)

    add("executed_action_quality", "train", "evaluative", "trajectory", (
        t.format(past=past, ing=ing, adj=adj)
        for t, (a, ing, past), adj in itertools.product([
            "That was {adj} when you {past}.", "Your {ing} was {adj}.",
            "{adj} work on {ing}.", "I thought it was {adj} that you {past}.",
        ], ACTION_RECORDS, GOOD + BAD) if not (adj == "a shame" and t.startswith("{adj}"))))
    add("completed_work_rating", "train", "evaluative", "trajectory", (
        f"You did {modifier} {adj} work {context}."
        for modifier, adj, context in itertools.product(["", "some"], GOOD + BAD[:-1],
            ["at the stove", "with those onions", "on that order", "during the last delivery"])))

    for family, templates in {
        "neutral_past_action": ["You {past}.", "I {past}.", "The chef {past}."],
        "neutral_present_action": ["You are {ing}.", "I am {ing}.", "The chef is {ing}."],
        "neutral_past_progressive": ["You were {ing}.", "The other chef was {ing}."],
        "neutral_action_observation": ["I noticed that you {past}.", "I saw you {ing}."],
        "neutral_negative_past": ["You did not {a}.", "You didn't {a}."],
        "neutral_action_plan": ["I will {a}.", "I'm going to {a}.", "My plan is to {a}."],
        "neutral_action_recent": ["You just {past}.", "You already {past}."],
    }.items():
        action_family(family, "train", "descriptive", "action" if family == "neutral_action_plan" else "trajectory", templates)
    action_family("hypothetical_action_fact", "train", "descriptive", "action", [
        "One possible action is to {a}.", "You could either {a} or wait.",
        "I am considering whether to {a}."])

    for family, templates in {
        "state_plain": ["{s} is {state}.", "{s} is currently {state}."],
        "state_still": ["{s} is still {state}.", "{s} is already {state}."],
        "state_negative": ["{s} is not {state}.", "{s} isn't {state}."],
        "state_observed": ["I can see that {s} is {state}.", "I noticed {s} is {state}."],
        "state_report": ["Right now, {s} is {state}.", "At the moment, {s} is {state}."],
        "state_reported": ["It looks like {s} is {state}.", "Apparently {s} is {state}."],
        "state_past": ["{s} was {state}.", "{s} was not {state}."],
    }.items():
        add(family, "train", "descriptive", "feature", (t.format(s=s, state=state)
            for t, (s, state) in itertools.product(templates, SUBJECT_STATES)))

    add("state_existential", "train", "descriptive", "feature", (
        f"{prefix} {f} {where}." for prefix, f, where in itertools.product(
            ["There are", "I can see"], ["onions", "tomatoes", "plates", "bowls", "dirty dishes"],
            ["on the counter", "by the sink", "near the pot", "in the corner", "on the left"])))
    add("state_count", "train", "descriptive", "feature", (
        f"{lead} {n} {f}." for lead, n, f in itertools.product(
            ["There are", "We have", "The kitchen contains"], ["no", "two", "three", "four"],
            ["onions", "clean plates", "ready orders", "empty pots", "bowls"])))
    add("state_location", "train", "descriptive", "feature", (
        f"The {item} {link} {place}." for item, link, place in itertools.product(
            ["stove", "sink", "serving window", "counter", "onion dispenser", "dish dispenser"],
            ["is located", "sits", "stands"], ["on the left", "on the right", "at the top", "at the bottom"])))
    add("state_machine", "train", "descriptive", "feature", [
        "The soup is cooking.", "The soup has finished cooking.", "The pot needs one more onion.",
        "The order requires three onions.", "This recipe uses tomatoes.", "The timer says ten seconds.",
        "We have no plates left.", "We don't have any onions.", "There isn't any soup ready.",
        "The stove has not started yet.", "The pot contains two onions.", "The path is blocked by a plate.",
        "I am holding a dish.", "You are carrying an onion.", "Your hands are empty.",
        "My inventory is full.", "You have a bowl in your hands.", "I am next to the sink.",
        "The dish is dirty.", "The plate is clean.", "The soup is cold.", "The pot is hot."])

    for family, templates in {
        "feature_valued": ["{f} {be} valuable.", "{f} {be} useful.", "{f} {be} important.",
                           "{f} {be} worthless.", "{f} {be} a waste.", "{f} {be} not useful."],
        "feature_preference": ["I prefer {f}.", "I like {f}.", "I dislike {f}.",
                               "I love {f}.", "I hate {f}.", "I don't like {f}.",
                               "I do not like {f}.", "I do not value {f}."],
        "feature_reward": ["{f} {matter} to me.", "I value {f}.", "I care about {f}.",
                           "{f} {be} worth more points.", "{f} {be} worth less."],
        "feature_cost": ["We get rewarded for {f}.", "{f} {cost} us points.",
                         "{f} {be} beneficial.", "{f} {be} undesirable."],
        "feature_should_priority": ["Prioritize {f}.", "Focus on {f}.", "Pay attention to {f}."],
    }.items():
        add(family, "train", "imperative" if family == "feature_should_priority" else "evaluative",
            "feature", (t.format(f=f, be=feature_be(f), matter="matter" if feature_be(f) == "are" else "matters",
                                cost="cost" if feature_be(f) == "are" else "costs")
                        for t, f in itertools.product(templates, FEATURES)))
    add("feature_comparative", "train", "evaluative", "feature", (
        f"{f1} {feature_be(f1)} {comparison} {f2}." for f1, f2 in zip(FEATURES, reversed(FEATURES))
        for comparison in ["better than", "more valuable than", "less important than"]))
    add("feature_instruction_target", "train", "imperative", "feature", (
        f"{prefix} {f}." for prefix, f in itertools.product(
            ["Concentrate on", "Give priority to", "Aim for", "Avoid accumulating", "Remember the value of",
             "Focus more on", "Pay more attention to", "Think more about"], FEATURES)))
    add("feature_elliptical_rating", "train", "evaluative", "feature", (
        f"{f}: {rating}." for f, rating in itertools.product(FEATURES, ["useful", "worthless", "valuable", "unimportant"])))
    add("feature_worthwhile", "train", "evaluative", "feature", (
        f"{f} {feature_be(f)} {rating}." for f, rating in itertools.product(FEATURES,
            ["not worthwhile", "worthwhile", "not worth the effort", "a waste of time", "not beneficial"])))

    add("overall_rating", "train", "evaluative", "trajectory", (
        f"{p} {copula} {adj}." for p, copula, adj in itertools.product(
            PERFORMANCE, ["was", "is", "was not", "wasn't"], GOOD + BAD)))
    add("overall_like", "train", "evaluative", "trajectory", (
        f"{reaction} {p}." for reaction, p in itertools.product(
            ["I liked", "I loved", "I hated", "I didn't like", "I appreciate", "I'm happy with", "I'm disappointed with"], PERFORMANCE)))
    add("overall_short_rating", "train", "evaluative", "trajectory", (
        f"{prefix}{a}{suffix}" for prefix, a, suffix in itertools.product(
            ["", "Really ", "Very ", "Not ", "That was "], GOOD + BAD[:-1], [".", " work!"])))
    add("overall_completed", "train", "evaluative", "trajectory", [
        "Well done!", "Good job!", "Nice work!", "Not bad.", "Too slow.",
        "You did well.", "You did badly.", "You are doing great.", "You have done a great job.",
        "You have been doing poorly.", "You are not helping.", "That did not help.",
        "That helped a lot.", "This is much better.", "Much worse this time.",
        "That was not what I wanted.", "Exactly what I wanted.", "You let me down.",
        "We worked well together.", "We were too slow.", "Your timing was poor.",
        "Your timing was perfect.", "Good teamwork.", "No, that was wrong.",
        "Yes, that was right.", "Nice!", "Thanks!", "Thank you.", "Perfect!", "Bravo!",
        "I am satisfied with your performance.", "I am not satisfied with your performance."])
    add("trajectory_factual_result", "train", "descriptive", "trajectory", (
        f"{who} {event} {count} {unit if count != 'one' else {'orders': 'order', 'deliveries': 'delivery'}[unit]} {when}." for who, event, count, unit, when in itertools.product(
            ["We", "You", "The team"], ["completed", "finished"], ["one", "two", "three"],
            ["orders", "deliveries"], ["this round", "in the last episode", "during that run"])))
    add("trajectory_factual_score", "train", "descriptive", "trajectory", (
        f"{prefix} {n} {suffix}." for prefix, n, suffix in itertools.product(
            ["We scored", "The team earned", "Your total was"], ["zero", "ten", "twenty", "forty"],
            ["points this round", "points in total", "points during the episode"])))
    add("trajectory_factual_summary", "train", "descriptive", "trajectory", [
        "The round is over.", "The episode has ended.", "We ran out of time.",
        "We delivered no soup this round.", "Our total score stayed the same.",
        "You spent most of the round walking.", "You waited for half the episode.",
        "The team finished three orders in total.", "Both chefs cooked during the round.",
        "We changed roles during this episode.", "Our run lasted a minute."])
    add("trajectory_instruction", "train", "imperative", "trajectory", [
        "Repeat that performance.", "Use the same strategy again.", "Try a different overall strategy.",
        "Continue working as a team.", "Coordinate your overall plan with me.",
        "Play the next round the same way.", "Do the same thing for the rest of the round.",
        "Follow the strategy from the previous episode.", "Improve your overall performance.",
        "Change the way you play this round.", "Keep working in the same way.",
        "Don't repeat that performance.", "Don't use the same strategy again.", "Keep it up!"])
    add("tiny_commands", "train", "imperative", "action", [
        "Left!", "Right!", "Up!", "Down!", "Wait!", "Stop!", "Go!", "Move!", "Serve!",
        "Cook!", "Faster!", "Hurry!", "Again!", "Don't stop!", "Come here!", "Over here!",
        "This way!", "Out of the way!", "Move aside!", "Step back!", "Turn left!", "Turn right!",
        "Leave it!", "Drop it!", "Take it!", "Put it down!", "Not that one!", "Use this one!",
        "Give me that!", "Stay there!", "Wait here!", "Go ahead!", "Keep going!", "Hold on!",
        "Don't do that!", "Do that again!", "Please hurry up!", "Help me!", "Watch out!",
        "Don't get in my way!", "Stop blocking the aisle!", "Get the soup!", "No more onions!", "More onions, please!",
        "One plate, please!", "A bowl, please!", "Bring it here!", "Do it now!", "Let's go!"])

    # Development families are separate constructions, authored before fitting.
    # They measure synthetic generalization only and are not the independent test.
    for family, templates in {
        "dev_polite_grateful": ["I'd be grateful if you could {a}."],
        "dev_next_task": ["Your next task: {a}."],
        "dev_suggestion_question": ["How about {ing}?"],
        "dev_prohibition_reminder": ["Make sure you do not {a}."],
        "dev_instruction_after": ["After that, please {a}."],
        "dev_urgent_action": ["We urgently need you to {a}."],
    }.items():
        action_family(family, "dev", "imperative", "action", templates)
    for family, templates in {
        "dev_past_regret": ["It is a shame that you {past}."],
        "dev_past_skill": ["You did a wonderful job of {ing}."],
        "dev_counterfactual_preference": ["I would have preferred it if you had {pp}."],
        "dev_action_choice_verdict": ["Your decision to {a} was unwise."],
    }.items():
        action_family(family, "dev", "evaluative", "action" if family == "dev_counterfactual_preference" else "trajectory", templates)
    for family, templates in {
        "dev_action_observed_fact": ["I watched as you {past}."],
        "dev_action_current_fact": ["I am watching you {a}."],
    }.items():
        action_family(family, "dev", "descriptive", "trajectory", templates)
    add("dev_state_notification", "dev", "descriptive", "feature", (
        f"Status of {s}: {state}." for s, state in SUBJECT_STATES))
    add("dev_state_confirmation", "dev", "descriptive", "feature", (
        f"{s} remains {state}." for s, state in SUBJECT_STATES))
    add("dev_feature_priorities", "dev", "imperative", "feature", (
        f"Put more emphasis on {f}." for f in FEATURES))
    add("dev_feature_disvalue", "dev", "evaluative", "feature", (
        f"I do not find {f} worthwhile." for f in FEATURES))
    add("dev_feature_value", "dev", "evaluative", "feature", (
        f"For me, {f} {feature_be(f)} particularly valuable." for f in FEATURES))
    add("dev_overall_impression", "dev", "evaluative", "trajectory", (
        f"How {a} {p} turned out!" for p, a in itertools.product(PERFORMANCE, GOOD + BAD[:-1])))
    add("dev_trajectory_factual_account", "dev", "descriptive", "trajectory", (
        f"In that entire run we made {n} soup deliveries." for n in ["zero", "one", "two", "three", "four", "five"]))
    add("dev_trajectory_repeat_plan", "dev", "imperative", "trajectory", [
        "Please use that overall strategy for the next round.",
        "For the coming episode, repeat your previous performance.",
        "Would you change your entire strategy this round?",
        "For this run, please coordinate your whole plan with me."])
    random.Random(SEED).shuffle(rows)
    return ([r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"])


POLICY = {
    "language": "English only",
    "unit": "One clause with a dominant communicative purpose; split multi-purpose feedback before classification.",
    "speech_act": {
        "descriptive": "A neutral factual assertion about state, observed action, score, or declared speaker plan; no requested change or value judgment.",
        "evaluative": "Praise, criticism, value/preference judgments, or retrospective counterfactual criticism. 'You should have done X' evaluates a past choice.",
        "imperative": "An instruction, request, prohibition, or actionable suggestion about future behavior. Polite questions and 'You should do X' are requests.",
    },
    "grounding": {
        "action": "A future action proposal, requested next action, or counterfactual alternative action. Factual accounts and judgments of executed behavior instead refer to trajectory.",
        "feature": "A kitchen state/property, resource, spatial fact, objective, or explicitly valued feature, independent of a particular executed action.",
        "trajectory": "Actual current or past behavior, including an identified executed action; overall performance, teamwork, strategy, total episode outcome, or underspecified praise/criticism of recent behavior.",
    },
    "ambiguity": "No rule guarantees ambiguity has low softmax. Runtime must abstain on low scores, empty/OOV/unsupported input; short context-dependent or mixed clauses may still be wrong. Explicit labels represent the intended synthetic reading.",
    "limitations": "Synthetic family evaluation is not human accuracy. English only; no human training or testing data is read. This uses the paper's classifier family, not a claim of exact original-domain reproduction.",
    "score_policy": {"kind": "raw_model_softmax_score", "temperature": 1, "probability_of_correctness": False},
}


def fit_head(train, dev, target, output, data_paths):
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words=None)
    matrix = vectorizer.fit_transform([r["text"] for r in train])
    dev_matrix = vectorizer.transform([r["text"] for r in dev])
    labels = sorted({r[target] for r in train})
    expected = [r[target] for r in dev]
    trials = []
    best = None
    for c_value in C_GRID:
        clf = LogisticRegression(C=c_value, max_iter=3000, solver="lbfgs", random_state=SEED, class_weight="balanced")
        clf.fit(matrix, [r[target] for r in train])
        pred = clf.predict(dev_matrix)
        macro = f1_score(expected, pred, labels=labels, average="macro", zero_division=0)
        trials.append({"C": c_value, "dev_macro_f1": float(macro), "dev_accuracy": float(accuracy_score(expected, pred))})
        if best is None or macro > best[0]:
            best = macro, clf, c_value
    _, classifier, selected_c = best
    pred = classifier.predict(dev_matrix)
    probability = classifier.predict_proba(dev_matrix)
    confidence = probability.max(axis=1)
    artifact = {"vectorizer": vectorizer, "classifier": classifier, "target_semantics": target,
        "training_scope": "synthetic_only", "selected_C": selected_c, "policy": POLICY,
        "source_train_sha256": sha256(data_paths[0]), "source_dev_sha256": sha256(data_paths[1])}
    joblib_path = output / f"{target}.joblib"
    joblib.dump(artifact, joblib_path)
    transformer = {"name": "word", "analyzer": "word", "ngram_range": [1, 2],
        "lowercase": True, "strip_accents": None, "sublinear_tf": True, "norm": "l2",
        "token_pattern": vectorizer.token_pattern, "feature_offset": 0,
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(), "stop_words": None}
    browser = {"schema_version": "durf-synthetic-linear-v1", "classes": classifier.classes_.tolist(),
        "transformers": [transformer], "classifier": {"kind": "multinomial_logistic_regression",
            "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist()},
        "source": {"model_sha256": sha256(joblib_path), "training_script_sha256": sha256(Path(__file__)),
            "train_sha256": sha256(data_paths[0]), "dev_sha256": sha256(data_paths[1])},
        "target_semantics": target, "minimum_confidence": 0.55, "training_scope": "synthetic_only",
        "calibration": {"method": "none", "temperature": 1, "independently_validated": False},
        "canonical_labels": {label: label.capitalize() for label in labels},
        "score_policy": POLICY["score_policy"], "language": "en",
        "model_card": {"method": "word unigram/bigram TF-IDF plus multinomial logistic regression",
            "selection": "train-only fit; C chosen by macro F1 on family-disjoint synthetic dev; ties choose smaller C",
            "selected_C": selected_c, "stop_words": None, "class_weight": "balanced",
            "scope": "Synthetic English feedback only; no demonstrated human accuracy.",
            "routing_threshold": "0.55 is a prespecified policy starting value, not calibrated correctness; it was not tuned against dev or test accuracy."}}
    browser_path = output / f"{target}.json"
    # Compact deterministic serialization matters for browser payload sizes and hash checks.
    browser_path.write_text(json.dumps(browser, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    errors = [{"text": row["text"], "group_id": row["group_id"], "expected": row[target],
        "predicted": str(actual), "raw_model_score": float(score)}
        for row, actual, score in zip(dev, pred, confidence) if row[target] != actual]
    save_json(output / f"{target}.dev_errors.json", errors)
    # Independently reimplement exported sparse inference to detect a corrupt export.
    exported_logits = dev_matrix @ np.asarray(browser["classifier"]["coefficients"]).T + browser["classifier"]["intercept"]
    exported_logits -= exported_logits.max(axis=1, keepdims=True)
    exported_prob = np.exp(exported_logits)
    exported_prob /= exported_prob.sum(axis=1, keepdims=True)
    max_diff = float(np.abs(exported_prob - probability).max())
    if max_diff > 1e-10:
        raise ValueError("export inference does not match sklearn")
    return {"selected_C": selected_c, "selection_grid": trials,
        "classes": labels, "vocabulary_size": len(vectorizer.vocabulary_),
        "dev_accuracy": float(accuracy_score(expected, pred)),
        "dev_classification": classification_report(expected, pred, labels=labels, output_dict=True, zero_division=0),
        "dev_confusion_matrix": confusion_matrix(expected, pred, labels=labels).tolist(),
        "dev_coverage_at_055": float((confidence >= 0.55).mean()),
        "dev_selective_accuracy_at_055": float(np.asarray(expected)[confidence >= 0.55].__eq__(pred[confidence >= 0.55]).mean()),
        "model_sha256": sha256(joblib_path), "browser_sha256": sha256(browser_path),
        "export_sklearn_max_probability_difference": max_diff,
        "model_frozen_before_independent_test": True,
        "train_labels": dict(Counter(r[target] for r in train)), "dev_labels": dict(Counter(expected))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic_feedback_v1" / "final")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "synthetic_feedback_v1" / "final")
    args = parser.parse_args()
    if args.data_dir.exists() or args.output_dir.exists():
        raise SystemExit("Refusing to overwrite a synthetic corpus or frozen run; supply new directories.")
    import sklearn
    train, dev = make_corpus()
    train_groups, dev_groups = ({r["group_id"] for r in split} for split in (train, dev))
    train_texts, dev_texts = ({normalized(r["text"]) for r in split} for split in (train, dev))
    if train_groups & dev_groups or train_texts & dev_texts:
        raise ValueError("split leakage")
    core_audit = audit_family_cores(train, dev)
    args.data_dir.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)
    paths = [args.data_dir / "train.json", args.data_dir / "dev.json"]
    save_json(paths[0], train)
    save_json(paths[1], dev)
    save_json(args.data_dir / "label_policy.json", POLICY)
    report = {"schema_version": "durf-synthetic-training-report-v1", "training_scope": "synthetic_only",
        "language": "en", "seed": SEED, "sklearn_version": sklearn.__version__,
        "training_script_sha256": sha256(Path(__file__)), "human_data_read": False,
        "independent_test_read": False, "external_data_read": False,
        "source": "Fresh programmatic kitchen clauses with manually specified synthetic label rules, no participant records.",
        "paper_method_reference": "https://arxiv.org/abs/2009.14715",
        "recipe_scope": "Paper classifier family, kitchen-specific synthetic ontology and raw tokens. This is not exact paper reproduction or a claim of best measured human performance.",
        "train_rows": len(train), "dev_rows": len(dev),
        "train_sha256": sha256(paths[0]), "dev_sha256": sha256(paths[1]),
        "train_groups": len(train_groups), "dev_groups": len(dev_groups),
        "split_audit": {"group_overlap": 0, "normalized_text_overlap": 0,
            **core_audit,
            "family_assignment": "each explicit structural family wholly train or dev",
            "shared_slot_vocabulary": True, "semantic_paraphrase_overlap": "expected across the same task; not an independent natural-language test"},
        "label_cross_tab_train": dict(Counter(f'{r["speech_act"]}/{r["grounding"]}' for r in train)),
        "vectorizer": {"fit_split": "train_only", "word_ngram_range": [1, 2], "min_df": 1,
            "sublinear_tf": True, "stop_words": None, "negation_retained": True},
        "score_policy": POLICY["score_policy"], "heads": {}}
    report["development_history"] = {
        "development_only_iterations": 4,
        "dev_used_for_corpus_improvement": True,
        "dev_is_independent_acceptance": False,
        "changes": ["Corrected completed-action grounding to trajectory before final freezing.",
            "Expanded evaluation vocabulary and polite/indirect instruction families using developer-owned dev errors.",
            "Corrected verb participles and noun agreement.",
            "Replaced prefixed copies in developer-owned dev with distinct constructions before final freezing."],
        "independent_test_access": "none",
    }
    print(f"Fresh synthetic corpus: {len(train)} train / {len(dev)} dev; {len(train_groups)} / {len(dev_groups)} disjoint families", flush=True)
    for target in ("speech_act", "grounding"):
        report["heads"][target] = fit_head(train, dev, target, args.output_dir, paths)
        head = report["heads"][target]
        print(f'{target}: C={head["selected_C"]}, dev accuracy={head["dev_accuracy"]:.4f}, macro F1={head["dev_classification"]["macro avg"]["f1-score"]:.4f}', flush=True)
    save_json(args.output_dir / "training_report.json", report)
    save_json(args.output_dir / "freeze_receipt.json", {
        "status": "frozen_before_independent_test", "training_report_sha256": sha256(args.output_dir / "training_report.json"),
        "training_script_sha256": sha256(Path(__file__)),
        "artifacts": {target: {"joblib_sha256": report["heads"][target]["model_sha256"],
            "browser_sha256": report["heads"][target]["browser_sha256"]} for target in report["heads"]},
        "rule": "Do not retrain using independent test feedback; any changed model requires a fresh independent test."})
    print(json.dumps({key: report["heads"][key]["browser_sha256"] for key in report["heads"]}), flush=True)


if __name__ == "__main__":
    main()
