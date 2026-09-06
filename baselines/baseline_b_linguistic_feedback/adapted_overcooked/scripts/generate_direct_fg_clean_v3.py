"""Generate the isolated current-ontology direct-fG clean-v3 corpus.

This module is deliberately data-input free.  Every row is rendered from the
typed scenarios below; no legacy synthetic, human, dev, test, frozen, or
membership file is opened.  The three-class target is assigned directly by the
speech-act render branch and is never derived from a reference classifier.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "feedback_form_classifier_candidates"
    / "direct_fg_clean_v3"
    / "generated_corpus.json"
)

VERSION = "direct-fg-clean-v3-current-ontology-v1"
SOURCE = "direct_fg_current_ontology_template_v3"
LABEL_SOURCE = "direct_surface_semantic_contract"
LABELS = ("evaluative", "imperative", "descriptive")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
VARIANTS_PER_FRAME = 10
FRAMES_PER_SCENARIO_PER_LABEL = 2

ALLOWED_ITEMS = frozenset({"tomato", "onion", "dish", "soup", None})
ALLOWED_ACTIONS = frozenset(
    {
        "GET_TOMATO",
        "PUT_TOMATO_IN_POT",
        "GET_ONION",
        "PUT_ONION_IN_POT",
        "GET_DISH",
        "PICKUP_SOUP",
        "SERVE_SOUP",
        "STASH_HELD_OBJECT",
        "YIELD_PATH",
        "WAIT",
    }
)
ALLOWED_LOCATIONS = frozenset(
    {
        None,
        "tomato dispenser",
        "onion dispenser",
        "dish dispenser",
        "pot",
        "serving station",
        "accessible counter",
        "upper corridor",
        "lower corridor",
        "pot access tile",
        "serving station access tile",
    }
)

# Word-boundary checks avoid false positives such as ``cut`` inside ``current``.
FORBIDDEN_PATTERNS = (
    r"\bchop(?:ped|ping)?\b",
    r"\bcut(?:ting)?\b",
    r"\bknife\b",
    r"\b(?:chopping\s+)?board\b",
    r"\bsink\b",
    r"\bwash(?:ed|ing)?\b",
    r"\bdirty\b",
    r"\bclean(?:ed|ing)?\s+(?:dish|plate)\b",
    r"\bdish\s+return\b",
    r"\bdish\s+rack\b",
    r"\bfridge\b",
    r"\brefrigerator\b",
    r"\boven\b",
    r"\bgrill\b",
    r"\bstove\b",
    r"\btrash\b",
    r"\bbin\b",
    r"\b(?:lettuce|meat|burger|rice|fish|cheese|bun)\b",
    r"\bprep\s+table\b",
    r"\bworktop\b",
    r"\bcenter\s+table\b",
    r"\b(?:tomato|onion)\s+crate\b",
    r"\bcentral\s+aisle\b",
    r"\bmain\s+aisle\b",
    r"\bcenter\s+lane\b",
    r"\bmiddle\s+path\b",
    r"\border\s+expires?\b",
    r"\b(?:floor|ground)\b",
    r"\bstart\s+the\s+pot\b",
    r"\bfourth\s+ingredient\b",
)
FORBIDDEN_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS)
BAD_GERUND_CONSTRUCTION_RE = re.compile(
    r"\b(?:start|continue|finish)\s+by\s+"
    r"(?:pick|get|collect|take|fetch|move|go|interact|retrieve|use|put|add|"
    r"place|carry|bring|complete|wait|stay|hold|remain|serve|deliver|set|"
    r"leave|stash|store|yield|step|clear|avoid|make)\b",
    re.IGNORECASE,
)

EVALUATIVE_TERMS = (
    "good choice",
    "helpful",
    "done well",
    "effective",
    "useful",
    "smart move",
    "efficient",
    "excellent",
    "correct",
    "valuable",
    "great",
    "well timed",
    "smooth",
    "strong decision",
    "good coordination",
    "well handled",
    "productive",
    "thoughtful",
    "good decision",
    "nicely coordinated",
    "successful",
    "good timing",
    "strong move",
    "positive choice",
    "bad choice",
    "unhelpful",
    "done poorly",
    "ineffective",
    "a mistake",
    "inefficient",
    "incorrect",
    "disruptive",
    "awkward",
    "too slow",
    "wasteful",
    "not useful",
    "poorly timed",
    "made coordination worse",
    "careless",
    "not effective",
    "frustrating",
    "counterproductive",
    "a weak decision",
    "not well handled",
    "badly coordinated",
    "a poor decision",
    "not helpful",
    "a negative choice",
)
EVALUATIVE_ANCHORS = (
    "just now",
    "last step",
    "a moment ago",
    "recent move",
    "recent step",
    "recent action",
    "last action",
    "completed move",
    "completed action",
    "what you did",
)
IMPERATIVE_PREFIXES = (
    "please ",
    "kindly ",
    "could you ",
    "would you ",
    "can you ",
    "you should ",
    "you need to ",
    "i need you to ",
    "go ahead and ",
    "make sure you ",
    "for your next move, ",
    "on this step, ",
    "before anything else, ",
    "your next task is to ",
    "focus on this: ",
    "i want you to ",
    "i would like you to ",
    "remember to ",
    "be sure to ",
    "right now, make sure you ",
    "next, ",
    "for this order, please ",
    "as your current task, ",
    "without delay, ",
    "take care to ",
    "the next thing to do is ",
    "your priority is to ",
    "work on this now: ",
    "handle this next: ",
    "do this now: ",
    "follow this instruction: ",
    "act on this: ",
    "start with this action: ",
    "continue with this action: ",
    "finish with this action: ",
    "use this move to ",
    "choose to ",
    "plan to ",
    "keep working on this: ",
    "stay focused and ",
    "make your next action this: ",
    "the immediate action is to ",
    "why don't you ",
    "would you please ",
    "take this action: ",
    "use the current step to ",
    "make this your next move: ",
    "carry out this request: ",
)
NEUTRAL_STATE_TERMS = (
    " is ",
    " are ",
    " has ",
    " holds ",
    " contains ",
    " requires ",
    " uses ",
    " accepts ",
    " remains ",
    " shows ",
    " carries ",
)


def normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class GroundingVariant:
    command: str
    past: str
    fact: str


@dataclass(frozen=True)
class TypedScenario:
    scenario_id: str
    canonical_action: str
    object_name: str | None
    source: str | None
    target: str | None
    precondition: str
    result: str
    variants: tuple[GroundingVariant, ...]


def _pickup_variants(item: str, source: str) -> tuple[GroundingVariant, ...]:
    commands = (
        f"pick up a {item} from the {source}",
        f"get a {item} from the {source}",
        f"collect a {item} at the {source}",
        f"take a {item} from the {source}",
        f"fetch a {item} from the {source}",
        f"move to the {source} and pick up a {item}",
        f"go to the {source} for a {item}",
        f"interact with the {source} to take a {item}",
        f"retrieve a {item} from the {source}",
        f"use the {source} to get a {item}",
    )
    past = (
        f"you picked up a {item} from the {source}",
        f"you got a {item} from the {source}",
        f"you collected a {item} at the {source}",
        f"you took a {item} from the {source}",
        f"you fetched a {item} from the {source}",
        f"you moved to the {source} and picked up a {item}",
        f"you went to the {source} for a {item}",
        f"you interacted with the {source} to take a {item}",
        f"you retrieved a {item} from the {source}",
        f"you used the {source} to get a {item}",
    )
    facts = (
        f"the AI chef is holding a {item} from the {source}",
        f"a {item} from the {source} is in the AI chef's hands",
        f"the AI chef has a {item} from the {source}",
        f"the AI chef holds one {item} from the {source}",
        f"one {item} from the {source} is held by the AI chef",
        f"the AI chef is carrying one {item} from the {source}",
        f"the held item is a {item} from the {source}",
        f"the AI chef's current item is a {item} from the {source}",
        f"the AI chef has exactly one {item} from the {source}",
        f"a {item} from the {source} remains in the AI chef's hands",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _pot_variants(item: str, contents: str, status: str) -> tuple[GroundingVariant, ...]:
    commands = (
        f"put the held {item} into the pot",
        f"add the held {item} to the pot",
        f"place the held {item} in the pot",
        f"carry the held {item} to the pot and add it",
        f"take the held {item} to the pot",
        f"interact with the pot while holding the {item}",
        f"use the held {item} as the next pot ingredient",
        f"move the held {item} from your hands into the pot",
        f"bring the held {item} to the pot and put it in",
        f"complete this pot interaction with the held {item}",
    )
    past = (
        f"you put the held {item} into the pot",
        f"you added the held {item} to the pot",
        f"you placed the held {item} in the pot",
        f"you carried the held {item} to the pot and added it",
        f"you took the held {item} to the pot",
        f"you interacted with the pot while holding the {item}",
        f"you used the held {item} as the next pot ingredient",
        f"you moved the held {item} from your hands into the pot",
        f"you brought the held {item} to the pot and put it in",
        f"you completed the pot interaction with the held {item}",
    )
    facts = (
        f"the pot contains {contents}",
        f"the pot has {contents}",
        f"the {status} pot contains {contents}",
        f"the current pot state is {status} with {contents}",
        f"the ingredient mix in the pot is {contents}",
        f"the pot holds {contents}",
        f"the pot is {status} and contains {contents}",
        f"the fixed order in the pot has {contents}",
        f"the pot's ingredients are {contents}",
        f"the recipe state shows {contents} in the {status} pot",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _wait_variants() -> tuple[GroundingVariant, ...]:
    commands = (
        "wait while the pot is cooking",
        "stay in place while the soup cooks",
        "hold your position during the cooking state",
        "remain still until the pot becomes ready",
        "use this step to wait for the cooking pot",
        "keep waiting while the full pot cooks",
        "stay where you are during this cooking step",
        "wait for the ready state before changing tasks",
        "remain in place while the fixed order cooks",
        "take no movement action while the pot is cooking",
    )
    past = (
        "you waited while the pot was cooking",
        "you stayed in place while the soup cooked",
        "you held your position during the cooking state",
        "you remained still until the pot became ready",
        "you used the step to wait for the cooking pot",
        "you kept waiting while the full pot cooked",
        "you stayed where you were during the cooking step",
        "you waited for the ready state before changing tasks",
        "you remained in place while the fixed order cooked",
        "you took no movement action while the pot was cooking",
    )
    facts = (
        "the pot is cooking",
        "the soup is in the cooking state",
        "the full pot is cooking the fixed order",
        "the cooking pot contains two tomatoes and one onion",
        "the pot remains in the cooking state",
        "the current pot status is cooking",
        "the fixed order is cooking in the pot",
        "the cooking state lasts twenty environment steps",
        "the pot has three ingredients and is cooking",
        "the recipe state shows a cooking pot",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _plate_soup_variants() -> tuple[GroundingVariant, ...]:
    commands = (
        "use a dish to pick up the ready soup",
        "plate the ready soup with the held dish",
        "pick up the ready soup with a dish",
        "take the ready soup from the pot with the held dish",
        "interact with the ready pot while holding a dish",
        "collect the ready soup in the held dish",
        "lift the ready soup from the pot with a dish",
        "fill the held dish with the ready soup",
        "use the held dish at the ready pot",
        "move the ready soup from the pot into the held dish",
    )
    past = (
        "you used a dish to pick up the ready soup",
        "you plated the ready soup with the held dish",
        "you picked up the ready soup with a dish",
        "you took the ready soup from the pot with the held dish",
        "you interacted with the ready pot while holding a dish",
        "you collected the ready soup in the held dish",
        "you lifted the ready soup from the pot with a dish",
        "you filled the held dish with the ready soup",
        "you used the held dish at the ready pot",
        "you moved the ready soup from the pot into the held dish",
    )
    facts = (
        "the AI chef is holding soup in a dish",
        "the ready soup is in the AI chef's dish",
        "the AI chef has soup from the ready pot",
        "the AI chef holds the soup in a dish",
        "the held item is soup from the ready pot",
        "the AI chef is carrying soup in a dish",
        "the ready pot is available to a chef holding a dish",
        "the AI chef's current item is soup in a dish",
        "the dish in the AI chef's hands contains soup",
        "the ready soup is held by the AI chef",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _serve_variants() -> tuple[GroundingVariant, ...]:
    commands = (
        "serve the held soup at the serving station",
        "deliver the soup in the dish to the serving station",
        "take the held soup to the serving station",
        "carry the soup in the dish to the serving station",
        "interact with the serving station while holding soup",
        "move the held soup to the serving station",
        "bring the soup in the dish to the serving station",
        "finish the delivery at the serving station",
        "use the serving station while holding soup",
        "complete the soup delivery at the serving station",
    )
    past = (
        "you served the held soup at the serving station",
        "you delivered the soup in the dish to the serving station",
        "you took the held soup to the serving station",
        "you carried the soup in the dish to the serving station",
        "you interacted with the serving station while holding soup",
        "you moved the held soup to the serving station",
        "you brought the soup in the dish to the serving station",
        "you finished the delivery at the serving station",
        "you used the serving station while holding soup",
        "you completed the soup delivery at the serving station",
    )
    facts = (
        "the serving station accepts soup held in a dish",
        "the AI chef is holding soup at the serving station",
        "the soup in the dish is at the serving station",
        "the serving station is the target for the held soup",
        "the AI chef has soup beside the serving station",
        "the delivery target is the serving station",
        "the AI chef holds soup in front of the serving station",
        "the soup delivery uses the serving station",
        "the serving station receives soup in a dish",
        "the held soup is at the serving station access tile",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _stash_variants(item: str) -> tuple[GroundingVariant, ...]:
    commands = (
        f"put the held {item} on an empty accessible counter",
        f"place the held {item} on an empty accessible counter",
        f"set the held {item} on an empty accessible counter",
        f"leave the held {item} on an empty accessible counter",
        f"move the held {item} onto an empty accessible counter",
        f"use an empty accessible counter for the held {item}",
        f"stash the held {item} on an empty accessible counter",
        f"interact with an empty accessible counter while holding the {item}",
        f"move the {item} from your hands to an empty accessible counter",
        f"store the held {item} on an empty accessible counter",
    )
    past = (
        f"you put the held {item} on an empty accessible counter",
        f"you placed the held {item} on an empty accessible counter",
        f"you set the held {item} on an empty accessible counter",
        f"you left the held {item} on an empty accessible counter",
        f"you moved the held {item} onto an empty accessible counter",
        f"you used an empty accessible counter for the held {item}",
        f"you stashed the held {item} on an empty accessible counter",
        f"you interacted with an empty accessible counter while holding the {item}",
        f"you moved the {item} from your hands to an empty accessible counter",
        f"you stored the held {item} on an empty accessible counter",
    )
    facts = (
        f"an occupied accessible counter is holding one {item}",
        f"one {item} is on an accessible counter",
        f"an accessible counter has one {item}",
        f"the staged item on the accessible counter is a {item}",
        f"one accessible counter contains a {item}",
        f"the {item} is staged on an accessible counter",
        f"an accessible counter holds the {item}",
        f"the counter inventory has one {item}",
        f"one {item} remains on an accessible counter",
        f"the current counter item is a {item}",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


def _yield_variants(location: str, target_phrase: str) -> tuple[GroundingVariant, ...]:
    commands = (
        f"move aside in the {location}",
        f"yield the {location} to the other chef",
        f"step away from the {target_phrase}",
        f"clear the {target_phrase} for the other chef",
        f"leave the {location} open for the teammate",
        f"avoid blocking the {target_phrase}",
        f"move out of the teammate's route in the {location}",
        f"use one move to clear the {target_phrase}",
        f"step out of the other chef's path in the {location}",
        f"make room for the teammate at the {target_phrase}",
    )
    past = (
        f"you moved aside in the {location}",
        f"you yielded the {location} to the other chef",
        f"you stepped away from the {target_phrase}",
        f"you cleared the {target_phrase} for the other chef",
        f"you left the {location} open for the teammate",
        f"you avoided blocking the {target_phrase}",
        f"you moved out of the teammate's route in the {location}",
        f"you used one move to clear the {target_phrase}",
        f"you stepped out of the other chef's path in the {location}",
        f"you made room for the teammate at the {target_phrase}",
    )
    facts = (
        f"the {location} is clear for the other chef",
        f"the teammate's route in the {location} is open",
        f"the AI chef is outside the {target_phrase}",
        f"the {target_phrase} is clear for the other chef",
        f"the other chef has an open route through the {location}",
        f"the AI chef is not blocking the {target_phrase}",
        f"the teammate's path in the {location} remains clear",
        f"the current path state shows a clear {target_phrase}",
        f"the other chef's route is open at the {target_phrase}",
        f"the AI chef has moved outside the {target_phrase}",
    )
    return tuple(GroundingVariant(*parts) for parts in zip(commands, past, facts))


SCENARIOS: tuple[TypedScenario, ...] = (
    TypedScenario("get_tomato_dispenser", "GET_TOMATO", "tomato", "tomato dispenser", None, "AI chef empty-handed", "AI chef holding tomato", _pickup_variants("tomato", "tomato dispenser")),
    TypedScenario("get_onion_dispenser", "GET_ONION", "onion", "onion dispenser", None, "AI chef empty-handed", "AI chef holding onion", _pickup_variants("onion", "onion dispenser")),
    TypedScenario("get_dish_dispenser", "GET_DISH", "dish", "dish dispenser", None, "AI chef empty-handed", "AI chef holding dish", _pickup_variants("dish", "dish dispenser")),
    TypedScenario("get_tomato_counter", "GET_TOMATO", "tomato", "accessible counter", None, "accessible counter holding tomato and AI chef empty-handed", "AI chef holding tomato", _pickup_variants("tomato", "accessible counter")),
    TypedScenario("get_onion_counter", "GET_ONION", "onion", "accessible counter", None, "accessible counter holding onion and AI chef empty-handed", "AI chef holding onion", _pickup_variants("onion", "accessible counter")),
    TypedScenario("get_dish_counter", "GET_DISH", "dish", "accessible counter", None, "accessible counter holding dish and AI chef empty-handed", "AI chef holding dish", _pickup_variants("dish", "accessible counter")),
    TypedScenario("get_soup_counter", "PICKUP_SOUP", "soup", "accessible counter", None, "accessible counter holding soup and AI chef empty-handed", "AI chef holding soup", _pickup_variants("soup", "accessible counter")),
    TypedScenario("put_tomato_empty_pot", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot empty", "pot filling with one tomato", _pot_variants("tomato", "one tomato", "filling")),
    TypedScenario("put_second_tomato", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot containing one tomato", "pot filling with two tomatoes", _pot_variants("tomato", "two tomatoes", "filling")),
    TypedScenario("put_onion_empty_pot", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot empty", "pot filling with one onion", _pot_variants("onion", "one onion", "filling")),
    TypedScenario("put_onion_after_tomato", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot containing one tomato", "pot filling with one tomato and one onion", _pot_variants("onion", "one tomato and one onion", "filling")),
    TypedScenario("complete_with_tomato", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot containing one tomato and one onion", "pot cooking two tomatoes and one onion", _pot_variants("tomato", "two tomatoes and one onion", "cooking")),
    TypedScenario("complete_with_onion", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot containing two tomatoes", "pot cooking two tomatoes and one onion", _pot_variants("onion", "two tomatoes and one onion", "cooking")),
    TypedScenario("wait_while_cooking", "WAIT", None, None, None, "pot cooking two tomatoes and one onion", "AI chef waiting while pot cooks", _wait_variants()),
    TypedScenario("plate_ready_soup", "PICKUP_SOUP", "soup", "pot", None, "AI chef holding dish and pot ready", "AI chef holding soup in dish", _plate_soup_variants()),
    TypedScenario("serve_held_soup", "SERVE_SOUP", "soup", None, "serving station", "AI chef holding soup", "soup delivered at serving station", _serve_variants()),
    TypedScenario("stash_tomato", "STASH_HELD_OBJECT", "tomato", None, "accessible counter", "AI chef holding tomato and accessible counter empty", "accessible counter holding tomato", _stash_variants("tomato")),
    TypedScenario("stash_onion", "STASH_HELD_OBJECT", "onion", None, "accessible counter", "AI chef holding onion and accessible counter empty", "accessible counter holding onion", _stash_variants("onion")),
    TypedScenario("stash_dish", "STASH_HELD_OBJECT", "dish", None, "accessible counter", "AI chef holding dish and accessible counter empty", "accessible counter holding dish", _stash_variants("dish")),
    TypedScenario("stash_soup", "STASH_HELD_OBJECT", "soup", None, "accessible counter", "AI chef holding soup and accessible counter empty", "accessible counter holding soup", _stash_variants("soup")),
    TypedScenario("yield_upper_corridor", "YIELD_PATH", None, "upper corridor", "upper corridor", "other chef approaching in upper corridor", "upper corridor clear", _yield_variants("upper corridor", "other chef's path")),
    TypedScenario("yield_lower_corridor", "YIELD_PATH", None, "lower corridor", "lower corridor", "other chef approaching in lower corridor", "lower corridor clear", _yield_variants("lower corridor", "other chef's path")),
    TypedScenario("clear_pot_access", "YIELD_PATH", None, "pot access tile", "pot access tile", "AI chef blocking pot access tile", "pot access tile clear", _yield_variants("pot access tile", "access tile in front of the pot")),
    TypedScenario("clear_serving_access", "YIELD_PATH", None, "serving station access tile", "serving station access tile", "AI chef blocking serving station access tile", "serving station access tile clear", _yield_variants("serving station access tile", "access tile in front of the serving station")),
)


EVALUATIVE_POSITIVE_FRAMES = (
    "{past_cap} just now, and that was a good choice.",
    "On the last step, {past}, which was helpful.",
    "{past_cap} a moment ago; that was done well.",
    "Your recent move was effective when {past}.",
    "What you did just now was useful: {past}.",
    "It was a smart move when {past} on the last step.",
    "That completed move was efficient because {past}.",
    "The way {past} just now was excellent.",
    "{past_cap} on the recent step, and that was correct.",
    "Your last action was valuable when {past}.",
    "What you did a moment ago was great: {past}.",
    "That recent action was well timed when {past}.",
    "{past_cap} just now, and the result was smooth.",
    "Your completed move was a strong decision: {past}.",
    "The last step showed good coordination when {past}.",
    "What you did on the recent move was well handled: {past}.",
    "{past_cap} a moment ago, which was productive.",
    "Your recent action was thoughtful when {past}.",
    "It was a good decision that {past} on the last step.",
    "That completed move was nicely coordinated: {past}.",
    "What you did just now was successful: {past}.",
    "Your last step showed good timing when {past}.",
    "{past_cap} on the recent move, and that was a strong move.",
    "That completed action was a positive choice because {past}.",
)
EVALUATIVE_NEGATIVE_FRAMES = (
    "{past_cap} just now, and that was a bad choice.",
    "On the last step, {past}, which was unhelpful.",
    "{past_cap} a moment ago; that was done poorly.",
    "Your recent move was ineffective when {past}.",
    "What you did just now was a mistake: {past}.",
    "It was inefficient when {past} on the last step.",
    "That completed move was incorrect because {past}.",
    "The way {past} just now was disruptive.",
    "{past_cap} on the recent step, and that was awkward.",
    "Your last action was too slow when {past}.",
    "What you did a moment ago was wasteful: {past}.",
    "That recent action was not useful when {past}.",
    "{past_cap} just now, and that was poorly timed.",
    "Your completed move made coordination worse: {past}.",
    "The last step was careless when {past}.",
    "What you did on the recent move was not effective: {past}.",
    "{past_cap} a moment ago, which was frustrating.",
    "Your recent action was counterproductive when {past}.",
    "It was a weak decision that {past} on the last step.",
    "That completed move was not well handled: {past}.",
    "What you did just now was badly coordinated: {past}.",
    "Your last step was a poor decision when {past}.",
    "{past_cap} on the recent move, and that was not helpful.",
    "That completed action was a negative choice because {past}.",
)

def _imperative_frame(prefix: str) -> str:
    question = prefix.startswith(
        ("could you ", "would you ", "can you ", "why don't you ")
    )
    return f"{prefix}{{command}}{'?' if question else '.'}"


IMPERATIVE_FRAMES_A = tuple(
    _imperative_frame(prefix) for prefix in IMPERATIVE_PREFIXES[:24]
)
IMPERATIVE_FRAMES_B = tuple(
    _imperative_frame(prefix) for prefix in IMPERATIVE_PREFIXES[24:48]
)

DESCRIPTIVE_FRAMES_A = (
    "{fact_cap}.",
    "Right now, {fact}.",
    "At the moment, {fact}.",
    "Currently, {fact}.",
    "The current state is this: {fact}.",
    "Here is the situation: {fact}.",
    "The task state shows that {fact}.",
    "For this order, {fact}.",
    "In the kitchen state, {fact}.",
    "As things stand, {fact}.",
    "The observed state is that {fact}.",
    "The present situation shows that {fact}.",
    "According to the current state, {fact}.",
    "The state record shows that {fact}.",
    "For the active order, {fact}.",
    "The current kitchen fact is that {fact}.",
    "In the present state, {fact}.",
    "The task facts show that {fact}.",
    "The current observation is that {fact}.",
    "The live state shows that {fact}.",
    "At this point, {fact}.",
    "The present task state is that {fact}.",
    "The kitchen observation shows that {fact}.",
    "The state at this moment is that {fact}.",
)
DESCRIPTIVE_FRAMES_B = (
    "One current fact is that {fact}.",
    "The observable situation is that {fact}.",
    "The current order state shows that {fact}.",
    "The present kitchen condition is that {fact}.",
    "The task currently shows that {fact}.",
    "The live observation is that {fact}.",
    "The current facts are clear: {fact}.",
    "The order state currently shows that {fact}.",
    "The visible kitchen state is that {fact}.",
    "The situation at this step is that {fact}.",
    "The state information shows that {fact}.",
    "The current task condition is that {fact}.",
    "The active kitchen state shows that {fact}.",
    "The observable task fact is that {fact}.",
    "The current scene shows that {fact}.",
    "The present order condition is that {fact}.",
    "The kitchen currently shows that {fact}.",
    "The task observation is that {fact}.",
    "The visible state shows that {fact}.",
    "The current condition is that {fact}.",
    "The order facts show that {fact}.",
    "The present observation is that {fact}.",
    "The current task record shows that {fact}.",
    "The kitchen state currently shows that {fact}.",
)


def speech_act_matches(text: str) -> tuple[str, ...]:
    lowered = f" {text.casefold().strip()} "
    evaluative = any(term in lowered for term in EVALUATIVE_TERMS) and any(
        anchor in lowered for anchor in EVALUATIVE_ANCHORS
    )
    stripped = text.casefold().strip()
    imperative = any(stripped.startswith(prefix) for prefix in IMPERATIVE_PREFIXES)
    descriptive = (
        not evaluative
        and not imperative
        and any(marker in lowered for marker in NEUTRAL_STATE_TERMS)
    )
    return tuple(
        label
        for label, matched in (
            ("evaluative", evaluative),
            ("imperative", imperative),
            ("descriptive", descriptive),
        )
        if matched
    )


def _render_text(label: str, frame: str, variant: GroundingVariant) -> str:
    if label == "evaluative":
        return frame.format(past=variant.past, past_cap=_cap(variant.past))
    if label == "imperative":
        return frame.format(command=variant.command, command_cap=_cap(variant.command))
    if label == "descriptive":
        return frame.format(fact=variant.fact, fact_cap=_cap(variant.fact))
    raise ValueError(f"unknown direct-fG label: {label}")


def _frames_for(label: str, scenario_index: int) -> tuple[tuple[str, str], ...]:
    if label == "evaluative":
        return (
            (EVALUATIVE_POSITIVE_FRAMES[scenario_index], "positive"),
            (EVALUATIVE_NEGATIVE_FRAMES[scenario_index], "negative"),
        )
    if label == "imperative":
        return (
            (IMPERATIVE_FRAMES_A[scenario_index], "directive"),
            (IMPERATIVE_FRAMES_B[scenario_index], "directive"),
        )
    if label == "descriptive":
        return (
            (DESCRIPTIVE_FRAMES_A[scenario_index], "neutral"),
            (DESCRIPTIVE_FRAMES_B[scenario_index], "neutral"),
        )
    raise ValueError(label)


def validate_scenarios() -> None:
    if len(SCENARIOS) != 24:
        raise ValueError(f"clean-v3 requires 24 typed scenarios, got {len(SCENARIOS)}")
    scenario_ids = [scenario.scenario_id for scenario in SCENARIOS]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("typed scenario ids must be unique")
    for scenario in SCENARIOS:
        if scenario.canonical_action not in ALLOWED_ACTIONS:
            raise ValueError(f"{scenario.scenario_id}: disallowed action")
        if scenario.object_name not in ALLOWED_ITEMS:
            raise ValueError(f"{scenario.scenario_id}: disallowed object")
        if scenario.source not in ALLOWED_LOCATIONS or scenario.target not in ALLOWED_LOCATIONS:
            raise ValueError(f"{scenario.scenario_id}: disallowed source/target")
        if len(scenario.variants) != VARIANTS_PER_FRAME:
            raise ValueError(f"{scenario.scenario_id}: expected 10 grounding variants")


def generate_rows() -> list[dict]:
    validate_scenarios()
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        scenario_group = f"clean-v3:scenario:{scenario.scenario_id}"
        for label in LABELS:
            for frame_offset, (frame, stance) in enumerate(
                _frames_for(label, scenario_index)
            ):
                family_index = scenario_index * FRAMES_PER_SCENARIO_PER_LABEL + frame_offset
                surface_family = f"clean-v3:{label}:frame-{family_index:02d}"
                for variant_index, variant in enumerate(scenario.variants):
                    text = _render_text(label, frame, variant)
                    normalized = normalize_text(text)
                    grounding_id = (
                        f"clean-v3:grounding:{scenario.scenario_id}:{variant_index:02d}"
                    )
                    temporal_frame = {
                        "evaluative": "recent_completed_event",
                        "imperative": "current_or_future_action",
                        "descriptive": "current_state_or_mechanism_fact",
                    }[label]
                    rows.append(
                        {
                            "feedback_id": hashlib.sha256(
                                f"{surface_family}|{variant_index}|{normalized}".encode("utf-8")
                            ).hexdigest()[:24],
                            "text": text,
                            "normalized_text": normalized,
                            "expected_feedback_type": label,
                            "classification_label": CANONICAL_LABELS[label],
                            "source": SOURCE,
                            "label_source": LABEL_SOURCE,
                            "scenario_id": scenario.scenario_id,
                            "scenario_group": scenario_group,
                            "grounding_id": grounding_id,
                            "surface_family": surface_family,
                            "paraphrase_group": surface_family,
                            "frame_stance": stance,
                            "semantic_payload": {
                                "actor": "AI chef",
                                "action": scenario.canonical_action,
                                "object": scenario.object_name,
                                "source": scenario.source,
                                "target": scenario.target,
                                "precondition": scenario.precondition,
                                "result": scenario.result,
                                "temporal_frame": temporal_frame,
                            },
                            "speech_act_contract": {
                                "direct_target": True,
                                "single_speech_act": True,
                                "validator_expected_match": label,
                            },
                            "generation_provenance": {
                                "ai_assisted": True,
                                "prior_error_informed": True,
                                "independent_human_semantic_audit": False,
                                "human_gold": False,
                                "real_player_data": False,
                            },
                        }
                    )
    audit_rows(rows)
    return rows


def audit_rows(rows: list[dict]) -> dict:
    expected_rows = len(SCENARIOS) * len(LABELS) * FRAMES_PER_SCENARIO_PER_LABEL * VARIANTS_PER_FRAME
    if len(rows) != expected_rows or len(rows) < 1440:
        raise ValueError(f"expected at least 1440 balanced rows, got {len(rows)}")
    labels = Counter(str(row.get("expected_feedback_type")) for row in rows)
    if len(set(labels.values())) != 1 or set(labels) != set(LABELS):
        raise ValueError(f"direct-fG labels are not balanced: {dict(labels)}")
    normalized = [str(row.get("normalized_text") or "") for row in rows]
    if any(not value for value in normalized):
        raise ValueError("empty normalized text")
    if len(set(normalized)) != len(normalized):
        duplicate_count = len(normalized) - len(set(normalized))
        raise ValueError(f"normalized exact duplicates: {duplicate_count}")
    bad_terms: list[dict] = []
    bad_contracts: list[dict] = []
    bad_grammar: list[dict] = []
    for row in rows:
        text = str(row["text"])
        matches = speech_act_matches(text)
        expected = str(row["expected_feedback_type"])
        if matches != (expected,):
            bad_contracts.append(
                {"text": text, "expected": expected, "matches": list(matches)}
            )
        for pattern, compiled in zip(FORBIDDEN_PATTERNS, FORBIDDEN_RE):
            if compiled.search(text):
                bad_terms.append({"text": text, "pattern": pattern})
        if "{" in text or "}" in text:
            bad_grammar.append({"text": text, "reason": "unresolved_placeholder"})
        if BAD_GERUND_CONSTRUCTION_RE.search(text):
            bad_grammar.append({"text": text, "reason": "by_plus_base_verb"})
        if len(re.findall(r"[.!?]", text)) != 1:
            bad_grammar.append({"text": text, "reason": "not_one_terminal_sentence"})
    if bad_terms:
        raise ValueError(f"forbidden ontology terms found: {bad_terms[:3]}")
    if bad_contracts:
        raise ValueError(f"speech-act contract failures: {bad_contracts[:3]}")
    if bad_grammar:
        raise ValueError(f"surface grammar failures: {bad_grammar[:3]}")
    family_counts = Counter(str(row["expected_feedback_type"]) for row in rows)
    family_sets = {
        label: {str(row["surface_family"]) for row in rows if row["expected_feedback_type"] == label}
        for label in LABELS
    }
    if any(len(families) != 48 for families in family_sets.values()):
        raise ValueError("each direct-fG label must have 48 surface families")
    if any("reference_type" in row for row in rows):
        raise ValueError("reference_type is forbidden in direct-fG clean-v3 rows")
    return {
        "version": VERSION,
        "rows": len(rows),
        "label_counts": dict(sorted(family_counts.items())),
        "scenario_count": len({row["scenario_id"] for row in rows}),
        "surface_family_counts": {
            label: len(family_sets[label]) for label in LABELS
        },
        "normalized_unique_rows": len(set(normalized)),
        "forbidden_ontology_hits": 0,
        "single_speech_act_failures": 0,
        "surface_grammar_failures": 0,
        "target_source": LABEL_SOURCE,
        "reference_type_used": False,
        "external_input_files_opened": 0,
        "ai_assisted": True,
        "prior_error_informed": True,
        "independent_human_semantic_audit": False,
        "human_gold": False,
        "real_player_accuracy_claim": False,
        "style_limit": "synthetic finite-frame English; not natural-player gold",
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows = generate_rows()
    audit = audit_rows(rows)
    if not args.audit_only:
        _write_json(args.output, rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
