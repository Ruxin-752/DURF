"""Generate the audit-only clean-v6 direct-fG surface corpus.

The generator is deliberately input-free.  It renders 24 typed scenarios from
the current tomato/onion Overcooked layout through five independently authored
lexical banks.  No legacy corpus, human row, development set, test set, frozen
set, or membership file is opened.  The output is synthetic research data; it
is not human gold, player-domain evidence, or a production classifier artifact.

Only the typed semantic payload is shared across banks.  Every bank owns its
event, command, state, and outer-frame wording.  This prevents the clean-v5
failure mode in which a long wrapper hid a verbatim shared core clause.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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
    / "direct_fg_clean_v6"
    / "generated_corpus.json"
)
VERSION = "direct-fg-clean-v6-independent-core-banks-v1"
SOURCE = "direct_fg_current_ontology_independent_surface_banks_v6"
LABEL_SOURCE = "direct_surface_semantic_contract"
LABELS = ("evaluative", "imperative", "descriptive")
CANONICAL_LABELS = {
    "evaluative": "Evaluative",
    "imperative": "Imperative",
    "descriptive": "Descriptive",
}
GROUNDINGS_PER_SCENARIO = 6

ALLOWED_ITEMS = frozenset({None, "tomato", "onion", "dish", "soup"})
ALLOWED_ACTIONS = frozenset(
    {
        "GET_TOMATO",
        "GET_ONION",
        "GET_DISH",
        "PUT_TOMATO_IN_POT",
        "PUT_ONION_IN_POT",
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
        "accessible counter",
        "pot",
        "serving station",
        "upper corridor",
        "lower corridor",
        "access tile in front of the pot",
        "access tile in front of the serving station",
    }
)
ALLOWED_FAMILIES = frozenset(
    {"pickup", "pot", "wait", "dish_soup", "serve", "stash", "yield"}
)

# Text ontology is narrower than metadata: ``plate`` is excluded altogether so
# it cannot accidentally be used as a noun.  The valid object term is ``dish``.
FORBIDDEN_PATTERNS = (
    r"\bchop(?:ped|ping)?\b",
    r"\bcut(?:ting)?\b",
    r"\bknife\b",
    r"\bboard\b",
    r"\bsink\b",
    r"\bwash(?:ed|ing)?\b",
    r"\bdirty\b",
    r"\bclean(?:ed|ing)?\s+(?:dish|plate)\b",
    r"\bdish\s+(?:return|rack)\b",
    r"\b(?:fridge|refrigerator|oven|grill|stove|trash|bin)\b",
    r"\b(?:lettuce|meat|burger|rice|fish|cheese|bun)\b",
    r"\bprep\s+table\b",
    r"\bworktop\b",
    r"\bcenter\s+table\b",
    r"\b(?:tomato|onion)\s+crate\b",
    r"\b(?:central aisle|main aisle|center lane|middle path)\b",
    r"\border\s+expires?\b",
    r"\b(?:floor|ground)\b",
    r"\bstart\s+the\s+pot\b",
    r"\bfourth\s+ingredient\b",
    r"\bplate\b",
    r"\bdrop\s+there\b",
)
FORBIDDEN_RE = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS
)
FORBIDDEN_METADATA_TEXT_RE = re.compile(
    r"\b(?:bank\s*(?:one|two|three|four|five|0?\d)|final\s+set|"
    r"eval(?:uation)?\s+group|train(?:ing)?\s+split|calibration\s+split|"
    r"evaluative|imperative|descriptive|classifier)\b",
    re.IGNORECASE,
)
MIXED_EVALUATION_RE = re.compile(
    r"\b(?:good|bad|great|poor|helpful|unhelpful|excellent|ineffective|"
    r"effective|wrong|right(?!\s+(?:now|away))|mistake|sound|misguided|useful|harmful|"
    r"productive|counterproductive|strong|weak|valuable|wasteful|"
    r"successful|unsuccessful|well judged|badly judged|well timed|"
    r"poorly timed)\b",
    re.IGNORECASE,
)
RECENT_COMPLETION_RE = re.compile(
    r"\b(?:just|recent|last|previous|finished|completed|closed|ended|"
    r"latest|changed|complete|passed|close|moments? ago|a second ago|"
    r"just now|just made|just passed|just gone)\b",
    re.IGNORECASE,
)
REQUEST_RE = re.compile(
    r"\b(?:please|request|need(?:ed)?|asks?|asking|want|should|must|"
    r"priority|instruction|necessary|assignment|your action|"
    r"next action|next move|next job|next task|right away|at once|time to|"
    r"time for you|open task|open duty|now calls|calls (?:on you|for)|"
    r"awaiting you|requires you)\b",
    re.IGNORECASE,
)
CURRENT_FACT_RE = re.compile(
    r"\b(?:current kitchen|current condition|current layout|current order|"
    r"present kitchen|present fact|present order|present situation|present condition|"
    r"present layout|presently (?:observe|note|confirm)|currently (?:see|true)|"
    r"live status|status reading|observed condition|active order record|"
    r"order (?:record|status)|kitchen state|kitchen shows|kitchen situation|"
    r"kitchen (?:currently|right now|as it stands|has this fact)|"
    r"layout (?:shows|confirms|reading)|view of the layout now|"
    r"visible (?:state|layout|condition|fact)|fact (?:is|in view)|"
    r"situation at this moment|happening (?:at the moment|at this moment|right now|in the kitchen)|"
    r"case at this moment|thing visible now|what we see|what is visible|"
    r"round as it stands|fact in the kitchen|kitchen at this moment|"
    r"see (?:this fact|right now|at this moment)|true (?:right now|at this moment)|"
    r"visibly the case now)\b",
    re.IGNORECASE,
)
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def _article(item: str | None) -> str:
    if item is None:
        return ""
    if item == "onion":
        return "an onion"
    if item == "soup":
        return "soup"
    return f"a {item}"


@dataclass(frozen=True)
class TypedScenario:
    scenario_id: str
    family: str
    canonical_action: str
    item: str | None
    source: str | None
    target: str | None
    precondition: str
    result: str
    contents: str | None = None
    pot_status: str | None = None


@dataclass(frozen=True)
class CoreTriad:
    event: str
    act: str
    state: str


@dataclass(frozen=True)
class OuterTriad:
    evaluative: str
    imperative: str
    descriptive: str


@dataclass(frozen=True)
class BankRealizer:
    bank_id: str
    split_role: str
    positive_verdicts: tuple[str, ...]
    negative_verdicts: tuple[str, ...]
    outer_frames: tuple[OuterTriad, ...]


SCENARIOS: tuple[TypedScenario, ...] = (
    TypedScenario("get_tomato_dispenser", "pickup", "GET_TOMATO", "tomato", "tomato dispenser", None, "AI chef empty-handed", "AI chef holding tomato"),
    TypedScenario("get_onion_dispenser", "pickup", "GET_ONION", "onion", "onion dispenser", None, "AI chef empty-handed", "AI chef holding onion"),
    TypedScenario("get_dish_dispenser", "pickup", "GET_DISH", "dish", "dish dispenser", None, "AI chef empty-handed", "AI chef holding dish"),
    TypedScenario("get_tomato_counter", "pickup", "GET_TOMATO", "tomato", "accessible counter", None, "accessible counter holding tomato and AI chef empty-handed", "AI chef holding tomato"),
    TypedScenario("get_onion_counter", "pickup", "GET_ONION", "onion", "accessible counter", None, "accessible counter holding onion and AI chef empty-handed", "AI chef holding onion"),
    TypedScenario("get_dish_counter", "pickup", "GET_DISH", "dish", "accessible counter", None, "accessible counter holding dish and AI chef empty-handed", "AI chef holding dish"),
    TypedScenario("get_soup_counter", "pickup", "PICKUP_SOUP", "soup", "accessible counter", None, "accessible counter holding soup and AI chef empty-handed", "AI chef holding soup"),
    TypedScenario("put_tomato_empty_pot", "pot", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot empty", "pot filling with one tomato", "one tomato", "filling"),
    TypedScenario("put_second_tomato", "pot", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot containing one tomato", "pot filling with two tomatoes", "two tomatoes", "filling"),
    TypedScenario("put_onion_empty_pot", "pot", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot empty", "pot filling with one onion", "one onion", "filling"),
    TypedScenario("put_onion_after_tomato", "pot", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot containing one tomato", "pot filling with one tomato and one onion", "one tomato and one onion", "filling"),
    TypedScenario("complete_with_tomato", "pot", "PUT_TOMATO_IN_POT", "tomato", None, "pot", "AI chef holding tomato and pot containing one tomato and one onion", "pot cooking two tomatoes and one onion", "two tomatoes and one onion", "cooking"),
    TypedScenario("complete_with_onion", "pot", "PUT_ONION_IN_POT", "onion", None, "pot", "AI chef holding onion and pot containing two tomatoes", "pot cooking two tomatoes and one onion", "two tomatoes and one onion", "cooking"),
    TypedScenario("wait_while_cooking", "wait", "WAIT", None, None, "pot", "pot cooking two tomatoes and one onion", "AI chef waiting while pot cooks", "two tomatoes and one onion", "cooking"),
    TypedScenario("plate_ready_soup", "dish_soup", "PICKUP_SOUP", "soup", "pot", None, "AI chef holding dish and pot ready", "AI chef holding soup in dish"),
    TypedScenario("serve_held_soup", "serve", "SERVE_SOUP", "soup", None, "serving station", "AI chef holding soup", "soup delivered at serving station"),
    TypedScenario("stash_tomato", "stash", "STASH_HELD_OBJECT", "tomato", None, "accessible counter", "AI chef holding tomato and accessible counter empty", "accessible counter holding tomato"),
    TypedScenario("stash_onion", "stash", "STASH_HELD_OBJECT", "onion", None, "accessible counter", "AI chef holding onion and accessible counter empty", "accessible counter holding onion"),
    TypedScenario("stash_dish", "stash", "STASH_HELD_OBJECT", "dish", None, "accessible counter", "AI chef holding dish and accessible counter empty", "accessible counter holding dish"),
    TypedScenario("stash_soup", "stash", "STASH_HELD_OBJECT", "soup", None, "accessible counter", "AI chef holding soup and accessible counter empty", "accessible counter holding soup"),
    TypedScenario("yield_upper_corridor", "yield", "YIELD_PATH", None, "upper corridor", "upper corridor", "other chef approaching in upper corridor", "upper corridor clear"),
    TypedScenario("yield_lower_corridor", "yield", "YIELD_PATH", None, "lower corridor", "lower corridor", "other chef approaching in lower corridor", "lower corridor clear"),
    TypedScenario("clear_pot_access", "yield", "YIELD_PATH", None, "access tile in front of the pot", "access tile in front of the pot", "AI chef blocking access tile in front of the pot", "access tile in front of the pot clear"),
    TypedScenario("clear_serving_access", "yield", "YIELD_PATH", None, "access tile in front of the serving station", "access tile in front of the serving station", "AI chef blocking access tile in front of the serving station", "access tile in front of the serving station clear"),
)


def _format_core(pattern: CoreTriad, scenario: TypedScenario) -> CoreTriad:
    values = {
        "item": scenario.item or "item",
        "object": _article(scenario.item),
        "source": scenario.source or "source",
        "target": scenario.target or "target",
        "contents": scenario.contents or "contents",
        "status": scenario.pot_status or "state",
    }
    return CoreTriad(
        pattern.event.format(**values),
        pattern.act.format(**values),
        pattern.state.format(**values),
    )


def _six(*values: CoreTriad) -> tuple[CoreTriad, ...]:
    if len(values) != GROUNDINGS_PER_SCENARIO:
        raise ValueError("each bank/family needs exactly six complete core triads")
    return values


# Every string below is bank-owned.  There is intentionally no common command,
# tense conversion, or state renderer shared across lexical banks.
CORE_PATTERNS: dict[str, dict[str, tuple[CoreTriad, ...]]] = {
    "bank_01": {
        "pickup": _six(
            CoreTriad("you picked up {object} from the {source}", "pick up {object} from the {source}", "the AI chef now holds {object} sourced from the {source}"),
            CoreTriad("you collected {object} at the {source}", "collect {object} at the {source}", "one {item} from the {source} is currently with the AI chef"),
            CoreTriad("you fetched {object} using the {source}", "fetch {object} using the {source}", "the AI chef is carrying {object} supplied by the {source}"),
            CoreTriad("you retrieved {object} out of the {source}", "retrieve {object} out of the {source}", "the held object is {object} taken from the {source}"),
            CoreTriad("you took {object} away from the {source}", "take {object} away from the {source}", "the {source} has provided {object} to the AI chef"),
            CoreTriad("you visited the {source} and lifted {object}", "visit the {source} and lift {object}", "the AI chef has {object} after using the {source}"),
        ),
        "pot": _six(
            CoreTriad("you put the held {item} into the pot", "put the held {item} into the pot", "the pot currently contains {contents}"),
            CoreTriad("you added the carried {item} to the pot", "add the carried {item} to the pot", "{contents} are now inside the {status} pot"),
            CoreTriad("you placed your {item} within the pot", "place your {item} within the pot", "the visible pot contents are {contents}"),
            CoreTriad("you loaded the {item} from your hands into the pot", "load the {item} from your hands into the pot", "the {status} vessel presently holds {contents}"),
            CoreTriad("you fed the held {item} into the recipe pot", "feed the held {item} into the recipe pot", "the recipe in the pot has {contents} at present"),
            CoreTriad("you brought your {item} over and inserted it into the pot", "bring your {item} over and insert it into the pot", "the pot shows {contents} while it is {status}"),
        ),
        "wait": _six(
            CoreTriad("you waited while the pot cooked", "wait while the pot cooks", "the pot is currently cooking"),
            CoreTriad("you stayed still during the cooking cycle", "stay still during the cooking cycle", "cooking is under way in the pot"),
            CoreTriad("you remained in place as the soup heated", "remain in place as the soup heats", "the soup is presently in its cooking phase"),
            CoreTriad("you held position until the pot became ready", "hold position until the pot becomes ready", "the full pot has not finished cooking yet"),
            CoreTriad("you paused movement for the active cook time", "pause movement for the active cook time", "the pot continues its twenty-step cooking cycle"),
            CoreTriad("you stood by while the recipe finished", "stand by while the recipe finishes", "two tomatoes and one onion are cooking in the pot"),
        ),
        "dish_soup": _six(
            CoreTriad("you used the dish to lift ready soup from the pot", "use the dish to lift ready soup from the pot", "the AI chef currently carries soup in the dish"),
            CoreTriad("you collected the cooked soup with your dish", "collect the cooked soup with your dish", "ready soup now rests in the AI chef's dish"),
            CoreTriad("you picked up a serving from the ready pot using the dish", "pick up a serving from the ready pot using the dish", "the held dish contains a serving of soup"),
            CoreTriad("you filled the dish from the pot after cooking finished", "fill the dish from the pot after cooking finishes", "the AI chef has taken soup out of the ready pot"),
            CoreTriad("you drew the finished soup into the held dish", "draw the finished soup into the held dish", "a dish of soup is in the AI chef's hands"),
            CoreTriad("you moved one soup serving from the pot into the dish", "move one soup serving from the pot into the dish", "the ready pot has supplied soup to the held dish"),
        ),
        "serve": _six(
            CoreTriad("you served the held soup at the serving station", "serve the held soup at the serving station", "the serving station has received the soup"),
            CoreTriad("you delivered the soup to the serving station", "deliver the soup to the serving station", "the current order includes one completed soup delivery"),
            CoreTriad("you carried the finished soup into the serving station", "carry the finished soup into the serving station", "the soup is now registered at the serving station"),
            CoreTriad("you brought the held soup over for service", "bring the held soup over for service", "the serving station currently holds the finished order"),
            CoreTriad("you completed the handoff of soup at the serving station", "complete the handoff of soup at the serving station", "one soup handoff has been accepted for this order"),
            CoreTriad("you finished the order by placing soup at the serving station", "finish the order by placing soup at the serving station", "the delivered soup has added twenty points"),
        ),
        "stash": _six(
            CoreTriad("you put the held {item} on an accessible counter", "put the held {item} on an accessible counter", "an accessible counter currently supports {object}"),
            CoreTriad("you placed your {item} onto an empty accessible counter", "place your {item} onto an empty accessible counter", "{object} is resting on an accessible counter"),
            CoreTriad("you set the carried {item} on a free accessible counter", "set the carried {item} on a free accessible counter", "the AI chef is empty-handed beside a counter holding {object}"),
            CoreTriad("you staged the {item} at an open accessible counter", "stage the {item} at an open accessible counter", "one accessible counter has {object} available"),
            CoreTriad("you stored the held {item} on an empty accessible counter", "store the held {item} on an empty accessible counter", "the {item} occupies an accessible counter at present"),
            CoreTriad("you left the {item} sitting on an accessible counter", "leave the {item} sitting on an accessible counter", "the counter item is {object} and the AI chef holds nothing"),
        ),
        "yield": _six(
            CoreTriad("you moved aside from the {target}", "move aside from the {target}", "the {target} is currently open to the other chef"),
            CoreTriad("you yielded space along the {target}", "yield space along the {target}", "the teammate can now pass through the {target}"),
            CoreTriad("you cleared your position out of the {target}", "clear your position out of the {target}", "no chef is blocking the {target} at present"),
            CoreTriad("you stepped away and opened the {target}", "step away and open the {target}", "the route through the {target} is available"),
            CoreTriad("you made room for the teammate at the {target}", "make room for the teammate at the {target}", "the other chef has clear access through the {target}"),
            CoreTriad("you left the {target} free for movement", "leave the {target} free for movement", "the AI chef stands outside the {target}"),
        ),
    },
    "bank_02": {
        "pickup": _six(
            CoreTriad("you secured {object} through the {source}", "secure {object} through the {source}", "the {source} has supplied the AI chef with {object}"),
            CoreTriad("you obtained {object} by using the {source}", "obtain {object} by using the {source}", "{object} supplied at the {source} is in the chef's possession"),
            CoreTriad("you acquired {object} directly at the {source}", "acquire {object} directly at the {source}", "the chef possesses {object} originating at the {source}"),
            CoreTriad("you drew {object} out through the {source}", "draw {object} out through the {source}", "the {source} has released {object} into the chef's hands"),
            CoreTriad("you claimed {object} available from the {source}", "claim {object} available from the {source}", "{object} issued by the {source} is now being carried"),
            CoreTriad("you removed {object} into your hands at the {source}", "remove {object} into your hands at the {source}", "the chef's possession includes {object} from the {source}"),
        ),
        "pot": _six(
            CoreTriad("you inserted the held {item} into the pot", "insert the held {item} into the pot", "{contents} make up the pot's present contents"),
            CoreTriad("you tipped your {item} in with the pot mixture", "tip your {item} in with the pot mixture", "the {status} recipe presently consists of {contents}"),
            CoreTriad("you emptied the carried {item} into the pot", "empty the carried {item} into the pot", "inside the pot there are {contents} at this moment"),
            CoreTriad("you incorporated the {item} held by the chef into the pot", "incorporate the {item} held by the chef into the pot", "{contents} form the active pot mixture"),
            CoreTriad("you slotted the {item} into the recipe vessel", "slot the {item} into the recipe vessel", "the vessel is {status} with {contents} inside"),
            CoreTriad("you deposited your {item} within the pot mixture", "deposit your {item} within the pot mixture", "the present recipe count is {contents} in the pot"),
        ),
        "wait": _six(
            CoreTriad("you suspended movement while cooking continued", "suspend movement while cooking continues", "the active cooking interval is still running"),
            CoreTriad("you idled through part of the pot cycle", "idle through part of the pot cycle", "the recipe remains midway through its cook time"),
            CoreTriad("you deferred movement until the soup was ready", "defer movement until the soup is ready", "the finished state has not yet arrived for the soup"),
            CoreTriad("you kept still as the ingredients cooked", "keep still as the ingredients cook", "three ingredients are undergoing cooking in the pot"),
            CoreTriad("you stopped moving for the ongoing recipe cycle", "stop moving for the ongoing recipe cycle", "the cooking timer is presently advancing"),
            CoreTriad("you left your position unchanged during heating", "leave your position unchanged during heating", "the pot remains occupied with a cooking recipe"),
        ),
        "dish_soup": _six(
            CoreTriad("you portioned ready soup into the dish", "portion ready soup into the dish", "the dish presently carries one soup portion"),
            CoreTriad("you ladled finished soup from the pot using the dish", "ladle finished soup from the pot using the dish", "the chef possesses a dish filled from the ready pot"),
            CoreTriad("you loaded a serving of soup into your dish", "load a serving of soup into your dish", "one cooked serving is contained by the held dish"),
            CoreTriad("you drew soup from the ready vessel with a dish", "draw soup from the ready vessel with a dish", "the ready vessel has transferred soup to the chef's dish"),
            CoreTriad("you received the cooked recipe into the held dish", "receive the cooked recipe into the held dish", "the AI chef possesses the completed recipe in a dish"),
            CoreTriad("you transferred one portion out of the pot and into the dish", "transfer one portion out of the pot and into the dish", "the held dish now contains soup removed from the pot"),
        ),
        "serve": _six(
            CoreTriad("you submitted the soup at the serving station", "submit the soup at the serving station", "the station presently records a delivered soup"),
            CoreTriad("you completed the soup handoff through the serving station", "complete the soup handoff through the serving station", "one finished order has passed through the serving station"),
            CoreTriad("you presented the cooked soup for service", "present the cooked soup for service", "the completed soup is accepted at its delivery point"),
            CoreTriad("you sent the held order into the serving station", "send the held order into the serving station", "the delivery point currently contains the submitted order"),
            CoreTriad("you passed the soup across at the serving station", "pass the soup across at the serving station", "the soup handover is registered as complete"),
            CoreTriad("you closed the order with a soup delivery", "close the order with a soup delivery", "twenty points are attached to the accepted soup"),
        ),
        "stash": _six(
            CoreTriad("you deposited the carried {item} on an accessible counter", "deposit the carried {item} on an accessible counter", "{object} presently occupies an accessible counter"),
            CoreTriad("you parked your {item} at an empty accessible counter", "park your {item} at an empty accessible counter", "the chef has free hands while {object} rests on the counter"),
            CoreTriad("you rested the held {item} upon an accessible counter", "rest the held {item} upon an accessible counter", "an accessible surface is supporting {object} now"),
            CoreTriad("you reserved an open counter position with your {item}", "reserve an open counter position with your {item}", "{object} marks one occupied accessible counter"),
            CoreTriad("you laid the carried {item} down on an accessible counter", "lay the carried {item} down on an accessible counter", "the counter rather than the chef is holding {object}"),
            CoreTriad("you kept the {item} at a vacant accessible counter", "keep the {item} at a vacant accessible counter", "a counter position contains {object} at this time"),
        ),
        "yield": _six(
            CoreTriad("you vacated the {target} for the other chef", "vacate the {target} for the other chef", "the other chef has unobstructed use of the {target}"),
            CoreTriad("you unblocked travel through the {target}", "unblock travel through the {target}", "movement through the {target} is presently unobstructed"),
            CoreTriad("you freed the passage at the {target}", "free the passage at the {target}", "the teammate's route crosses the {target} without a block"),
            CoreTriad("you exited the lane around the {target}", "exit the lane around the {target}", "the AI chef no longer occupies the {target}"),
            CoreTriad("you released access to the {target}", "release access to the {target}", "access across the {target} belongs to the approaching chef"),
            CoreTriad("you opened a route over the {target}", "open a route over the {target}", "the teammate currently faces an open {target}"),
        ),
    },
    "bank_03": {
        "pickup": _six(
            CoreTriad("you grabbed {object} beside the {source}", "grab {object} beside the {source}", "{object} from the {source} is with the chef at this moment"),
            CoreTriad("you snagged {object} on your visit to the {source}", "snag {object} on your visit to the {source}", "the chef came away from the {source} carrying {object}"),
            CoreTriad("you came away from the {source} with {object}", "come away from the {source} with {object}", "the AI chef has {object} that came through the {source}"),
            CoreTriad("you got hold of {object} over at the {source}", "get hold of {object} over at the {source}", "{object} obtained near the {source} is being carried"),
            CoreTriad("you pulled {object} free at the {source}", "pull {object} free at the {source}", "the chef's hands contain {object} supplied there"),
            CoreTriad("you scooped up {object} by the {source}", "scoop up {object} by the {source}", "the item with the AI chef is {object} from the {source}"),
        ),
        "pot": _six(
            CoreTriad("you popped the held {item} into the pot", "pop the held {item} into the pot", "the pot has reached {contents} in its present mix"),
            CoreTriad("you tossed your {item} in with the recipe", "toss your {item} in with the recipe", "{contents} sit together inside the {status} pot"),
            CoreTriad("you mixed the carried {item} into the pot", "mix the carried {item} into the pot", "the pot mixture now amounts to {contents}"),
            CoreTriad("you sent the {item} from your hands into the pot", "send the {item} from your hands into the pot", "inside the vessel, {contents} are presently {status}"),
            CoreTriad("you worked the {item} into the soup mixture", "work the {item} into the soup mixture", "the active soup mixture contains {contents}"),
            CoreTriad("you tucked your {item} inside the pot", "tuck your {item} inside the pot", "{contents} can be seen within the pot now"),
        ),
        "wait": _six(
            CoreTriad("you took a beat while the pot kept cooking", "take a beat while the pot keeps cooking", "the recipe is still going through its cook cycle"),
            CoreTriad("you hung on without moving during the heat cycle", "hang on without moving during the heat cycle", "the heated mixture is not ready at this moment"),
            CoreTriad("you stayed put while the soup finished up", "stay put while the soup finishes up", "the soup presently needs more cooking time"),
            CoreTriad("you gave the recipe a moment before moving", "give the recipe a moment before moving", "the pot is partway through cooking its contents"),
            CoreTriad("you held up movement as the pot ran", "hold up movement as the pot runs", "a full recipe is actively cooking nearby"),
            CoreTriad("you let the soup cook while keeping your tile", "let the soup cook while keeping your tile", "the cooking vessel has three ingredients underway"),
        ),
        "dish_soup": _six(
            CoreTriad("you dished up ready soup using the held dish", "dish up ready soup using the held dish", "the chef has a dish with cooked soup in it"),
            CoreTriad("you scooped the finished soup into your dish", "scoop the finished soup into your dish", "cooked soup sits in the dish being carried"),
            CoreTriad("you got the meal from the pot into the dish", "get the meal from the pot into the dish", "the dish in hand now has the completed meal"),
            CoreTriad("you pulled a soup serving out with the dish", "pull a soup serving out with the dish", "one serving from the pot is with the chef"),
            CoreTriad("you took the cooked serving in your dish", "take the cooked serving in your dish", "the AI chef's dish is holding ready soup"),
            CoreTriad("you brought the ready soup onto the dish", "bring the ready soup onto the dish", "the pot's finished contents are now in a dish"),
        ),
        "serve": _six(
            CoreTriad("you handed the soup in at the serving station", "hand the soup in at the serving station", "the station shows a soup turned in for the order"),
            CoreTriad("you ran the cooked order over for service", "run the cooked order over for service", "the completed order is with the serving station now"),
            CoreTriad("you turned in the soup you were carrying", "turn in the soup you are carrying", "the carried soup has become a recorded delivery"),
            CoreTriad("you left the finished soup at the serving station", "leave the finished soup at the serving station", "the delivery location now has the finished soup"),
            CoreTriad("you took the soup across to its station", "take the soup across to its station", "service has accepted one soup from the AI chef"),
            CoreTriad("you gave the order over at the serving station", "give the order over at the serving station", "the score reflects a completed soup order"),
        ),
        "stash": _six(
            CoreTriad("you set the {item} down on an accessible counter", "set the {item} down on an accessible counter", "{object} is sitting at an accessible counter now"),
            CoreTriad("you left the held {item} at a free counter position", "leave the held {item} at a free counter position", "the AI chef has open hands after leaving {object} nearby"),
            CoreTriad("you cleared your hands by using an accessible counter for the {item}", "clear your hands by using an accessible counter for the {item}", "an accessible counter is carrying {object} for the team"),
            CoreTriad("you put the {item} aside on an empty accessible counter", "put the {item} aside on an empty accessible counter", "the counter space contains {object} rather than being empty"),
            CoreTriad("you tucked the carried {item} away at an accessible counter", "tuck the carried {item} away at an accessible counter", "{object} remains available on a reachable counter"),
            CoreTriad("you left your {item} sitting at an open counter spot", "leave your {item} sitting at an open counter spot", "the reachable counter has {object} while the chef holds nothing"),
        ),
        "yield": _six(
            CoreTriad("you got out of the way at the {target}", "get out of the way at the {target}", "the teammate can get by at the {target} at this moment"),
            CoreTriad("you let the other chef through the {target}", "let the other chef through the {target}", "the AI chef is no longer in the teammate's way there"),
            CoreTriad("you gave the teammate room near the {target}", "give the teammate room near the {target}", "enough space is available for passage across the {target}"),
            CoreTriad("you backed off from the {target}", "back off from the {target}", "the approaching chef faces no collision at the {target}"),
            CoreTriad("you stepped clear of the {target}", "step clear of the {target}", "the route at the {target} has room for the teammate"),
            CoreTriad("you stopped blocking movement around the {target}", "stop blocking movement around the {target}", "both chefs can use the area without a blocked path"),
        ),
    },
    "bank_04": {
        "pickup": _six(
            CoreTriad("the pickup transferred {object} out of the {source} and into your hands", "transfer {object} out of the {source} and into your hands", "a transfer from the {source} leaves the AI chef holding {object}"),
            CoreTriad("you loaded {object} into hand from the {source}", "load {object} into hand from the {source}", "the chef's present load is {object} obtained at the {source}"),
            CoreTriad("you sourced {object} through interaction with the {source}", "source {object} through interaction with the {source}", "an interaction at the {source} has put {object} with the chef"),
            CoreTriad("you received {object} from the {source} into your grasp", "receive {object} from the {source} into your grasp", "the AI chef's grasp now contains {object} from the {source}"),
            CoreTriad("you extracted {object} by operating the {source}", "extract {object} by operating the {source}", "operating the {source} has supplied {object} to the chef"),
            CoreTriad("you moved {object} out from the {source} and into hand", "move {object} out from the {source} and into hand", "{object} has moved from the {source} to the AI chef"),
        ),
        "pot": _six(
            CoreTriad("you transferred the held {item} into the pot", "transfer the held {item} into the pot", "the pot's current inventory comprises {contents}"),
            CoreTriad("you introduced your {item} to the recipe vessel", "introduce your {item} to the recipe vessel", "{contents} constitute the present recipe inventory"),
            CoreTriad("you routed the carried {item} from hand to pot", "route the carried {item} from hand to pot", "the vessel's observed load consists of {contents}"),
            CoreTriad("you shifted the {item} across into the pot", "shift the {item} across into the pot", "the {status} process now includes {contents}"),
            CoreTriad("you merged the held {item} with the pot contents", "merge the held {item} with the pot contents", "the combined ingredients currently total {contents}"),
            CoreTriad("you moved your {item} from the chef's hand to the pot", "move your {item} from the chef's hand to the pot", "the pot inventory reads {contents} during {status}"),
        ),
        "wait": _six(
            CoreTriad("you maintained your tile during the cooking interval", "maintain your tile during the cooking interval", "the cooking interval currently remains active"),
            CoreTriad("you preserved position throughout the ongoing cycle", "preserve position throughout the ongoing cycle", "the recipe cycle has not reached ready status"),
            CoreTriad("you allowed the pot cycle to advance without moving", "allow the pot cycle to advance without moving", "the pot timer is advancing toward completion"),
            CoreTriad("you avoided displacement while the recipe processed", "avoid displacement while the recipe processes", "the recipe is presently processing in the pot"),
            CoreTriad("you occupied the same square during active cooking", "occupy the same square during active cooking", "the three-ingredient mixture is still being cooked"),
            CoreTriad("you retained your location until the cook cycle ended", "retain your location until the cook cycle ends", "twenty environment steps govern the active cooking period"),
        ),
        "dish_soup": _six(
            CoreTriad("you decanted the ready soup from the pot into the dish", "decant the ready soup from the pot into the dish", "the dish now contains soup decanted from the pot"),
            CoreTriad("you moved a serving from the vessel into tableware", "move a serving from the vessel into the dish", "the chef's dish presently contains one ready serving"),
            CoreTriad("you routed cooked soup out of the pot with the dish", "route cooked soup out of the pot with the dish", "the ready soup has moved into the held dish"),
            CoreTriad("you shifted the ready contents from pot to dish", "shift the ready contents from pot to dish", "the AI chef currently possesses soup in the dish"),
            CoreTriad("you transferred a finished portion into the held dish", "transfer a finished portion into the held dish", "one completed portion occupies the chef's dish"),
            CoreTriad("you received soup in the dish from the ready pot", "receive soup in the dish from the ready pot", "the pot has released its ready soup into the dish"),
        ),
        "serve": _six(
            CoreTriad("you transferred the soup to the serving station", "transfer the soup to the serving station", "the serving station's current intake is one soup"),
            CoreTriad("you released the completed order at its serving point", "release the completed order at its serving point", "the serving point has accepted the completed order"),
            CoreTriad("you conveyed the held soup into the station", "convey the held soup into the station", "a soup delivery is presently recorded by the station"),
            CoreTriad("you routed the finished soup to its delivery target", "route the finished soup to its delivery target", "the target now registers the finished soup"),
            CoreTriad("you moved the completed serving from hand to station", "move the completed serving from hand to station", "the AI chef is empty-handed after the accepted serving"),
            CoreTriad("you completed the exchange at the serving station", "complete the exchange at the serving station", "the exchange has produced a twenty-point delivery"),
        ),
        "stash": _six(
            CoreTriad("you relocated the held {item} onto an accessible counter", "relocate the held {item} onto an accessible counter", "the accessible counter presently contains {object}"),
            CoreTriad("you positioned your {item} at a vacant counter space", "position your {item} at a vacant counter space", "{object} occupies one reachable counter position"),
            CoreTriad("you shifted the carried {item} onto an empty accessible counter", "shift the carried {item} onto an empty accessible counter", "the chef's hands are free while the counter bears {object}"),
            CoreTriad("you assigned an accessible surface to the {item}", "assign an accessible surface to the {item}", "an accessible surface is presently assigned to {object}"),
            CoreTriad("you moved the {item} from hand to an open counter", "move the {item} from hand to an open counter", "the transfer leaves {object} on a reachable counter"),
            CoreTriad("you transferred your {item} onto an accessible counter", "transfer your {item} onto an accessible counter", "the counter inventory now includes {object}"),
        ),
        "yield": _six(
            CoreTriad("you repositioned outside the {target}", "reposition outside the {target}", "the {target} presently lies outside the AI chef's occupied tile"),
            CoreTriad("you withdrew from the route at the {target}", "withdraw from the route at the {target}", "the teammate has an available route through the {target}"),
            CoreTriad("you relocated beyond the {target}", "relocate beyond the {target}", "the AI chef and the {target} no longer share a blocking position"),
            CoreTriad("you removed your position from the {target}", "remove your position from the {target}", "the approaching chef can traverse the {target}"),
            CoreTriad("you shifted off the tile at the {target}", "shift off the tile at the {target}", "the target tile currently allows passage"),
            CoreTriad("you stood beyond the {target} after moving", "stand beyond the {target} after moving", "the route remains available to the other chef"),
        ),
    },
    "bank_05": {
        "pickup": _six(
            CoreTriad("you went and got {object} from the {source}", "go get {object} from the {source}", "the chef has {object} from its {source} in hand at the moment"),
            CoreTriad("you brought {object} back from the {source}", "bring {object} back from the {source}", "{object} came back with the chef from the {source}"),
            CoreTriad("you headed over for {object} at the {source}", "head over for {object} at the {source}", "the AI chef is now carrying what the {source} supplied"),
            CoreTriad("you returned from the {source} carrying {object}", "return from the {source} carrying {object}", "the chef currently has {object} after visiting the {source}"),
            CoreTriad("you got {object} into your hand through the {source}", "get {object} into your hand through the {source}", "the held item at this moment is {object} from the {source}"),
            CoreTriad("you took one {item} off the {source}", "take one {item} off the {source}", "one {item} taken there is presently with the AI chef"),
        ),
        "pot": _six(
            CoreTriad("you got the held {item} into the pot", "get the held {item} into the pot", "at this moment the pot carries {contents}"),
            CoreTriad("you stuck your {item} in with the ingredients", "stick your {item} in with the ingredients", "the mixture at this moment has {contents}"),
            CoreTriad("you set the carried {item} inside the pot", "set the carried {item} inside the pot", "{contents} are sitting in the {status} pot now"),
            CoreTriad("you carried the {item} over and put it in", "carry the {item} over and put it in the pot", "the pot currently shows an ingredient count of {contents}"),
            CoreTriad("you worked your {item} into the pot recipe", "work your {item} into the pot recipe", "the recipe stands at {contents} during its {status} phase"),
            CoreTriad("you put the {item} down inside the pot", "put the {item} down inside the pot", "at the moment {contents} make up the pot recipe"),
        ),
        "wait": _six(
            CoreTriad("you waited here while the soup cooked", "wait here while the soup cooks", "the soup is still cooking at this moment"),
            CoreTriad("you kept where you were during cooking", "keep where you are during cooking", "at this moment the pot needs more time"),
            CoreTriad("you stood around as the recipe heated", "stand around as the recipe heats", "the current recipe has not become ready yet"),
            CoreTriad("you stuck there until the cooking moved on", "stick there until the cooking moves on", "the three ingredients are heating together now"),
            CoreTriad("you held off moving while the pot ran", "hold off moving while the pot runs", "the active pot is partway through twenty steps"),
            CoreTriad("you let the soup cook before taking another move", "let the soup cook before taking another move", "two tomatoes and one onion remain in a cooking pot"),
        ),
        "dish_soup": _six(
            CoreTriad("you got the soup with the dish at the pot", "get the soup with the dish at the pot", "the chef has the ready soup held in a dish at the moment"),
            CoreTriad("you filled the dish from the ready soup", "fill the dish from the ready soup", "a filled dish is currently with the AI chef"),
            CoreTriad("you took the soup out using your dish", "take the soup out using your dish", "the soup has left the pot and now sits in the dish"),
            CoreTriad("you scooped cooked soup into the held dish", "scoop cooked soup into the held dish", "the dish presently holds what finished cooking"),
            CoreTriad("you lifted the ready serving into your dish", "lift the ready serving into your dish", "one ready serving is with the chef in a dish"),
            CoreTriad("you got a serving out of the pot with the dish", "get a serving out of the pot with the dish", "at this point the chef carries soup instead of an empty dish"),
        ),
        "serve": _six(
            CoreTriad("you served up the soup at the station", "serve up the soup at the serving station", "the serving station has the soup at this moment"),
            CoreTriad("you got the finished soup over to service", "get the finished soup over to service", "service currently has the completed soup"),
            CoreTriad("you handed the order over at the serving station", "hand the order over at the serving station", "the serving station shows that the order arrived"),
            CoreTriad("you left the soup at its station for delivery", "leave the soup at its station for delivery", "the delivered soup is presently at the station"),
            CoreTriad("you took the order in to the serving station", "take the order in to the serving station", "the latest order handoff is complete"),
            CoreTriad("you finished delivery with the held soup", "finish delivery with the held soup", "the current score includes twenty points from that soup"),
        ),
        "stash": _six(
            CoreTriad("you got the held {item} out of hand and onto an accessible counter", "get the held {item} out of hand and onto an accessible counter", "the {item} is off the chef's hands and on a counter now"),
            CoreTriad("you used a free accessible counter for your {item}", "use a free accessible counter for your {item}", "{object} is presently waiting on the accessible counter"),
            CoreTriad("you left the {item} on a clear counter space", "leave the {item} on a clear counter space", "the chef currently has empty hands beside {object}"),
            CoreTriad("you set the carried {item} somewhere open on an accessible counter", "set the carried {item} somewhere open on an accessible counter", "an open counter became occupied by {object}"),
            CoreTriad("you rested your {item} at an empty reachable counter", "rest your {item} at an empty reachable counter", "the reachable counter now keeps {object} for later"),
            CoreTriad("you put the {item} down at an accessible counter", "put the {item} down at an accessible counter", "at this moment {object} sits on the counter"),
        ),
        "yield": _six(
            CoreTriad("you moved over from the {target}", "move over from the {target}", "the other chef can use the {target} now"),
            CoreTriad("you got clear of the {target}", "get clear of the {target}", "the teammate currently finds the {target} clear"),
            CoreTriad("you let the teammate by through the {target}", "let the teammate by through the {target}", "the AI chef is standing away from the passing route"),
            CoreTriad("you got off the spot at the {target}", "get off the spot at the {target}", "that spot is open for the approaching chef at present"),
            CoreTriad("you scooted away from the {target}", "scoot away from the {target}", "there is room for both chefs around the {target}"),
            CoreTriad("you opened things up around the {target}", "open things up around the {target}", "movement beside the {target} is currently free"),
        ),
    },
}


def _outer(*values: OuterTriad) -> tuple[OuterTriad, ...]:
    if len(values) != 12:
        raise ValueError("each bank needs exactly twelve E/I/D outer-frame triads")
    return values


FIRST_TOKENS = (
    "The", "That", "This", "Your", "We", "I",
    "It", "There", "now", "for", "in", "At",
)
NO_TERMINAL_SLOTS = frozenset({8, 10})


BANKS: tuple[BankRealizer, ...] = (
    BankRealizer(
        "bank_01",
        "train",
        ("good", "helpful", "effective", "sound", "useful", "well judged", "right", "productive", "strong", "well timed", "valuable", "successful"),
        ("poor", "unhelpful", "ineffective", "misguided", "harmful", "badly judged", "wrong", "counterproductive", "weak", "poorly timed", "wasteful", "unsuccessful"),
        _outer(
            OuterTriad("The recent move earns a {verdict} assessment because {event}.", "The action needed from you now is to {act}.", "The current kitchen view records that {state}."),
            OuterTriad("That step you just finished was {verdict}: {event}.", "That request for this moment is to {act}.", "That present kitchen fact is that {state}."),
            OuterTriad("This completed action looks {verdict} since {event}.", "This immediate task asks you to {act}.", "This current condition shows that {state}."),
            OuterTriad("Your last move was {verdict} when {event}.", "Your next move should be to {act}.", "Your current kitchen state has this fact: {state}."),
            OuterTriad("We rate the action just taken as {verdict}: {event}.", "We need you to {act} on this turn.", "We can currently see that {state}."),
            OuterTriad("I can say right now that the recent choice was {verdict} because {event}.", "I want you to {act} right now.", "I can report this present fact right now: {state}."),
            OuterTriad("It was {verdict} on the move just completed because {event}.", "It is time for you to {act}.", "It is currently true that {state}."),
            OuterTriad("There was a {verdict} outcome on the last action: {event}.", "There is one task awaiting you now: {act}.", "There is a current condition: {state}."),
            OuterTriad("now that the move has ended, it seems {verdict}: {event}", "now the request is for you to {act}", "now the kitchen shows that {state}"),
            OuterTriad("for the action just finished, the verdict is {verdict}: {event}.", "for this next action, please {act}.", "for the current kitchen state, the fact is that {state}."),
            OuterTriad("in reviewing the previous move, the result is {verdict}: {event}", "in this moment, the instruction is to {act}", "in the present situation, it is true that {state}"),
            OuterTriad("At the close of the last step, {event}, which was {verdict}.", "At this point, I ask you to {act}.", "At present, the kitchen state indicates that {state}."),
        ),
    ),
    BankRealizer(
        "bank_02",
        "train",
        ("good", "helpful", "effective", "sound", "useful", "well judged", "right", "productive", "strong", "well timed", "valuable", "successful"),
        ("poor", "unhelpful", "ineffective", "misguided", "harmful", "badly judged", "wrong", "counterproductive", "weak", "poorly timed", "wasteful", "unsuccessful"),
        _outer(
            OuterTriad("The turn now closed receives a {verdict} assessment because {event}.", "The open assignment requires you to {act}.", "The active order record indicates that {state}."),
            OuterTriad("That finished contribution merits a {verdict} mark: {event}.", "That pending duty calls for you to {act}.", "That live status entry confirms that {state}."),
            OuterTriad("This task you completed is judged {verdict}, given that {event}.", "This available instruction needs you to {act}.", "This observed condition establishes that {state}."),
            OuterTriad("Your previous turn rates as {verdict} because {event}.", "Your immediate assignment is to {act}.", "Your present order record contains this fact: {state}."),
            OuterTriad("We assess the completed contribution as {verdict}: {event}.", "We require you to {act} for the open task.", "We presently observe that {state}."),
            OuterTriad("I can rate the closed action right now as {verdict} because {event}.", "I need you to {act} right now for the active order.", "I can note this present condition right now: {state}."),
            OuterTriad("It receives a {verdict} review after the task ended because {event}.", "It is your priority to {act} next.", "It remains a present fact that {state}."),
            OuterTriad("There is a {verdict} judgment for the contribution just completed: {event}.", "There is an open instruction for you to {act}.", "There is a live status reading: {state}."),
            OuterTriad("now that the turn is closed, the assessment is {verdict}: {event}", "now the active assignment calls on you to {act}", "now the order record states that {state}"),
            OuterTriad("for the task recently completed, the rating is {verdict}: {event}.", "for the open duty, you must {act}.", "for the present order, the record shows that {state}."),
            OuterTriad("in the review of the finished turn, the result is {verdict} because {event}", "in the active assignment, you need to {act}", "in the live status record, it follows that {state}"),
            OuterTriad("At the end of the closed contribution, {event}, earning a {verdict} mark.", "At this stage, the task requires you to {act}.", "At this stage, the order status confirms that {state}."),
        ),
    ),
    BankRealizer(
        "bank_03",
        "train",
        ("good", "helpful", "effective", "sound", "useful", "well judged", "right", "productive", "strong", "well timed", "valuable", "successful"),
        ("poor", "unhelpful", "ineffective", "misguided", "harmful", "badly judged", "wrong", "counterproductive", "weak", "poorly timed", "wasteful", "unsuccessful"),
        _outer(
            OuterTriad("The thing you did a second ago felt {verdict}: {event}.", "The thing I need from you now is to {act}.", "The thing happening at this moment is that {state}."),
            OuterTriad("That bit of play just then looked {verdict} because {event}.", "That move I am asking for next is to {act}.", "That situation at this moment shows that {state}."),
            OuterTriad("This last bit of teamwork came off {verdict}: {event}.", "This request is simple at this moment: {act}.", "This is what the kitchen shows at this moment: {state}."),
            OuterTriad("Your move from a moment ago seems {verdict} since {event}.", "Your next action needs to be to {act}.", "Your kitchen situation currently includes this: {state}."),
            OuterTriad("We saw a {verdict} play just now when {event}.", "We want you to {act} on the next beat.", "We can see this happening at the moment: {state}."),
            OuterTriad("I can say right now that the thing just seen was {verdict} because {event}.", "I am asking you to {act} right now.", "I can see this fact right now: {state}."),
            OuterTriad("It looked {verdict} in the moment that just passed because {event}.", "It is your next move to {act}.", "It is the case at this moment that {state}."),
            OuterTriad("There was something {verdict} in the last bit of play: {event}.", "There is one thing I want now: {act}.", "There is one thing visible now: {state}."),
            OuterTriad("now the previous beat is over, it looks {verdict}: {event}", "now I am asking for this move: {act}", "now this is happening in the kitchen: {state}"),
            OuterTriad("for what happened just then, the result feels {verdict} because {event}.", "for the next beat, I want you to {act}.", "for what is visible at this moment, {state}."),
            OuterTriad("in looking back at the thing just done, the outcome seems {verdict} because {event}", "in the next moment, please {act}", "in the kitchen as it stands, {state}"),
            OuterTriad("At the moment after that play ended, it felt {verdict} because {event}.", "At this moment, I need you to {act}.", "At this moment, what we see is that {state}."),
        ),
    ),
    BankRealizer(
        "bank_04",
        "calibration",
        ("good", "helpful", "effective", "sound", "useful", "well judged", "right", "productive", "strong", "well timed", "valuable", "successful"),
        ("poor", "unhelpful", "ineffective", "misguided", "harmful", "badly judged", "wrong", "counterproductive", "weak", "poorly timed", "wasteful", "unsuccessful"),
        _outer(
            OuterTriad("The position change just made was {verdict}, as {event}.", "The position now calls on you to {act}.", "The visible layout currently confirms that {state}."),
            OuterTriad("That layout outcome from the last move was {verdict}: {event}.", "That immediate priority directs you to {act}.", "That layout reading at present establishes that {state}."),
            OuterTriad("This latest layout shift appears {verdict} because {event}.", "This current position requires you to {act}.", "This visible state presently indicates that {state}."),
            OuterTriad("Your recently completed position change rates {verdict}: {event}.", "Your action for the visible position is to {act}.", "Your view of the layout now includes this fact: {state}."),
            OuterTriad("We judge the layout change just completed as {verdict} because {event}.", "We ask you to {act} from the current position.", "We can presently confirm from the layout that {state}."),
            OuterTriad("I read the last position outcome right now as {verdict}, since {event}.", "I need you to {act} right now in the visible position.", "I can confirm this visible condition right now: {state}."),
            OuterTriad("It was a {verdict} result after the layout changed because {event}.", "It is necessary now for you to {act}.", "It is visibly the case now that {state}."),
            OuterTriad("There was a {verdict} result in the position just left: {event}.", "There is a position-based request to {act}.", "There is a present layout fact: {state}."),
            OuterTriad("now the layout shift is complete, the outcome is {verdict}: {event}", "now the visible position calls for you to {act}", "now the layout confirms that {state}"),
            OuterTriad("for the position change just made, the result was {verdict} because {event}.", "for this visible position, you should {act}.", "for the current layout, the condition is that {state}."),
            OuterTriad("in reviewing the latest layout outcome, it appears {verdict} because {event}", "in response to the present position, please {act}", "in the visible layout at present, it follows that {state}"),
            OuterTriad("At the close of that position change, the result was {verdict} because {event}.", "At the current position, I ask you to {act}.", "At the current position, the layout shows that {state}."),
        ),
    ),
    BankRealizer(
        "bank_05",
        "final_eval",
        ("good", "helpful", "effective", "sound", "useful", "well judged", "right", "productive", "strong", "well timed", "valuable", "successful"),
        ("poor", "unhelpful", "ineffective", "misguided", "harmful", "badly judged", "wrong", "counterproductive", "weak", "poorly timed", "wasteful", "unsuccessful"),
        _outer(
            OuterTriad("The play that just landed was {verdict}: {event}.", "The play I want right away is to {act}.", "The round as it stands has this fact: {state}."),
            OuterTriad("That previous beat was {verdict} because {event}.", "That next move should be to {act}.", "That fact in the kitchen at this moment is that {state}."),
            OuterTriad("This turn just gone looks {verdict}: {event}.", "This move is needed now: {act}.", "This is the present kitchen picture: {state}."),
            OuterTriad("Your last play came out {verdict} when {event}.", "Your next job is to {act}.", "Your kitchen currently has this situation: {state}."),
            OuterTriad("We call the move just passed {verdict}: {event}.", "We want you to {act} next.", "We can see at this moment that {state}."),
            OuterTriad("I think right now that the last action was {verdict} because {event}.", "I need you to {act} right now.", "I can state the present fact right now: {state}."),
            OuterTriad("It was {verdict} on that previous move since {event}.", "It is time to {act} next.", "It is true at this moment that {state}."),
            OuterTriad("There was a {verdict} result on the turn just gone: {event}.", "There is a next action waiting: {act}.", "There is a fact in view: {state}."),
            OuterTriad("now that play has passed, it was {verdict}: {event}", "now the move I want is to {act}", "now the kitchen has this fact: {state}"),
            OuterTriad("for the move that just ended, the result was {verdict} because {event}.", "for the move ahead, please {act}.", "for the kitchen at this moment, {state}."),
            OuterTriad("in the turn just gone, the outcome came out {verdict} because {event}", "in the next turn, I need you to {act}", "in the kitchen at this moment, {state}"),
            OuterTriad("At the end of the previous beat, it looked {verdict} because {event}.", "At this point, please {act}.", "At this point, the visible fact is that {state}."),
        ),
    ),
)


class GenerationAuditError(ValueError):
    def __init__(self, failure_codes: list[str] | tuple[str, ...], report: dict):
        self.failure_codes = tuple(sorted(set(failure_codes)))
        self.report = report
        super().__init__("clean-v6 generation audit failed: " + ", ".join(self.failure_codes))


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_model_input(value: str) -> str:
    """Return the sole allowed feature input and reject metadata dictionaries."""
    if not isinstance(value, str):
        raise TypeError("clean-v6 model input must be a raw text string")
    return value


def _scenario_map() -> dict[str, TypedScenario]:
    return {scenario.scenario_id: scenario for scenario in SCENARIOS}


def validate_scenarios() -> None:
    if len(SCENARIOS) != 24 or len(_scenario_map()) != 24:
        raise ValueError("clean-v6 requires 24 unique typed scenarios")
    for scenario in SCENARIOS:
        if type(scenario) is not TypedScenario:
            raise TypeError("scenario must be an exact TypedScenario instance")
        if scenario.family not in ALLOWED_FAMILIES:
            raise ValueError(f"unsupported scenario family: {scenario.family}")
        if scenario.canonical_action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {scenario.canonical_action}")
        if scenario.item not in ALLOWED_ITEMS:
            raise ValueError(f"unsupported item: {scenario.item}")
        if scenario.source not in ALLOWED_LOCATIONS or scenario.target not in ALLOWED_LOCATIONS:
            raise ValueError(f"unsupported location in {scenario.scenario_id}")
        required = (scenario.scenario_id, scenario.precondition, scenario.result)
        if any(not str(value).strip() for value in required):
            raise ValueError(f"missing typed field in {scenario.scenario_id}")
    if {scenario.canonical_action for scenario in SCENARIOS} != ALLOWED_ACTIONS:
        raise ValueError("typed scenarios must cover all ten canonical actions")
    bank_ids = [bank.bank_id for bank in BANKS]
    if len(bank_ids) != 5 or len(set(bank_ids)) != 5:
        raise ValueError("clean-v6 requires five unique lexical banks")
    if Counter(bank.split_role for bank in BANKS) != Counter(
        {"train": 3, "calibration": 1, "final_eval": 1}
    ):
        raise ValueError("bank roles must be three train, one calibration, one final_eval")
    for bank in BANKS:
        if set(CORE_PATTERNS.get(bank.bank_id, {})) != ALLOWED_FAMILIES:
            raise ValueError(f"incomplete core family map for {bank.bank_id}")
        if len(bank.outer_frames) != 12:
            raise ValueError(f"bank {bank.bank_id} must own twelve outer frame triads")
        if len(bank.positive_verdicts) != 12 or len(bank.negative_verdicts) != 12:
            raise ValueError(f"bank {bank.bank_id} must own twelve verdicts per polarity")
        for family, patterns in CORE_PATTERNS[bank.bank_id].items():
            if len(patterns) != 6 or any(type(value) is not CoreTriad for value in patterns):
                raise ValueError(f"invalid core bank/family cell: {bank.bank_id}/{family}")
        for slot, triad in enumerate(bank.outer_frames):
            expected_first = FIRST_TOKENS[slot]
            for label, template in zip(LABELS, asdict(triad).values()):
                if template.split(maxsplit=1)[0] != expected_first:
                    raise ValueError(
                        f"first-token Latin slot mismatch: {bank.bank_id}/{slot}/{label}"
                    )
                has_terminal = template[-1:] in ".!?"
                if has_terminal != (slot not in NO_TERMINAL_SLOTS):
                    raise ValueError(
                        f"terminal punctuation quota mismatch: {bank.bank_id}/{slot}/{label}"
                    )


def _semantic_payload(scenario: TypedScenario, variant_index: int) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "actor": "AI chef",
        "action": scenario.canonical_action,
        "item": scenario.item,
        "source": scenario.source,
        "target": scenario.target,
        "precondition": scenario.precondition,
        "result": scenario.result,
        "contents": scenario.contents,
        "pot_status": scenario.pot_status,
        "state_scope": "current kitchen state",
        "grounding_variant_index": variant_index,
    }


def _frame_slot(scenario_index: int, variant_index: int, bank_index: int) -> int:
    return (5 * scenario_index + 7 * variant_index + 3 * bank_index) % 12


def _core_for(bank: BankRealizer, scenario: TypedScenario, variant_index: int) -> CoreTriad:
    pattern = CORE_PATTERNS[bank.bank_id][scenario.family][variant_index]
    return _format_core(pattern, scenario)


def _render(
    bank: BankRealizer,
    bank_index: int,
    scenario: TypedScenario,
    scenario_index: int,
    variant_index: int,
    label: str,
) -> tuple[str, str, str, int]:
    core = _core_for(bank, scenario, variant_index)
    slot = _frame_slot(scenario_index, variant_index, bank_index)
    outer = bank.outer_frames[slot]
    polarity = "neutral"
    if label == "evaluative":
        positive = (variant_index + bank_index) % 2 == 0
        polarity = "positive" if positive else "negative"
        verdict = (
            bank.positive_verdicts[slot]
            if positive
            else bank.negative_verdicts[slot]
        )
        core_text = core.event
        text = outer.evaluative.format(verdict=verdict, event=core.event)
    elif label == "imperative":
        core_text = core.act
        text = outer.imperative.format(act=core.act)
    elif label == "descriptive":
        core_text = core.state
        text = outer.descriptive.format(state=core.state)
    else:
        raise ValueError(f"unsupported direct-fG label: {label}")
    return text, core_text, polarity, slot


def speech_act_matches(text: str) -> tuple[str, ...]:
    """Independent lexical validator used by generation and mutation audits."""
    value = str(text)
    matches: list[str] = []
    if MIXED_EVALUATION_RE.search(value):
        matches.append("evaluative")
    if REQUEST_RE.search(value):
        matches.append("imperative")
    descriptive_hit = bool(CURRENT_FACT_RE.search(value))
    descriptive_hit = descriptive_hit or bool(
        re.match(r"^The pot is ready\b", value, re.IGNORECASE)
    )
    if descriptive_hit:
        matches.append("descriptive")
    return tuple(matches)


def generate_rows() -> list[dict]:
    validate_scenarios()
    rows: list[dict] = []
    for bank_index, bank in enumerate(BANKS):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for variant_index in range(GROUNDINGS_PER_SCENARIO):
                payload = _semantic_payload(scenario, variant_index)
                grounding_id = (
                    f"clean-v6:grounding:{scenario.scenario_id}:{variant_index:02d}"
                )
                triad_id = (
                    f"clean-v6:{bank.bank_id}:eid:{scenario.scenario_id}:"
                    f"{variant_index:02d}"
                )
                for label in LABELS:
                    text, core_text, polarity, slot = _render(
                        bank,
                        bank_index,
                        scenario,
                        scenario_index,
                        variant_index,
                        label,
                    )
                    normalized = normalize_text(text)
                    feedback_id = hashlib.sha256(
                        (
                            f"{VERSION}|{bank.bank_id}|{scenario.scenario_id}|"
                            f"{variant_index}|{label}|{normalized}"
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    rows.append(
                        {
                            "feedback_id": feedback_id,
                            "text": text,
                            "normalized_text": normalized,
                            "core_text": core_text,
                            "normalized_core_text": normalize_text(core_text),
                            "expected_feedback_type": label,
                            "classification_label": CANONICAL_LABELS[label],
                            "source": SOURCE,
                            "label_source": LABEL_SOURCE,
                            "bank_id": bank.bank_id,
                            "bank_split_role": bank.split_role,
                            "scenario_id": scenario.scenario_id,
                            "scenario_family": scenario.family,
                            "grounding_variant_index": variant_index,
                            "grounding_id": grounding_id,
                            "contrast_triad_id": triad_id,
                            "outer_frame_slot": slot,
                            "outer_frame_id": (
                                f"clean-v6:{bank.bank_id}:{label}:outer-{slot:02d}"
                            ),
                            "outer_frame_triad_id": (
                                f"clean-v6:{bank.bank_id}:outer-triad-{slot:02d}"
                            ),
                            "surface_family": (
                                f"clean-v6:{bank.bank_id}:{label}:outer-{slot:02d}"
                            ),
                            "frame_stance": polarity,
                            "semantic_payload": dict(payload),
                            "model_input_contract": {
                                "allowed_fields": ["text"],
                                "metadata_excluded": True,
                                "raw_text_only": True,
                            },
                            "speech_act_contract": {
                                "single_speech_act": True,
                                "validator_expected_match": label,
                                "temporal_frame": {
                                    "evaluative": "recent_completed_event_with_judgment",
                                    "imperative": "current_or_future_request",
                                    "descriptive": "neutral_current_state_or_fact",
                                }[label],
                            },
                            "generation_provenance": {
                                "ai_assisted": True,
                                "prior_error_informed": True,
                                "human_gold": False,
                                "real_player_data": False,
                                "independent_human_semantic_audit": False,
                                "surface_banks": 5,
                                "single_locked_synthetic_eval_bank": True,
                            },
                        }
                    )
    audit_rows(rows)
    return rows


def _expected_feedback_id(row: dict) -> str:
    return hashlib.sha256(
        (
            f"{VERSION}|{row.get('bank_id')}|{row.get('scenario_id')}|"
            f"{row.get('grounding_variant_index')}|"
            f"{row.get('expected_feedback_type')}|{row.get('normalized_text')}"
        ).encode("utf-8")
    ).hexdigest()[:24]


def audit_rows(rows: list[dict], *, raise_on_failure: bool = True) -> dict:
    """Run the complete inexpensive generation contract without short-circuiting."""
    failures: set[str] = set()
    checks: dict[str, bool] = {}

    def record(code: str, passed: bool) -> None:
        checks[code] = bool(passed)
        if not passed:
            failures.add(code)

    expected_bank_roles = {bank.bank_id: bank.split_role for bank in BANKS}
    expected_scenarios = _scenario_map()
    expected_row_count = 5 * 24 * 6 * 3
    record("row_count_exact", isinstance(rows, list) and len(rows) == expected_row_count)
    well_formed = all(isinstance(row, dict) for row in rows)
    record("rows_are_objects", well_formed)
    safe_rows = [row for row in rows if isinstance(row, dict)]

    feedback_ids = [str(row.get("feedback_id") or "") for row in safe_rows]
    record(
        "feedback_id_unique",
        len(feedback_ids) == len(rows)
        and all(feedback_ids)
        and len(set(feedback_ids)) == len(feedback_ids),
    )
    normalized_match = all(
        isinstance(row.get("text"), str)
        and row.get("normalized_text") == normalize_text(row.get("text"))
        for row in safe_rows
    )
    record("normalized_text_matches_recomputed", normalized_match)
    normalized_values = [str(row.get("normalized_text") or "") for row in safe_rows]
    record(
        "normalized_text_unique_global",
        len(normalized_values) == len(rows)
        and all(normalized_values)
        and len(set(normalized_values)) == len(normalized_values),
    )
    core_required = all(
        isinstance(row.get("core_text"), str) and bool(row.get("core_text", "").strip())
        for row in safe_rows
    )
    record("core_text_required_nonempty", core_required)
    record(
        "normalized_core_text_matches_recomputed",
        core_required
        and all(
            row.get("normalized_core_text") == normalize_text(row.get("core_text"))
            for row in safe_rows
        ),
    )
    record(
        "feedback_id_matches_content",
        normalized_match
        and all(row.get("feedback_id") == _expected_feedback_id(row) for row in safe_rows),
    )

    observed_banks = {row.get("bank_id") for row in safe_rows}
    record("bank_manifest_closed", observed_banks == set(expected_bank_roles))
    record(
        "bank_split_role_exact",
        all(
            row.get("bank_id") in expected_bank_roles
            and row.get("bank_split_role") == expected_bank_roles[row.get("bank_id")]
            for row in safe_rows
        ),
    )
    labels = Counter(row.get("expected_feedback_type") for row in safe_rows)
    record("global_cardinality_and_label_balance_exact", labels == Counter({label: 720 for label in LABELS}))

    bank_counts = Counter(row.get("bank_id") for row in safe_rows)
    bank_label_counts = Counter(
        (row.get("bank_id"), row.get("expected_feedback_type")) for row in safe_rows
    )
    record(
        "bank_cardinality_and_label_balance_exact",
        all(bank_counts[bank.bank_id] == 432 for bank in BANKS)
        and all(
            bank_label_counts[(bank.bank_id, label)] == 144
            for bank in BANKS
            for label in LABELS
        ),
    )
    partition_counts = Counter(row.get("bank_split_role") for row in safe_rows)
    partition_labels = Counter(
        (row.get("bank_split_role"), row.get("expected_feedback_type"))
        for row in safe_rows
    )
    expected_partition_rows = {"train": 1296, "calibration": 432, "final_eval": 432}
    expected_partition_per_label = {"train": 432, "calibration": 144, "final_eval": 144}
    record(
        "partition_cardinality_and_label_balance_exact",
        partition_counts == Counter(expected_partition_rows)
        and all(
            partition_labels[(role, label)] == per_label
            for role, per_label in expected_partition_per_label.items()
            for label in LABELS
        ),
    )

    scenario_bank_label = Counter(
        (row.get("bank_id"), row.get("scenario_id"), row.get("expected_feedback_type"))
        for row in safe_rows
    )
    expected_scenario_cells = {
        (bank.bank_id, scenario.scenario_id, label)
        for bank in BANKS
        for scenario in SCENARIOS
        for label in LABELS
    }
    record(
        "scenario_x_label_support_exact",
        set(scenario_bank_label) == expected_scenario_cells
        and all(value == 6 for value in scenario_bank_label.values()),
    )
    record("scenario_bank_label_six", checks["scenario_x_label_support_exact"])

    action_bank_label = Counter(
        (
            row.get("bank_id"),
            (row.get("semantic_payload") or {}).get("action")
            if isinstance(row.get("semantic_payload"), dict)
            else None,
            row.get("expected_feedback_type"),
        )
        for row in safe_rows
    )
    scenario_action_counts = Counter(scenario.canonical_action for scenario in SCENARIOS)
    expected_action_cells = {
        (bank.bank_id, action, label): scenario_action_counts[action] * 6
        for bank in BANKS
        for action in ALLOWED_ACTIONS
        for label in LABELS
    }
    record(
        "action_x_label_support_exact",
        set(action_bank_label) == set(expected_action_cells)
        and all(action_bank_label[key] == value for key, value in expected_action_cells.items()),
    )

    by_triad: dict[str, list[dict]] = defaultdict(list)
    for row in safe_rows:
        by_triad[str(row.get("contrast_triad_id") or "")].append(row)
    expected_triad_count = 5 * 24 * 6
    triad_shapes_ok = len(by_triad) == expected_triad_count and all(
        len(group) == 3
        and Counter(row.get("expected_feedback_type") for row in group)
        == Counter({label: 1 for label in LABELS})
        for group in by_triad.values()
    )
    record("eid_triad_shape_exact", triad_shapes_ok)
    coordinate_fields = (
        "bank_id",
        "bank_split_role",
        "scenario_id",
        "scenario_family",
        "grounding_variant_index",
        "grounding_id",
        "outer_frame_slot",
        "outer_frame_triad_id",
    )
    record(
        "eid_triad_coordinates_identical",
        bool(by_triad)
        and all(
            all(len({row.get(field) for row in group}) == 1 for field in coordinate_fields)
            for group in by_triad.values()
        ),
    )
    record(
        "eid_triad_payload_identical",
        bool(by_triad)
        and all(
            len({_canonical_hash(row.get("semantic_payload")) for row in group}) == 1
            for group in by_triad.values()
        ),
    )

    by_grounding: dict[str, list[dict]] = defaultdict(list)
    for row in safe_rows:
        by_grounding[str(row.get("grounding_id") or "")].append(row)
    expected_grounding_ids = {
        f"clean-v6:grounding:{scenario.scenario_id}:{variant_index:02d}"
        for scenario in SCENARIOS
        for variant_index in range(6)
    }
    grid_shape_ok = set(by_grounding) == expected_grounding_ids and all(
        len(group) == 15
        and {row.get("bank_id") for row in group} == set(expected_bank_roles)
        and Counter(row.get("expected_feedback_type") for row in group)
        == Counter({label: 5 for label in LABELS})
        for group in by_grounding.values()
    )
    record("cross_bank_payload_grid_exact_144x15", grid_shape_ok)
    record(
        "cross_bank_payload_identity",
        bool(by_grounding)
        and all(
            len({_canonical_hash(row.get("semantic_payload")) for row in group}) == 1
            for group in by_grounding.values()
        ),
    )

    frozen_payload_ok = True
    frozen_coordinate_ok = True
    for row in safe_rows:
        scenario = expected_scenarios.get(str(row.get("scenario_id") or ""))
        variant = row.get("grounding_variant_index")
        if scenario is None or type(variant) is not int or not 0 <= variant < 6:
            frozen_payload_ok = False
            frozen_coordinate_ok = False
            continue
        if row.get("scenario_family") != scenario.family:
            frozen_coordinate_ok = False
        if row.get("grounding_id") != f"clean-v6:grounding:{scenario.scenario_id}:{variant:02d}":
            frozen_coordinate_ok = False
        if row.get("semantic_payload") != _semantic_payload(scenario, variant):
            frozen_payload_ok = False
    record("payload_equals_frozen_ontology", frozen_payload_ok)
    record("coordinates_equal_frozen_ontology", frozen_coordinate_ok)

    single_target_ok = True
    exactly_one_ok = True
    evaluative_temporal_ok = True
    imperative_command_ok = True
    descriptive_neutral_ok = True
    for row in safe_rows:
        text = str(row.get("text") or "")
        expected = row.get("expected_feedback_type")
        matches = speech_act_matches(text)
        single_target_ok = single_target_ok and matches == (expected,)
        exactly_one_ok = exactly_one_ok and len(matches) == 1
        if expected == "evaluative":
            evaluative_temporal_ok = evaluative_temporal_ok and bool(
                RECENT_COMPLETION_RE.search(text) and MIXED_EVALUATION_RE.search(text)
            )
        elif expected == "imperative":
            imperative_command_ok = imperative_command_ok and bool(REQUEST_RE.search(text))
            imperative_command_ok = imperative_command_ok and not bool(
                re.fullmatch(r"\s*You\s+\w+ed\b.*", text, re.IGNORECASE)
            )
        elif expected == "descriptive":
            descriptive_neutral_ok = descriptive_neutral_ok and bool(CURRENT_FACT_RE.search(text))
            descriptive_neutral_ok = descriptive_neutral_ok and not bool(
                MIXED_EVALUATION_RE.search(text) or REQUEST_RE.search(text)
            )
        else:
            single_target_ok = False
    record("direct_single_speech_act_matches_target", single_target_ok)
    record("exactly_one_speech_act", exactly_one_ok)
    record("evaluative_recent_completed_anchor", evaluative_temporal_ok)
    record("imperative_current_command", imperative_command_ok)
    record("descriptive_neutral_current_fact", descriptive_neutral_ok)

    old_schema_keys = {
        "reference_type",
        "reference_label",
        "reference_class",
        "feedback_reference_type",
        "five_class_label",
    }
    schema_ok = all(
        not (set(row) & old_schema_keys)
        and row.get("expected_feedback_type") in LABELS
        and row.get("classification_label")
        == CANONICAL_LABELS.get(row.get("expected_feedback_type"))
        and row.get("source") == SOURCE
        and row.get("label_source") == LABEL_SOURCE
        for row in safe_rows
    )
    record("current_three_class_schema_only", schema_ok)

    forbidden_hits: list[dict] = []
    hidden_hits: list[dict] = []
    grammar_ok = True
    for row in safe_rows:
        text = str(row.get("text") or "")
        core_text = str(row.get("core_text") or "")
        for pattern, compiled in zip(FORBIDDEN_PATTERNS, FORBIDDEN_RE):
            if compiled.search(text) or compiled.search(core_text):
                forbidden_hits.append({"feedback_id": row.get("feedback_id"), "pattern": pattern})
        if FORBIDDEN_METADATA_TEXT_RE.search(text):
            hidden_hits.append({"feedback_id": row.get("feedback_id"), "kind": "metadata_marker"})
        if INVISIBLE_RE.search(text) or any(
            unicodedata.category(character).startswith("C") for character in text
        ):
            hidden_hits.append({"feedback_id": row.get("feedback_id"), "kind": "control_or_invisible"})
        if "{" in text or "}" in text or "{" in core_text or "}" in core_text:
            grammar_ok = False
        if len(re.findall(r"[.!?]", text)) > 1:
            grammar_ok = False
        if core_text[-1:] in ".!?":
            grammar_ok = False
    record("forbidden_legacy_ontology_zero", not forbidden_hits)
    record("no_hidden_identifier_nonce_or_control_text", not hidden_hits)
    record("surface_grammar_contract", grammar_ok)

    outer_usage = Counter(
        (row.get("bank_id"), row.get("expected_feedback_type"), row.get("outer_frame_id"))
        for row in safe_rows
    )
    expected_outer_ids = {
        (
            bank.bank_id,
            label,
            f"clean-v6:{bank.bank_id}:{label}:outer-{slot:02d}",
        )
        for bank in BANKS
        for label in LABELS
        for slot in range(12)
    }
    record(
        "outer_frame_usage_exact",
        set(outer_usage) == expected_outer_ids
        and all(value == 12 for value in outer_usage.values()),
    )
    frame_style_ok = True
    for row in safe_rows:
        try:
            slot = int(row.get("outer_frame_slot"))
        except (TypeError, ValueError):
            frame_style_ok = False
            continue
        text = str(row.get("text") or "")
        first = text.split(maxsplit=1)[0] if text.split() else ""
        frame_style_ok = frame_style_ok and 0 <= slot < 12 and first == FIRST_TOKENS[slot]
        frame_style_ok = frame_style_ok and ((text[-1:] in ".!?") == (slot not in NO_TERMINAL_SLOTS))
    record("latin_square_first_token_and_style_quota", frame_style_ok)

    polarity = Counter(
        (row.get("bank_id"), row.get("frame_stance"))
        for row in safe_rows
        if row.get("expected_feedback_type") == "evaluative"
    )
    non_eval_neutral = all(
        row.get("frame_stance") == "neutral"
        for row in safe_rows
        if row.get("expected_feedback_type") != "evaluative"
    )
    record(
        "evaluative_polarity_72_72",
        non_eval_neutral
        and all(
            polarity[(bank.bank_id, "positive")] == 72
            and polarity[(bank.bank_id, "negative")] == 72
            for bank in BANKS
        ),
    )
    record(
        "model_feature_allowlist_raw_text_only",
        all(
            row.get("model_input_contract")
            == {"allowed_fields": ["text"], "metadata_excluded": True, "raw_text_only": True}
            and extract_model_input(row.get("text")) == row.get("text")
            for row in safe_rows
            if isinstance(row.get("text"), str)
        )
        and all(isinstance(row.get("text"), str) for row in safe_rows),
    )

    report = {
        "version": VERSION,
        "status": "passed_generation_contract" if not failures else "failed_generation_contract",
        "quality_gate_passed": not failures,
        "failure_codes": sorted(failures),
        "checks": checks,
        "rows": len(rows) if isinstance(rows, list) else None,
        "normalized_unique_rows": len(set(normalized_values)),
        "label_counts": dict(sorted((str(key), value) for key, value in labels.items())),
        "bank_count": len(observed_banks),
        "scenario_count": len({row.get("scenario_id") for row in safe_rows}),
        "canonical_action_count": len(
            {
                row.get("semantic_payload", {}).get("action")
                for row in safe_rows
                if isinstance(row.get("semantic_payload"), dict)
            }
        ),
        "contrast_triad_count": len(by_triad),
        "cross_bank_payload_grid_count": len(by_grounding),
        "outer_frame_count": len(outer_usage),
        "forbidden_ontology_hits": forbidden_hits[:10],
        "hidden_identifier_or_control_hits": hidden_hits[:10],
        "external_input_files_opened": 0,
        "old_human_dev_test_frozen_rows_read": 0,
        "production_artifact_written": False,
    }
    if failures and raise_on_failure:
        raise GenerationAuditError(sorted(failures), report)
    return report


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows = generate_rows()
    report = audit_rows(rows)
    if not args.audit_only:
        _write_json(args.output.resolve(), rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
