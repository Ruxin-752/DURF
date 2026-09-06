"""Independent synthetic acceptance authoring. NEVER import into training.

Authored by acceptance agent without reading any candidate generator, train or dev
text. The acceptance text was fixed before candidate inference. AI-authored labels
measure an operational synthetic semantic contract, not human generalization.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent

# Each row is text | independent grounding | diagnostic stratum.
EVALUATIVE = r"""
You should have left that onion on the worktop.|action|retrospective_should
Turning toward the empty pot was a bad decision.|action|past_action_judgment
That last trip to the serving hatch was excellent.|action|past_action_judgment
You ought to have waited until the soup was ready.|action|retrospective_should
Picking up the dirty bowl was the wrong move.|action|past_action_judgment
I liked how you stepped aside for the other cook.|action|past_action_judgment
You did not make a mistake by facing the stove.|action|negated_judgment
Delivering that order so quickly was impressive.|action|past_action_judgment
It was unhelpful to carry the tomato back again.|action|past_action_judgment
You should not have put the full bowl down there.|action|retrospective_should
Great job fetching the plate from the lower shelf.|action|past_action_judgment
Your choice to take the upper passage was sensible.|action|spatial_judgment
I disapprove of your last ingredient pickup.|action|past_action_judgment
Staying by that counter was not useful.|action|negated_judgment
Well done for letting your partner through.|action|past_action_judgment
The turn you just made to the right was wasteful.|action|spatial_judgment
I wish you had served that bowl before the timer ran out.|action|retrospective_regret
You would have done better to leave the ladle alone.|action|retrospective_counterfactual
Failing to collect the soup was disappointing.|action|past_action_judgment
That was a clever way to pass the ingredient across.|action|past_action_judgment
I prefer the counter beside the onion crate.|feature|preference
The left hand passage is a poor route.|feature|spatial_value
The serving hatch on the right is very convenient.|feature|spatial_value
An unoccupied stove is valuable.|feature|object_value
Dirty bowls are no use to us.|feature|object_value
That waiting area is a terrible place to stand.|feature|spatial_value
A clear path between the cooks is helpful.|feature|object_value
This short route is better than the long one.|feature|comparison
I dislike the cramped corner by the sink.|feature|preference
Finished soup is worth more than loose onions.|feature|comparison
The northern worktop is not a good storage spot.|feature|negated_value
I value clean plates more than spare tomatoes.|feature|preference
The lower corridor is wonderfully spacious.|feature|spatial_value
A pot with three ingredients is ideal.|feature|object_value
The rightmost shelf is the worst one.|feature|spatial_value
I am pleased with the empty space near the hatch.|feature|object_value
That blocked square is a nuisance.|feature|object_value
The spare ladle is useless here.|feature|object_value
The central stove is my favorite.|feature|preference
Having an onion ready beside the pot is beneficial.|feature|object_value
Your teamwork throughout that round was excellent.|trajectory|whole_round_judgment
The entire run was a disappointment.|trajectory|whole_round_judgment
Overall, you played that kitchen very well.|trajectory|whole_round_judgment
I am unhappy with how the whole shift went.|trajectory|whole_round_judgment
That sequence of deliveries was well coordinated.|trajectory|sequence_judgment
Your overall performance was not bad.|trajectory|negated_judgment
The way you managed the last five orders was inefficient.|trajectory|sequence_judgment
I loved the cooperation across the whole session.|trajectory|whole_round_judgment
Taken as a whole, that attempt was successful.|trajectory|whole_round_judgment
All that wandering around the kitchen was pointless.|trajectory|sequence_judgment
"""

IMPERATIVE = r"""
Carry the next onion over to the pot.|action|direct_request
Please face the worktop beneath you.|action|polite_spatial_request
Could you leave the upper passage clear?|action|indirect_request
Would you bring me an empty bowl?|action|indirect_request
Do not take the tomato from my side.|action|negated_request
Stop walking toward the blocked corner.|action|negated_spatial_request
Let the other cook pass before you move.|action|direct_request
I would like you to fetch a plate now.|action|indirect_request
Make your next turn to the left.|action|spatial_request
Put that cooked soup through the serving hatch.|action|direct_request
Would you mind waiting beside the empty shelf?|action|indirect_request
Keep your hands free until the onion arrives.|action|direct_request
Leave that dirty dish where it is.|action|direct_request
Go around the stove on its eastern side.|action|spatial_request
Please do not step into my path.|action|negated_request
Take one step backward from the counter.|action|spatial_request
Pick up the ingredient directly in front of you.|action|spatial_request
Will you handle the next delivery, please?|action|indirect_request
I need you to fill the empty pot.|action|indirect_request
Turn away from the sink and face the crate.|action|spatial_request
Hold on to that bowl for a moment.|action|direct_request
Avoid putting onions on the delivery tile.|action|negated_request
Bring the finished dish here after it cooks.|action|future_request
Let us use the lower passage for the next trip.|action|collaborative_request
Be sure to collect the soup before leaving.|action|direct_request
How about you carry the plate this time?|action|indirect_request
Get out of the narrow corridor, please.|action|polite_spatial_request
Try staying near the stove on this turn.|action|direct_request
It would help if you passed me that onion now.|action|indirect_request
Could you please stop circling the island?|action|negated_indirect_request
Do not serve the unfinished soup.|action|negated_request
Start chopping the next ingredient.|action|direct_request
Wait until I have crossed the doorway.|action|direct_request
Set the bowl on the counter to your right.|action|spatial_request
For the next order, collect the tomato first.|action|future_request
You should bring a clean dish on your next trip.|action|future_should
You need to step aside now.|action|immediate_request
Please swap places with me beside the pot.|action|polite_request
Make sure the spare plate stays on the shelf.|action|direct_request
I want you to take the shorter passage next.|action|indirect_request
Could I ask you to fetch the missing onion?|action|indirect_request
Do me a favor and serve the waiting bowl.|action|indirect_request
Please wait instead of picking anything up.|action|negated_request
Stay out of the square in front of the hatch.|action|negated_spatial_request
Let go of the tomato at the nearest worktop.|action|direct_request
Move two squares south when the path opens.|action|future_spatial_request
Check the pot before you grab another ingredient.|action|direct_request
Please return that ladle to the sink.|action|polite_request
Would you be able to give your partner the bowl now?|action|indirect_request
On the next turn, choose the passage above the stove.|action|future_spatial_request
"""

DESCRIPTIVE = r"""
The onion crate stands beyond the lower counter.|feature|spatial_state
There are two clean bowls by the sink.|feature|object_state
The serving hatch is opposite the stove.|feature|spatial_state
No soup is ready yet.|feature|negated_state
The right hand corridor has a cook in it.|feature|spatial_state
This pot contains one tomato and two onions.|feature|object_state
The plate rack is not on your left.|feature|negated_spatial_state
An empty square separates the two worktops.|feature|spatial_state
The onion on the north shelf is still there.|feature|spatial_state
The timer shows twelve seconds remaining.|feature|object_state
Your partner is holding a full bowl.|feature|agent_state
Nothing is blocking the bottom passage.|feature|negated_state
The pot behind you has begun to simmer.|feature|object_state
Both cooks are facing the same stove.|feature|agent_state
The nearest counter has no ingredients on it.|feature|negated_state
The tomato crate sits above the dish rack.|feature|spatial_state
One bowl remains beside the delivery window.|feature|object_state
The eastern doorway is closed.|feature|spatial_state
Your hands are empty.|feature|agent_state
There is a spare plate between the pots.|feature|spatial_state
The soup needs another six ticks to finish.|feature|object_state
The top passage connects the two halves of the kitchen.|feature|spatial_state
Three onions are sitting in the central pot.|feature|object_state
I am standing next to the lower stove.|feature|agent_state
The order displayed above us is tomato soup.|feature|object_state
The left hand tile is a counter, not a floor tile.|feature|negated_spatial_state
That bowl has soup in it.|feature|object_state
The nearest ingredient source is behind the wall.|feature|spatial_state
The southern pot is empty.|feature|object_state
Your partner is not carrying an onion.|feature|negated_agent_state
The serving station lies to the west of both pots.|feature|spatial_state
Two worktops occupy the corner beside the crate.|feature|spatial_state
You placed an onion on the shelf a moment ago.|action|past_action_fact
The other cook just turned north.|action|past_spatial_fact
You did not pick up the bowl on that turn.|action|negated_past_fact
I watched you cross the lower passage.|action|past_spatial_fact
The last action was a step toward the sink.|action|past_action_fact
You served one soup while I was waiting.|action|past_action_fact
Your partner dropped the tomato by the pot.|action|past_action_fact
You stayed still during the previous tick.|action|past_action_fact
The onion was moved from the rack to the worktop.|action|past_action_fact
You turned left after reaching the corner.|action|past_spatial_fact
No ingredient was collected on the last turn.|action|negated_past_fact
The cook handed the plate across the island.|action|past_action_fact
During that round, you delivered four bowls of soup.|trajectory|whole_round_fact
The session ended after two minutes.|trajectory|whole_round_fact
Across the entire run, there were nine deliveries.|trajectory|whole_round_fact
You spent half of that round near the stove.|trajectory|whole_round_fact
The last three orders all used onions.|trajectory|sequence_fact
Neither cook served any soup during that session.|trajectory|negated_whole_round_fact
"""

MIXED = [
    ("Nice delivery, but fetch another bowl now.", [("Nice delivery", "evaluative", "action"), ("fetch another bowl now", "imperative", "action")]),
    ("The upper stove is empty, so put an onion in it.", [("The upper stove is empty", "descriptive", "feature"), ("put an onion in it", "imperative", "action")]),
    ("You should have waited; please serve the bowl now.", [("You should have waited", "evaluative", "action"), ("please serve the bowl now", "imperative", "action")]),
    ("That corner is inconvenient, and a bowl is sitting there.", [("That corner is inconvenient", "evaluative", "feature"), ("a bowl is sitting there", "descriptive", "feature")]),
    ("You collected the tomato, but that was a poor choice.", [("You collected the tomato", "descriptive", "action"), ("that was a poor choice", "evaluative", "action")]),
    ("Move beside the pot; your partner is at the sink.", [("Move beside the pot", "imperative", "action"), ("your partner is at the sink", "descriptive", "feature")]),
    ("I like the shorter route, so use it for the next delivery.", [("I like the shorter route", "evaluative", "feature"), ("use it for the next delivery", "imperative", "action")]),
    ("Do not take the plate; it already holds soup.", [("Do not take the plate", "imperative", "action"), ("it already holds soup", "descriptive", "feature")]),
    ("Great teamwork this round; there are two orders remaining.", [("Great teamwork this round", "evaluative", "trajectory"), ("there are two orders remaining", "descriptive", "feature")]),
    ("The doorway is clear, and the southern path is excellent.", [("The doorway is clear", "descriptive", "feature"), ("the southern path is excellent", "evaluative", "feature")]),
    ("You ought to have used the empty stove, but the central pot is full now.", [("You ought to have used the empty stove", "evaluative", "action"), ("the central pot is full now", "descriptive", "feature")]),
    ("The soup is ready; bring a bowl, and serve it.", [("The soup is ready", "descriptive", "feature"), ("bring a bowl", "imperative", "action"), ("serve it", "imperative", "action")]),
    ("That was a wasteful trip; keep the next onion near the stove.", [("That was a wasteful trip", "evaluative", "action"), ("keep the next onion near the stove", "imperative", "action")]),
    ("Could you step aside? The passage behind you is blocked.", [("Could you step aside?", "imperative", "action"), ("The passage behind you is blocked", "descriptive", "feature")]),
    ("You moved toward the crate, and your partner picked up a bowl.", [("You moved toward the crate", "descriptive", "action"), ("your partner picked up a bowl", "descriptive", "action")]),
    ("Please carry the tomato; I do not like this crowded shelf.", [("Please carry the tomato", "imperative", "action"), ("I do not like this crowded shelf", "evaluative", "feature")]),
    ("Your whole run was efficient, but leaving that bowl was a mistake.", [("Your whole run was efficient", "evaluative", "trajectory"), ("leaving that bowl was a mistake", "evaluative", "action")]),
    ("There is no onion in the pot; could you add one, please?", [("There is no onion in the pot", "descriptive", "feature"), ("could you add one, please?", "imperative", "action")]),
    ("Excellent handoff! The other cook now has the plate; wait here.", [("Excellent handoff!", "evaluative", "action"), ("The other cook now has the plate", "descriptive", "feature"), ("wait here", "imperative", "action")]),
    ("The last shift had six deliveries; overall, the teamwork was poor.", [("The last shift had six deliveries", "descriptive", "trajectory"), ("overall, the teamwork was poor", "evaluative", "trajectory")]),
]

UNCERTAIN = [
    ("Hello there!", "out_of_domain", "Greeting supplies no task feedback."),
    ("What time is it in London?", "out_of_domain", "Unrelated information question."),
    ("My favorite movie has a chef in it.", "out_of_domain", "Kitchen vocabulary does not make a game evaluation."),
    ("Purple elephants dance on the moon.", "out_of_domain", "No game-relevant referent."),
    ("Thank you for chatting with me.", "out_of_domain", "Social thanks without a game referent."),
    ("Do you speak Spanish?", "out_of_domain", "Meta conversation question."),
    ("I will be away tomorrow.", "out_of_domain", "Personal scheduling, not kitchen state."),
    ("What is the capital of France?", "out_of_domain", "External knowledge question."),
    ("asdf qwer zxcv", "uninterpretable", "Keyboard noise."),
    ("...", "uninterpretable", "No semantic proposition."),
    ("Left?", "ambiguous", "Could ask position, suggest motion, or question a past turn."),
    ("That bowl.", "ambiguous", "Bare referent gives no speech act."),
    ("We need soup.", "ambiguous", "Could report an objective or indirectly request production."),
    ("You should go left.", "ambiguous", "Without timing may advise a future action or criticize prior play."),
    ("The pot is empty again.", "pragmatically_ambiguous", "Literal state may function as a reproach or request."),
    ("Really?", "ambiguous", "Could be surprise, criticism, or a question."),
    ("I could use a plate.", "pragmatically_ambiguous", "May be a preference, capability statement, or indirect request."),
    ("Interesting move.", "ambiguous", "Valence and purpose depend on context."),
    ("Are you going to serve that?", "pragmatically_ambiguous", "Could ask intention or pressure an action."),
    ("Sure, keep blocking the door.", "sarcasm_ambiguous", "Literal permission and sarcastic criticism differ."),
]


def main():
    singles = []
    for label, rows in [("evaluative", EVALUATIVE), ("imperative", IMPERATIVE), ("descriptive", DESCRIPTIVE)]:
        parsed = [line.strip().split("|") for line in rows.strip().splitlines()]
        assert len(parsed) == 50, (label, len(parsed))
        for n, (text, grounding, tag) in enumerate(parsed, 1):
            singles.append({"id": f"{label[0].upper()}{n:03}", "text": text, "speech_act": label,
                            "grounding": grounding, "stratum": tag, "origin": "synthetic_acceptance_agent"})
    mixed = [{"id": f"M{n:03}", "text": text,
              "expected_speech_acts": sorted(set(c[1] for c in clauses)),
              "clauses": [{"text": c[0], "speech_act": c[1], "grounding": c[2]} for c in clauses],
              "origin": "synthetic_acceptance_agent"}
             for n, (text, clauses) in enumerate(MIXED, 1)]
    uncertain = [{"id": f"U{n:03}", "text": text, "kind": kind,
                  "reason": reason, "expected_policy": "abstain_or_explicit_uncertainty_no_automatic_learning",
                  "origin": "synthetic_acceptance_agent"}
                 for n, (text, kind, reason) in enumerate(UNCERTAIN, 1)]
    corpus = {
        "schema_version": 1,
        "dataset_id": "durf-independent-synthetic-acceptance-20260906",
        "source": "AI-authored synthetic only; no human participant data",
        "independence": {
            "candidate_training_text_read": False, "candidate_generator_read": False,
            "candidate_inference_before_freeze": False,
            "use": "one final evaluation after candidate and inference code freeze; no tuning on this set",
            "limitation": "independent authorship within the same AI session, not independently human-validated ground truth",
        },
        "contract": {
            "speech_act": {
                "evaluative": "Judgment, praise, criticism, regret about past choices, or expressed object preference/value.",
                "imperative": "Request or instruction for a future/immediate action, including polite indirect requests.",
                "descriptive": "Factual state or event report without explicit judgment or requested action.",
            },
            "grounding": {
                "action": "A particular action, move, or requested next action, including its retrospective judgment.",
                "feature": "An object, spatial relation, current state, or its value/preference.",
                "trajectory": "An entire run/round/session or multi-action sequence considered together.",
            },
            "mixed": "Evaluate component speech-act presence and gold clauses separately; one class need not fit the whole message.",
            "uncertain": "Excluded from forced three-class accuracy; report confidence/abstention and learning-gate behavior separately.",
        },
        "predeclared_checks": {
            "speech_act_macro_f1_min": 0.87, "speech_act_each_class_recall_min": 0.85,
            "grounding_macro_f1_min": 0.87, "grounding_each_class_recall_min": 0.85,
            "mixed_message_component_set_accuracy_min": 0.85,
            "critical_strata": ["retrospective_should", "future_should", "spatial_state", "spatial_request", "object_value", "past_action_fact"],
            "notes": "Synthetic acceptance only. Critical systematic failures and unsupported learning behavior must be reported regardless of aggregate score.",
        },
        "singles": singles, "mixed": mixed, "uncertain": uncertain,
    }
    target = ROOT / "frozen.json"
    if target.exists():
        raise SystemExit("Refusing to overwrite a frozen acceptance dataset.")
    data = (json.dumps(corpus, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    summary = {"dataset_id": corpus["dataset_id"], "sha256": sha,
               "single_count": len(singles), "speech_act_counts": dict(Counter(x["speech_act"] for x in singles)),
               "grounding_counts": dict(Counter(x["grounding"] for x in singles)),
               "cross_label_counts": dict(Counter(x["speech_act"]+"/"+x["grounding"] for x in singles)),
               "mixed_count": len(mixed), "mixed_clause_count": sum(len(x["clauses"]) for x in mixed),
               "uncertain_count": len(uncertain), "source": corpus["source"],
               "independence": corpus["independence"], "predeclared_checks": corpus["predeclared_checks"]}
    (ROOT / "seal.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    (ROOT / "frozen.sha256").write_text(sha+"  frozen.json\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
