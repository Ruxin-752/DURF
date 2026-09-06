"""Second independently authored synthetic acceptance corpus.

Created after first-round aggregate results, without reading any training/dev
texts or candidate generator. Previous acceptance may now be development data.
These texts must remain hidden from model development until final freeze.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent

EVALUATIVE = r"""
Your timing on that soup pickup was perfect.|trajectory|completed_judgment
You handled the rush brilliantly.|trajectory|completed_judgment
That handoff was a complete mess.|trajectory|completed_judgment
I give your performance in this round a thumbs up.|trajectory|overall_judgment
What an awful decision to leave the burner idle!|trajectory|completed_judgment
That was not the right moment to grab a tomato.|trajectory|negated_judgment
The way you recovered after dropping the bowl was admirable.|trajectory|completed_judgment
You deserve credit for the last delivery.|trajectory|completed_judgment
I was impressed by how smoothly you worked around me.|trajectory|completed_judgment
Your coordination over those ten turns was dreadful.|trajectory|overall_judgment
Losing that cooked soup was such a shame.|trajectory|completed_judgment
That last move gets a thumbs down from me.|trajectory|completed_judgment
Your handling of the dinner rush deserves praise.|trajectory|overall_judgment
I am disappointed in the route you just took.|trajectory|completed_judgment
The way you used the spare dish was smart.|trajectory|completed_judgment
There was nothing useful about that detour.|trajectory|negated_judgment
You did a lovely job stocking the counter.|trajectory|completed_judgment
That round went much worse than I hoped.|trajectory|overall_judgment
I am satisfied with your work on the last order.|trajectory|completed_judgment
Your last-minute rescue of the soup was outstanding.|trajectory|completed_judgment
You should have handed me the empty dish instead.|action|retrospective_counterfactual
It would have been better if you had taken the open lane.|action|retrospective_counterfactual
You ought not to have discarded that ingredient.|action|retrospective_counterfactual
I wish you had made room for me at the hatch.|action|retrospective_counterfactual
You could have avoided that mistake by waiting one turn.|action|retrospective_counterfactual
For me, an unobstructed aisle matters more than an extra dish.|feature|preference
My least favorite storage area is the bench by the door.|feature|preference
Having every pot occupied is a disadvantage.|feature|feature_value
The spacious end of the kitchen is a real asset.|feature|feature_value
Freshly washed dishes are more useful than dirty ones.|feature|feature_comparison
I am a fan of the worktop between the cookers.|feature|preference
The supply cupboard is in an awkward position.|feature|spatial_value
An ingredient within reach is a big help.|feature|feature_value
Two cooks sharing that tiny nook is an inefficient arrangement.|feature|feature_value
I consider a free delivery lane essential.|feature|preference
That unused rack is a waste of space.|feature|feature_value
The layout around the burners is poorly designed.|feature|spatial_value
I cannot stand the cramped approach to the hatch.|feature|preference
To my mind, spare dishes are worth keeping.|feature|preference
The bottleneck next to the cupboard is horrible.|feature|spatial_value
An onion on an accessible bench is more valuable than one across the room.|feature|feature_comparison
I would rate this serving area highly.|feature|feature_value
The long walk from storage to the cooker is a drawback.|feature|spatial_value
I find this corner easier to work in than the opposite one.|feature|preference
A blocked delivery point is the worst possible setup.|feature|feature_value
That attempt was a huge improvement on the previous round.|trajectory|overall_judgment
There is a lot to like about how you completed those orders.|trajectory|overall_judgment
Your cooperation during this shift fell short.|trajectory|overall_judgment
I cannot fault your performance on that run.|trajectory|negated_judgment
You nailed the final handoff.|trajectory|completed_judgment
"""

IMPERATIVE = r"""
Pass the dish across before I reach the stove.|action|direct_request
Please fill the saucepan that is still empty.|action|polite_request
Can you collect what is on the middle bench?|action|indirect_request
Give me room to reach the serving point.|action|direct_request
Do not leave a tomato in the walkway.|action|negated_request
Could you deal with the bowl waiting at the cooker?|action|indirect_request
Bring your onion to the burner with two tomatoes.|action|direct_request
Leave the only clear tile unoccupied.|action|negated_request
Once I put this down, take over the delivery.|action|conditional_request
Please get behind me before entering that lane.|action|polite_spatial_request
I am asking you to stay at the delivery end for now.|action|indirect_request
Keep moving until you reach the far wall.|action|spatial_request
Would you please carry the spare dish over here?|action|indirect_request
Do not discard anything from the full saucepan.|action|negated_request
Take care of the soup as soon as it finishes.|action|conditional_request
Face me so I can pass you the ingredient.|action|spatial_request
Put the bowl within my reach.|action|direct_request
Back up a square to let me into the corner.|action|spatial_request
Please check whether the burner needs another tomato.|action|polite_request
Make the next delivery yourself.|action|direct_request
Could you switch to collecting dishes for this order?|action|indirect_request
Wait for the steam before taking the soup out.|action|direct_request
Let me get past the rack first.|action|direct_request
Hand over the ingredient you are carrying.|action|direct_request
You need to make some room at the serving point now.|action|immediate_request
Please use the other cooker for this batch.|action|polite_request
On your next move, head toward the unoccupied tile.|action|future_spatial_request
Help me by bringing a dish to this corner.|action|indirect_request
Do not start a second trip until I have returned.|action|negated_request
Take the object off the bench nearest the door.|action|direct_request
Please leave the bottom cooker for my tomatoes.|action|polite_request
Can you wait one more tick before moving forward?|action|indirect_request
Set down what you are holding so you can take this plate.|action|direct_request
I would appreciate it if you handled this delivery now.|action|indirect_request
Turn your back to the wall and face the saucepan.|action|spatial_request
Use the free space beyond the dish rack.|action|spatial_request
You should collect a fresh plate after this delivery.|action|future_should
Please do not reach for the ingredient I am carrying.|action|negated_request
Give the steaming soup to the customer.|action|direct_request
The next time you cross the room, bring a bowl back.|action|conditional_request
Leave one bench empty for our handoffs.|action|direct_request
Please pause while I get around the island.|action|polite_request
I need you over at the cooker right away.|action|indirect_request
Can you put your ingredient into the unfinished recipe?|action|indirect_request
Keep that soup in your hands until you reach the hatch.|action|direct_request
Try the open aisle on the far side this time.|action|spatial_request
Do not turn toward me on the next tick.|action|negated_spatial_request
Make a space on the bench for this bowl.|action|direct_request
Would you collect a tomato while I fetch the dish?|action|indirect_request
Take your next step away from the occupied doorway.|action|future_spatial_request
"""

DESCRIPTIVE = r"""
One of the burners is in use.|feature|current_state
The dish in your hands is empty.|feature|current_state
There is a cooker at each end of this bench.|feature|spatial_state
Steam is rising from the saucepan nearest me.|feature|current_state
Only the middle aisle is unoccupied.|feature|spatial_state
Your partner has two counter tiles between them and the hatch.|feature|spatial_state
The remaining order calls for three tomatoes.|feature|current_state
The storage cupboard and the serving point share a wall.|feature|spatial_state
I have an onion in my hands.|feature|current_state
Both saucepans contain the same ingredients.|feature|current_state
The bench beneath the timer holds a plate.|feature|spatial_state
No cooker is empty at present.|feature|negated_state
The way around the island passes the dish rack.|feature|spatial_state
There are seven ticks left in the cooking cycle.|feature|current_state
The cook in blue is looking toward the wall.|feature|agent_state
An onion occupies the tile where the bowl was.|feature|current_state
The dish supply is across the aisle from you.|feature|spatial_state
The unfinished soup has two ingredients in it.|feature|current_state
Your partner is on the opposite side of the island.|feature|agent_state
The hatch accepts bowls of completed soup.|feature|general_rule
An interaction uses the counter a cook is facing.|feature|general_rule
The narrowest aisle has room for one cook.|feature|spatial_state
This recipe requires tomatoes rather than onions.|feature|general_rule
The clock is still running.|feature|current_state
Neither of the nearby counters holds a dish.|feature|negated_state
There is no gap between the wall and that stove.|feature|negated_spatial_state
The tile below your partner is a serving point.|feature|spatial_state
The cooker by the cupboard has not started heating.|feature|negated_state
We currently have one completed order and two pending orders.|feature|current_state
The island has a worktop on all four sides.|feature|spatial_state
You brought back a plate on the preceding trip.|trajectory|completed_fact
The blue cook deposited a tomato in the saucepan.|trajectory|completed_fact
I saw you put the dish beside the door.|trajectory|completed_fact
Your most recent interaction removed an onion from storage.|trajectory|completed_fact
You crossed the kitchen twice during that delivery.|trajectory|completed_fact
The last bowl reached the hatch before the bell.|trajectory|completed_fact
You have taken three steps since picking up that ingredient.|trajectory|completed_fact
The other cook stopped at the end of the aisle.|trajectory|completed_fact
No dish changed hands on the previous turn.|trajectory|negated_completed_fact
You faced the cupboard for two ticks and then walked away.|trajectory|completed_sequence_fact
The score increased when your last soup was delivered.|trajectory|completed_fact
That run included five tomato orders.|trajectory|whole_run_fact
Your partner carried the empty bowl all the way across.|trajectory|completed_fact
The first delivery happened thirty ticks into the round.|trajectory|whole_run_fact
You spent three trips gathering ingredients for that batch.|trajectory|completed_sequence_fact
The completed shift lasted eight hundred steps.|trajectory|whole_run_fact
There were no collisions during the last ten turns.|trajectory|negated_completed_fact
You dropped the ingredient before collecting the plate.|trajectory|completed_sequence_fact
The previous game ended with four dishes on the counters.|trajectory|whole_run_fact
The bowl was handed over while you were next to the burner.|trajectory|completed_fact
"""

MIXED = [
    ("You nailed that delivery. Now bring me a dish.", [("You nailed that delivery.", "evaluative", "trajectory"), ("Now bring me a dish.", "imperative", "action")]),
    ("We have one tomato left, so put it into this saucepan.", [("We have one tomato left", "descriptive", "feature"), ("put it into this saucepan", "imperative", "action")]),
    ("This work area is cramped, and I dislike its layout.", [("This work area is cramped", "descriptive", "feature"), ("I dislike its layout", "evaluative", "feature")]),
    ("Your first trip was excellent; the second trip used the upper aisle.", [("Your first trip was excellent", "evaluative", "trajectory"), ("the second trip used the upper aisle", "descriptive", "trajectory")]),
    ("I wish you had passed the plate earlier; collect another one now.", [("I wish you had passed the plate earlier", "evaluative", "action"), ("collect another one now", "imperative", "action")]),
    ("The burner has stopped steaming. Please check the soup.", [("The burner has stopped steaming.", "descriptive", "feature"), ("Please check the soup.", "imperative", "action")]),
    ("Do not enter that gap; a cook is standing there.", [("Do not enter that gap", "imperative", "action"), ("a cook is standing there", "descriptive", "feature")]),
    ("I value the spare bowls, but the nearest rack is empty.", [("I value the spare bowls", "evaluative", "feature"), ("the nearest rack is empty", "descriptive", "feature")]),
    ("You delivered the bowl on that turn, and your timing was superb.", [("You delivered the bowl on that turn", "descriptive", "trajectory"), ("your timing was superb", "evaluative", "trajectory")]),
    ("Please handle the dish while I collect the onion; that last handoff was clumsy.", [("Please handle the dish while I collect the onion", "imperative", "action"), ("that last handoff was clumsy", "evaluative", "trajectory")]),
    ("The recipe needs one more ingredient; fetch an onion, then wait here.", [("The recipe needs one more ingredient", "descriptive", "feature"), ("fetch an onion", "imperative", "action"), ("wait here", "imperative", "action")]),
    ("Your teamwork was poor; nevertheless, that final delivery was impressive.", [("Your teamwork was poor", "evaluative", "trajectory"), ("that final delivery was impressive", "evaluative", "trajectory")]),
    ("The long detour is wasteful, so take the opening behind me.", [("The long detour is wasteful", "evaluative", "feature"), ("take the opening behind me", "imperative", "action")]),
    ("Can you face the bench? The bowl is already on it.", [("Can you face the bench?", "imperative", "action"), ("The bowl is already on it.", "descriptive", "feature")]),
    ("You should have stayed near the serving point, but you walked toward storage.", [("You should have stayed near the serving point", "evaluative", "action"), ("you walked toward storage", "descriptive", "trajectory")]),
    ("The lower saucepan is cooking, and the top one is empty.", [("The lower saucepan is cooking", "descriptive", "feature"), ("the top one is empty", "descriptive", "feature")]),
    ("That attempt deserves praise; give the next order the same attention.", [("That attempt deserves praise", "evaluative", "trajectory"), ("give the next order the same attention", "imperative", "action")]),
    ("Keep this aisle free! It is the only route to the hatch.", [("Keep this aisle free!", "imperative", "action"), ("It is the only route to the hatch.", "descriptive", "feature")]),
    ("I prefer the space by the dish rack; it contains two benches.", [("I prefer the space by the dish rack", "evaluative", "feature"), ("it contains two benches", "descriptive", "feature")]),
    ("Well handled on that order! The next soup is ready; take it out.", [("Well handled on that order!", "evaluative", "trajectory"), ("The next soup is ready", "descriptive", "feature"), ("take it out", "imperative", "action")]),
]

UNCERTAIN = [
    ("Good morning, everyone.", "out_of_domain", "Greeting without game feedback."),
    ("I enjoyed a tomato sandwich for breakfast.", "out_of_domain", "Personal meal opinion, not a kitchen-game target."),
    ("Can you explain photosynthesis?", "out_of_domain", "External information request."),
    ("My real kitchen is being painted this week.", "out_of_domain", "Real-world personal statement, not this game."),
    ("Have a pleasant weekend!", "out_of_domain", "Social wish without a task referent."),
    ("The restaurant in my hometown closed yesterday.", "out_of_domain", "Outside-world news despite kitchen topic words."),
    ("I am testing my microphone.", "out_of_domain", "Conversation infrastructure statement."),
    ("What language is this website written in?", "out_of_domain", "Question about the website, not kitchen feedback."),
    ("12345 $$$", "uninterpretable", "No interpretable game instruction."),
    ("hmm...", "ambiguous", "Hesitation supplies no definite act."),
    ("The other one.", "ambiguous", "Missing antecedent and speech act."),
    ("Soup, perhaps?", "ambiguous", "Could suggest, request, or ask about an object."),
    ("Already?", "ambiguous", "Timing question or evaluative surprise."),
    ("That is something.", "ambiguous", "Unspecified referent and valence."),
    ("Can you reach it?", "pragmatically_ambiguous", "Capability query or action request with unresolved referent."),
    ("Would another bowl help?", "pragmatically_ambiguous", "Information question, offer, or indirect suggestion."),
    ("A bit much.", "ambiguous", "Missing referent and evaluative dimension."),
    ("I suppose so.", "ambiguous", "Dependent conversational agreement without known target."),
    ("Oh, wonderful, another collision.", "sarcasm_ambiguous", "Likely sarcastic criticism; literal valence differs."),
    ("You know what to do.", "pragmatically_ambiguous", "Could encourage, delegate, or merely state knowledge."),
]

def main():
    previous = json.loads((ROOT / 'frozen-v2.json').read_text(encoding='utf-8'))
    old_texts = {r['text'].casefold().strip() for group in ('singles', 'mixed', 'uncertain') for r in previous[group]}
    singles = []
    for label, source in [('evaluative', EVALUATIVE), ('imperative', IMPERATIVE), ('descriptive', DESCRIPTIVE)]:
        rows = [line.strip().split('|') for line in source.strip().splitlines()]
        assert len(rows) == 50, (label, len(rows))
        for n, (text, grounding, stratum) in enumerate(rows, 1):
            assert text.casefold().strip() not in old_texts
            singles.append({'id': f'N{label[0].upper()}{n:03}', 'text': text, 'speech_act': label,
                            'grounding': grounding, 'stratum': stratum, 'origin': 'synthetic_acceptance_agent_round2'})
    mixed = [{'id': f'NM{n:03}', 'text': text, 'expected_speech_acts': sorted({c[1] for c in clauses}),
              'clauses': [{'text': c[0], 'speech_act': c[1], 'grounding': c[2]} for c in clauses],
              'origin': 'synthetic_acceptance_agent_round2'} for n, (text, clauses) in enumerate(MIXED, 1)]
    uncertain = [{'id': f'NU{n:03}', 'text': text, 'kind': kind, 'reason': reason,
                  'expected_policy': 'abstain_or_explicit_uncertainty_no_automatic_learning',
                  'origin': 'synthetic_acceptance_agent_round2'} for n, (text, kind, reason) in enumerate(UNCERTAIN, 1)]
    for row in mixed + uncertain:
        assert row['text'].casefold().strip() not in old_texts
    dataset = {
        'schema_version': 1, 'dataset_id': 'durf-independent-synthetic-acceptance-20260906-round2',
        'source': 'AI-authored synthetic only; no human participant data',
        'independence': {
            'candidate_training_text_read': False, 'candidate_generator_read': False,
            'candidate_inference_before_freeze': False,
            'previous_acceptance_now_development': True,
            'authorship': 'Fresh complete sentences and scenarios authored without reading training/dev text; no exact previous-acceptance sentence reused.',
            'use': 'one final evaluation after next candidate and inference code freeze; no tuning on this set',
            'limitation': 'independent authorship within the same AI session, not independently human-validated ground truth',
        },
        'contract': previous['contract'], 'predeclared_checks': previous['predeclared_checks'],
        'singles': singles, 'mixed': mixed, 'uncertain': uncertain,
    }
    target = ROOT / 'frozen-next.json'
    assert not target.exists(), 'Refusing to overwrite frozen round2'
    content = (json.dumps(dataset, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    target.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    summary = {'dataset_id': dataset['dataset_id'], 'sha256': sha,
               'single_count': len(singles), 'speech_act_counts': dict(Counter(r['speech_act'] for r in singles)),
               'grounding_counts': dict(Counter(r['grounding'] for r in singles)),
               'mixed_count': len(mixed), 'mixed_clause_count': sum(len(r['clauses']) for r in mixed),
               'uncertain_count': len(uncertain), 'independence': dataset['independence'],
               'predeclared_checks': dataset['predeclared_checks']}
    (ROOT / 'seal-next.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'frozen-next.sha256').write_text(sha+'  frozen-next.json\n', encoding='ascii')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
