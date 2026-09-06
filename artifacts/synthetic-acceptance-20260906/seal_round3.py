"""Third blind synthetic corpus: common kitchen vocabulary, unchanged semantics.

The author has not read candidate train/dev files, data/, or generators. Earlier
acceptance sets are now development material and cannot measure this release.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter
ROOT = Path(__file__).resolve().parent

EVALUATIVE = r"""
Good work getting the soup out on time.|trajectory|completed_judgment
You wasted that last move.|trajectory|completed_judgment
Your last onion delivery was very helpful.|trajectory|completed_judgment
I liked the way you kept the pot busy.|trajectory|completed_judgment
That was a bad time to go for a bowl.|trajectory|completed_judgment
You were too slow with the last dish.|trajectory|completed_judgment
Well played on that final order.|trajectory|completed_judgment
You made the right choice when you waited for me.|trajectory|completed_judgment
I am not happy with that last turn.|trajectory|negated_judgment
Your quick handoff made that round much better.|trajectory|completed_judgment
The extra walking in that round was unnecessary.|trajectory|whole_round_judgment
I think you did a good job with the soup.|trajectory|completed_judgment
That last pickup was a mistake.|trajectory|completed_judgment
I really enjoyed how well we worked together in that run.|trajectory|whole_round_judgment
You did not handle that order well.|trajectory|negated_judgment
Getting in my way on that turn was unhelpful.|trajectory|completed_judgment
Your movement around the pot was excellent.|trajectory|completed_judgment
The last round showed poor teamwork.|trajectory|whole_round_judgment
I approve of the choice you just made.|trajectory|completed_judgment
You did a great job keeping up with the orders.|trajectory|whole_round_judgment
Your last trip took far too long.|trajectory|completed_judgment
I am pleased with how you used that bowl.|trajectory|completed_judgment
It was wrong to take the onion I had ready.|trajectory|completed_judgment
There was no need for the extra turn you just took.|trajectory|negated_judgment
That round was much more efficient.|trajectory|whole_round_judgment
You should have picked up the ready soup first.|action|retrospective_should
You ought to have put your tomato into the other pot.|action|retrospective_should
I wish you had waited for me before taking the dish.|action|retrospective_counterfactual
It would have been smarter to stay next to the soup.|action|retrospective_counterfactual
You should not have carried that onion away from the pot.|action|retrospective_should
The counter near the dishes is a useful place.|feature|feature_value
I prefer having more room around the pots.|feature|preference
The far corner is a bad place for spare bowls.|feature|spatial_value
A clear serving area is better than a crowded one.|feature|feature_comparison
Tomatoes on a nearby counter are helpful.|feature|feature_value
The route past the dish supply is too long.|feature|spatial_value
I do not like having both cooks in this narrow space.|feature|preference
This empty counter is just what we need.|feature|feature_value
The lower pot is in a much better position.|feature|spatial_value
In my opinion, keeping a spare dish nearby is useful.|feature|preference
That short path is my preferred route.|feature|preference
Loose ingredients in the serving area are a problem.|feature|feature_value
I find the left side of this kitchen easier to use.|feature|preference
An extra onion is not useful when both pots are full.|feature|negated_value
The space next to the serving point is valuable.|feature|feature_value
I dislike the long path around the counters.|feature|preference
An empty-handed cook beside the ready soup is no help.|feature|feature_value
The pot closest to the dishes is the best one.|feature|feature_value
A bowl within reach of the soup makes things easier.|feature|feature_value
To me, a clear path is more important than a spare tomato.|feature|preference
"""

IMPERATIVE = r"""
Please pick up the bowl next to my onion.|action|polite_request
Go to the pot with the soup in it.|action|spatial_request
Could you add your tomato to this recipe?|action|indirect_request
Do not put anything on the serving counter.|action|negated_request
Take the soup to the delivery point now.|action|direct_request
Let me use the pot on this turn.|action|direct_request
I want you to wait for the next bowl.|action|indirect_request
Please collect an onion before the timer ends.|action|polite_request
Turn toward the dish supply on your next move.|action|future_spatial_request
Leave the ready soup for me to collect.|action|direct_request
Can you move to the empty square next to us?|action|indirect_request
Hold that dish until the pot finishes cooking.|action|direct_request
Please put your ingredient in the pot with room in it.|action|polite_request
Do not walk through the serving area yet.|action|negated_request
Come back to this counter when you have a bowl.|action|conditional_request
Bring me the onion that is nearest to you.|action|direct_request
Would you take the dish from this counter?|action|indirect_request
Please give your partner a chance to pass.|action|polite_request
Use your next move to get closer to the soup.|action|future_request
Stop taking tomatoes from that counter.|action|negated_request
I need you to handle the pot while I am away.|action|indirect_request
Wait here for one turn, please.|action|polite_request
Keep the last empty bowl in your hands.|action|direct_request
Put down the onion so that I can take it.|action|direct_request
Please use the path beside the serving point.|action|spatial_request
Do not move into the square I am using.|action|negated_spatial_request
Could you bring the finished soup straight over?|action|indirect_request
Take a step toward the counter with the dishes.|action|spatial_request
You should leave the next dish beside the pot.|action|future_should
Please get another tomato for the new order.|action|polite_request
Let your partner collect this bowl.|action|direct_request
Move away from the serving counter after this delivery.|action|future_spatial_request
Can you take care of the next onion pickup?|action|indirect_request
Put the full dish down on an empty counter.|action|direct_request
Do not take a second ingredient while holding the first.|action|negated_request
Please face the pot before trying to use it.|action|polite_request
Go around me to reach the empty space.|action|spatial_request
Could you stay on that side until I get through?|action|indirect_request
Return to the onion supply for the next trip.|action|future_request
Keep this corner open for the other cook.|action|direct_request
Please pick the soup up with a clean bowl.|action|polite_request
You need to clear the way to the dishes now.|action|immediate_request
Try waiting at the nearby counter for a turn.|action|direct_request
Would you leave me the last tomato on the counter?|action|indirect_request
Take the empty route toward the serving point.|action|spatial_request
Do not put the soup back into the pot.|action|negated_request
Please set the spare bowl where I can reach it.|action|polite_request
When this order is done, get the next onion.|action|conditional_request
I would like you to serve the bowl you are holding.|action|indirect_request
Stay beside this pot until your partner arrives.|action|direct_request
"""

DESCRIPTIVE = r"""
The pot closest to me contains two tomatoes.|feature|current_state
Your bowl is on the counter behind the player.|feature|spatial_state
The cook on the right has no item in their hands.|feature|negated_agent_state
There is room for one more ingredient in this pot.|feature|current_state
The dish supply is next to the top corner.|feature|spatial_state
Both players are standing in the lower half of the kitchen.|feature|agent_state
The soup in the left pot is still cooking.|feature|current_state
One tomato remains on the far counter.|feature|current_state
The path between the counters is one tile wide.|feature|spatial_state
I am carrying the bowl that was beside the pot.|feature|agent_state
The current order needs one onion.|feature|current_state
The serving point is two squares away from you.|feature|spatial_state
There are no dishes on the counter in front of me.|feature|negated_state
The pot has enough ingredients to start cooking.|feature|current_state
Your partner is looking at the onion supply.|feature|agent_state
The upper and lower paths meet beside the dishes.|feature|spatial_state
That soup will be ready in four ticks.|feature|current_state
The counter between us contains a full bowl.|feature|spatial_state
The player is facing a wall at the moment.|feature|agent_state
The two pots are on different sides of the counter.|feature|spatial_state
An empty dish is required to collect cooked soup.|feature|general_rule
Only completed soup can be handed in for an order.|feature|general_rule
The onion supply does not contain tomatoes.|feature|negated_state
There is a counter directly below the serving point.|feature|spatial_state
Neither player is beside the empty pot.|feature|negated_agent_state
The round has twenty seconds remaining.|feature|current_state
One side of the dish counter faces the open path.|feature|spatial_state
The next listed order uses the same recipe.|feature|current_state
The corner behind you has no free floor tile.|feature|negated_spatial_state
Your partner has a dish, and you have a tomato.|feature|agent_state
You collected the last bowl before I arrived.|trajectory|completed_fact
The other cook put an onion beside the serving point.|trajectory|completed_fact
You used the upper path for your previous trip.|trajectory|completed_fact
The last turn left you facing the dish counter.|trajectory|completed_fact
You walked past the empty pot without stopping.|trajectory|completed_fact
The most recent delivery increased our order count by one.|trajectory|completed_fact
Your partner picked up the soup after the timer finished.|trajectory|completed_fact
No ingredient went into the pot on that turn.|trajectory|negated_completed_fact
You spent two turns waiting beside me.|trajectory|completed_fact
The previous order was made with tomatoes.|trajectory|completed_fact
You took the dish off the counter after dropping your onion.|trajectory|completed_sequence_fact
The last game included three soup deliveries.|trajectory|whole_run_fact
Your partner crossed from the left side to the right side.|trajectory|completed_fact
The bowl stayed in your hands throughout the trip.|trajectory|completed_fact
You returned to the serving point twice in that round.|trajectory|whole_run_fact
The last soup began cooking while you were collecting dishes.|trajectory|completed_fact
There were two full pots at the end of that run.|trajectory|whole_run_fact
You did not interact with any counter on the previous tick.|trajectory|negated_completed_fact
Your last two moves both took you toward the tomato supply.|trajectory|completed_sequence_fact
The score was forty when the round ended.|trajectory|whole_run_fact
"""

MIXED = [
    ("Good work with that bowl; please collect the next ingredient.", [("Good work with that bowl", "evaluative", "trajectory"), ("please collect the next ingredient", "imperative", "action")]),
    ("The pot is nearly ready, so bring a dish over.", [("The pot is nearly ready", "descriptive", "feature"), ("bring a dish over", "imperative", "action")]),
    ("I like the open space here, and the counter has two dishes on it.", [("I like the open space here", "evaluative", "feature"), ("the counter has two dishes on it", "descriptive", "feature")]),
    ("You took the long path; that choice was unhelpful.", [("You took the long path", "descriptive", "trajectory"), ("that choice was unhelpful", "evaluative", "trajectory")]),
    ("Please stay near the dishes. Your partner is at the serving point.", [("Please stay near the dishes.", "imperative", "action"), ("Your partner is at the serving point.", "descriptive", "feature")]),
    ("You should have used a bowl earlier, but there is still time to finish.", [("You should have used a bowl earlier", "evaluative", "action"), ("there is still time to finish", "descriptive", "feature")]),
    ("The soup is finished, and your quick work was helpful.", [("The soup is finished", "descriptive", "feature"), ("your quick work was helpful", "evaluative", "trajectory")]),
    ("Do not pick up that onion; I have one already.", [("Do not pick up that onion", "imperative", "action"), ("I have one already", "descriptive", "feature")]),
    ("The empty counter is useful, so leave it free for now.", [("The empty counter is useful", "evaluative", "feature"), ("leave it free for now", "imperative", "action")]),
    ("Your last round was good, but that final trip was too slow.", [("Your last round was good", "evaluative", "trajectory"), ("that final trip was too slow", "evaluative", "trajectory")]),
    ("Can you get a bowl? There is one behind the player.", [("Can you get a bowl?", "imperative", "action"), ("There is one behind the player.", "descriptive", "feature")]),
    ("The lower path is open; use it, then wait by the pot.", [("The lower path is open", "descriptive", "feature"), ("use it", "imperative", "action"), ("wait by the pot", "imperative", "action")]),
    ("You picked up an onion, and your partner took the empty bowl.", [("You picked up an onion", "descriptive", "trajectory"), ("your partner took the empty bowl", "descriptive", "trajectory")]),
    ("That extra trip was a mistake. Bring the soup here now.", [("That extra trip was a mistake.", "evaluative", "trajectory"), ("Bring the soup here now.", "imperative", "action")]),
    ("I prefer the pot near the dishes; the other one is full.", [("I prefer the pot near the dishes", "evaluative", "feature"), ("the other one is full", "descriptive", "feature")]),
    ("There is one order remaining, so keep working on this soup.", [("There is one order remaining", "descriptive", "feature"), ("keep working on this soup", "imperative", "action")]),
    ("Your turn toward the counter was smart; please stay there.", [("Your turn toward the counter was smart", "evaluative", "trajectory"), ("please stay there", "imperative", "action")]),
    ("You ought to have left the onion for me; it is on the far counter now.", [("You ought to have left the onion for me", "evaluative", "action"), ("it is on the far counter now", "descriptive", "feature")]),
    ("Please do not block this path, and the pot beside you needs a tomato.", [("Please do not block this path", "imperative", "action"), ("the pot beside you needs a tomato", "descriptive", "feature")]),
    ("Nice job on that order! The next bowl is ready; take it to the serving point.", [("Nice job on that order!", "evaluative", "trajectory"), ("The next bowl is ready", "descriptive", "feature"), ("take it to the serving point", "imperative", "action")]),
]

UNCERTAIN = [
    ("Hi, how is your day going?", "out_of_domain", "Social conversation without a game referent."),
    ("I like making soup at home on Sundays.", "out_of_domain", "A personal cooking preference, not this game's reward."),
    ("Please help me with my maths homework.", "out_of_domain", "An unrelated request despite imperative syntax."),
    ("My laptop is running out of battery.", "out_of_domain", "Device status, not kitchen state."),
    ("Thanks for your time today.", "out_of_domain", "Social thanks with no game action specified."),
    ("Did you see the football match last night?", "out_of_domain", "External conversation question."),
    ("The soup I bought yesterday tasted nice.", "out_of_domain", "Real-world food evaluation, not agent feedback."),
    ("Can I change the page background?", "out_of_domain", "Interface question rather than kitchen guidance."),
    ("qzxv qzxv", "uninterpretable", "No meaningful task vocabulary."),
    ("?!?", "uninterpretable", "Punctuation with no recoverable instruction."),
    ("Over there.", "ambiguous", "Direction with unresolved target and intent."),
    ("Another one?", "ambiguous", "Missing object and conversational intent."),
    ("Maybe later.", "ambiguous", "Missing requested action and context."),
    ("This time?", "ambiguous", "Unspecified timing question."),
    ("I see.", "ambiguous", "Acknowledgment supplies no task judgment or action."),
    ("What about that?", "ambiguous", "Referent and intended feedback are missing."),
    ("More soup?", "pragmatically_ambiguous", "Could ask quantity, request production, or suggest an order."),
    ("Is that what you wanted?", "pragmatically_ambiguous", "Question about an unknown intention and referent."),
    ("Just great, now we are stuck.", "sarcasm_ambiguous", "Literal praise conflicts with a likely complaint."),
    ("You might want a bowl.", "pragmatically_ambiguous", "Could describe a preference or suggest an action."),
]

def main():
    prior = json.loads((ROOT / 'frozen-v2.json').read_text(encoding='utf-8'))
    old_texts = set()
    for file in ('frozen-v2.json', 'frozen-next-v2.json'):
        old = json.loads((ROOT / file).read_text(encoding='utf-8'))
        old_texts.update(row['text'].strip().casefold() for group in ('singles', 'mixed', 'uncertain') for row in old[group])
    singles = []
    for label, source in [('evaluative', EVALUATIVE), ('imperative', IMPERATIVE), ('descriptive', DESCRIPTIVE)]:
        rows = [line.strip().split('|') for line in source.strip().splitlines()]
        assert len(rows) == 50, (label, len(rows))
        for n, (text, grounding, stratum) in enumerate(rows, 1):
            singles.append({'id': f'R3{label[0].upper()}{n:03}', 'text': text, 'speech_act': label,
                            'grounding': grounding, 'stratum': stratum, 'origin': 'synthetic_acceptance_agent_round3'})
    mixed = [{'id': f'R3M{n:03}', 'text': text, 'expected_speech_acts': sorted({c[1] for c in clauses}),
              'clauses': [{'text': c[0], 'speech_act': c[1], 'grounding': c[2]} for c in clauses],
              'origin': 'synthetic_acceptance_agent_round3'} for n, (text, clauses) in enumerate(MIXED, 1)]
    uncertain = [{'id': f'R3U{n:03}', 'text': text, 'kind': kind, 'reason': reason,
                  'expected_policy': 'abstain_or_explicit_uncertainty_no_automatic_learning',
                  'origin': 'synthetic_acceptance_agent_round3'} for n, (text, kind, reason) in enumerate(UNCERTAIN, 1)]
    seen = set()
    for row in singles + mixed + uncertain:
        normalized = row['text'].strip().casefold()
        assert normalized not in old_texts, row['id']
        assert normalized not in seen, row['id']
        seen.add(normalized)
    dataset = {
        'schema_version': 1, 'dataset_id': 'durf-independent-synthetic-acceptance-20260906-round3',
        'source': 'AI-authored synthetic only; no human participant data',
        'independence': {
            'candidate_training_text_read': False, 'candidate_generator_read': False,
            'candidate_inference_before_freeze': False,
            'previous_acceptance_now_development': True,
            'authorship': 'Independently authored common kitchen feedback; no exact sentence from the two earlier sets reused; no candidate train/dev or generator consulted.',
            'use': 'one final evaluation after third candidate and inference code freeze; no tuning on this set',
            'limitation': 'independent authorship within the same AI session, not independently human-validated ground truth',
        },
        'contract': prior['contract'], 'predeclared_checks': prior['predeclared_checks'],
        'singles': singles, 'mixed': mixed, 'uncertain': uncertain,
    }
    target = ROOT / 'frozen-round3.json'
    assert not target.exists()
    content = (json.dumps(dataset, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    target.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    summary = {'dataset_id': dataset['dataset_id'], 'sha256': sha,
               'single_count': len(singles), 'speech_act_counts': dict(Counter(r['speech_act'] for r in singles)),
               'grounding_counts': dict(Counter(r['grounding'] for r in singles)),
               'mixed_count': len(mixed), 'mixed_clause_count': sum(len(r['clauses']) for r in mixed),
               'uncertain_count': len(uncertain), 'independence': dataset['independence'],
               'predeclared_checks': dataset['predeclared_checks']}
    (ROOT / 'seal-round3.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'frozen-round3.sha256').write_text(sha+'  frozen-round3.json\n', encoding='ascii')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
