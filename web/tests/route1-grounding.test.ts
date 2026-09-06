import { describe, expect, it } from 'vitest';
import { createFullGaussianPrior } from '../lib/browser-models';
import { chooseAiDecision, createGameState, stepGame, type GameState } from '../lib/game';
import { applyGroundedPragmaticFeedback } from '../lib/pragmatic-route1';
import { groundRoute1Feedback, type Route1GroundingPrediction } from '../lib/route1-grounding';
import { REWARD_FEATURES, type RewardWeights } from '../lib/subgoal-policy';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

const initial = createGameState('running');
const fork: GameState = { ...initial, partner: { ...initial.partner, x: 1, y: 4, facing: 'left' } };
const prior = () => createFullGaussianPrior([...REWARD_FEATURES]);
const weights = (mean: number[]): RewardWeights => Object.fromEntries(REWARD_FEATURES.map((feature, index) => [feature, mean[index]]));
function bind(text: string, label: Route1GroundingPrediction['label'] = 'action', state = fork, trajectoryFeatures = {}) {
  return groundRoute1Feedback({ text, grounding: { label, confidence: 0.9, modelHash: 'independent-grounding-test' }, state, trajectoryFeatures });
}
function update(text: string, state = fork) {
  return applyGroundedPragmaticFeedback(prior(), { grounding: bind(text, 'action', state), sentiment: scoreVaderSentiment(text).compound });
}

describe('independent Route1 kitchen grounding and live policy', () => {
  it('binds an onion counterfactual to the exact executable candidate and changes the atomic action', () => {
    const before = chooseAiDecision(fork);
    const result = update('Pick an onion.');
    const candidate = before.ranking.find((item) => item.subgoal === 'GET_ONION');
    expect(result.grounding).toMatchObject({ status: 'grounded', label: 'action', selectedSubgoal: 'GET_ONION', modelHash: 'independent-grounding-test' });
    expect(result.targetFeatures).toEqual(candidate?.features);
    expect(result.targetFeatures).not.toHaveProperty('pick_tomato');
    expect(result.targetFeatures).not.toHaveProperty('near_pot');
    expect(before).toMatchObject({ chosenSubgoal: 'GET_TOMATO', action: 'interact' });
    const after = chooseAiDecision(fork, weights(result.state.mean));
    expect(after).toMatchObject({ chosenSubgoal: 'GET_ONION', action: 'right', usedLearnedWeights: true });
    expect(stepGame(fork, before.action, 'stay').partner.held).toBe('tomato');
    expect(stepGame(fork, after.action, 'stay').partner).toMatchObject({ x: 2, y: 4, held: null });
  });

  it('uses the other ingredient for the minimal text counterfactual', () => {
    const onion = update('Pick an onion.');
    const tomato = update('Pick a tomato.');
    expect(chooseAiDecision(fork, weights(onion.state.mean)).chosenSubgoal).toBe('GET_ONION');
    expect(chooseAiDecision(fork, weights(tomato.state.mean)).chosenSubgoal).toBe('GET_TOMATO');
  });

  it.each(["Don't pick onions.", 'Do not get an onion.', 'Avoid picking onions.', 'Stop picking onions.'])('keeps raw sentiment and separately handles prohibition: %s', (text) => {
    const result = update(text);
    expect(result.status).toBe('updated');
    expect(result.rawSentiment).toBe(scoreVaderSentiment(text).compound);
    expect(result.effectiveValence).toBe(-30);
    expect(result.valenceSource).toBe('explicit_prohibition');
    expect(result.pragmatic).toBeNull();
    expect(result.delta.pick_onion).toBeLessThan(0);
    expect(chooseAiDecision(fork, weights(result.state.mean)).chosenSubgoal).not.toBe('GET_ONION');
  });

  it('uses put-in-pot features for a held ingredient instead of rewarding pickup', () => {
    const state = { ...fork, partner: { ...fork.partner, held: 'onion' as const } };
    const result = update('Put the onion in the pot.', state);
    expect(result.status).toBe('updated');
    expect(result.grounding.selectedSubgoal).toBe('PUT_ONION_IN_POT');
    expect(result.targetFeatures).toHaveProperty('adds_needed_onion', 1);
    expect(result.targetFeatures).not.toHaveProperty('pick_onion');
    expect(chooseAiDecision(state, weights(result.state.mean)).chosenSubgoal).toBe('PUT_ONION_IN_POT');
    expect(bind('Pick an onion.', 'action', state).reason).toBe('infeasible_action_reference');
  });

  it('grounds ready-soup pickup and serving in the correct held-object states', () => {
    const ready = { ...fork, partner: { ...fork.partner, held: 'dish' as const }, pot: { ...fork.pot, stage: 'ready' as const, tomatoes: 2, onions: 1 } };
    const pickup = update('Scoop the soup.', ready);
    expect(pickup.grounding.selectedSubgoal).toBe('PICKUP_SOUP');
    expect(pickup.targetFeatures).toHaveProperty('pick_ready_soup', 1);
    expect(chooseAiDecision(ready, weights(pickup.state.mean)).chosenSubgoal).toBe('PICKUP_SOUP');
    const holdingSoup = { ...ready, partner: { ...ready.partner, held: 'soup' as const } };
    const serve = update('Serve the soup.', holdingSoup);
    expect(serve.grounding.selectedSubgoal).toBe('SERVE_SOUP');
    expect(chooseAiDecision(holdingSoup, weights(serve.state.mean)).chosenSubgoal).toBe('SERVE_SOUP');
    expect(bind('Serve the soup.').reason).toBe('infeasible_action_reference');
  });

  it('takes historical credit solely from the recent trajectory, independently of UI grammar', () => {
    const result = bind('That was good.', 'trajectory', fork, { pick_tomato: 2, pot_cooking: 0, invalid: 99 });
    expect(result.targetFeatures).toEqual({ pick_tomato: 2 });
    expect(result.selectedSubgoal).toBeNull();
    expect(bind('Good job.', 'trajectory').reason).toBe('empty_target_features');
  });

  it('uses the original full feature complement before normalization', () => {
    const result = update('Pick an onion.');
    const expected = Object.fromEntries(REWARD_FEATURES.filter(feature => !(feature in result.targetFeatures)).map(feature => [feature, 1]));
    expect(result.grounding.adaptation).toBe('paper-full-feature-complement-v1');
    expect(result.grounding.pragmaticAlternatives).toEqual(expected);
    for (const feature of ['collision_risk', 'blocks_partner_on_ring', 'near_pot', 'avoids_duplicate_human_task']) {
      if (!(feature in result.targetFeatures)) expect(result.delta[feature]).toBeLessThan(0);
    }
  });

  it.each(['Move left.', 'Pick onion and tomato.', 'Get the onion instead of tomato.', 'Do something.', 'Please do not avoid onions.', 'Do not avoid picking onions.', 'Put the onion on the floor.'])('rejects unresolved or ambiguous actions without changing the posterior: %s', (text) => {
    const initialPrior = prior();
    const grounding = bind(text);
    const result = applyGroundedPragmaticFeedback(initialPrior, { grounding, sentiment: scoreVaderSentiment(text).compound });
    expect(result.status).toBe('rejected');
    expect(result.state).toBe(initialPrior);
    expect(result.delta).toEqual({});
    expect(result.pragmatic).toBeNull();
  });

  it('rejects state-only descriptions and accepts explicit feature preferences', () => {
    expect(bind('The onion dispenser is on the left.', 'feature').reason).toBe('empty_target_features');
    expect(bind('The pot contains two tomatoes.', 'feature').reason).toBe('empty_target_features');
    expect(bind('Onions are useful.', 'feature').targetFeatures).toEqual({ ingredient_onion: 1 });
  });

  it('rejects low-score independent grounding and unsupported language', () => {
    expect(groundRoute1Feedback({ text: 'Pick an onion.', grounding: { label: 'action', confidence: 0.2 }, state: fork }).reason).toBe('grounding_low_confidence');
    expect(bind('拿一个洋葱。').reason).toBe('unsupported_language');
  });
});
