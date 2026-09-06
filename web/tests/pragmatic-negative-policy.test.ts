import { describe, expect, it } from 'vitest';
import { createFullGaussianPrior } from '../lib/browser-models';
import { chooseAiDecision, createGameState, stepGame, type GameState } from '../lib/game';
import { applyGroundedPragmaticFeedback, applyPragmaticRoute1Update } from '../lib/pragmatic-route1';
import { groundRoute1Feedback } from '../lib/route1-grounding';
import { POLICY_ACTIVE_FEATURES, REWARD_FEATURES, type RewardWeights } from '../lib/subgoal-policy';
import { featurizeTrajectorySteps, recordTrajectoryStep } from '../lib/trajectory-featurizer';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

const initial = createGameState('running');
const pickupBefore: GameState = { ...initial, partner: { ...initial.partner, x: 4, y: 4, facing: 'down' } };
const pickupAfter = stepGame(pickupBefore, 'interact', 'stay');
const trajectory = featurizeTrajectorySteps([recordTrajectoryStep(pickupBefore, pickupAfter, 'interact', 'stay')]);
// An explicitly synthetic decision snapshot, reusing the actual observed pickup
// feature vector. Existing positive task reward makes the relative ordering
// visible as a real action, instead of both negative options losing to WAIT.
const state: GameState = {
  ...initial, partner: { ...initial.partner, x: 1, y: 4, facing: 'left' },
  pot: { stage: 'cooking', tomatoes: 2, onions: 1, secondsRemaining: 15 },
};
function prior() {
  const value = createFullGaussianPrior([...REWARD_FEATURES]);
  value.mean[REWARD_FEATURES.indexOf('moves_toward_needed_object')] = 30;
  return value;
}
const weights = (mean: number[]): RewardWeights => Object.fromEntries(REWARD_FEATURES.map((feature, index) => [feature, mean[index]]));
function bind(text: string, label: 'trajectory' | 'action' = 'trajectory') {
  return groundRoute1Feedback({ text, grounding: { label, confidence: 0.99 }, state, trajectoryFeatures: trajectory });
}
function margin(mean: number[]) {
  const scores = Object.fromEntries(chooseAiDecision(state, weights(mean)).ranking.map(row => [row.subgoal, row.score]));
  return scores.GET_ONION - scores.GET_TOMATO;
}

describe('original full complement preserves criticism direction in kitchen policy', () => {
  it.each(['bad', 'terrible'])('fixes the actual action reversal caused by restricting the %s complement', adjective => {
    expect(pickupAfter.partner.held).toBe('onion');
    expect(trajectory).toEqual({ ingredient_onion: 1, pick_onion: 1, pot_empty: 1 });
    const text = `That onion pickup was ${adjective}.`;
    const grounding = bind(text);
    const sentiment = scoreVaderSentiment(text).compound;
    expect(grounding.status).toBe('grounded');
    expect(sentiment).toBeLessThan(0);
    const active = new Set<string>(POLICY_ACTIVE_FEATURES);
    const oldComplement = Object.fromEntries(chooseAiDecision(state).ranking.flatMap(row =>
      Object.keys(row.features).filter(feature => active.has(feature) && !(feature in grounding.targetFeatures)).map(feature => [feature, 1]),
    ));
    const defective = applyGroundedPragmaticFeedback(prior(), { grounding: { ...grounding, pragmaticAlternatives: oldComplement }, sentiment });
    const result = applyGroundedPragmaticFeedback(prior(), { grounding, sentiment });
    const original = applyPragmaticRoute1Update(prior(), { feedbackForm: 'evaluative', feedbackFormConfidence: 0.99, trajectoryFeatures: trajectory, sentiment });
    expect(result.state).toEqual(original.state);
    expect(margin(prior().mean)).toBe(0);
    expect(margin(defective.state.mean)).toBeGreaterThan(0);
    expect(margin(result.state.mean)).toBeLessThan(0);
    const before = chooseAiDecision(state, weights(prior().mean));
    const broken = chooseAiDecision(state, weights(defective.state.mean));
    const after = chooseAiDecision(state, weights(result.state.mean));
    expect(before).toMatchObject({ chosenSubgoal: 'GET_TOMATO', action: 'interact' });
    expect(broken).toMatchObject({ chosenSubgoal: 'GET_ONION', action: 'right' });
    expect(after).toMatchObject({ chosenSubgoal: 'GET_TOMATO', action: 'interact' });
    expect(stepGame(state, broken.action, 'stay').partner).toMatchObject({ x: 2, y: 4, held: null });
    expect(stepGame(state, after.action, 'stay').partner.held).toBe('tomato');
  });

  it.each(['That onion pickup was good.', 'You picked an onion.'])('matches original positive/neutral evidence for %s', text => {
    const grounding = bind(text);
    const sentiment = scoreVaderSentiment(text).compound;
    const result = applyGroundedPragmaticFeedback(prior(), { grounding, sentiment });
    const original = applyPragmaticRoute1Update(prior(), { feedbackForm: 'evaluative', feedbackFormConfidence: 0.99, trajectoryFeatures: trajectory, sentiment });
    expect(result.state).toEqual(original.state);
    expect(result.effectiveValence).toBe(sentiment === 0 ? 15 : sentiment * 30);
    expect(margin(result.state.mean)).toBeGreaterThan(0);
    const decision = chooseAiDecision(state, weights(result.state.mean));
    expect(decision).toMatchObject({ chosenSubgoal: 'GET_ONION', action: 'right' });
    expect(stepGame(state, decision.action, 'stay').partner).toMatchObject({ x: 2, y: 4, held: null });
  });

  it.each(["Don't pick onions.", 'Avoid picking onions.'])('retains explicit prohibition without complementary punishment: %s', text => {
    const grounding = bind(text, 'action');
    const sentiment = scoreVaderSentiment(text).compound;
    const result = applyGroundedPragmaticFeedback(prior(), { grounding, sentiment });
    expect(result.rawSentiment).toBe(sentiment);
    expect(result.grounding.pragmaticAlternatives).toEqual({});
    expect(result.pragmatic).toBeNull();
    expect(result.effectiveValence).toBe(-30);
    expect(result.valenceSource).toBe('explicit_prohibition');
    expect(margin(result.state.mean)).toBeLessThan(0);
    const decision = chooseAiDecision(state, weights(result.state.mean));
    expect(decision).toMatchObject({ chosenSubgoal: 'GET_TOMATO', action: 'interact' });
    expect(stepGame(state, decision.action, 'stay').partner.held).toBe('tomato');
  });
});
