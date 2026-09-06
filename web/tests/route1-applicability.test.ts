import { describe, expect, it } from 'vitest';
import { createFullGaussianPrior } from '../lib/browser-models';
import { createGameState } from '../lib/game';
import { applyGroundedPragmaticFeedback } from '../lib/pragmatic-route1';
import { groundRoute1Feedback, type Route1GroundingPrediction } from '../lib/route1-grounding';
import { REWARD_FEATURES } from '../lib/subgoal-policy';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

const state = createGameState('running');
const recent = { pick_onion: 1, moves_toward_needed_object: 1 };
function apply(text: string, label: Route1GroundingPrediction['label'] = 'trajectory', trajectoryFeatures: Record<string, number> = recent) {
  const prior = createFullGaussianPrior([...REWARD_FEATURES]);
  const grounding = groundRoute1Feedback({ text, grounding: { label, confidence: 0.99 }, state, trajectoryFeatures });
  return { prior, result: applyGroundedPragmaticFeedback(prior, { grounding, sentiment: scoreVaderSentiment(text).compound }) };
}

describe('independent grounding applicability before reward learning', () => {
  // These are exposed development regressions from the stateful audit, not a
  // new blind test. No frozen-next/round2 test cases are read by this suite.
  it.each([
    ['U003', 'My favorite movie has a chef in it.'],
    ['U005', 'Thank you for chatting with me.'],
    ['U012', 'That bowl.'],
    ['U016', 'Really?'],
    ['U019', 'Are you going to serve that?'],
  ])('does not apply exposed uncertain case %s even with recent real features', (_, text) => {
    const { prior, result } = apply(text);
    expect(result.status).toBe('rejected');
    expect(result.state).toBe(prior);
    expect(result.delta).toEqual({});
    expect(result.pragmatic).toBeNull();
  });

  it.each(['Great job!', 'Well done.', 'Good work.', 'That was good.', 'That was not good.', 'You did well.', 'Taking onions was good.', 'Thank you for picking the onion.', 'You picked an onion.'])('retains clear recent-play feedback: %s', (text) => {
    const { result } = apply(text);
    expect(result.status).toBe('updated');
    expect(result.targetFeatures).toEqual(recent);
  });

  it.each([
    ['Great job!', 'Great job chatting with me.'],
    ['Well done.', 'Well done in that movie.'],
    ['That was good.', 'Was that good?'],
    ['You picked an onion.', 'Did you pick an onion?'],
    ['Taking onions was good.', 'Those onions.'],
    ['Thank you for picking the onion.', 'Thank you for chatting with me.'],
  ])('distinguishes actionable appraisal from missing context: %s / %s', (accepted, rejected) => {
    expect(apply(accepted).result.status).toBe('updated');
    expect(apply(rejected).result.status).toBe('rejected');
  });

  it('requires both a clear appraisal and matching observed kitchen context', () => {
    expect(apply('Great job!', 'trajectory', {}).result.reason).toBe('empty_target_features');
    expect(apply('Taking tomatoes was good.').result.reason).toBe('trajectory_reference_not_observed');
    expect(apply('Taking tomatoes was good.', 'trajectory', { pick_tomato: 1, moves_toward_needed_object: 1 }).result.status).toBe('updated');
  });

  it('keeps clear polite action requests while rejecting requests for information', () => {
    expect(apply('Could you please pick an onion?', 'action').result.status).toBe('updated');
    expect(apply('Can you tell me how to pick an onion?', 'action').result.reason).toBe('unresolved_question_reference');
    expect(apply('Where is the onion?', 'feature').result.reason).toBe('unresolved_question_reference');
  });

  it.each(['trajectory', 'feature', 'action'] as const)('does not let the predicted class bypass domain and reference checks: %s', (label) => {
    expect(apply('That onion.', label).result.status).toBe('rejected');
    expect(apply('The onions in my movie are great.', label).result.status).toBe('rejected');
  });
});
