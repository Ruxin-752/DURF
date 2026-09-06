import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  applyRoute1PaperUpdate,
  applyRoute2Gaussian,
  createBrowserModels,
  createFullGaussianPrior,
  createIndependentGaussianPrior,
} from '../lib/browser-models';
import { chooseAiDecision, createGameState } from '../lib/game';
import { REWARD_FEATURES, type RewardWeights } from '../lib/subgoal-policy';

const modelRoot = join(import.meta.dirname, '..', 'public', 'models');
const browserModels = createBrowserModels(
  JSON.parse(readFileSync(join(modelRoot, 'feedback-form-v3.json'), 'utf8')),
  JSON.parse(readFileSync(join(modelRoot, 'route2-v5.json'), 'utf8')),
);

function weightsFrom(mean: number[]): RewardWeights {
  return Object.fromEntries(
    REWARD_FEATURES.map((feature, index) => [feature, mean[index]]),
  );
}

describe('feedback posterior to live AI behavior', () => {
  const forkState = {
    ...createGameState('running'),
    partner: {
      ...createGameState('running').partner,
      x: 1,
      y: 4,
      facing: 'left' as const,
    },
  };

  it('uses an accepted Route1 posterior in both subgoal and atomic action selection', () => {
    const before = chooseAiDecision(forkState, {});
    const prediction = browserModels.classify('Please take an onion.');
    const update = applyRoute1PaperUpdate(
      createFullGaussianPrior([...REWARD_FEATURES]),
      {
        feedbackForm: prediction.label,
        feedbackFormConfidence: prediction.confidence,
        feedbackFormThreshold: prediction.threshold,
        feedbackFormAbstained: prediction.abstained,
        actionFeatures: { pick_onion: 1 },
        valence: 30,
        basePrecision: 2,
      },
    );
    const after = chooseAiDecision(forkState, weightsFrom(update.state.mean));

    expect(prediction).toMatchObject({ label: 'imperative', abstained: false });
    expect(update.status).toBe('updated');
    expect(before).toMatchObject({
      chosenSubgoal: 'GET_TOMATO',
      action: 'interact',
    });
    expect(after).toMatchObject({
      chosenSubgoal: 'GET_ONION',
      action: 'right',
      decisionSource: 'reward_argmax',
      usedLearnedWeights: true,
    });
  });

  it('keeps behavior unchanged when the classifier gate rejects feedback', () => {
    const prior = createFullGaussianPrior([...REWARD_FEATURES]);
    const update = applyRoute1PaperUpdate(prior, {
      feedbackForm: 'imperative',
      feedbackFormConfidence: 0.4,
      feedbackFormThreshold: 0.55,
      feedbackFormAbstained: true,
      actionFeatures: { pick_onion: 1 },
      valence: 30,
    });

    expect(update.status).toBe('rejected');
    expect(update.state).toBe(prior);
    expect(chooseAiDecision(forkState, weightsFrom(update.state.mean))).toMatchObject({
      chosenSubgoal: 'GET_TOMATO',
      action: 'interact',
    });
  });

  it('uses a Route2 posterior in the same live policy adapter', () => {
    const observation = Object.fromEntries(
      REWARD_FEATURES.map((feature) => [feature, feature === 'pick_onion' ? 4 : 0]),
    );
    const update = applyRoute2Gaussian(
      createIndependentGaussianPrior([...REWARD_FEATURES]),
      observation,
      2,
    );
    const decision = chooseAiDecision(forkState, weightsFrom(update.state.mean));

    expect(decision).toMatchObject({
      chosenSubgoal: 'GET_ONION',
      action: 'right',
      decisionSource: 'reward_argmax',
    });
  });
});
