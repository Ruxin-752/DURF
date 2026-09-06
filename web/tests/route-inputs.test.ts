import { describe, expect, it } from 'vitest';
import {
  gameFeatureCounts,
  lowConfidenceRouteMessage,
  splitFeedbackPhrases,
  toResearchPrediction,
} from '../lib/route-inputs';
import type { FeedbackFormPrediction } from '../lib/browser-models';
import { createGameState, type GameEvent } from '../lib/game';

describe('mixed-language feedback splitting', () => {
  it('splits Chinese contrast while preserving a continuous command plan', () => {
    expect(
      splitFeedbackPhrases('刚才路线很好，但是请拿盘子，然后去出餐。'),
    ).toEqual(['刚才路线很好，', '请拿盘子，然后去出餐。']);
  });

  it('splits English contrast while preserving sequential actions', () => {
    expect(
      splitFeedbackPhrases('That route was good, but take a dish and then serve it.'),
    ).toEqual(['That route was good,', 'take a dish and then serve it.']);
  });

  it('splits independent English sentences and line breaks', () => {
    expect(
      splitFeedbackPhrases(
        'That last route was bad. Please take a dish instead.\nThe pot is empty.',
      ),
    ).toEqual([
      'That last route was bad.',
      'Please take a dish instead.',
      'The pot is empty.',
    ]);
  });

  it('keeps polite commands intact across ordinary commas', () => {
    expect(splitFeedbackPhrases('Please, get an onion.')).toEqual([
      'Please, get an onion.',
    ]);
    expect(splitFeedbackPhrases('If possible, take a dish next.')).toEqual([
      'If possible, take a dish next.',
    ]);
  });

  it('does not split decimals or common abbreviations', () => {
    expect(splitFeedbackPhrases('Keep 2.5 tiles away. Then take a dish.')).toEqual([
      'Keep 2.5 tiles away.',
      'Then take a dish.',
    ]);
    expect(splitFeedbackPhrases('For example, e.g. take a dish next.')).toEqual([
      'For example, e.g. take a dish next.',
    ]);
    expect(splitFeedbackPhrases('In other words, i.e. stop waiting.')).toEqual([
      'In other words, i.e. stop waiting.',
    ]);
  });

  it('separates complete clauses while preserving object lists and conditions', () => {
    expect(splitFeedbackPhrases('That helped, please grab the plate.')).toEqual([
      'That helped', 'please grab the plate.',
    ]);
    expect(splitFeedbackPhrases('You did well and the pot is full.')).toEqual([
      'You did well', 'the pot is full.',
    ]);
    expect(splitFeedbackPhrases('Fetch tomatoes, onions and a dish.')).toEqual([
      'Fetch tomatoes, onions and a dish.',
    ]);
    expect(splitFeedbackPhrases('When the pot is ready, serve the soup.')).toEqual([
      'When the pot is ready, serve the soup.',
    ]);
  });
});

describe('low-confidence route copy', () => {
  it('states that low fG is diagnostic and does not gate Route2 updates', () => {
    const copy = lowConfidenceRouteMessage('route2', 0.55);
    expect(copy).toContain('fG is diagnostic only');
    expect(copy).toContain('Route 2 still updates');
    expect(copy).toContain('only Route 1 rejects');
  });

  it('states that Route1 rejects the whole utterance', () => {
    const copy = lowConfidenceRouteMessage('route1', 0.55);
    expect(copy).toContain('Route 1 rejects the whole utterance');
    expect(copy).toContain('Route 2 is not gated by fG');
  });
});

describe('phrase score provenance', () => {
  const basePrediction: FeedbackFormPrediction = {
    label: 'imperative',
    confidence: 0.8,
    probabilities: { descriptive: 0.1, evaluative: 0.1, imperative: 0.8 },
    threshold: 0.55,
    abstained: false,
    calibrated: false,
    temperatureScaled: false,
    independentlyCalibrated: false,
    calibrationVersion: null,
    modelHash: 'model',
    scoreKind: 'raw_model_softmax_score',
  };

  it('keeps raw score and threshold semantics on each experimental phrase', () => {
    expect(toResearchPrediction('take it', basePrediction)).toMatchObject({
      scoreKind: 'raw_model_softmax_score',
      thresholdPolicy: 'existing_web_preview_policy_not_validated_in_raw_score_space',
    });
  });

  it('does not change the production phrase payload shape', () => {
    const phrase = toResearchPrediction('take it', {
      ...basePrediction,
      temperatureScaled: true,
      scoreKind: 'temperature_scaled_selection_dev_probability',
    });
    expect(phrase).not.toHaveProperty('scoreKind');
    expect(phrase).not.toHaveProperty('thresholdPolicy');
  });
});

describe('structured game features', () => {
  it('uses event codes and items, so display-copy changes cannot alter features', () => {
    const events: GameEvent[] = [
      {
        code: 'pick_tomato',
        actor: 'human',
        item: 'tomato',
        message: 'The player picked up a tomato.',
      },
      {
        code: 'pickup_counter',
        actor: 'ai',
        item: 'onion',
        message: 'The AI picked up an onion from a counter.',
      },
      {
        code: 'pickup_counter',
        actor: 'human',
        item: 'dish',
        message: 'The player picked up a dish from a counter.',
      },
      {
        code: 'pickup_counter',
        actor: 'ai',
        item: 'soup',
        message: 'The AI picked up soup from a counter.',
      },
      {
        code: 'serve_correct_soup',
        actor: 'human',
        item: 'soup',
        message: 'The player delivered the correct soup.',
      },
    ];
    const state = {
      ...createGameState('running'),
      lastAction: 'Visible English copy can change independently.',
      lastStepEvents: events,
    };
    const expected = gameFeatureCounts(state);

    expect(expected).toMatchObject({
      pick_tomato: 1,
      pick_onion: 1,
      pick_dish: 1,
      pick_ready_soup: 1,
      serve_ready_soup: 1,
    });

    expect(
      gameFeatureCounts({
        ...state,
        lastAction: '界面文案已改为其他语言',
        lastStepEvents: events.map((event) => ({
          ...event,
          message: `Reworded display message for ${event.code}`,
        })),
      }),
    ).toEqual(expected);
  });

  it('does not infer actions from lastAction text', () => {
    const state = {
      ...createGameState('running'),
      lastAction: 'Picked up tomato, onion, and dish; delivered the correct soup.',
      lastStepEvents: [
        { code: 'move', actor: 'human', message: 'The player moved.' },
      ] satisfies GameEvent[],
    };

    expect(gameFeatureCounts(state)).toMatchObject({
      pick_tomato: 0,
      pick_onion: 0,
      pick_dish: 0,
      pick_ready_soup: 0,
      serve_ready_soup: 0,
    });
  });
});
