import { describe, expect, it } from 'vitest';
import {
  gameFeatureCounts,
  lowConfidenceRouteMessage,
  splitFeedbackPhrases,
} from '../lib/route-inputs';
import { createGameState, type GameEvent } from '../lib/game';

describe('mixed-language feedback splitting', () => {
  it('splits Chinese commas and connective words into grounded phrases', () => {
    expect(
      splitFeedbackPhrases('刚才路线很好，但是请拿盘子，然后去出餐。'),
    ).toEqual(['刚才路线很好，', '请拿盘子，', '去出餐。']);
  });

  it('supports English contrast and sequence connectives', () => {
    expect(
      splitFeedbackPhrases('That route was good, but take a dish and then serve it.'),
    ).toEqual(['That route was good,', 'take a dish', 'serve it.']);
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
