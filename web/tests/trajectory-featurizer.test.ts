import { describe, expect, it } from 'vitest';

import { createGameState, type GameState } from '../lib/game';
import {
  appendTrajectoryStep,
  featurizeTrajectorySteps,
  normalizeTrajectoryFeatures,
  recordTrajectoryStep,
} from '../lib/trajectory-featurizer';

function state(overrides: Partial<GameState> = {}): GameState {
  return { ...createGameState('running'), ...overrides };
}

describe('paper trajectory feature bridge', () => {
  it('matches the Python transition rules for pickup, potting, serving, and wait', () => {
    const initial = state();
    const picked = state({
      partner: { ...initial.partner, held: 'tomato' },
      pot: { ...initial.pot, stage: 'filling' },
    });
    const potted = state({
      partner: { ...initial.partner, held: null },
      pot: { ...initial.pot, stage: 'filling', tomatoes: 1 },
    });
    const holdingSoup = state({ partner: { ...initial.partner, held: 'soup' } });
    const served = state({
      partner: { ...initial.partner, held: null },
      score: 20,
      ordersCompleted: 1,
    });

    expect(
      featurizeTrajectorySteps([
        recordTrajectoryStep(initial, picked, 'interact', 'stay'),
        recordTrajectoryStep(picked, potted, 'interact', 'stay'),
        recordTrajectoryStep(holdingSoup, served, 'interact', 'stay', 20),
        recordTrajectoryStep(served, served, 'stay', 'stay'),
      ]),
    ).toMatchObject({
      ingredient_tomato: 1,
      pick_tomato: 1,
      adds_needed_tomato: 1,
      serve_ready_soup: 1,
      time_cost: 1,
    });
  });

  it('keeps pot facts as window indicators and L1-normalizes the frozen input', () => {
    const before = state();
    const ready = state({ pot: { ...before.pot, stage: 'ready' } });
    const counts = featurizeTrajectorySteps([
      recordTrajectoryStep(before, ready, 'right', 'stay'),
      recordTrajectoryStep(ready, ready, 'left', 'stay'),
    ]);

    expect(counts.soup_ready).toBe(1);
    expect(counts.dish_needed_for_ready_soup).toBe(1);
    expect(Object.values(normalizeTrajectoryFeatures(counts)).reduce((a, b) => a + Math.abs(b), 0)).toBeCloseTo(1);
  });

  it('bounds the recent trajectory window', () => {
    const current = state();
    const step = recordTrajectoryStep(current, current, 'stay', 'stay');
    expect(appendTrajectoryStep([step, step], step, 2)).toHaveLength(2);
    expect(() => appendTrajectoryStep([], step, 0)).toThrow('positive integer');
  });
});
