import { describe, expect, it } from 'vitest';

import { createGameState, startGame, stepGame } from '../lib/game';
import {
  aiExecutionResult,
  canonicalReceiptValue,
  counterfactualBehaviorChanged,
  observeAiRelevantOutcome,
} from '../lib/policy-execution-receipt';

describe('counterfactual policy execution receipts', () => {
  it('does not attribute the same human and environment transition to the AI policy', () => {
    const started = startGame(createGameState());
    const previous = {
      ...started,
      pot: {
        stage: 'cooking' as const,
        secondsRemaining: 5,
        tomatoes: 2,
        onions: 1,
      },
    };

    const actual = stepGame(previous, 'stay', 'right');
    const counterfactual = stepGame(previous, 'stay', 'right');
    const comparison = counterfactualBehaviorChanged(actual, counterfactual);

    expect(actual.player).not.toEqual(previous.player);
    expect(actual.pot.secondsRemaining).toBe(4);
    expect(comparison.behaviorChangeConfirmed).toBe(false);
    expect(comparison.actualObservableOutcome).toEqual(
      comparison.counterfactualObservableOutcome,
    );
  });

  it('confirms a change only when same-state, same-human-action worlds diverge', () => {
    const previous = startGame(createGameState());
    const humanAction = 'stay' as const;
    const actual = stepGame(previous, 'right', humanAction);
    const counterfactual = stepGame(previous, 'stay', humanAction);
    const comparison = counterfactualBehaviorChanged(actual, counterfactual);

    expect(comparison.behaviorChangeConfirmed).toBe(true);
    expect(comparison.actualObservableOutcome.partner).not.toEqual(
      comparison.counterfactualObservableOutcome.partner,
    );
  });

  it('ignores non-observable policy metadata such as a changed subgoal name', () => {
    const previous = startGame(createGameState());
    const next = stepGame(previous, 'stay', 'left');

    expect(counterfactualBehaviorChanged(next, next).behaviorChangeConfirmed).toBe(false);
    expect(observeAiRelevantOutcome(next)).not.toHaveProperty('chosenSubgoal');
  });

  it('compares observable objects independently of property insertion order', () => {
    expect(canonicalReceiptValue({ b: 2, a: { d: 4, c: 3 } })).toBe(
      canonicalReceiptValue({ a: { c: 3, d: 4 }, b: 2 }),
    );
  });

  it('classifies blocked and successful movement from the realized transition', () => {
    const previous = startGame(createGameState());
    const moved = stepGame(previous, 'right', 'stay');
    const blocked = stepGame(previous, 'up', 'stay');

    expect(aiExecutionResult(previous, moved, 'right')).toEqual({
      actuallyExecuted: true,
      executionOutcome: 'moved',
    });
    expect(aiExecutionResult(previous, blocked, 'up')).toEqual({
      actuallyExecuted: false,
      executionOutcome: 'movement_blocked',
    });
  });
});
