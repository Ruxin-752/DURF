import { describe, expect, it } from 'vitest';
import {
  humanActionForJointStep,
  movementDirectionForKey,
  shouldIgnoreGameHotkeys,
} from '../lib/game-hotkeys';

function mockTarget(tag: string): EventTarget {
  return {
    closest: (selector: string) =>
      selector
        .split(',')
        .map((part) => part.trim())
        .includes(tag)
        ? { tag }
        : null,
  } as unknown as EventTarget;
}

describe('game keyboard focus guard', () => {
  it.each(['summary', 'details', 'input', 'textarea', 'button'])(
    'does not capture Space from %s',
    (tag) => {
      expect(shouldIgnoreGameHotkeys(mockTarget(tag))).toBe(true);
    },
  );

  it('keeps game hotkeys active on non-interactive content', () => {
    expect(shouldIgnoreGameHotkeys(mockTarget('main'))).toBe(false);
    expect(shouldIgnoreGameHotkeys(null)).toBe(false);
  });
});

describe('fixed-rate human movement sampling', () => {
  it('maps WASD and arrow keys to the same four movement actions', () => {
    expect(movementDirectionForKey('w')).toBe('up');
    expect(movementDirectionForKey('ArrowUp')).toBe('up');
    expect(movementDirectionForKey('d')).toBe('right');
    expect(movementDirectionForKey('Enter')).toBeNull();
  });

  it('keeps a held direction active on every joint step without key repeat', () => {
    const heldKeys = new Set(['d']);
    expect(humanActionForJointStep('right', heldKeys)).toBe('right');
    expect(humanActionForJointStep('stay', heldKeys)).toBe('right');
    expect(humanActionForJointStep('stay', heldKeys)).toBe('right');
    heldKeys.delete('d');
    expect(humanActionForJointStep('stay', heldKeys)).toBe('stay');
  });

  it('gives a queued one-shot interaction priority over held movement', () => {
    expect(humanActionForJointStep('interact', new Set(['a']))).toBe('interact');
  });

  it('uses the most recently pressed movement key when several are held', () => {
    const heldKeys = new Set(['w', 'd']);
    expect(humanActionForJointStep('stay', heldKeys)).toBe('right');
    heldKeys.delete('d');
    expect(humanActionForJointStep('stay', heldKeys)).toBe('up');
  });
});
