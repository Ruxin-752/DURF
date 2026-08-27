import { describe, expect, it } from 'vitest';
import { shouldIgnoreGameHotkeys } from '../lib/game-hotkeys';

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
