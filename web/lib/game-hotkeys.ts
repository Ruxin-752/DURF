import type { Direction, GameAction } from './game';

const GAME_HOTKEY_IGNORE_SELECTOR = [
  'input',
  'textarea',
  'select',
  'button',
  'a',
  'summary',
  'details',
  '[contenteditable="true"]',
  '[role="button"]',
  '[role="radio"]',
].join(', ');

interface ClosestCapableTarget {
  closest: (selector: string) => unknown;
}

const MOVEMENT_KEYS: Readonly<Record<string, Direction>> = {
  w: 'up',
  arrowup: 'up',
  a: 'left',
  arrowleft: 'left',
  s: 'down',
  arrowdown: 'down',
  d: 'right',
  arrowright: 'right',
};

export function shouldIgnoreGameHotkeys(target: EventTarget | null): boolean {
  if (!target) {
    return false;
  }
  const closestTarget = target as unknown as Partial<ClosestCapableTarget>;
  if (typeof closestTarget.closest !== 'function') return false;
  return Boolean(closestTarget.closest(GAME_HOTKEY_IGNORE_SELECTOR));
}

export function movementDirectionForKey(key: string): Direction | null {
  return MOVEMENT_KEYS[key.toLowerCase()] ?? null;
}

/**
 * Select the human action at the same fixed-rate boundary as the AI action.
 * A one-shot click or interaction wins for this step; otherwise the most
 * recently pressed movement key remains active until keyup.
 */
export function humanActionForJointStep(
  pendingAction: GameAction,
  pressedMovementKeys: ReadonlySet<string>,
): GameAction {
  if (pendingAction !== 'stay') return pendingAction;
  let heldDirection: Direction | null = null;
  for (const key of pressedMovementKeys) {
    heldDirection = movementDirectionForKey(key) ?? heldDirection;
  }
  return heldDirection ?? 'stay';
}
