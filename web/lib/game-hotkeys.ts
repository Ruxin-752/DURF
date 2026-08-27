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

export function shouldIgnoreGameHotkeys(target: EventTarget | null): boolean {
  if (!target) {
    return false;
  }
  const closestTarget = target as unknown as Partial<ClosestCapableTarget>;
  if (typeof closestTarget.closest !== 'function') return false;
  return Boolean(closestTarget.closest(GAME_HOTKEY_IGNORE_SELECTOR));
}
