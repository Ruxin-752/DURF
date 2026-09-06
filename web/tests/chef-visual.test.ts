import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const root = join(import.meta.dirname, '..');
const component = readFileSync(join(root, 'components', 'kitchen-game-app.tsx'), 'utf8');
const css = readFileSync(join(root, 'app', 'globals.css'), 'utf8');

describe('simple CSS chefs', () => {
  const layers = [
    'chef-body',
    'chef-face',
    'chef-hat',
  ];

  it('renders every decorative layer without adding accessible noise', () => {
    for (const layer of layers) {
      expect(component).toContain(`aria-hidden="true" className="${layer}"`);
      expect(css).toContain(`.${layer}`);
    }

    for (const removedLayer of ['chef-arms', 'chef-apron', 'chef-hair', 'chef-scarf']) {
      expect(component).not.toContain(`className="${removedLayer}"`);
    }
  });

  it('keeps the established chef container geometry and same-tile offset', () => {
    const chefRule = css.match(/\.chef \{([\s\S]*?)\n\}/u)?.[1] ?? '';
    const partnerRule = css.match(/\.chef-partner \{([\s\S]*?)\n\}/u)?.[1] ?? '';

    expect(chefRule).toContain('bottom: 6%');
    expect(chefRule).toContain('left: 12%');
    expect(chefRule).toContain('width: 72%');
    expect(chefRule).toContain('height: 88%');
    expect(partnerRule).toContain('left: 17%');
  });

  it('keeps held items above chef details and places them for every facing', () => {
    expect(css).toMatch(/\.held-item \{[\s\S]*?z-index: 12/u);
    expect(css).toContain(".chef[data-facing='left'] .held-item");
    expect(css).toContain(".chef[data-facing='up'] .held-item");
    expect(css).toContain(".chef[data-facing='down'] .held-item");
  });
});
