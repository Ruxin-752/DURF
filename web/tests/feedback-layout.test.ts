import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const styles = readFileSync(join(import.meta.dirname, '..', 'app', 'globals.css'), 'utf8');
const component = readFileSync(
  join(import.meta.dirname, '..', 'components', 'kitchen-game-app.tsx'),
  'utf8',
);

describe('feedback probability layout', () => {
  it('places each probability bar below its label and percentage', () => {
    expect(styles).toMatch(
      /\.probability-row\s*{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s+auto;/s,
    );
    expect(styles).toMatch(
      /\.probability-track\s*{[^}]*grid-column:\s*1\s*\/\s*-1;[^}]*grid-row:\s*2;/s,
    );
  });

  it('allows long labels and uncertainty copy to wrap without covering siblings', () => {
    expect(styles).toMatch(/\.probability-row\s*>\s*span\s*{[^}]*overflow-wrap:\s*anywhere;/s);
    expect(styles).toMatch(/\.confidence-warning\s*{[^}]*overflow-wrap:\s*anywhere;/s);
    expect(styles).toMatch(/\.prediction-line strong\s*{[^}]*white-space:\s*nowrap;/s);
  });

  it('keeps technical score explanations out of the player view', () => {
    expect(component).not.toContain('Model confidence · not measured accuracy');
    expect(component).not.toContain('accuracy estimates');
    expect(component).not.toContain('diagnostic threshold');
    expect(component).not.toContain('Calibrated confidence');
  });
});
