import { describe, expect, it } from 'vitest';
import fixture from './fixtures/vader-python-parity.json';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

describe('official Python VADER 3.3.2 browser parity', () => {
  it.each(fixture.cases)('$text', ({ text, scores }) => {
    expect(scoreVaderSentiment(text)).toEqual(scores);
  });
  it('preserves neutral commands as zero sentiment, leaving prohibition to grounding', () => {
    expect(scoreVaderSentiment('Do not pick onions.').compound).toBe(0);
    expect(scoreVaderSentiment('Not good.').compound).toBeLessThan(0);
    expect(scoreVaderSentiment('Very GOOD!!!').compound).toBeGreaterThan(scoreVaderSentiment('good').compound);
  });
});
