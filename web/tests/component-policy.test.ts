import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';
import { configuredFeedbackRoute } from '../lib/feedback-route-config';

const component = readFileSync(
  join(import.meta.dirname, '..', 'components', 'kitchen-game-app.tsx'),
  'utf8',
);
const environmentExample = readFileSync(join(import.meta.dirname, '..', '.env.example'), 'utf8');

describe('hidden feedback-route policy', () => {
  it('uses deployment configuration without exposing a route selector', () => {
    expect(configuredFeedbackRoute('')).toBe('route1');
    expect(configuredFeedbackRoute('route1')).toBe('route1');
    expect(configuredFeedbackRoute('route2')).toBe('route2');
    expect(() => configuredFeedbackRoute('invalid')).toThrow(
      'NEXT_PUBLIC_FEEDBACK_ROUTE must be route1 or route2',
    );
    expect(component).toContain('const DEPLOYMENT_FEEDBACK_ROUTE = configuredFeedbackRoute()');
    expect(component).not.toContain('setRoute');
    expect(component).not.toContain('route-switch');
    expect(component).not.toContain('Feedback update route');
    expect(environmentExample).toContain('NEXT_PUBLIC_FEEDBACK_ROUTE=route1');
  });

  it('retains an executable Route2 update path and its research trace', () => {
    expect(component).toContain("if (DEPLOYMENT_FEEDBACK_ROUTE === 'route1')");
    expect(component).toContain('models.route2(utterance');
    expect(component).toContain('applyRoute2Gaussian(prior, prediction.weights, 2)');
    expect(component).toContain('10-model ensemble → 53D reward vector');
  });
});

describe('pagehide queue lifecycle', () => {
  it('does not close the queue when the page enters BFCache', () => {
    const start = component.indexOf('const handlePageHide');
    const end = component.indexOf("window.addEventListener('pagehide'", start);
    const handler = component.slice(start, end);

    expect(handler).toContain('(event: PageTransitionEvent)');
    expect(handler).toContain('if (!event.persisted) queue.end(gameSummary(gameRef.current));');
    expect(handler).toContain('queue.flushWithBeacon();');
  });
});

describe('experimental classifier disclosure', () => {
  it('marks the opt-in shadow as non-production and explains its live effect', () => {
    expect(component).toContain('Experimental shadow classifier');
    expect(component).toContain('Not production. Scores are not independently calibrated.');
    expect(component).toContain('drives the displayed result and Route1 update');
    expect(component).toContain("{uncertain ? 'Uncertain' : copy.name}");
  });
});
