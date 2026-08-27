import { describe, expect, it } from 'vitest';
import {
  lowConfidenceRouteMessage,
  splitFeedbackPhrases,
} from '../lib/route-inputs';

describe('mixed-language feedback splitting', () => {
  it('splits Chinese commas and connective words into grounded phrases', () => {
    expect(
      splitFeedbackPhrases('刚才路线很好，但是请拿盘子，然后去出餐。'),
    ).toEqual(['刚才路线很好，', '请拿盘子，', '去出餐。']);
  });

  it('supports English contrast and sequence connectives', () => {
    expect(
      splitFeedbackPhrases('That route was good, but take a dish and then serve it.'),
    ).toEqual(['That route was good,', 'take a dish', 'serve it.']);
  });

  it('splits independent English sentences and line breaks', () => {
    expect(
      splitFeedbackPhrases(
        'That last route was bad. Please take a dish instead.\nThe pot is empty.',
      ),
    ).toEqual([
      'That last route was bad.',
      'Please take a dish instead.',
      'The pot is empty.',
    ]);
  });
});

describe('low-confidence route copy', () => {
  it('states that low fG is diagnostic and does not gate Route2 updates', () => {
    const copy = lowConfidenceRouteMessage('route2', 0.55);
    expect(copy).toContain('fG 仅作诊断');
    expect(copy).toContain('Route2 仍');
    expect(copy).toContain('只有 Route1 会拒绝');
  });

  it('states that Route1 rejects the whole utterance', () => {
    const copy = lowConfidenceRouteMessage('route1', 0.55);
    expect(copy).toContain('Route1 拒绝整句且不更新');
    expect(copy).toContain('Route2 不由 fG 门控');
  });
});
