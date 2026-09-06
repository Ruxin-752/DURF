import { describe, expect, it } from "vitest";

import { createFullGaussianPrior, type FullGaussianState } from "../lib/browser-models";
import { applyPragmaticRoute1Update } from "../lib/pragmatic-route1";

describe("paper pragmatic Route1 update", () => {
  it("updates cited features, then unmentioned features with negative sentiment", () => {
    const prior = createFullGaussianPrior(["onion", "tomato", "dish"]);
    const snapshot = structuredClone(prior);
    const result = applyPragmaticRoute1Update(prior, {
      feedbackForm: "descriptive",
      feedbackFormConfidence: 0.9,
      namedFeatures: { onion: 1, tomato: 0 },
      sentiment: 0.8,
    });
    expect(result.status).toBe("updated");
    expect(result.effectiveValence).toBe(24);
    expect(result.literal.state.mean[0]).toBeCloseTo((50 * 24) / 51);
    expect(result.state.mean[0]).toBeCloseTo(result.literal.state.mean[0]);
    expect(result.state.mean[1]).toBeCloseTo(-375 / 26);
    expect(result.state.mean[2]).toBeCloseTo(-375 / 26);
    expect(result.pragmatic?.targetFeatures).toEqual({ tomato: 1, dish: 1 });
    expect(result.targetFeatures).toEqual({ onion: 1 });
    expect(result.delta.onion).toBeCloseTo(result.state.mean[0]);
    expect(result.delta.tomato).toBeCloseTo(result.state.mean[1]);
    expect(prior).toEqual(snapshot);
  });

  it("uses +15 only for exactly neutral raw sentiment and preserves negative sentiment", () => {
    const prior = createFullGaussianPrior(["onion"]);
    const observation = {
      feedbackForm: "descriptive" as const,
      feedbackFormConfidence: 0.9,
      namedFeatures: { onion: 1 },
    };
    const neutral = applyPragmaticRoute1Update(prior, { ...observation, sentiment: 0 });
    const negative = applyPragmaticRoute1Update(prior, { ...observation, sentiment: -0.5 });
    expect(neutral.effectiveValence).toBe(15);
    expect(neutral.state.mean[0]).toBeCloseTo(750 / 51);
    expect(negative.effectiveValence).toBe(-15);
    expect(negative.state.mean[0]).toBeCloseTo(-750 / 51);
    expect(neutral.pragmatic).toBeNull();
  });

  it.each([
    { feedbackFormConfidence: 0.1 },
    { feedbackFormAbstained: true },
    { requestedReferenceSubtype: "other" as const },
    { namedFeatures: {} },
  ])("does not make either update when the literal observation is rejected: %j", (override) => {
    const prior = createFullGaussianPrior(["onion", "tomato"]);
    const result = applyPragmaticRoute1Update(prior, {
      feedbackForm: "descriptive",
      feedbackFormConfidence: 0.9,
      namedFeatures: { onion: 1 },
      sentiment: 0,
      ...override,
    });
    expect(result.status).toBe("rejected");
    expect(result.state).toBe(prior);
    expect(result.delta).toEqual({});
    expect(result.pragmatic).toBeNull();
  });

  it("uses the phrase form to select trajectory or counterfactual action grounding", () => {
    const prior = createFullGaussianPrior(["onion", "tomato"]);
    const observation = {
      feedbackFormConfidence: 0.9,
      trajectoryFeatures: { onion: 1 },
      actionFeatures: { tomato: 1 },
      sentiment: 0.5,
    };
    const evaluative = applyPragmaticRoute1Update(prior, { ...observation, feedbackForm: "evaluative" });
    const imperative = applyPragmaticRoute1Update(prior, { ...observation, feedbackForm: "imperative" });
    expect(evaluative.state.mean[0]).toBeGreaterThan(0);
    expect(evaluative.state.mean[1]).toBeLessThan(0);
    expect(imperative.state.mean[0]).toBeLessThan(0);
    expect(imperative.state.mean[1]).toBeGreaterThan(0);
    expect(imperative.feedbackForm).toBe("imperative");
    expect(imperative.effectiveReferenceSubtype).toBe("action_spatial");
  });

  it("matches direct Gaussian precision multiplication for a correlated prior", () => {
    const prior: FullGaussianState = {
      features: ["onion", "tomato"],
      mean: [3, -2],
      covariance: [[4, 1], [1, 3]],
    };
    const result = applyPragmaticRoute1Update(prior, {
      feedbackForm: "descriptive",
      feedbackFormConfidence: 0.9,
      namedFeatures: { onion: 1 },
      sentiment: 0.5,
    });
    // Invert Sigma^-1 + 2 e1 e1^T + 2 e2 e2^T independently of rank-one code.
    const p00 = 3 / 11 + 2;
    const p01 = -1 / 11;
    const p11 = 4 / 11 + 2;
    const determinant = p00 * p11 - p01 * p01;
    const covariance = [[p11 / determinant, -p01 / determinant], [-p01 / determinant, p00 / determinant]];
    const naturalMean = [1 + 2 * 15, -1 + 2 * -30];
    const mean = covariance.map((row) => row[0] * naturalMean[0] + row[1] * naturalMean[1]);
    for (let row = 0; row < 2; row += 1) {
      expect(result.state.mean[row]).toBeCloseTo(mean[row], 12);
      for (let column = 0; column < 2; column += 1) {
        expect(result.state.covariance[row][column]).toBeCloseTo(covariance[row][column], 12);
      }
    }
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, 1.01, -1.01])("rejects invalid raw sentiment %s", (sentiment) => {
    expect(() => applyPragmaticRoute1Update(createFullGaussianPrior(["onion"]), {
      feedbackForm: "descriptive",
      feedbackFormConfidence: 0.9,
      namedFeatures: { onion: 1 },
      sentiment,
    })).toThrow("compound score in [-1, 1]");
  });
});
