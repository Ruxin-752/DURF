import {
  applyRoute1PaperUpdate,
  type FullGaussianState,
  type Route1PaperObservation,
  type Route1PaperResult,
} from "./browser-models";
import type { GroundedRoute1Feedback } from './route1-grounding';

/** Sentiment is the unscaled VADER compound score, not classifier confidence. */
export interface PragmaticRoute1Observation
  extends Omit<Route1PaperObservation, "valence"> {
  sentiment: number;
}

export interface PragmaticRoute1Result extends Route1PaperResult {
  effectiveValence: number;
  literal: Route1PaperResult;
  pragmatic: Route1PaperResult | null;
}

/**
 * Sumers et al. (2021), Sections 5.1-5.2 and released observations.py:
 * sentiment * 30 (neutral -> +15), then -30 on the normalized complement.
 * This is an opt-in calculation; it does not select the deployed feedback route.
 */
export function applyPragmaticRoute1Update(
  state: FullGaussianState,
  observation: PragmaticRoute1Observation,
): PragmaticRoute1Result {
  return calculatePragmaticUpdate(state, observation);
}

function calculatePragmaticUpdate(
  state: FullGaussianState,
  observation: PragmaticRoute1Observation,
  alternatives?: Record<string, number>,
  explicitValence?: number,
): PragmaticRoute1Result {
  if (!Number.isFinite(observation.sentiment) || Math.abs(observation.sentiment) > 1) {
    throw new Error("Route1 sentiment must be a finite compound score in [-1, 1]");
  }
  const effectiveValence = explicitValence ?? (observation.sentiment === 0 ? 15 : 30 * observation.sentiment);
  const literal = applyRoute1PaperUpdate(state, {
    ...observation,
    valence: effectiveValence,
  });
  if (literal.status === "rejected") {
    return { ...literal, effectiveValence, literal, pragmatic: null };
  }

  const inverseFeatures = alternatives ?? Object.fromEntries(
    state.features
      .filter((feature) => !(feature in literal.targetFeatures))
      .map((feature) => [feature, 1]),
  );
  if (Object.keys(inverseFeatures).length === 0) {
    return { ...literal, effectiveValence, literal, pragmatic: null };
  }

  // Reuse the accepted-observation Gaussian update and its L1 normalization.
  // The internal descriptive subtype only supplies an explicit feature vector;
  // the returned top-level form and grounding still describe the actual phrase.
  const pragmatic = applyRoute1PaperUpdate(literal.state, {
    feedbackForm: "descriptive",
    feedbackFormConfidence: observation.feedbackFormConfidence,
    feedbackFormThreshold: observation.feedbackFormThreshold,
    feedbackFormAbstained: observation.feedbackFormAbstained,
    namedFeatures: inverseFeatures,
    valence: -30,
    basePrecision: observation.basePrecision,
  });

  return {
    ...literal,
    state: pragmatic.state,
    delta: Object.fromEntries(
      state.features
        .map((feature, index) => [feature, pragmatic.state.mean[index] - state.mean[index]] as const)
        .filter(([, value]) => Math.abs(value) > 1e-12),
    ),
    effectiveValence,
    literal,
    pragmatic,
  };
}

export interface GroundedPragmaticRoute1Result extends PragmaticRoute1Result {
  grounding: GroundedRoute1Feedback;
  rawSentiment: number;
  valenceSource: 'vader_compound' | 'paper_neutral_prior' | 'explicit_prohibition';
}

export interface GroundedPragmaticObservation {
  grounding: GroundedRoute1Feedback;
  sentiment: number;
  basePrecision?: number;
}

/**
 * Live API: the UI speech-act prediction is deliberately not an input.
 * Legacy form names below only select an existing Gaussian-math code path.
 * The independent grounding object is the authoritative reference for logging.
 *
 * The grounded complement follows the paper over the full reward feature set.
 * Explicit prohibitions impose -30 on the cited action and do not punish
 * alternatives. The original VADER score is retained without alteration.
 */
export function applyGroundedPragmaticFeedback(
  state: FullGaussianState,
  observation: GroundedPragmaticObservation,
): GroundedPragmaticRoute1Result {
  const { grounding, sentiment } = observation;
  if (observation.basePrecision !== undefined &&
      (!Number.isFinite(observation.basePrecision) || observation.basePrecision <= 0)) {
    throw new Error('Route1 observation precision must be finite and positive');
  }
  const feedbackForm = grounding.label === 'action' ? 'imperative'
    : grounding.label === 'trajectory' ? 'evaluative' : 'descriptive';
  const prohibit = grounding.directivePolarity === 'prohibit';
  const result = calculatePragmaticUpdate(state, {
    feedbackForm,
    feedbackFormConfidence: grounding.confidence,
    feedbackFormThreshold: grounding.threshold,
    feedbackFormAbstained: grounding.status === 'rejected',
    trajectoryFeatures: grounding.targetFeatures,
    actionFeatures: grounding.targetFeatures,
    namedFeatures: grounding.targetFeatures,
    sentiment,
    basePrecision: observation.basePrecision,
  }, grounding.pragmaticAlternatives, prohibit ? -30 : undefined);
  return {
    ...result,
    ...(grounding.status === 'rejected' ? { reason: grounding.reason } : {}),
    grounding,
    rawSentiment: sentiment,
    valenceSource: prohibit ? 'explicit_prohibition'
      : sentiment === 0 ? 'paper_neutral_prior' : 'vader_compound',
  };
}

/** Candidate phrase results are diagnostic until the entire utterance succeeds. */
export function applyGroundedPragmaticUtterance(
  state: FullGaussianState,
  observations: GroundedPragmaticObservation[],
): {
  state: FullGaussianState;
  status: 'updated' | 'rejected';
  reason?: string;
  results: GroundedPragmaticRoute1Result[];
} {
  let candidate = state;
  const results = observations.map((observation) => {
    const result = applyGroundedPragmaticFeedback(candidate, observation);
    candidate = result.state;
    return result;
  });
  const rejection = results.find((result) => result.status === 'rejected');
  if (rejection || results.length === 0) {
    return { state, status: 'rejected', reason: rejection?.reason ?? 'empty_utterance', results };
  }
  return { state: candidate, status: 'updated', results };
}
