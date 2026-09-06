import type { GroundedPragmaticRoute1Result } from './pragmatic-route1';

type FeatureVector = Record<string, number>;
interface WithFeatureVectors {
  targetFeatures: FeatureVector;
  pragmaticAlternatives: FeatureVector;
}
export interface IndexedGroundingFeatureTable {
  groundingVectorEncoding: 'indexed-dense-v1';
  groundingFeatureNames: string[];
  groundingFeatureVectors: Record<string, (number | null)[]>;
}
export interface GroundingModelContext {
  modelHash?: string;
  threshold: number;
  adaptation: string;
}
export interface GroundingModelContextTable {
  groundingModelContexts: Record<string, GroundingModelContext>;
}

/** Lossless, payload-local vector sharing; no rounding, pruning, or truncation. */
export function compactGroundingPredictions<T extends WithFeatureVectors>(predictions: readonly T[]) {
  const groundingFeatureNames = [...new Set(predictions.flatMap(({ targetFeatures, pragmaticAlternatives }) =>
    [...Object.keys(targetFeatures), ...Object.keys(pragmaticAlternatives)]))].sort();
  const groundingFeatureVectors: Record<string, (number | null)[]> = {};
  const references = new Map<string, string>();

  function intern(vector: FeatureVector): string {
    if (Object.values(vector).some((value) => !Number.isFinite(value))) {
      throw new Error('Research grounding features must contain finite numeric values');
    }
    const values = groundingFeatureNames.map((name) => Object.hasOwn(vector, name) ? vector[name] : null);
    // Preserve the distinction between a present zero and a missing property.
    // Negative zero also receives its own identity before JSON serialization.
    const key = JSON.stringify(values.map((value) => Object.is(value, -0) ? '-0' : value));
    const existing = references.get(key);
    if (existing !== undefined) return existing;
    const reference = `v${references.size}`;
    references.set(key, reference);
    groundingFeatureVectors[reference] = values;
    return reference;
  }

  const groundingPredictions = predictions.map(({ targetFeatures, pragmaticAlternatives, ...prediction }) => ({
    ...prediction,
    targetFeaturesRef: intern(targetFeatures),
    pragmaticAlternativesRef: intern(pragmaticAlternatives),
  }));
  return { groundingPredictions, groundingFeatureNames, groundingFeatureVectors,
    groundingVectorEncoding: 'indexed-dense-v1' as const };
}

/** Restore exactly the present finite numeric entries; null denotes absence. */
export function restoreGroundingFeatureVector(table: IndexedGroundingFeatureTable, reference: string): FeatureVector {
  const names = table.groundingFeatureNames;
  const values = Object.hasOwn(table.groundingFeatureVectors, reference) ? table.groundingFeatureVectors[reference] : undefined;
  if (table.groundingVectorEncoding !== 'indexed-dense-v1' || !Array.isArray(names) ||
      names.some((name) => typeof name !== 'string') || new Set(names).size !== names.length ||
      !Array.isArray(values) || values.length !== names.length ||
      values.some((value) => value !== null && !Number.isFinite(value))) {
    throw new Error('Invalid indexed grounding feature reference');
  }
  return Object.fromEntries(names.flatMap((name, index) => values[index] === null ? [] : [[name, values[index] as number]]));
}

export function restoreGroundingModelContext(table: GroundingModelContextTable, reference: string): GroundingModelContext {
  const context = Object.hasOwn(table.groundingModelContexts, reference) ? table.groundingModelContexts[reference] : undefined;
  if (!context || (context.modelHash !== undefined && typeof context.modelHash !== 'string') ||
      !Number.isFinite(context.threshold) || typeof context.adaptation !== 'string') {
    throw new Error('Invalid grounding model context reference');
  }
  return { ...context };
}

/** Phrase diagnostics remain uncommitted when the whole utterance rolls back. */
export function buildGroundingResearchTrace(
  results: readonly GroundedPragmaticRoute1Result[],
  phrases: readonly string[],
  utteranceCommitted: boolean,
) {
  const groundingModelContexts: Record<string, GroundingModelContext> = {};
  const contexts = new Map<string, string>();
  function contextReference(result: GroundedPragmaticRoute1Result): string {
    const { modelHash, threshold, adaptation } = result.grounding;
    if (!Number.isFinite(threshold)) throw new Error('Research grounding threshold must be finite');
    const key = JSON.stringify([modelHash, threshold, adaptation]);
    const existing = contexts.get(key);
    if (existing !== undefined) return existing;
    const reference = `c${contexts.size}`;
    contexts.set(key, reference);
    groundingModelContexts[reference] = { modelHash, threshold, adaptation };
    return reference;
  }
  const trace = compactGroundingPredictions(results.map((result, index) => ({
    phrase: phrases[index],
    label: result.grounding.label,
    confidence: result.grounding.confidence,
    contextRef: contextReference(result),
    status: result.status,
    committed: utteranceCommitted && result.status === 'updated',
    reason: result.reason,
    selectedSubgoal: result.grounding.selectedSubgoal,
    directivePolarity: result.grounding.directivePolarity,
    targetFeatures: result.grounding.targetFeatures,
    pragmaticAlternatives: result.grounding.pragmaticAlternatives,
    rawSentiment: result.rawSentiment,
    effectiveValence: result.effectiveValence,
    valenceSource: result.valenceSource,
  })));
  return { ...trace, groundingModelContexts };
}
