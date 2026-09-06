import { describe, expect, it } from 'vitest';

import { createFullGaussianPrior } from '../lib/browser-models';
import { chooseAiDecision, createGameState } from '../lib/game';
import { buildGroundingResearchTrace, compactGroundingPredictions, restoreGroundingFeatureVector, restoreGroundingModelContext } from '../lib/grounding-research-trace';
import { applyGroundedPragmaticUtterance } from '../lib/pragmatic-route1';
import { groundRoute1Feedback } from '../lib/route1-grounding';
import { REWARD_FEATURES } from '../lib/subgoal-policy';
import { CLIENT_VERSION, CONSENT_VERSION, SCHEMA_VERSION } from '../lib/research-types';
import { parseResearchBatch } from '../lib/validation';
import { gameFeatureCounts } from '../lib/route-inputs';
import { scoreVaderSentiment } from '../lib/vader-sentiment';

const modelHash = 'a'.repeat(64);

describe('lossless research grounding vector sharing', () => {
  it('shares equal vectors regardless of property insertion order and preserves other fields', () => {
    const input: { targetFeatures: Record<string, number>; pragmaticAlternatives: Record<string, number>; phrase: string; committed: boolean; source: string; status: string }[] = [
      { phrase: 'first', targetFeatures: { z: 0.125, a: 2 / 3 }, pragmaticAlternatives: { other: 1 }, committed: true, source: 'raw', status: 'updated' },
      { phrase: 'second', targetFeatures: { a: 2 / 3, z: 0.125 }, pragmaticAlternatives: { other: 1 }, committed: false, source: 'raw', status: 'rejected' },
    ];
    const before = structuredClone(input);
    const encoded = compactGroundingPredictions(input);
    const decoded = JSON.parse(JSON.stringify(encoded)) as typeof encoded;
    expect(Object.keys(decoded.groundingFeatureVectors)).toEqual(['v0', 'v1']);
    expect(decoded.groundingPredictions[0].targetFeaturesRef).toBe(decoded.groundingPredictions[1].targetFeaturesRef);
    const reconstructed = decoded.groundingPredictions.map(({ targetFeaturesRef, pragmaticAlternativesRef, ...row }) => ({
      ...row,
      targetFeatures: restoreGroundingFeatureVector(decoded, targetFeaturesRef),
      pragmaticAlternatives: restoreGroundingFeatureVector(decoded, pragmaticAlternativesRef),
    }));
    expect(reconstructed).toEqual(input);
    expect(input).toEqual(before);
    expect(encoded.groundingFeatureNames).toEqual(['a', 'other', 'z']);
    expect(encoded.groundingFeatureVectors.v0).toEqual([2 / 3, null, 0.125]);
    expect(encoded.groundingVectorEncoding).toBe('indexed-dense-v1');
  });

  it('retains empty vectors and distinguishes missing features from explicit zero and changed values', () => {
    const input: { targetFeatures: Record<string, number>; pragmaticAlternatives: Record<string, number> }[] = [
      { targetFeatures: {}, pragmaticAlternatives: {} },
      { targetFeatures: { x: 0 }, pragmaticAlternatives: { x: 0.1 } },
      { targetFeatures: { x: 0.10000000000000002 }, pragmaticAlternatives: { y: 0.1 } },
    ];
    const result = compactGroundingPredictions(input);
    expect(Object.keys(result.groundingFeatureVectors)).toHaveLength(5);
    expect(result.groundingPredictions[0].targetFeaturesRef).toBe(result.groundingPredictions[0].pragmaticAlternativesRef);
    result.groundingPredictions.forEach((prediction, index) => {
      expect(restoreGroundingFeatureVector(result, prediction.targetFeaturesRef)).toEqual(input[index].targetFeatures);
      expect(restoreGroundingFeatureVector(result, prediction.pragmaticAlternativesRef)).toEqual(input[index].pragmaticAlternatives);
    });
    expect(compactGroundingPredictions([])).toEqual({ groundingPredictions: [], groundingFeatureVectors: {},
      groundingFeatureNames: [], groundingVectorEncoding: 'indexed-dense-v1' });
    const onlyEmpty = compactGroundingPredictions([{ targetFeatures: {}, pragmaticAlternatives: {} }]);
    expect(onlyEmpty.groundingFeatureVectors.v0).toEqual([]);
    expect(restoreGroundingFeatureVector(onlyEmpty, 'v0')).toEqual({});
  });

  it('takes an independent numeric snapshot and rejects numbers JSON would erase', () => {
    const original = { count: 0.12345678901234568 };
    const result = compactGroundingPredictions([{ targetFeatures: original, pragmaticAlternatives: {} }]);
    original.count = 10;
    expect(restoreGroundingFeatureVector(result, 'v0').count).toBe(0.12345678901234568);
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, null as unknown as number]) {
      expect(() => compactGroundingPredictions([{ targetFeatures: { count: value }, pragmaticAlternatives: {} }])).toThrow('finite numeric values');
    }
  });

  it('keeps the complete 12-phrase dense 53-feature trace below 10 KB', () => {
    expect(REWARD_FEATURES).toHaveLength(53);
    const phrases = Array.from({ length: 12 }, () => 'Great job!');
    const trajectoryFeatures = Object.fromEntries(REWARD_FEATURES.map((name, index) => [name, (index + 1) / 53]));
    const state = createGameState('running');
    const observations = phrases.map((text) => ({
      grounding: groundRoute1Feedback({ text, state, trajectoryFeatures,
        grounding: { label: 'trajectory', confidence: 0.9912345678901234, modelHash } }),
      sentiment: 0.8,
    }));
    const transaction = applyGroundedPragmaticUtterance(createFullGaussianPrior([...REWARD_FEATURES]), observations);
    expect(transaction.status).toBe('updated');
    const trace = buildGroundingResearchTrace(transaction.results, phrases, transaction.status === 'updated');
    const bytes = new TextEncoder().encode(JSON.stringify(trace)).length;
    expect(bytes).toBeLessThan(10_000);
    expect(trace.groundingPredictions).toHaveLength(12);
    expect(Object.keys(trace.groundingFeatureVectors)).toHaveLength(2);
    for (const [index, prediction] of trace.groundingPredictions.entries()) {
      expect(prediction.committed).toBe(true);
      expect(restoreGroundingFeatureVector(trace, prediction.targetFeaturesRef)).toEqual(transaction.results[index].grounding.targetFeatures);
      expect(restoreGroundingFeatureVector(trace, prediction.pragmaticAlternativesRef)).toEqual(transaction.results[index].grounding.pragmaticAlternatives);
      expect(Object.keys(restoreGroundingFeatureVector(trace, prediction.targetFeaturesRef))).toHaveLength(53);
    }
    // Exercise the actual ingestion validator, not only a helper byte count.
    // These identities exist only in this in-memory unit fixture.
    const now = Date.now();
    const sessionId = '11111111-1111-4111-8111-111111111111';
    const probabilities = { Evaluative: 0.97, Imperative: 0.02, Descriptive: 0.01 };
    const payload = {
      ...trace, utterance: phrases.join(' '), selectedRoute: 'route1',
      feedbackSemanticsVersion: 'speech-act-grounding-v1', trainingScope: 'synthetic_only',
      classifierModelHash: 'b'.repeat(64), updaterModelHash: modelHash, groundingModelHash: modelHash,
      gameSnapshot: { tick: 40, score: 0, features: trajectoryFeatures, recentTrajectoryFeatures: trajectoryFeatures },
    };
    const batch = { session: {
      sessionId, anonymousUserId: '22222222-2222-4222-8222-222222222222',
      consentVersion: CONSENT_VERSION, consentedAt: now, startedAt: now,
      clientVersion: CLIENT_VERSION, schemaVersion: SCHEMA_VERSION,
    }, events: [{
      eventId: '33333333-3333-4333-8333-333333333333', sessionId,
      sequenceNumber: 0, eventType: 'feedback', occurredAt: now,
      payload, schemaVersion: SCHEMA_VERSION, modelHash,
      feedback: {
        feedbackId: '44444444-4444-4444-8444-444444444444', utterance: phrases.join(' '),
        route: 'route1', topLabel: 'Evaluative', lowConfidence: false, probabilities,
        modelHash: 'b'.repeat(64), schemaVersion: SCHEMA_VERSION, routeTrace: 'route1',
        phrases: phrases.map((phrase) => ({ phrase, label: 'Evaluative', confidence: .97,
          probabilities, abstained: false, scoreKind: 'raw_model_softmax_score',
          thresholdPolicy: 'synthetic_dev_threshold_v1' })),
      },
    }] };
    const parsed = parseResearchBatch(JSON.parse(JSON.stringify(batch)));
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) throw new Error(parsed.error);
    expect(parsed.value.events[0].payload).toEqual(payload);
    expect(parsed.value.events[0].feedback?.phrases).toHaveLength(12);
  });

  it('records every phrase as uncommitted after whole-utterance rollback', () => {
    const phrases = ['Great job!', 'Serve the soup.'];
    const state = createGameState('running');
    const prior = createFullGaussianPrior([...REWARD_FEATURES]);
    const observations = phrases.map((text, index) => ({
      grounding: groundRoute1Feedback({ text, state, trajectoryFeatures: { pick_onion: 2 },
        grounding: { label: index === 0 ? 'trajectory' : 'action', confidence: 0.99, modelHash } }),
      sentiment: 0.8,
    }));
    const transaction = applyGroundedPragmaticUtterance(prior, observations);
    expect(transaction.status).toBe('rejected');
    expect(transaction.state).toBe(prior);
    expect(transaction.results[0].status).toBe('updated');
    const trace = buildGroundingResearchTrace(transaction.results, phrases, transaction.status === 'updated');
    expect(trace.groundingPredictions.map((row) => row.committed)).toEqual([false, false]);
    expect(trace.groundingPredictions[0]).toMatchObject({ phrase: 'Great job!', status: 'updated', rawSentiment: 0.8, valenceSource: 'vader_compound' });
    expect(trace.groundingPredictions[1]).toMatchObject({ phrase: 'Serve the soup.', status: 'rejected', reason: 'infeasible_action_reference' });
  });

  it('round-trips 12 distinct targets with full complements through the complete ingestion payload', () => {
    const names = [...REWARD_FEATURES];
    const phrases = Array.from({ length: 12 }, (_, index) => `Reward feature ${index + 1} matters.`);
    const state = createGameState('running');
    const basis = groundRoute1Feedback({ text: 'Great job!', state, trajectoryFeatures: { pick_onion: 1 },
      grounding: { label: 'trajectory', confidence: 0.9912345678901234, modelHash } });
    const observations = phrases.map((_, index) => ({
      grounding: { ...basis,
        targetFeatures: { [names[index]]: 1 },
        pragmaticAlternatives: Object.fromEntries(names.filter((_, other) => other !== index).map((name) => [name, 1])),
      },
      sentiment: scoreVaderSentiment(phrases[index]).compound,
    }));
    const transaction = applyGroundedPragmaticUtterance(createFullGaussianPrior(names), observations);
    expect(transaction.status).toBe('updated');
    const trace = buildGroundingResearchTrace(transaction.results, phrases, true);
    expect(trace.groundingFeatureNames).toHaveLength(53);
    expect(Object.keys(trace.groundingFeatureVectors)).toHaveLength(24);
    expect(Object.keys(trace.groundingModelContexts)).toEqual(['c0']);
    for (const vector of Object.values(trace.groundingFeatureVectors)) expect(vector).toHaveLength(53);
    for (const [index, row] of trace.groundingPredictions.entries()) {
      expect(restoreGroundingFeatureVector(trace, row.targetFeaturesRef)).toEqual(observations[index].grounding.targetFeatures);
      expect(restoreGroundingFeatureVector(trace, row.pragmaticAlternativesRef)).toEqual(observations[index].grounding.pragmaticAlternatives);
      expect(Object.keys(restoreGroundingFeatureVector(trace, row.pragmaticAlternativesRef))).toHaveLength(52);
      expect(restoreGroundingModelContext(trace, row.contextRef)).toEqual({
        modelHash: observations[index].grounding.modelHash,
        threshold: observations[index].grounding.threshold,
        adaptation: observations[index].grounding.adaptation,
      });
    }
    const now = Date.now();
    const sessionId = '11111111-1111-4111-8111-111111111111';
    const feedbackId = '44444444-4444-4444-8444-444444444444';
    const probabilities = { Evaluative: 0.97, Imperative: 0.02, Descriptive: 0.01 };
    const decision = chooseAiDecision(state, Object.fromEntries(names.map((name, index) => [name, transaction.state.mean[index]])));
    const payload = {
      ...trace, roundId: '55555555-5555-4555-8555-555555555555', submittedAtTick: 40,
      utterance: phrases.join(' '), selectedRoute: 'route1',
      feedbackSemanticsVersion: 'speech-act-grounding-v1', trainingScope: 'synthetic_only',
      classifierModelHash: 'b'.repeat(64), updaterModelHash: modelHash, groundingModelHash: modelHash,
      calibrated: false, temperatureScaled: false, independentlyCalibrated: false,
      scoreKinds: phrases.map(() => 'raw_model_softmax_score'), scorePolicyVersion: 'synthetic-raw-softmax-v1',
      thresholdPolicy: 'synthetic_dev_threshold_v1', diagnosticThresholds: phrases.map(() => 0.55),
      classifierVariant: 'synthetic-v1', classifierManifestPath: '/models/manifest-synthetic-v1.json',
      classifierArtifactPath: `/models/speech-act-${'b'.repeat(64)}.json`, calibrationVersions: phrases.map(() => null),
      outcome: 'Route 1 applied 12 independently grounded pragmatic updates', policyRevision: 1,
      policyImpact: { feedbackId, roundId: '55555555-5555-4555-8555-555555555555', applied: true, revision: 1,
        evaluatedAtTick: 40, beforeSubgoal: 'GET_TOMATO', beforeAction: 'left', afterSubgoal: 'GET_ONION', afterAction: 'right', planChanged: true },
      proposedAiDecision: { evaluatedAtTick: 40, snapshotStatus: 'running', proposalEvaluationStatus: 'running',
        chosenSubgoal: decision.chosenSubgoal, action: decision.action, decisionSource: decision.decisionSource, rewardMargin: decision.rewardMargin,
        candidateScores: decision.ranking.map(({ subgoal, score }) => ({ subgoal, score })),
        topContributions: decision.topContributions },
      changedFeatures: Object.fromEntries(names.slice(0, 5).map((name, index) => [name, transaction.state.mean[index]])),
      gameSnapshot: { tick: 40, score: 0,
        features: gameFeatureCounts(state),
        recentTrajectoryFeatures: Object.fromEntries(names.map((name, index) => [name, (index + 1) / 53])) },
    };
    expect(JSON.stringify(payload).length).toBeLessThan(16_384);
    const batch = { session: {
      sessionId, anonymousUserId: '22222222-2222-4222-8222-222222222222', consentVersion: CONSENT_VERSION,
      consentedAt: now, startedAt: now, clientVersion: CLIENT_VERSION, schemaVersion: SCHEMA_VERSION,
    }, events: [{
      eventId: '33333333-3333-4333-8333-333333333333', sessionId, sequenceNumber: 0,
      eventType: 'feedback', occurredAt: now, payload, schemaVersion: SCHEMA_VERSION, modelHash,
      feedback: { feedbackId, utterance: phrases.join(' '), route: 'route1', topLabel: 'Evaluative',
        lowConfidence: false, probabilities, modelHash: 'b'.repeat(64), schemaVersion: SCHEMA_VERSION, routeTrace: 'route1',
        phrases: phrases.map((phrase) => ({ phrase, label: 'Evaluative', confidence: 0.97,
          probabilities, abstained: false, scoreKind: 'raw_model_softmax_score', thresholdPolicy: 'synthetic_dev_threshold_v1' })) },
    }] };
    const parsed = parseResearchBatch(JSON.parse(JSON.stringify(batch)));
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) throw new Error(parsed.error);
    expect(parsed.value.events[0].payload).toEqual(payload);
    const savedTrace = parsed.value.events[0].payload as typeof payload;
    savedTrace.groundingPredictions.forEach((row, index) => {
      expect(restoreGroundingFeatureVector(savedTrace, row.targetFeaturesRef)).toEqual(observations[index].grounding.targetFeatures);
      expect(restoreGroundingFeatureVector(savedTrace, row.pragmaticAlternativesRef)).toEqual(observations[index].grounding.pragmaticAlternatives);
      expect(restoreGroundingModelContext(savedTrace, row.contextRef)).toEqual({ modelHash,
        threshold: observations[index].grounding.threshold, adaptation: observations[index].grounding.adaptation });
    });
  });

  it('rejects corrupt indexed references instead of inventing a missing vector', () => {
    const table = compactGroundingPredictions([{ targetFeatures: { x: 0 }, pragmaticAlternatives: {} }]);
    expect(() => restoreGroundingFeatureVector(table, 'v99')).toThrow('Invalid indexed grounding feature reference');
    expect(() => restoreGroundingFeatureVector({ ...table, groundingFeatureNames: [] }, 'v0')).toThrow();
    expect(() => restoreGroundingFeatureVector({ ...table, groundingFeatureVectors: { v0: [Number.NaN] } }, 'v0')).toThrow();
  });

  it('shares only identical model contexts and snapshots different hashes or thresholds without loss', () => {
    const state = createGameState('running');
    const grounding = groundRoute1Feedback({ text: 'Great job!', state, trajectoryFeatures: { pick_onion: 1 },
      grounding: { label: 'trajectory', confidence: 0.99, modelHash } });
    const otherAdaptation = { ...grounding };
    Object.assign(otherAdaptation, { adaptation: 'historical-test-adaptation' });
    const observations = [
      { grounding, sentiment: 0.8 },
      { grounding: { ...grounding }, sentiment: 0.8 },
      { grounding: { ...grounding, threshold: 0.5512345678901234 }, sentiment: 0.8 },
      { grounding: { ...grounding, modelHash: 'b'.repeat(64) }, sentiment: 0.8 },
      { grounding: otherAdaptation, sentiment: 0.8 },
    ];
    const result = applyGroundedPragmaticUtterance(createFullGaussianPrior([...REWARD_FEATURES]), observations);
    const trace = buildGroundingResearchTrace(result.results, observations.map(() => 'Great job!'), true);
    expect(trace.groundingPredictions.map((row) => row.contextRef)).toEqual(['c0', 'c0', 'c1', 'c2', 'c3']);
    const saved = JSON.parse(JSON.stringify(trace)) as typeof trace;
    saved.groundingPredictions.forEach((row, index) => {
      expect(restoreGroundingModelContext(saved, row.contextRef)).toEqual({
        modelHash: observations[index].grounding.modelHash, threshold: observations[index].grounding.threshold,
        adaptation: observations[index].grounding.adaptation,
      });
    });
    grounding.modelHash = 'c'.repeat(64);
    expect(restoreGroundingModelContext(trace, 'c0').modelHash).toBe(modelHash);
    expect(() => restoreGroundingModelContext(trace, 'c99')).toThrow('Invalid grounding model context reference');
  });
});
