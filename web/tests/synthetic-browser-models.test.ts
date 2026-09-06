import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  loadSyntheticGameModels,
  predictSynthetic,
  validateSyntheticArtifact,
  type SyntheticLinearArtifact,
  type SyntheticManifest,
} from '../lib/synthetic-browser-models';

// These fixtures come only from the trainer's own development corpus. No
// independent acceptance examples or labels are read by this engineering test.
const publicDirectory = join(import.meta.dirname, '..', 'public');
const releaseManifest = JSON.parse(readFileSync(join(publicDirectory, 'models', 'manifest-synthetic-v1.json'), 'utf8')) as SyntheticManifest;
const speechBytes = readFileSync(join(publicDirectory, releaseManifest.models.speech_act.path.slice(1)));
const groundingBytes = readFileSync(join(publicDirectory, releaseManifest.models.grounding.path.slice(1)));
const route2Bytes = readFileSync(join(import.meta.dirname, '..', 'public', 'models', 'route2-v5.json'));
const speech = JSON.parse(speechBytes.toString()) as SyntheticLinearArtifact;
const grounding = JSON.parse(groundingBytes.toString()) as SyntheticLinearArtifact;
interface ParityFixture {
  text: string;
  speech_act_scores: Record<string, number>;
  grounding_scores: Record<string, number>;
}
const fixtures = JSON.parse(readFileSync(join(import.meta.dirname, 'fixtures', 'synthetic-browser-parity.json'), 'utf8')) as ParityFixture[];

function digest(value: Uint8Array | string) {
  return createHash('sha256').update(value).digest('hex');
}

function manifestFixture(): SyntheticManifest {
  return {
    schema_version: 'durf-synthetic-feedback-release-v1',
    training_scope: 'synthetic_only',
    supported_language: 'en',
    release_id: 'unit-test-fixture-not-a-release',
    models: {
      speech_act: { path: '/models/unit-speech.json', sha256: digest(speechBytes) },
      grounding: { path: '/models/unit-grounding.json', sha256: digest(groundingBytes) },
      route2: { path: '/models/unit-route2.json', sha256: digest(route2Bytes) },
    },
    evidence: {
      speech_model_sha256: speech.source.model_sha256,
      grounding_model_sha256: grounding.source.model_sha256,
      synthetic_test_sha256: '1'.repeat(64),
      report_sha256: '2'.repeat(64),
      speech_accuracy: 0.9,
      speech_macro_f1: 0.9,
      grounding_accuracy: 0.9,
      rows: 10,
      human_accuracy: null,
      passed: true,
    },
  };
}

function mockFetcher(manifest = manifestFixture(), overrides: Record<string, string | Uint8Array> = {}) {
  const bodies: Record<string, string | Uint8Array> = {
    '/models/manifest-synthetic-v1.json': JSON.stringify(manifest),
    '/models/unit-speech.json': speechBytes,
    '/models/unit-grounding.json': groundingBytes,
    '/models/unit-route2.json': route2Bytes,
    ...overrides,
  };
  return async (input: RequestInfo | URL): Promise<Response> => {
    const path = new URL(String(input), 'https://unit.invalid').pathname;
    const body = bodies[path];
    return body === undefined ? new Response('Not found', { status: 404 })
      : new Response(typeof body === 'string' ? body : new Uint8Array(body));
  };
}

describe('frozen synthetic classifier parity', () => {
  it('binds the engineering tests to the exact frozen browser artifacts', () => {
    expect(digest(speechBytes)).toBe(releaseManifest.models.speech_act.sha256);
    expect(digest(groundingBytes)).toBe(releaseManifest.models.grounding.sha256);
    expect(fixtures).toHaveLength(20);
  });

  for (const [head, artifact] of [['speech_act', speech], ['grounding', grounding]] as const) {
    it(`${head} matches every sklearn class probability on all development fixtures`, () => {
      validateSyntheticArtifact(artifact, artifact.classes);
      for (const fixture of fixtures) {
        const actual = predictSynthetic(artifact, fixture.text);
        const expected = fixture[`${head}_scores`];
        expect(Object.keys(actual.probabilities)).toEqual(Object.keys(expected));
        for (const label of artifact.classes) {
          expect(Math.abs(actual.probabilities[label] - expected[label]), `${fixture.text}: ${label}`).toBeLessThanOrEqual(1e-10);
        }
        expect(actual.modelHash).toBe(artifact.source.model_sha256);
        expect(actual.threshold).toBe(0.55);
        expect(actual.confidence).toBe(Math.max(...Object.values(actual.probabilities)));
      }
    });

    it(`${head} rejects empty input and abstains on unsupported scripts or no vocabulary matches`, () => {
      expect(() => predictSynthetic(artifact, '')).toThrow('feedback text cannot be empty');
      for (const text of ['!!!', '123456789', '请拿一个盘子', 'Please 拿盘子', 'суп готов', 'français', 'qzxwvvv zxqqwwvv']) {
        expect(predictSynthetic(artifact, text).abstained, text).toBe(true);
      }
    });

    it(`${head} applies the raw-score threshold without changing the scores`, () => {
      const changed = structuredClone(artifact);
      changed.minimum_confidence = 1;
      const text = fixtures[0].text;
      const original = predictSynthetic(artifact, text);
      const thresholded = predictSynthetic(changed, text);
      expect(thresholded.abstained).toBe(true);
      expect(thresholded.probabilities).toEqual(original.probabilities);
    });
  }
});

describe('synthetic artifact contracts', () => {
  const corruptions: [string, (artifact: SyntheticLinearArtifact) => void][] = [
    ['class order', (a) => { a.classes.reverse(); }],
    ['source identity', (a) => { a.source.model_sha256 = 'invalid'; }],
    ['temperature', (a) => { Object.assign(a.calibration, { temperature: 0.5 }); }],
    ['calibration method', (a) => { Object.assign(a.calibration, { method: 'temperature_scaling' }); }],
    ['threshold', (a) => { a.minimum_confidence = Number.NaN; }],
    ['analyzer', (a) => { a.transformers[0].analyzer = 'char_wb'; }],
    ['IDF', (a) => { a.transformers[0].idf[0] = Number.POSITIVE_INFINITY; }],
    ['vocabulary indices', (a) => { a.transformers[0].vocabulary[Object.keys(a.transformers[0].vocabulary)[0]] = -1; }],
    ['coefficient dimensions', (a) => { a.classifier.coefficients[0].pop(); }],
    ['coefficient value', (a) => { a.classifier.coefficients[0][0] = Number.NaN; }],
    ['tokenization contract', (a) => { a.transformers[0].token_pattern = '[a-z]+'; }],
    ['TF-IDF scaling', (a) => { a.transformers[0].weight = 2; }],
    ['negation preservation', (a) => { Object.assign(a.transformers[0], { stop_words: ['not', 'no'] }); }],
  ];
  for (const [name, corrupt] of corruptions) {
    it(`rejects a changed ${name}`, () => {
      const artifact = structuredClone(speech);
      corrupt(artifact);
      expect(() => validateSyntheticArtifact(artifact, ['descriptive', 'evaluative', 'imperative'])).toThrow();
    });
  }
});

describe('synthetic release loader', () => {
  it('loads both verified heads and preserves raw-score claims', async () => {
    const models = await loadSyntheticGameModels('https://unit.invalid/', mockFetcher());
    expect(models.classifierVariant).toBe('synthetic-v1');
    expect(models.groundingModelHash).toBe(grounding.source.model_sha256);
    const result = models.classify(fixtures[0].text);
    expect(result.probabilities).toEqual(predictSynthetic(speech, fixtures[0].text).probabilities);
    expect(result.calibrated).toBe(false);
    expect(result.independentlyCalibrated).toBe(false);
    expect(result.scoreKind).toBe('raw_model_softmax_score');
    expect(models.ground(fixtures[0].text).probabilities).toEqual(predictSynthetic(grounding, fixtures[0].text).probabilities);
    expect(models.classifierEvidence.independentCurrentPlayerAccuracy).toBeNull();
  });

  it('rejects byte tampering before attempting to parse the model', async () => {
    await expect(loadSyntheticGameModels('', mockFetcher(manifestFixture(), {
      '/models/unit-speech.json': 'untrusted modified bytes',
    }))).rejects.toThrow('Model hash mismatch');
  });

  it('rejects a matching-hash artifact with an invalid schema', async () => {
    const invalid = JSON.stringify({ ...speech, schema_version: 'wrong-format' });
    const manifest = manifestFixture();
    manifest.models.speech_act.sha256 = digest(invalid);
    await expect(loadSyntheticGameModels('', mockFetcher(manifest, {
      '/models/unit-speech.json': invalid,
    }))).rejects.toThrow('Invalid synthetic classifier');
  });

  it('rejects a matching-hash file containing invalid JSON', async () => {
    const manifest = manifestFixture();
    manifest.models.speech_act.sha256 = digest('{invalid');
    await expect(loadSyntheticGameModels('', mockFetcher(manifest, {
      '/models/unit-speech.json': '{invalid',
    }))).rejects.toThrow();
  });

  it('rejects a manifest with a wrong schema', async () => {
    const manifest = manifestFixture();
    Object.assign(manifest, { schema_version: 'legacy-release' });
    await expect(loadSyntheticGameModels('', mockFetcher(manifest))).rejects.toThrow('Invalid synthetic release');
  });

  it('rejects a manifest pointing outside the model directory', async () => {
    const manifest = manifestFixture();
    manifest.models.speech_act.path = '/models/../../private.json';
    await expect(loadSyntheticGameModels('', mockFetcher(manifest))).rejects.toThrow('Invalid model path or hash');
  });

  it('rejects a head with mismatched source evidence', async () => {
    const manifest = manifestFixture();
    manifest.evidence.speech_model_sha256 = '3'.repeat(64);
    await expect(loadSyntheticGameModels('', mockFetcher(manifest))).rejects.toThrow('Synthetic release evidence');
  });

  it('rejects failed release evidence', async () => {
    const manifest = manifestFixture();
    manifest.evidence.passed = false;
    await expect(loadSyntheticGameModels('', mockFetcher(manifest))).rejects.toThrow('Synthetic release evidence');
  });

  it('rejects a head assigned to the wrong semantic target', async () => {
    const invalid = JSON.stringify({ ...speech, target_semantics: 'grounding' });
    const manifest = manifestFixture();
    manifest.models.speech_act.sha256 = digest(invalid);
    await expect(loadSyntheticGameModels('', mockFetcher(manifest, {
      '/models/unit-speech.json': invalid,
    }))).rejects.toThrow('Synthetic release evidence');
  });

  it('fails closed when the manifest cannot be downloaded', async () => {
    await expect(loadSyntheticGameModels('', async () => new Response('Not found', { status: 404 })))
      .rejects.toThrow('Synthetic model manifest is unavailable');
  });
});
