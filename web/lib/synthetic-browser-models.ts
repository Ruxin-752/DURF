import {
  createRoute2Predictor, linearTextProbabilities,
  type BrowserModels, type FeedbackFormPrediction, type FeedbackLabel,
  type Route2Artifact, type TfidfTransformerArtifact,
} from './browser-models';

export type GroundingLabel = 'action' | 'feature' | 'trajectory';
export interface GroundingPrediction {
  label: GroundingLabel;
  confidence: number;
  probabilities: Record<GroundingLabel, number>;
  threshold: number;
  abstained: boolean;
  modelHash: string;
}
export interface SyntheticLinearArtifact {
  schema_version: 'durf-synthetic-linear-v1';
  classes: string[];
  transformers: TfidfTransformerArtifact[];
  classifier: { kind: string; coefficients: number[][]; intercept: number[] };
  source: { model_sha256: string };
  target_semantics: string;
  minimum_confidence: number;
  training_scope: 'synthetic_only';
  calibration: { method: 'none'; temperature: 1 };
}
export interface GameModels extends Omit<BrowserModels, 'classifierVariant'> {
  classifierVariant: 'synthetic-v1';
  groundingModelHash: string;
  ground(text: string): GroundingPrediction;
}

interface Entry { path: string; sha256: string }
export interface SyntheticManifest {
  schema_version: 'durf-synthetic-feedback-release-v1';
  training_scope: 'synthetic_only';
  supported_language: 'en';
  release_id: string;
  models: { speech_act: Entry; grounding: Entry; route2: Entry };
  evidence: {
    speech_model_sha256: string;
    grounding_model_sha256: string;
    synthetic_test_sha256: string;
    report_sha256: string;
    speech_accuracy: number;
    speech_macro_f1: number;
    grounding_accuracy: number;
    rows: number;
    human_accuracy: null;
    passed: boolean;
  };
}
const HASH = /^[0-9a-f]{64}$/u;
const PATH = /^\/models\/[a-zA-Z0-9._-]+\.json$/u;
const SPEECH = ['descriptive', 'evaluative', 'imperative'];
const GROUNDING = ['action', 'feature', 'trajectory'];
type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function validateSyntheticArtifact(artifact: SyntheticLinearArtifact, labels: string[]): void {
  if (!artifact || artifact.schema_version !== 'durf-synthetic-linear-v1' ||
      artifact.training_scope !== 'synthetic_only' ||
      JSON.stringify(artifact.classes) !== JSON.stringify(labels) ||
      !HASH.test(artifact.source?.model_sha256 ?? '') ||
      artifact.calibration?.method !== 'none' || artifact.calibration.temperature !== 1 ||
      !Number.isFinite(artifact.minimum_confidence) || artifact.minimum_confidence < 0 || artifact.minimum_confidence > 1) {
    throw new Error('Invalid synthetic classifier identity or score contract');
  }
  let dimensions = 0;
  if (!Array.isArray(artifact.transformers) || artifact.transformers.length !== 1) {
    throw new Error('Synthetic classifier must use the declared word TF-IDF recipe');
  }
  for (const transformer of artifact.transformers) {
    if (transformer.analyzer !== 'word' || transformer.ngram_range?.[0] !== 1 || transformer.ngram_range[1] !== 2 ||
        transformer.norm !== 'l2' || transformer.lowercase !== true || transformer.sublinear_tf !== true ||
        transformer.feature_offset !== dimensions || transformer.strip_accents !== null ||
        transformer.token_pattern !== '(?u)\\b\\w\\w+\\b' ||
        (transformer.weight !== undefined && transformer.weight !== 1) ||
        ('stop_words' in transformer && transformer.stop_words !== null) ||
        !Array.isArray(transformer.idf) || !transformer.idf.length ||
        transformer.idf.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new Error('Unsupported synthetic text preprocessing');
    }
    const indexes = Object.values(transformer.vocabulary ?? {});
    if (indexes.length !== transformer.idf.length || new Set(indexes).size !== indexes.length ||
        indexes.some((index) => !Number.isInteger(index) || index < 0 || index >= transformer.idf.length)) {
      throw new Error('Invalid synthetic vocabulary');
    }
    dimensions += transformer.idf.length;
  }
  const classifier = artifact.classifier;
  if (classifier?.kind !== 'multinomial_logistic_regression' ||
      classifier.intercept?.length !== labels.length || classifier.coefficients?.length !== labels.length ||
      classifier.intercept.some((value) => !Number.isFinite(value)) ||
      classifier.coefficients.some((row) => row.length !== dimensions || row.some((value) => !Number.isFinite(value)))) {
    throw new Error('Invalid synthetic classifier dimensions');
  }
}

export function predictSynthetic(artifact: SyntheticLinearArtifact, text: string) {
  const result = linearTextProbabilities(artifact, text);
  const index = result.probabilities.indexOf(Math.max(...result.probabilities));
  const confidence = result.probabilities[index];
  const letters = text.match(/\p{L}/gu) ?? [];
  const english = letters.length > 0 && letters.every((letter) => /[a-z]/iu.test(letter));
  return {
    label: artifact.classes[index], confidence,
    probabilities: Object.fromEntries(artifact.classes.map((label, i) => [label, result.probabilities[i]])),
    threshold: artifact.minimum_confidence,
    abstained: !english || result.matchedFeatures === 0 || confidence < artifact.minimum_confidence,
    modelHash: artifact.source.model_sha256,
  };
}

export function createSyntheticGameModels(
  speech: SyntheticLinearArtifact,
  grounding: SyntheticLinearArtifact,
  route2: Route2Artifact,
  manifest: SyntheticManifest,
): GameModels {
  validateSyntheticArtifact(speech, SPEECH);
  validateSyntheticArtifact(grounding, GROUNDING);
  if (speech.target_semantics !== 'speech_act' || grounding.target_semantics !== 'grounding' ||
      manifest.evidence.speech_model_sha256 !== speech.source.model_sha256 ||
      manifest.evidence.grounding_model_sha256 !== grounding.source.model_sha256 ||
      manifest.evidence.human_accuracy !== null || !manifest.evidence.passed) {
    throw new Error('Synthetic release evidence does not match the models');
  }
  if (route2.schema_version !== 'durf-route2-browser-v1') throw new Error('Invalid Route2 artifact');
  return {
    features: [...route2.features],
    classifierMode: 'production', classifierVariant: 'synthetic-v1',
    classifierManifestPath: '/models/manifest-synthetic-v1.json',
    classifierArtifactPath: manifest.models.speech_act.path,
    groundingModelHash: grounding.source.model_sha256,
    classifierEvidence: {
      trained: true, frozen: true, modelVersion: manifest.release_id,
      modelHash: speech.source.model_sha256, releaseStatus: 'synthetic_validated',
      promotionEligible: true, defaultEligible: true,
      promotionReason: 'User-authorized synthetic-only release; human performance is unvalidated.',
      independentCurrentPlayerAccuracy: null,
      diagnosticAccuracy: manifest.evidence.speech_accuracy,
      diagnosticRows: manifest.evidence.rows, diagnosticScope: 'independently_authored_synthetic_test',
      requestedAccuracyTarget: null, targetPassed: manifest.evidence.passed,
      diagnosticPreviouslyExposed: false,
    },
    classify(text): FeedbackFormPrediction {
      const prediction = predictSynthetic(speech, text);
      return {
        ...prediction, label: prediction.label as FeedbackLabel,
        probabilities: prediction.probabilities as Record<FeedbackLabel, number>,
        calibrated: false, temperatureScaled: false, independentlyCalibrated: false,
        calibrationVersion: null, scoreKind: 'raw_model_softmax_score',
        thresholdPolicy: 'synthetic_dev_threshold_v1',
      };
    },
    ground: (text) => predictSynthetic(grounding, text) as GroundingPrediction,
    route2: createRoute2Predictor(route2),
  };
}

async function fetchArtifact<T>(entry: Entry, baseUrl: string, fetcher: Fetcher): Promise<T> {
  if (!entry || !PATH.test(entry.path) || !HASH.test(entry.sha256)) throw new Error('Invalid model path or hash');
  const response = await fetcher(`${baseUrl}${entry.path}`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Model download failed');
  const bytes = await response.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hash = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('');
  if (hash !== entry.sha256) throw new Error('Model hash mismatch');
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export async function loadSyntheticGameModels(baseUrl = '', fetcher: Fetcher = fetch): Promise<GameModels> {
  const base = baseUrl.replace(/\/$/u, '');
  const response = await fetcher(`${base}/models/manifest-synthetic-v1.json`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Synthetic model manifest is unavailable');
  const manifest = await response.json() as SyntheticManifest;
  if (manifest.schema_version !== 'durf-synthetic-feedback-release-v1' || manifest.training_scope !== 'synthetic_only' ||
      manifest.supported_language !== 'en' || !manifest.evidence || !HASH.test(manifest.evidence.synthetic_test_sha256) ||
      !HASH.test(manifest.evidence.report_sha256) || !manifest.models) throw new Error('Invalid synthetic release');
  const [speech, grounding, route2] = await Promise.all([
    fetchArtifact<SyntheticLinearArtifact>(manifest.models.speech_act, base, fetcher),
    fetchArtifact<SyntheticLinearArtifact>(manifest.models.grounding, base, fetcher),
    fetchArtifact<Route2Artifact>(manifest.models.route2, base, fetcher),
  ]);
  return createSyntheticGameModels(speech, grounding, route2, manifest);
}
