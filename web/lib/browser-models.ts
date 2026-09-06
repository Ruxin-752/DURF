export type FeedbackLabel = "evaluative" | "imperative" | "descriptive";
export type FeedbackClassifierVariant = "production" | "boundary-shadow-raw-v2";

export interface FeedbackFormPrediction {
  label: FeedbackLabel;
  confidence: number;
  probabilities: Record<FeedbackLabel, number>;
  threshold: number;
  abstained: boolean;
  calibrated: boolean;
  temperatureScaled: boolean;
  independentlyCalibrated: boolean;
  calibrationVersion: string | null;
  modelHash: string;
  thresholdPolicy?: 'synthetic_dev_threshold_v1';
  scoreKind:
    | "temperature_scaled_selection_dev_probability"
    | "temperature_scaled_train_oof_probability"
    | "raw_model_softmax_score"
    | "raw_probability";
}

export interface ClassifierEvidence {
  trained: true;
  frozen: true;
  modelVersion: string;
  modelHash: string;
  releaseStatus: string;
  promotionEligible: boolean;
  promotionReason: string;
  defaultEligible: boolean;
  independentCurrentPlayerAccuracy: number | null;
  diagnosticAccuracy: number | null;
  diagnosticRows: number | null;
  diagnosticScope: string;
  requestedAccuracyTarget: number | null;
  targetPassed: boolean | null;
  diagnosticPreviouslyExposed: boolean;
}

interface FeedbackReleaseMetadata {
  channel: "production" | "shadow-preview";
  status: string;
  default_eligible: boolean;
  promotion_eligible: boolean;
  promotion_status: string;
  promotion_reason: string;
}

interface ModelCardReport {
  id: string;
  role: string;
  report_sha256: string;
  source_model_sha256: string;
  binding_fields: string[];
}

interface ModelCardEvidence {
  id: string;
  report_sha256: string;
  source_model_sha256: string;
  role: string;
  scope: string;
  rows: number;
  accuracy: number;
  macro_f1: number;
  human_or_player_gold: boolean;
  independent_current_player: boolean;
  previously_exposed: boolean;
  requested_accuracy_target: number;
  target_passed: boolean;
}

interface FeedbackModelCard {
  schema_version: "durf-feedback-form-model-card-v1";
  source_model_sha256: string;
  release: FeedbackReleaseMetadata;
  training: {
    trained: true;
    frozen: true;
    status: string;
    report_sha256: string;
    source_model_sha256: string;
  };
  reports: ModelCardReport[];
  display_evidence_id: string;
  evidence: ModelCardEvidence[];
  claims: {
    independent_current_player_accuracy: number | null;
    calibration_independently_validated: boolean;
  };
}

export interface TfidfTransformerArtifact {
  name: string;
  analyzer: "word" | "char_wb";
  ngram_range: [number, number];
  lowercase: boolean;
  strip_accents: "unicode" | null;
  sublinear_tf: boolean;
  norm: "l2" | null;
  token_pattern: string | null;
  feature_offset: number;
  weight?: number;
  vocabulary: Record<string, number>;
  idf: number[];
}

interface FeedbackArtifact {
  schema_version: "durf-feedback-form-browser-v1";
  source: {
    model_sha256: string;
    report_sha256: string;
    frozen_config_sha256?: string;
    predictor_sha256?: string;
    candidate_manifest_sha256?: string;
    diagnostic_report_sha256?: string;
    release_contract_sha256?: string;
  };
  classes: FeedbackLabel[];
  model_type?: string;
  model_version?: string | number;
  minimum_confidence: number;
  calibration?: {
    method?: string;
    temperature?: number;
    version?: string;
    scope?: "train_oof" | string;
    independently_validated?: boolean;
  };
  transformers: TfidfTransformerArtifact[];
  classifier: {
    kind: string;
    coefficients: number[][];
    intercept: number[];
  };
  claim_scope?: {
    diagnostic_shadow_preview?: boolean;
    calibration_independently_validated?: boolean;
    displayed_score_kind?: string;
    displayed_score_is_probability_of_correctness?: boolean;
    minimum_confidence_source?: string;
    independent_current_player_accuracy?: number | null;
    paper_reference_proxy_accuracy?: number;
    paper_reference_proxy_examples?: number;
    paper_reference_proxy_is_previously_exposed?: boolean;
    synthetic_training_rows?: number;
    private_diagnostic_accuracy?: number;
    private_diagnostic_examples?: number;
  };
  score_policy?: {
    kind?: string;
    version?: string;
    temperature?: number;
    probability_of_correctness?: boolean;
    independently_calibrated?: boolean;
    routing_threshold?: number;
    threshold_policy?: string;
  };
  model_card: FeedbackModelCard;
}

const FEEDBACK_CLASS_ORDER: FeedbackLabel[] = [
  "descriptive",
  "evaluative",
  "imperative",
];
const SHA256_PATTERN = /^[0-9a-f]{64}$/u;
const SHADOW_SOURCE = {
  model_sha256: "40b03b2f9a88a71f7f5334d61f77176bcde01b86b7f8f277422c3038be4e1a55",
  report_sha256: "4b23b8ca052e11c530929d640f1a6c15bbf24db7ae8e20c78b13b99e66ef9019",
  frozen_config_sha256:
    "cb0e4ccf00c96030167428ee057dbdcf134f3f075d6af77c2616445483191327",
  predictor_sha256: "d76fe50fe2e7d079c2709d37c020c88227c5708d087595347f483d6b63c8c507",
  candidate_manifest_sha256:
    "60258ce84ed959aa75f91c6bb43cd02975c3377300b8b0426fd2aeb223db5bee",
  diagnostic_report_sha256:
    "5cef25eb4ddba21f182ec4ba9bdc8f1a135ad741b3ef49e9593ab055454ce5d2",
} as const;
const SHADOW_DISPLAY_TEMPERATURE = 1;

function assertReleasePolicy(
  release: FeedbackReleaseMetadata,
  shadow: boolean,
): void {
  const expected = shadow
    ? {
        channel: "shadow-preview",
        status: "shadow_diagnostic_only",
        default_eligible: false,
        promotion_eligible: false,
        promotion_status: "failed_target_keep_non_promotable_shadow",
      }
    : {
        channel: "production",
        status: "active_legacy_default",
        default_eligible: true,
        promotion_eligible: false,
        promotion_status: "legacy_active_not_requalified",
      };
  if (
    !release ||
    release.channel !== expected.channel ||
    release.status !== expected.status ||
    release.default_eligible !== expected.default_eligible ||
    release.promotion_eligible !== expected.promotion_eligible ||
    release.promotion_status !== expected.promotion_status ||
    typeof release.promotion_reason !== "string" ||
    release.promotion_reason.length === 0
  ) {
    throw new Error("feedback classifier release status or promotion contract changed");
  }
}

function assertUnitMetric(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`feedback classifier model-card ${label} is invalid`);
  }
}

function assertFeedbackModelCard(artifact: FeedbackArtifact): void {
  const card = artifact.model_card;
  const modelHash = artifact.source.model_sha256;
  const shadow = artifact.claim_scope?.diagnostic_shadow_preview === true;
  if (
    !card ||
    card.schema_version !== "durf-feedback-form-model-card-v1" ||
    !SHA256_PATTERN.test(modelHash) ||
    card.source_model_sha256 !== modelHash
  ) {
    throw new Error("feedback classifier model-card source binding changed");
  }
  assertReleasePolicy(card.release, shadow);
  if (
    card.training?.trained !== true ||
    card.training?.frozen !== true ||
    typeof card.training.status !== "string" ||
    !SHA256_PATTERN.test(card.training.report_sha256) ||
    card.training.report_sha256 !== artifact.source.report_sha256 ||
    card.training.source_model_sha256 !== modelHash
  ) {
    throw new Error("feedback classifier model-card training binding changed");
  }
  if (!Array.isArray(card.reports) || card.reports.length < 2) {
    throw new Error("feedback classifier model-card reports are missing");
  }
  const reportIds = new Set<string>();
  for (const report of card.reports) {
    if (
      !report.id ||
      reportIds.has(report.id) ||
      !report.role ||
      !SHA256_PATTERN.test(report.report_sha256) ||
      report.source_model_sha256 !== modelHash ||
      !Array.isArray(report.binding_fields) ||
      report.binding_fields.length === 0 ||
      report.binding_fields.some((field) => typeof field !== "string" || !field)
    ) {
      throw new Error("feedback classifier model-card report binding changed");
    }
    reportIds.add(report.id);
  }
  if (
    !card.reports.some(
      (report) =>
        report.role === "training" &&
        report.report_sha256 === card.training.report_sha256 &&
        report.source_model_sha256 === modelHash,
    )
  ) {
    throw new Error("feedback classifier model-card training report is unreferenced");
  }
  if (!Array.isArray(card.evidence) || card.evidence.length === 0) {
    throw new Error("feedback classifier model-card evidence is missing");
  }
  const displayEvidence = card.evidence.find(
    (evidence) => evidence.id === card.display_evidence_id,
  );
  const evidenceReport = card.reports.find(
    (report) => report.id === card.display_evidence_id,
  );
  if (
    !displayEvidence ||
    !evidenceReport ||
    displayEvidence.report_sha256 !== evidenceReport.report_sha256 ||
    displayEvidence.source_model_sha256 !== modelHash ||
    !displayEvidence.role ||
    !displayEvidence.scope ||
    !Number.isInteger(displayEvidence.rows) ||
    displayEvidence.rows <= 0 ||
    displayEvidence.human_or_player_gold !== false ||
    displayEvidence.independent_current_player !== false ||
    typeof displayEvidence.previously_exposed !== "boolean" ||
    typeof displayEvidence.target_passed !== "boolean"
  ) {
    throw new Error("feedback classifier display evidence binding changed");
  }
  assertUnitMetric(displayEvidence.accuracy, "accuracy");
  assertUnitMetric(displayEvidence.macro_f1, "macro F1");
  assertUnitMetric(displayEvidence.requested_accuracy_target, "accuracy target");
  if (
    card.claims?.independent_current_player_accuracy !== null ||
    card.claims.calibration_independently_validated !== false ||
    artifact.claim_scope?.independent_current_player_accuracy !==
      card.claims.independent_current_player_accuracy ||
    artifact.claim_scope?.calibration_independently_validated !==
      card.claims.calibration_independently_validated
  ) {
    throw new Error("feedback classifier model-card claim scope changed");
  }
  if (
    shadow &&
    (artifact.source.diagnostic_report_sha256 !== displayEvidence.report_sha256 ||
      artifact.claim_scope?.private_diagnostic_accuracy !== displayEvidence.accuracy ||
      artifact.claim_scope?.private_diagnostic_examples !== displayEvidence.rows)
  ) {
    throw new Error("shadow classifier diagnostic evidence binding changed");
  }
  if (
    !shadow &&
    (!SHA256_PATTERN.test(artifact.source.release_contract_sha256 ?? "") ||
      artifact.claim_scope?.paper_reference_proxy_accuracy !== displayEvidence.accuracy ||
      artifact.claim_scope?.paper_reference_proxy_examples !== displayEvidence.rows ||
      artifact.claim_scope?.paper_reference_proxy_is_previously_exposed !==
        displayEvidence.previously_exposed)
  ) {
    throw new Error("production classifier diagnostic evidence binding changed");
  }
}

function assertFeedbackArtifactShape(artifact: FeedbackArtifact): void {
  if (artifact.schema_version !== "durf-feedback-form-browser-v1") {
    throw new Error("unsupported feedback-form browser artifact");
  }
  if (
    artifact.classes.length !== FEEDBACK_CLASS_ORDER.length ||
    artifact.classes.some((label, index) => label !== FEEDBACK_CLASS_ORDER[index])
  ) {
    throw new Error("feedback classifier class order changed");
  }
  if (
    !Number.isFinite(artifact.minimum_confidence) ||
    artifact.minimum_confidence < 0 ||
    artifact.minimum_confidence > 1
  ) {
    throw new Error("feedback classifier threshold is invalid");
  }
  let expectedOffset = 0;
  for (const transformer of artifact.transformers) {
    const vocabularySize = Object.keys(transformer.vocabulary).length;
    const vocabularyIndices = Object.values(transformer.vocabulary).sort(
      (left, right) => left - right,
    );
    if (
      transformer.feature_offset !== expectedOffset ||
      transformer.idf.length !== vocabularySize ||
      vocabularyIndices.some((value, index) => value !== index) ||
      transformer.idf.some((value) => !Number.isFinite(value)) ||
      !Number.isFinite(transformer.weight ?? 1) ||
      (transformer.weight ?? 1) <= 0
    ) {
      throw new Error(`feedback transformer is invalid: ${transformer.name}`);
    }
    expectedOffset += vocabularySize;
  }
  if (
    artifact.classifier.kind !== "multinomial_logistic_regression" ||
    artifact.classifier.intercept.length !== FEEDBACK_CLASS_ORDER.length ||
    artifact.classifier.intercept.some((value) => !Number.isFinite(value)) ||
    artifact.classifier.coefficients.length !== FEEDBACK_CLASS_ORDER.length ||
    artifact.classifier.coefficients.some(
      (row) =>
        row.length !== expectedOffset || row.some((value) => !Number.isFinite(value)),
    )
  ) {
    throw new Error("feedback classifier parameters are invalid");
  }
  if (
    artifact.calibration?.method === "temperature_scaling" &&
    (!Number.isFinite(Number(artifact.calibration.temperature)) ||
      !(Number(artifact.calibration.temperature) > 0))
  ) {
    throw new Error("feedback classifier temperature is invalid");
  }
  assertFeedbackModelCard(artifact);
}

function assertShadowArtifactIdentity(artifact: FeedbackArtifact): void {
  for (const [field, expected] of Object.entries(SHADOW_SOURCE)) {
    if (artifact.source[field as keyof typeof artifact.source] !== expected) {
      throw new Error(`shadow classifier source identity changed: ${field}`);
    }
  }
  const [word, char] = artifact.transformers;
  if (
    artifact.model_version !== "direct-fg-boundary-shadow-model-v1" ||
    artifact.model_type !== "direct_fg_tfidf_logistic_regression_shadow" ||
    artifact.minimum_confidence !== 0.55 ||
    artifact.calibration?.method !== "none" ||
    artifact.calibration?.temperature !== undefined ||
    artifact.calibration?.version !== undefined ||
    artifact.calibration?.scope !== "raw_model_output" ||
    artifact.calibration?.independently_validated !== false ||
    artifact.claim_scope?.diagnostic_shadow_preview !== true ||
    artifact.claim_scope?.calibration_independently_validated !== false ||
    artifact.claim_scope?.displayed_score_kind !== "raw_model_softmax_score" ||
    artifact.claim_scope?.displayed_score_is_probability_of_correctness !== false ||
    artifact.claim_scope?.minimum_confidence_source !==
      "existing_web_preview_policy_not_validated_in_raw_score_space" ||
    artifact.score_policy?.kind !== "raw_softmax" ||
    artifact.score_policy?.version !== "boundary-shadow-raw-softmax-v2" ||
    artifact.score_policy?.temperature !== SHADOW_DISPLAY_TEMPERATURE ||
    artifact.score_policy?.probability_of_correctness !== false ||
    artifact.score_policy?.independently_calibrated !== false ||
    artifact.score_policy?.routing_threshold !== 0.55 ||
    artifact.score_policy?.threshold_policy !==
      "existing_web_preview_policy_not_validated_in_raw_score_space" ||
    artifact.transformers.length !== 2 ||
    word?.name !== "word" ||
    word?.analyzer !== "word" ||
    word?.ngram_range[0] !== 1 ||
    word?.ngram_range[1] !== 2 ||
    word?.lowercase !== true ||
    word?.sublinear_tf !== true ||
    word?.norm !== "l2" ||
    (word?.weight ?? 1) !== 1 ||
    char?.name !== "char" ||
    char?.analyzer !== "char_wb" ||
    char?.ngram_range[0] !== 3 ||
    char?.ngram_range[1] !== 5 ||
    char?.lowercase !== true ||
    char?.sublinear_tf !== true ||
    char?.norm !== "l2" ||
    char?.weight !== 0.7
  ) {
    throw new Error("shadow classifier release contract changed");
  }
}

interface EncodedFloat32 {
  shape: number[];
  dtype: "float32-le";
  base64: string;
}

interface Route2MemberArtifact {
  fold: number;
  checkpoint_sha256: string;
  vocab: Record<string, number>;
  config: {
    vocab_size: number;
    n_features: number;
    embedding_dim: number;
    hidden_dim: number;
    use_feature_counts: boolean;
  };
  parameters: Record<
    | "embedding.weight"
    | "fc1.weight"
    | "fc1.bias"
    | "fc2.weight"
    | "fc2.bias",
    EncodedFloat32
  >;
}

export interface Route2Artifact {
  schema_version: "durf-route2-browser-v1";
  source: {
    ensemble_manifest_sha256: string;
    ensemble_identity: string;
  };
  features: string[];
  members: Route2MemberArtifact[];
  update: {
    kind: "paper_independent_gaussian";
    prior_variance: number;
    observation_precision: number;
  };
}

export interface Route2Prediction {
  weights: Record<string, number>;
  uncertainty: Record<string, number>;
  ensembleSize: number;
  modelHash: string;
  tokenizerVersion: "paper-noun-lemma-browser-v1";
}

export interface IndependentGaussianState {
  features: string[];
  mean: number[];
  precision: number[];
}

export interface FullGaussianState {
  features: string[];
  mean: number[];
  covariance: number[][];
}

export type ReferenceSubtype =
  | "trajectory"
  | "action_behavioral"
  | "action_spatial"
  | "feature"
  | "other";

export interface Route1PaperObservation {
  feedbackForm: FeedbackLabel;
  feedbackFormConfidence: number;
  feedbackFormThreshold?: number;
  feedbackFormAbstained?: boolean;
  requestedReferenceSubtype?: ReferenceSubtype;
  trajectoryFeatures?: Record<string, number>;
  actionFeatures?: Record<string, number>;
  namedFeatures?: Record<string, number>;
  valence: number;
  basePrecision?: number;
}

export interface Route1PaperResult {
  state: FullGaussianState;
  status: "updated" | "rejected";
  reason?: string;
  feedbackForm: FeedbackLabel;
  effectiveReferenceSubtype: ReferenceSubtype;
  referenceConflict: boolean;
  targetFeatures: Record<string, number>;
  effectivePrecision: number;
  delta: Record<string, number>;
}

export interface BrowserModels {
  features: string[];
  classifierEvidence: ClassifierEvidence;
  classifierMode: "production" | "experimental-shadow-preview";
  classifierVariant: FeedbackClassifierVariant;
  classifierManifestPath: string | null;
  classifierArtifactPath: string | null;
  classify(text: string): FeedbackFormPrediction;
  route2(text: string, featureCounts?: number[] | Record<string, number>): Route2Prediction;
}

function classifierEvidence(artifact: FeedbackArtifact): ClassifierEvidence {
  const card = artifact.model_card;
  const evidence = card.evidence.find(
    (candidate) => candidate.id === card.display_evidence_id,
  );
  if (!evidence) throw new Error("feedback classifier display evidence is missing");
  return {
    trained: card.training.trained,
    frozen: card.training.frozen,
    modelVersion: String(artifact.model_version ?? "unknown"),
    modelHash: artifact.source.model_sha256,
    releaseStatus: card.release.status,
    promotionEligible: card.release.promotion_eligible,
    promotionReason: card.release.promotion_reason,
    defaultEligible: card.release.default_eligible,
    independentCurrentPlayerAccuracy:
      card.claims.independent_current_player_accuracy,
    diagnosticAccuracy: evidence.accuracy,
    diagnosticRows: evidence.rows,
    diagnosticScope: evidence.scope,
    requestedAccuracyTarget: evidence.requested_accuracy_target,
    targetPassed: evidence.target_passed,
    diagnosticPreviouslyExposed: evidence.previously_exposed,
  };
}

interface BrowserModelManifestEntry {
  path: string;
  sha256: string;
}

interface BrowserModelManifest {
  schema_version: "durf-browser-model-manifest-v2";
  release: FeedbackReleaseMetadata & {
    source_model_sha256: string;
    model_card_schema_version: "durf-feedback-form-model-card-v1";
  };
  models: {
    feedback_form: BrowserModelManifestEntry;
    route2: BrowserModelManifestEntry;
  };
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

const ASCII_PUNCTUATION = /[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;
const NUMBER_WORDS: Record<string, string> = {
  "0": "zero",
  "1": "one",
  "2": "two",
  "3": "three",
  "4": "four",
  "5": "five",
  "6": "six",
  "7": "seven",
  "8": "eight",
  "9": "nine",
};

function assertFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`${label} is not finite`);
  return value;
}

function softmax(values: number[]): number[] {
  const largest = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function stripUnicodeAccents(value: string): string {
  return value.normalize("NFKD").replace(/\p{M}/gu, "");
}

function wordAnalyzer(text: string, range: [number, number]): string[] {
  const tokens = text.match(/[\p{L}\p{N}_]{2,}/gu) ?? [];
  const output: string[] = [];
  for (let size = range[0]; size <= range[1]; size += 1) {
    for (let start = 0; start + size <= tokens.length; start += 1) {
      output.push(tokens.slice(start, start + size).join(" "));
    }
  }
  return output;
}

// Mirrors sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char_wb").
function charWbAnalyzer(text: string, range: [number, number]): string[] {
  const normalized = text.replace(/\s+/gu, " ");
  const output: string[] = [];
  for (const token of normalized.split(" ").filter(Boolean)) {
    const padded = ` ${token} `;
    for (let size = range[0]; size <= range[1]; size += 1) {
      let offset = 0;
      output.push(padded.slice(offset, offset + size));
      while (offset + size < padded.length) {
        offset += 1;
        output.push(padded.slice(offset, offset + size));
      }
      // sklearn stops after counting a short padded word once. Continuing to
      // larger n would make JS slice return the same truncated term again and
      // over-count it (notably for one- and two-letter words such as "a"/"is").
      if (offset === 0) break;
    }
  }
  return output;
}

function tfidfEntries(
  rawText: string,
  transformer: TfidfTransformerArtifact,
): Array<[number, number]> {
  let text = rawText;
  if (transformer.lowercase) text = text.toLowerCase();
  if (transformer.strip_accents === "unicode") text = stripUnicodeAccents(text);
  const terms =
    transformer.analyzer === "char_wb"
      ? charWbAnalyzer(text, transformer.ngram_range)
      : wordAnalyzer(text, transformer.ngram_range);
  const counts = new Map<number, number>();
  for (const term of terms) {
    const index = transformer.vocabulary[term];
    if (index === undefined) continue;
    counts.set(index, (counts.get(index) ?? 0) + 1);
  }
  const entries = [...counts.entries()].sort((left, right) => left[0] - right[0]);
  let squaredNorm = 0;
  const weighted = entries.map(([index, count]) => {
    const termFrequency = transformer.sublinear_tf ? 1 + Math.log(count) : count;
    const value = termFrequency * transformer.idf[index];
    squaredNorm += value * value;
    return [index + transformer.feature_offset, value] as [number, number];
  });
  const transformerWeight = transformer.weight ?? 1;
  if (transformer.norm !== "l2" || squaredNorm === 0) {
    return weighted.map(([index, value]) => [index, value * transformerWeight]);
  }
  const norm = Math.sqrt(squaredNorm);
  return weighted.map(([index, value]) => [
    index,
    (value / norm) * transformerWeight,
  ]);
}

/** Shared raw TF-IDF/logistic calculation for separately trained label heads. */
export function linearTextProbabilities(
  artifact: {
    classes: string[];
    transformers: TfidfTransformerArtifact[];
    classifier: { coefficients: number[][]; intercept: number[] };
  },
  text: string,
): { probabilities: number[]; matchedFeatures: number } {
  if (!text.trim()) throw new Error('feedback text cannot be empty');
  const sparse = artifact.transformers.flatMap((transformer) => tfidfEntries(text, transformer));
  const logits = artifact.classes.map((_, index) => {
    let value = artifact.classifier.intercept[index];
    for (const [feature, score] of sparse) value += artifact.classifier.coefficients[index][feature] * score;
    return assertFinite(value, 'linear classifier logit');
  });
  return { probabilities: softmax(logits), matchedFeatures: sparse.length };
}

function classifyWithArtifact(
  artifact: FeedbackArtifact,
  rawText: string,
): FeedbackFormPrediction {
  if (!rawText.trim()) throw new Error("feedback text cannot be empty");
  const sparse = artifact.transformers.flatMap((transformer) =>
    tfidfEntries(rawText, transformer),
  );
  const logits = artifact.classes.map((_, classIndex) => {
    let value = artifact.classifier.intercept[classIndex];
    const coefficients = artifact.classifier.coefficients[classIndex];
    for (const [featureIndex, featureValue] of sparse) {
      value += coefficients[featureIndex] * featureValue;
    }
    return assertFinite(value, "feedback-form logit");
  });
  const temperature = Number(artifact.calibration?.temperature);
  const temperatureScaled =
    artifact.calibration?.method === "temperature_scaling" && temperature > 0;
  let probabilities = softmax(logits);
  if (temperatureScaled) {
    probabilities = artifact.claim_scope?.diagnostic_shadow_preview
      ? softmax(logits.map((value) => value / temperature))
      : softmax(
          probabilities.map(
            (value) => Math.log(Math.max(value, 1e-12)) / temperature,
          ),
        );
  }
  let bestIndex = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[bestIndex]) bestIndex = index;
  }
  const label = artifact.classes[bestIndex];
  const confidence = probabilities[bestIndex];
  const probabilityMap = Object.fromEntries(
    artifact.classes.map((candidate, index) => [candidate, probabilities[index]]),
  ) as Record<FeedbackLabel, number>;
  return {
    label,
    confidence,
    probabilities: probabilityMap,
    threshold: artifact.minimum_confidence,
    abstained: confidence < artifact.minimum_confidence,
    calibrated:
      temperatureScaled && !artifact.claim_scope?.diagnostic_shadow_preview,
    temperatureScaled,
    independentlyCalibrated:
      temperatureScaled && artifact.calibration?.independently_validated === true,
    calibrationVersion: artifact.calibration?.version ?? null,
    modelHash: artifact.source.model_sha256,
    scoreKind: artifact.score_policy?.kind === "raw_softmax"
      ? "raw_model_softmax_score"
      : temperatureScaled
      ? artifact.calibration?.scope === "train_oof"
        ? "temperature_scaled_train_oof_probability"
        : "temperature_scaled_selection_dev_probability"
      : "raw_probability",
  };
}

function fallbackNounLemma(word: string): string {
  const irregular: Record<string, string> = {
    dishes: "dish",
    knives: "knife",
    potatoes: "potato",
    tomatoes: "tomato",
  };
  if (irregular[word]) return irregular[word];
  if (word.length > 4 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (
    word.length > 4 &&
    ["ches", "shes", "sses", "xes", "zes"].some((ending) => word.endsWith(ending))
  ) {
    return word.slice(0, -2);
  }
  if (word.length > 3 && word.endsWith("oes")) return word.slice(0, -2);
  if (
    word.length > 3 &&
    word.endsWith("s") &&
    !["ss", "us", "is"].some((ending) => word.endsWith(ending))
  ) {
    return word.slice(0, -1);
  }
  return word;
}

export function route2Tokenize(text: string): string[] {
  const normalized = text
    .replace(/\|/gu, " ")
    .replace(/'/gu, "")
    .replace(ASCII_PUNCTUATION, " ")
    .toLowerCase();
  const words = normalized.match(/[^\W_]+/gu) ?? [];
  const tokens = words.map((word) => NUMBER_WORDS[word] ?? fallbackNounLemma(word));
  return tokens.length ? tokens : [""];
}

function decodeFloat32(encoded: EncodedFloat32): Float32Array {
  if (encoded.dtype !== "float32-le") throw new Error("unsupported model dtype");
  const binary = globalThis.atob(encoded.base64);
  if (binary.length % 4 !== 0) throw new Error("invalid float32 payload length");
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const values = new Float32Array(binary.length / 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(index * 4, true);
  }
  const expected = encoded.shape.reduce((product, value) => product * value, 1);
  if (expected !== values.length) throw new Error("model parameter shape mismatch");
  return values;
}

interface DecodedRoute2Member {
  source: Route2MemberArtifact;
  embedding: Float32Array;
  fc1Weight: Float32Array;
  fc1Bias: Float32Array;
  fc2Weight: Float32Array;
  fc2Bias: Float32Array;
}

function decodeRoute2Member(source: Route2MemberArtifact): DecodedRoute2Member {
  return {
    source,
    embedding: decodeFloat32(source.parameters["embedding.weight"]),
    fc1Weight: decodeFloat32(source.parameters["fc1.weight"]),
    fc1Bias: decodeFloat32(source.parameters["fc1.bias"]),
    fc2Weight: decodeFloat32(source.parameters["fc2.weight"]),
    fc2Bias: decodeFloat32(source.parameters["fc2.bias"]),
  };
}

function countsArray(
  features: string[],
  supplied?: number[] | Record<string, number>,
): number[] {
  if (!supplied) return features.map(() => 0);
  if (Array.isArray(supplied)) {
    if (supplied.length !== features.length) {
      throw new Error(`expected ${features.length} trajectory features`);
    }
    return supplied.map((value) => assertFinite(Number(value), "trajectory feature"));
  }
  return features.map((feature) =>
    assertFinite(Number(supplied[feature] ?? 0), `trajectory feature ${feature}`),
  );
}

function predictRoute2Member(
  member: DecodedRoute2Member,
  tokens: string[],
  featureCounts: number[],
): number[] {
  const { config, vocab } = member.source;
  const unknown = vocab["<unk>"] ?? 0;
  const encoded = tokens.map((token) => vocab[token] ?? unknown);
  const embedding = Array.from({ length: config.embedding_dim }, () => 0);
  for (const tokenIndex of encoded) {
    const row = tokenIndex * config.embedding_dim;
    for (let column = 0; column < config.embedding_dim; column += 1) {
      embedding[column] += member.embedding[row + column];
    }
  }
  for (let column = 0; column < embedding.length; column += 1) {
    embedding[column] /= encoded.length;
  }
  const combined = [...embedding, ...featureCounts];
  const hidden = Array.from({ length: config.hidden_dim }, () => 0);
  for (let row = 0; row < config.hidden_dim; row += 1) {
    let value = member.fc1Bias[row];
    const offset = row * combined.length;
    for (let column = 0; column < combined.length; column += 1) {
      value += member.fc1Weight[offset + column] * combined[column];
    }
    hidden[row] = Math.max(0, value);
  }
  const output = Array.from({ length: config.n_features }, () => 0);
  for (let row = 0; row < config.n_features; row += 1) {
    let value = member.fc2Bias[row];
    const offset = row * config.hidden_dim;
    for (let column = 0; column < config.hidden_dim; column += 1) {
      value += member.fc2Weight[offset + column] * hidden[column];
    }
    output[row] = assertFinite(value, "Route2 prediction");
  }
  return output;
}

export function createRoute2Predictor(artifact: Route2Artifact) {
  if (artifact.members.length !== 10) throw new Error("Route2 requires 10 folds");
  const members = artifact.members.map(decodeRoute2Member);
  return (text: string, supplied?: number[] | Record<string, number>): Route2Prediction => {
    const tokens = route2Tokenize(text);
    const trajectory = countsArray(artifact.features, supplied);
    const rows = members.map((member) => predictRoute2Member(member, tokens, trajectory));
    const means = artifact.features.map((_, column) =>
      rows.reduce((sum, row) => sum + row[column], 0) / rows.length,
    );
    const deviations = artifact.features.map((_, column) => {
      const variance =
        rows.reduce((sum, row) => {
          const difference = row[column] - means[column];
          return sum + difference * difference;
        }, 0) / rows.length;
      return Math.sqrt(variance);
    });
    return {
      weights: Object.fromEntries(
        artifact.features.map((feature, index) => [feature, means[index]]),
      ),
      uncertainty: Object.fromEntries(
        artifact.features.map((feature, index) => [feature, deviations[index]]),
      ),
      ensembleSize: rows.length,
      modelHash: artifact.source.ensemble_manifest_sha256,
      tokenizerVersion: "paper-noun-lemma-browser-v1",
    };
  };
}

export function createBrowserModels(
  feedbackArtifact: FeedbackArtifact,
  route2Artifact: Route2Artifact,
): BrowserModels {
  assertFeedbackArtifactShape(feedbackArtifact);
  if (feedbackArtifact.claim_scope?.diagnostic_shadow_preview) {
    assertShadowArtifactIdentity(feedbackArtifact);
  }
  if (route2Artifact.schema_version !== "durf-route2-browser-v1") {
    throw new Error("unsupported Route2 browser artifact");
  }
  const route2 = createRoute2Predictor(route2Artifact);
  return {
    features: [...route2Artifact.features],
    classifierEvidence: classifierEvidence(feedbackArtifact),
    classifierMode: feedbackArtifact.claim_scope?.diagnostic_shadow_preview
      ? "experimental-shadow-preview"
      : "production",
    classifierVariant: feedbackArtifact.claim_scope?.diagnostic_shadow_preview
      ? "boundary-shadow-raw-v2"
      : "production",
    classifierManifestPath: null,
    classifierArtifactPath: null,
    classify: (text) => classifyWithArtifact(feedbackArtifact, text),
    route2,
  };
}

const modelPromises = new Map<string, Promise<BrowserModels>>();

const CLASSIFIER_RELEASES: Record<
  FeedbackClassifierVariant,
  { manifestPath: string; feedbackPath: string; feedbackSha256: string }
> = {
  production: {
    manifestPath: "/models/manifest.json",
    feedbackPath: "/models/feedback-form-v3.json",
    feedbackSha256: "2280f07b71b5790bbd53a90cea38dd4e84fa9e7f1cf34be1510e304b85bdcb48",
  },
  "boundary-shadow-raw-v2": {
    manifestPath: "/models/manifest-boundary-shadow-raw-v2.json",
    feedbackPath:
      "/models/feedback-form-boundary-shadow-raw-v2-2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef.json",
    feedbackSha256: "2f1e1f7ed8836680c4eebd6e6b89f28e565ef99f1dbbf1c6ed03023903719fef",
  },
};
const ROUTE2_RELEASE = {
  path: "/models/route2-v5.json",
  sha256: "fa9ddb2aff1a451713e62417fcd3bca56a9e06987fbc6d7dbec2cf1651b80177",
} as const;

const MODEL_PATH_PATTERN = /^\/models\/[A-Za-z0-9._-]+\.json$/u;

function modelUrl(baseUrl: string, path: string): string {
  if (!MODEL_PATH_PATTERN.test(path)) throw new Error("model manifest path is invalid");
  return `${baseUrl.replace(/\/$/u, "")}${path}`;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("SHA-256 verification is unavailable in this browser");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchJsonWithHash<T>(
  baseUrl: string,
  entry: BrowserModelManifestEntry,
  label: string,
  fetcher: FetchLike,
): Promise<T> {
  if (!SHA256_PATTERN.test(entry.sha256)) {
    throw new Error(`${label} model hash is invalid`);
  }
  const response = await fetcher(modelUrl(baseUrl, entry.path), { cache: "no-store" });
  if (!response.ok) throw new Error(`failed to load ${label} model`);
  const bytes = await response.arrayBuffer();
  const actualHash = await sha256Hex(bytes);
  if (actualHash !== entry.sha256) throw new Error(`${label} model hash mismatch`);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

function assertManifestRelease(
  release: BrowserModelManifest["release"],
  classifierVariant: FeedbackClassifierVariant,
): void {
  const shadow = classifierVariant === "boundary-shadow-raw-v2";
  assertReleasePolicy(release, shadow);
  if (
    !SHA256_PATTERN.test(release.source_model_sha256) ||
    release.model_card_schema_version !== "durf-feedback-form-model-card-v1"
  ) {
    throw new Error("browser model manifest release evidence is invalid");
  }
}

function assertManifestModelCardBinding(
  release: BrowserModelManifest["release"],
  artifact: FeedbackArtifact,
): void {
  const cardRelease = artifact.model_card.release;
  if (
    release.source_model_sha256 !== artifact.source.model_sha256 ||
    release.model_card_schema_version !== artifact.model_card.schema_version ||
    release.channel !== cardRelease.channel ||
    release.status !== cardRelease.status ||
    release.default_eligible !== cardRelease.default_eligible ||
    release.promotion_eligible !== cardRelease.promotion_eligible ||
    release.promotion_status !== cardRelease.promotion_status ||
    release.promotion_reason !== cardRelease.promotion_reason
  ) {
    throw new Error("browser model manifest/model-card release binding changed");
  }
}

export async function loadBrowserModelsFromManifest(
  baseUrl = "",
  fetcher: FetchLike = (input, init) => fetch(input, init),
  classifierVariant: FeedbackClassifierVariant = "production",
): Promise<BrowserModels> {
  const release = CLASSIFIER_RELEASES[classifierVariant];
  const response = await fetcher(`${baseUrl.replace(/\/$/u, "")}${release.manifestPath}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("failed to load browser model manifest");
  const manifest = (await response.json()) as BrowserModelManifest;
  if (
    manifest.schema_version !== "durf-browser-model-manifest-v2" ||
    !manifest.release ||
    !manifest.models?.feedback_form ||
    !manifest.models?.route2
  ) {
    throw new Error("unsupported browser model manifest");
  }
  assertManifestRelease(manifest.release, classifierVariant);
  if (
    manifest.models.feedback_form.path !== release.feedbackPath ||
    manifest.models.feedback_form.sha256 !== release.feedbackSha256 ||
    manifest.models.route2.path !== ROUTE2_RELEASE.path ||
    manifest.models.route2.sha256 !== ROUTE2_RELEASE.sha256
  ) {
    throw new Error("browser model manifest release identity changed");
  }
  const [feedback, route2] = await Promise.all([
    fetchJsonWithHash<FeedbackArtifact>(
      baseUrl,
      manifest.models.feedback_form,
      "feedback-form",
      fetcher,
    ),
    fetchJsonWithHash<Route2Artifact>(
      baseUrl,
      manifest.models.route2,
      "Route2",
      fetcher,
    ),
  ]);
  assertManifestModelCardBinding(manifest.release, feedback);
  if (classifierVariant === "boundary-shadow-raw-v2") {
    assertShadowArtifactIdentity(feedback);
  }
  const models = createBrowserModels(feedback, route2);
  if (
    (classifierVariant === "production" && models.classifierMode !== "production") ||
    (classifierVariant !== "production" &&
      models.classifierMode !== "experimental-shadow-preview")
  ) {
    throw new Error("feedback classifier release channel mismatch");
  }
  return {
    ...models,
    classifierVariant,
    classifierManifestPath: release.manifestPath,
    classifierArtifactPath: manifest.models.feedback_form.path,
  };
}

export function classifierVariantFromSearch(search: string): FeedbackClassifierVariant {
  const requested = new URLSearchParams(search).getAll("classifier");
  if (requested.length === 0) return "production";
  if (requested.length !== 1) {
    throw new Error("classifier query parameter must appear exactly once");
  }
  if (requested[0] !== "boundary-shadow-raw-v2") {
    throw new Error(`unsupported classifier query parameter: ${requested[0]}`);
  }
  return "boundary-shadow-raw-v2";
}

export function loadBrowserModels(
  baseUrl = "",
  classifierVariant: FeedbackClassifierVariant = "production",
): Promise<BrowserModels> {
  const key = `${baseUrl}\n${classifierVariant}`;
  let modelPromise = modelPromises.get(key);
  if (!modelPromise) {
    modelPromise = loadBrowserModelsFromManifest(
      baseUrl,
      undefined,
      classifierVariant,
    );
    modelPromises.set(key, modelPromise);
  }
  return modelPromise;
}

export function createIndependentGaussianPrior(
  features: string[],
  variance = 25,
): IndependentGaussianState {
  if (!(variance > 0)) throw new Error("prior variance must be positive");
  return {
    features: [...features],
    mean: features.map(() => 0),
    precision: features.map(() => 1 / variance),
  };
}

// Paper Route2: raw (u, trajectory) predicts a complete reward vector.  f_G is
// deliberately absent from this API and therefore cannot gate or route it.
export function applyRoute2Gaussian(
  state: IndependentGaussianState,
  predictedWeights: Record<string, number>,
  observationPrecision = 2,
): { state: IndependentGaussianState; delta: Record<string, number> } {
  if (!(observationPrecision > 0)) throw new Error("observation precision must be positive");
  const nextMean: number[] = [];
  const nextPrecision: number[] = [];
  const delta: Record<string, number> = {};
  state.features.forEach((feature, index) => {
    const observation = assertFinite(predictedWeights[feature], `Route2 weight ${feature}`);
    const precision = state.precision[index];
    const updatedPrecision = precision + observationPrecision;
    const updatedMean =
      (precision * state.mean[index] + observationPrecision * observation) /
      updatedPrecision;
    nextMean.push(updatedMean);
    nextPrecision.push(updatedPrecision);
    delta[feature] = updatedMean - state.mean[index];
  });
  return {
    state: { features: [...state.features], mean: nextMean, precision: nextPrecision },
    delta,
  };
}

export function createFullGaussianPrior(
  features: string[],
  variance = 25,
): FullGaussianState {
  if (!(variance > 0)) throw new Error("prior variance must be positive");
  return {
    features: [...features],
    mean: features.map(() => 0),
    covariance: features.map((_, row) =>
      features.map((__, column) => (row === column ? variance : 0)),
    ),
  };
}

const FORM_DEFAULT: Record<FeedbackLabel, ReferenceSubtype> = {
  evaluative: "trajectory",
  imperative: "action_spatial",
  descriptive: "feature",
};
const FORM_SUBTYPES: Record<FeedbackLabel, Set<ReferenceSubtype>> = {
  evaluative: new Set(["trajectory", "action_behavioral"]),
  imperative: new Set(["action_spatial"]),
  descriptive: new Set(["feature"]),
};

function normalizedReference(
  features: string[],
  values: Record<string, number>,
): { vector: number[]; selected: Record<string, number> } {
  const selected: Record<string, number> = {};
  const vector = features.map((feature) => {
    const value = Number(values[feature] ?? 0);
    if (!Number.isFinite(value)) throw new Error(`Route1 feature ${feature} is not finite`);
    if (value !== 0) selected[feature] = value;
    return value;
  });
  const total = vector.reduce((sum, value) => sum + Math.abs(value), 0);
  return {
    vector: total ? vector.map((value) => value / total) : vector,
    selected,
  };
}

export function applyRoute1PaperUpdate(
  state: FullGaussianState,
  observation: Route1PaperObservation,
): Route1PaperResult {
  const threshold = observation.feedbackFormThreshold ?? 0.55;
  const requested = observation.requestedReferenceSubtype;
  const defaultSubtype = FORM_DEFAULT[observation.feedbackForm];
  const conflict = Boolean(requested && requested !== "other" && !FORM_SUBTYPES[observation.feedbackForm].has(requested));
  const effectiveSubtype =
    requested === "other"
      ? "other"
      : requested && FORM_SUBTYPES[observation.feedbackForm].has(requested)
        ? requested
        : defaultSubtype;
  const source =
    observation.feedbackForm === "evaluative"
      ? observation.trajectoryFeatures ?? {}
      : observation.feedbackForm === "imperative"
        ? observation.actionFeatures ?? {}
        : observation.namedFeatures ?? {};
  const { vector, selected } = normalizedReference(state.features, source);
  const confidence = observation.feedbackFormConfidence;
  // The classifier score decides whether the observation is accepted. Once
  // accepted, the paper uses fixed feedback-noise variance 1/2 (precision 2),
  // not a confidence-scaled observation precision.
  const effectivePrecision = observation.basePrecision ?? 2;
  if (!(effectivePrecision > 0)) {
    throw new Error("Route1 observation precision must be positive");
  }
  const rejectedReason =
    observation.feedbackFormAbstained || !(confidence >= threshold)
      ? "feedback_form_low_confidence"
      : effectiveSubtype === "other"
        ? "reference_type_other"
        : vector.every((value) => value === 0)
          ? "empty_target_features"
          : null;
  if (rejectedReason) {
    return {
      state,
      status: "rejected",
      reason: rejectedReason,
      feedbackForm: observation.feedbackForm,
      effectiveReferenceSubtype: effectiveSubtype,
      referenceConflict: conflict,
      targetFeatures: selected,
      effectivePrecision,
      delta: {},
    };
  }
  const covarianceR = state.covariance.map((row) =>
    row.reduce((sum, value, column) => sum + value * vector[column], 0),
  );
  const rCovarianceR = vector.reduce(
    (sum, value, index) => sum + value * covarianceR[index],
    0,
  );
  const denominator = 1 + effectivePrecision * rCovarianceR;
  const activeTarget =
    vector.reduce((sum, value) => sum + value * value, 0) * observation.valence;
  const projectedMean = vector.reduce(
    (sum, value, index) => sum + value * state.mean[index],
    0,
  );
  const innovation = activeTarget - projectedMean;
  const gain = covarianceR.map((value) => (effectivePrecision * value) / denominator);
  const nextMean = state.mean.map((value, index) => value + gain[index] * innovation);
  const nextCovariance = state.covariance.map((row, rowIndex) =>
    row.map(
      (value, columnIndex) =>
        value -
        (effectivePrecision * covarianceR[rowIndex] * covarianceR[columnIndex]) /
          denominator,
    ),
  );
  // Symmetrize exactly as the Python implementation does after rank-one update.
  for (let row = 0; row < nextCovariance.length; row += 1) {
    for (let column = row + 1; column < nextCovariance.length; column += 1) {
      const average = (nextCovariance[row][column] + nextCovariance[column][row]) / 2;
      nextCovariance[row][column] = average;
      nextCovariance[column][row] = average;
    }
  }
  return {
    state: {
      features: [...state.features],
      mean: nextMean,
      covariance: nextCovariance,
    },
    status: "updated",
    feedbackForm: observation.feedbackForm,
    effectiveReferenceSubtype: effectiveSubtype,
    referenceConflict: conflict,
    targetFeatures: selected,
    effectivePrecision,
    delta: Object.fromEntries(
      state.features
        .map((feature, index) => [feature, nextMean[index] - state.mean[index]] as const)
        .filter(([, value]) => Math.abs(value) > 1e-12),
    ),
  };
}
