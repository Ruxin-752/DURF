export type FeedbackLabel = "evaluative" | "imperative" | "descriptive";

export interface FeedbackFormPrediction {
  label: FeedbackLabel;
  confidence: number;
  probabilities: Record<FeedbackLabel, number>;
  threshold: number;
  abstained: boolean;
  calibrated: boolean;
  calibrationVersion: string | null;
  modelHash: string;
  scoreKind: "temperature_scaled_selection_dev_probability" | "raw_probability";
}

interface TfidfTransformerArtifact {
  name: string;
  analyzer: "word" | "char_wb";
  ngram_range: [number, number];
  lowercase: boolean;
  strip_accents: "unicode" | null;
  sublinear_tf: boolean;
  norm: "l2" | null;
  token_pattern: string | null;
  feature_offset: number;
  vocabulary: Record<string, number>;
  idf: number[];
}

interface FeedbackArtifact {
  schema_version: "durf-feedback-form-browser-v1";
  source: { model_sha256: string; report_sha256: string };
  classes: FeedbackLabel[];
  minimum_confidence: number;
  calibration?: { method?: string; temperature?: number; version?: string };
  transformers: TfidfTransformerArtifact[];
  classifier: {
    kind: string;
    coefficients: number[][];
    intercept: number[];
  };
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

interface Route2Artifact {
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
  classify(text: string): FeedbackFormPrediction;
  route2(text: string, featureCounts?: number[] | Record<string, number>): Route2Prediction;
}

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
  if (transformer.norm !== "l2" || squaredNorm === 0) return weighted;
  const norm = Math.sqrt(squaredNorm);
  return weighted.map(([index, value]) => [index, value / norm]);
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
  let probabilities = softmax(logits);
  const temperature = Number(artifact.calibration?.temperature);
  const calibrated =
    artifact.calibration?.method === "temperature_scaling" && temperature > 0;
  if (calibrated) {
    const scaledLogProbabilities = probabilities.map(
      (value) => Math.log(Math.max(value, 1e-12)) / temperature,
    );
    probabilities = softmax(scaledLogProbabilities);
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
    calibrated,
    calibrationVersion: artifact.calibration?.version ?? null,
    modelHash: artifact.source.model_sha256,
    scoreKind: calibrated
      ? "temperature_scaled_selection_dev_probability"
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

function createRoute2Predictor(artifact: Route2Artifact) {
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
  if (feedbackArtifact.schema_version !== "durf-feedback-form-browser-v1") {
    throw new Error("unsupported feedback-form browser artifact");
  }
  if (route2Artifact.schema_version !== "durf-route2-browser-v1") {
    throw new Error("unsupported Route2 browser artifact");
  }
  const route2 = createRoute2Predictor(route2Artifact);
  return {
    features: [...route2Artifact.features],
    classify: (text) => classifyWithArtifact(feedbackArtifact, text),
    route2,
  };
}

let modelPromise: Promise<BrowserModels> | null = null;

export function loadBrowserModels(baseUrl = ""): Promise<BrowserModels> {
  if (!modelPromise) {
    modelPromise = Promise.all([
      fetch(`${baseUrl}/models/feedback-form-v3.json`).then((response) => {
        if (!response.ok) throw new Error("failed to load feedback-form model");
        return response.json() as Promise<FeedbackArtifact>;
      }),
      fetch(`${baseUrl}/models/route2-v5.json`).then((response) => {
        if (!response.ok) throw new Error("failed to load Route2 model");
        return response.json() as Promise<Route2Artifact>;
      }),
    ]).then(([feedback, route2]) => createBrowserModels(feedback, route2));
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
  const effectivePrecision = (observation.basePrecision ?? 2) * confidence;
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
