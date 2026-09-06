// Never invoke until the root has explicitly frozen models AND inference code.
// This loads source modules through TypeScript's transpiler without esbuild.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const Module = require('node:module');
const workspace = path.resolve(__dirname, '../..');
const sealedDatasets = {
  '4473f6647e428777f92f03e54a4c6c561976775bf64cb0fe567dfd5c50105ff1': 'frozen-v2.json',
  '23c4272190395efa3e9b9791f68b75fda502a9019c85ab2ae5e120d8fc999fea': 'frozen-next-v2.json',
  'ddbdc0ed38517865427c01d8ae24b72400fc163034acb857945045a064f258de': 'frozen-round3.json',
};
const expectedDatasetHash = process.env.DURF_ACCEPTANCE_FROZEN;
if (!sealedDatasets[expectedDatasetHash]) {
  throw new Error('Explicit final-inference freeze authorization is required');
}
const [speechPath, groundingPath, outputPath] = process.argv.slice(2).map((p) => path.resolve(p));
if (!speechPath || !groundingPath || !outputPath) throw new Error('Pass speech artifact, grounding artifact and output path');
if (fs.existsSync(outputPath)) throw new Error('Refusing to overwrite an existing inference dump');
const sha = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const datasetPath = path.join(__dirname, sealedDatasets[expectedDatasetHash]);
if (sha(datasetPath) !== expectedDatasetHash) throw new Error('Acceptance text changed after sealing');
const before = {};
for (const file of [speechPath, groundingPath, datasetPath]) before[file] = sha(file);
const ts = require(path.join(workspace, 'web/node_modules/typescript'));
Module._extensions['.ts'] = function loadTypeScript(module, filename) {
  before[filename] = sha(filename);
  const source = fs.readFileSync(filename, 'utf8');
  const result = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
    fileName: filename,
  });
  module._compile(result.outputText, filename);
};
const { predictSynthetic } = require(path.join(workspace, 'web/lib/synthetic-browser-models.ts'));
const { splitFeedbackPhrases } = require(path.join(workspace, 'web/lib/route-inputs.ts'));
const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf8'));
const speechArtifact = JSON.parse(fs.readFileSync(speechPath, 'utf8'));
const groundingArtifact = JSON.parse(fs.readFileSync(groundingPath, 'utf8'));
function predict(text) {
  const speech = predictSynthetic(speechArtifact, text);
  const grounding = predictSynthetic(groundingArtifact, text);
  return {
    speech_act: speech.label, confidence: speech.confidence, speech_probabilities: speech.probabilities,
    abstained: speech.abstained, threshold: speech.threshold,
    grounding: grounding.label, grounding_confidence: grounding.confidence,
    grounding_probabilities: grounding.probabilities, grounding_abstained: grounding.abstained,
    grounding_threshold: grounding.threshold,
    learning_applied: null,
    uncertainty_reason: speech.abstained ? 'classifier_abstention' : null,
  };
}
const records = [];
for (const row of dataset.singles) records.push({ id: row.id, ...predict(row.text) });
for (const row of dataset.mixed) {
  records.push({ id: row.id, ...predict(row.text), clauses: splitFeedbackPhrases(row.text).map((text) => ({ text, ...predict(text) })) });
  row.clauses.forEach((clause, i) => records.push({ id: `${row.id}:C${i + 1}`, ...predict(clause.text) }));
}
for (const row of dataset.uncertain) records.push({ id: row.id, ...predict(row.text) });
for (const [filename, hash] of Object.entries(before)) {
  if (sha(filename) !== hash) throw new Error(`A model or inference source changed during the final evaluation: ${filename}`);
}
fs.writeFileSync(outputPath, JSON.stringify({
  provenance: {
    candidate_frozen_before_inference: true,
    candidate_training_text_read_by_acceptance_agent: false,
    dataset_sha256: expectedDatasetHash,
    generated_at: new Date().toISOString(),
    hashes: before,
    speech_model_source_sha256: speechArtifact.source.model_sha256,
    grounding_model_source_sha256: groundingArtifact.source.model_sha256,
    learning_behavior_scope: 'Classifier and clause split only; no game action or learning update was executed.',
  },
  records,
}, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ output: outputPath, records: records.length, sha256: sha(outputPath), files_hashed: Object.keys(before).length }));
