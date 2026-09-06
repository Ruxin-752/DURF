const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const Module = require('node:module');
const workspace = path.resolve(__dirname, '../..');
const ts = require(path.join(workspace, 'web/node_modules/typescript'));
const hashes = {};
const sha = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
Module._extensions['.ts'] = function loadTypeScript(module, filename) {
  hashes[filename] = sha(filename);
  module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true }, fileName: filename,
  }).outputText, filename);
};
const { groundRoute1Feedback } = require(path.join(workspace, 'web/lib/route1-grounding.ts'));
const { applyGroundedPragmaticFeedback } = require(path.join(workspace, 'web/lib/pragmatic-route1.ts'));
const { createFullGaussianPrior } = require(path.join(workspace, 'web/lib/browser-models.ts'));
const { createGameState } = require(path.join(workspace, 'web/lib/game.ts'));
const { REWARD_FEATURES } = require(path.join(workspace, 'web/lib/subgoal-policy.ts'));
hashes[path.join(workspace, 'web/lib/vendor/vader-data.json')] = sha(path.join(workspace, 'web/lib/vendor/vader-data.json'));
const { scoreVaderSentiment } = require(path.join(workspace, 'web/lib/vader-sentiment.ts'));
const datasetName = process.argv[2] ?? 'frozen-v2.json';
const predictionName = process.argv[3] ?? 'final-predictions.json';
const outputName = process.argv[4] ?? 'uncertain-stateful-report.json';
const analysisKind = process.argv[5] ?? 'stateful_gate_evaluation_using_frozen_predictions';
const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, datasetName), 'utf8'));
const predictions = JSON.parse(fs.readFileSync(path.join(__dirname, predictionName), 'utf8'));
if (predictions.provenance.dataset_sha256 !== sha(path.join(__dirname, datasetName))) throw new Error('Predictions and dataset hashes do not match');
const byId = Object.fromEntries(predictions.records.map((row) => [row.id, row]));
const records = [];
for (const [contextName, trajectoryFeatures] of [
  ['fresh_game_no_recent_trajectory', {}],
  ['synthetic_recent_onion_pickup', { pick_onion: 1, moves_toward_needed_object: 1 }],
]) {
  for (const row of dataset.uncertain) {
    const prediction = byId[row.id];
    const grounding = groundRoute1Feedback({
      text: row.text, state: createGameState('running'), trajectoryFeatures,
      grounding: {
        label: prediction.grounding, confidence: prediction.grounding_confidence,
        threshold: prediction.grounding_threshold, abstained: prediction.grounding_abstained,
      },
    });
    const sentiment = scoreVaderSentiment(row.text).compound;
    const prior = createFullGaussianPrior(REWARD_FEATURES);
    const result = applyGroundedPragmaticFeedback(prior, { grounding, sentiment });
    records.push({
      id: row.id, kind: row.kind, text: row.text, context: contextName,
      speech_act: prediction.speech_act, speech_abstained: prediction.abstained,
      predicted_grounding: prediction.grounding, grounding_abstained: prediction.grounding_abstained,
      grounded_status: grounding.status, grounded_reason: grounding.reason,
      selected_subgoal: grounding.selectedSubgoal, target_features: grounding.targetFeatures,
      update_status: result.status, update_reason: result.reason,
      learning_applied: JSON.stringify(result.state) !== JSON.stringify(prior),
      sentiment, effective_valence: result.effectiveValence,
    });
  }
}
for (const [file, hash] of Object.entries(hashes)) if (sha(file) !== hash) throw new Error(`Source changed during audit: ${file}`);
const summary = Object.fromEntries([...new Set(records.map((r) => r.context))].map((context) => {
  const selected = records.filter((r) => r.context === context);
  return [context, { n: selected.length, grounded: selected.filter((r) => r.grounded_status === 'grounded').length,
    learning_applied: selected.filter((r) => r.learning_applied).length,
    updated_ids: selected.filter((r) => r.learning_applied).map((r) => r.id) }];
}));
const output = path.join(__dirname, outputName);
fs.writeFileSync(output, JSON.stringify({
  analysis_kind: analysisKind,
  not_a_new_classifier_evaluation: true,
  source: 'Synthetic state only; no participant records or model re-inference.',
  scope: (analysisKind === 'post_full53_update_rule_regression' ? 'Regression after restoring the full 53-feature Pragmatic complement; original round3 classifier predictions are unchanged. ' : '') + 'Actual groundRoute1Feedback and applyGroundedPragmaticFeedback functions using frozen classifier predictions, with two explicitly constructed local game contexts. This does not execute browser submission or persist any research rows.',
  dataset_sha256: sha(path.join(__dirname, datasetName)),
  predictions_sha256: sha(path.join(__dirname, predictionName)),
  source_hashes: hashes, summary, records,
}, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ output, summary }));
