// Independent diagnostic only: never use these authored examples for training.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBrowserModelsFromManifest } from '../../web/lib/browser-models.ts';

const out = dirname(fileURLToPath(import.meta.url));
const repo = resolve(out, '../..');
const labels = ['evaluative', 'imperative', 'descriptive'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const examples = {
  evaluative: [
    'Nice work!', 'That was a terrible move.', 'You handled that order well.',
    'Your last delivery was excellent.', 'I liked how you avoided the collision.',
    'You did a poor job of sharing the counter.', 'That detour was unnecessary.',
    'Great teamwork on the soup!', 'Waiting there was a bad decision.',
    'You made the right choice.', 'That was not a good handoff.',
    'Your timing was perfect.', 'I appreciate your help with the dishes.',
    'Your last action was helpful.', 'The way you fetched that onion was inefficient.',
    'Dropping the plate there was a mistake.', 'Well done on clearing the route.',
    'Your move to the right was careless.', 'You were too slow with that order.',
    'That was a clever way to get around the pot.',
  ],
  imperative: [
    'Go to the stove.', 'Bring me a clean plate.', 'Please collect another onion.',
    'Could you take the soup to the serving window?', 'Stop blocking the doorway.',
    "Don't put the plate on the floor.", 'Do not move into my path.',
    'You should fetch a dish.', 'Would you mind waiting beside the sink?',
    'Let me pass.', 'Keep the middle counter clear.', 'Avoid the narrow corridor.',
    'Pick up the cooked soup now.', 'Leave the onion on the counter.',
    'Please do not stand in front of the pot.', 'Can you move one tile west?',
    'Try handing me the dish.', 'Make room for the other chef.',
    'Stay near the serving area.', 'Wait until the soup is ready.',
  ],
  descriptive: [
    'The serving window is on the right.', 'The onion dispenser is on the left.',
    'There are two onions in the pot.', 'I am holding a clean plate.',
    'The soup is not ready yet.', 'You are standing beside the sink.',
    'The corridor is blocked.', 'The counter next to the pot is empty.',
    'I have already delivered the soup.', 'There is no plate by the stove.',
    'You did not pick up the onion.', 'The other chef is carrying a tomato.',
    'Our score is twenty.', 'The order needs three onions.',
    'The pot contains no soup.', 'My route goes through the middle.',
    'A dish is sitting on the bottom counter.', 'You are not in my way.',
    'The onion is still on the floor.', 'The top counter has a clean plate.',
  ],
};
const rows = labels.flatMap(label => examples[label].map((text, n) => ({
  id: `${label}-${n + 1}`, text, expected: label, stratum: 'single_clause',
})));
for (const direction of ['left', 'right', 'top', 'bottom']) {
  for (const [expected, text] of [
    ['imperative', `Go to the ${direction} counter.`],
    ['descriptive', `The soup is on the ${direction} counter.`],
    ['evaluative', `Leaving the soup on the ${direction} counter was a bad decision.`],
  ]) rows.push({ id: `location-${direction}-${expected}`, text, expected, stratum: 'location_counterfactual' });
}
const mixed = [
  { text: 'Nice work! Please fetch a plate.', expected: ['evaluative', 'imperative'] },
  { text: 'The soup is ready; take it to the serving window.', expected: ['descriptive', 'imperative'] },
  { text: 'You did well, but keep the corridor clear.', expected: ['evaluative', 'imperative'] },
  { text: 'The counter is empty. Your last delivery was excellent.', expected: ['descriptive', 'evaluative'] },
  { text: 'Please fetch an onion, but do not block the sink.', expected: ['imperative', 'imperative'] },
  { text: 'That was a good handoff, and the next soup is ready.', expected: ['evaluative', 'descriptive'] },
  { text: 'The soup is ready, so bring me a plate.', expected: ['descriptive', 'imperative'] },
  { text: 'Nice job, please move to the left.', expected: ['evaluative', 'imperative'] },
  { text: 'The pot is full and you did a great job.', expected: ['descriptive', 'evaluative'] },
  { text: 'Great job. The pot is full. Please serve the soup.', expected: ['evaluative', 'descriptive', 'imperative'] },
];
const source = readFileSync(resolve(repo, 'web/lib/route-inputs.ts'), 'utf8');
const splitterSource = source.slice(source.indexOf('export function splitFeedbackPhrases'), source.indexOf('export function lowConfidenceRouteMessage'));
const split = new Function(`${splitterSource.replace('export ', '').replace('(text: string): string[]', '(text)')}\nreturn splitFeedbackPhrases;`)();
const fetcher = async input => {
  const bytes = readFileSync(resolve(repo, `web/public${String(input)}`));
  return new Response(new Uint8Array(bytes));
};
function summarize(predictions) {
  const matrix = labels.map(expected => labels.map(predicted => predictions.filter(p => p.expected === expected && p.prediction.label === predicted).length));
  const perClass = Object.fromEntries(labels.map((label, k) => {
    const support = matrix[k].reduce((a, b) => a + b, 0);
    const predicted = matrix.reduce((a, row) => a + row[k], 0);
    const tp = matrix[k][k];
    const recall = tp / support;
    const precision = predicted ? tp / predicted : 0;
    return [label, { support, correct: tp, recall, precision, f1: recall + precision ? 2 * recall * precision / (recall + precision) : 0 }];
  }));
  const accepted = predictions.filter(p => !p.prediction.abstained);
  return {
    rows: predictions.length, correct: predictions.filter(p => p.correct).length,
    accuracy: predictions.filter(p => p.correct).length / predictions.length,
    macro_f1: labels.reduce((sum, label) => sum + perClass[label].f1, 0) / 3,
    labels, confusion_matrix: matrix, per_class: perClass,
    accepted: accepted.length, accepted_accuracy: accepted.length ? accepted.filter(p => p.correct).length / accepted.length : null,
    high_score_wrong: predictions.filter(p => !p.correct && p.prediction.confidence >= 0.8).length,
  };
}
const report = {
  schema: 'durf-independent-browser-semantic-diagnostic-v1', date: '2026-09-06',
  scope: 'Agent-authored English speech-act diagnostic; not human gold, not independent current-player accuracy. All models frozen before this audit. No training or production edits.',
  cautions: ['Simple utterances can overlap prior training; overlap is audited separately.', 'Implicit feedback can be pragmatically ambiguous; the expected label records the literal speech act.', 'Mixed examples test clause segmentation and sequence; whole-utterance softmax is not a composition percentage.'],
  expected_labels_frozen_sha256: sha(JSON.stringify({ rows, mixed })),
  splitter_source_sha256: sha(splitterSource), models: {},
};
for (const variant of ['production', 'boundary-shadow-raw-v2']) {
  const model = await loadBrowserModelsFromManifest('', fetcher, variant);
  const predictions = rows.map(row => {
    const prediction = model.classify(row.text);
    return { ...row, prediction, correct: prediction.label === row.expected };
  });
  const mixedPredictions = mixed.map(row => {
    const phrases = split(row.text);
    const predictions = phrases.map(text => model.classify(text));
    return { ...row, phrases, predictions, segmentation_matches_count: phrases.length === row.expected.length,
      exact_label_sequence: predictions.map(p => p.label).join('|') === row.expected.join('|') };
  });
  report.models[variant] = {
    artifact_path: model.classifierArtifactPath,
    artifact_sha256: sha(readFileSync(resolve(repo, `web/public${model.classifierArtifactPath}`))),
    evidence: model.classifierEvidence,
    summary: summarize(predictions),
    single_clause_summary: summarize(predictions.filter(p => p.stratum === 'single_clause')),
    location_counterfactual_summary: summarize(predictions.filter(p => p.stratum === 'location_counterfactual')),
    mixed: { rows: mixed.length, correct_label_sequences: mixedPredictions.filter(p => p.exact_label_sequence).length,
      correct_segment_counts: mixedPredictions.filter(p => p.segmentation_matches_count).length, predictions: mixedPredictions },
    predictions,
  };
}
writeFileSync(resolve(out, 'probes.frozen.json'), `${JSON.stringify({ scope: report.scope, rows, mixed }, null, 2)}\n`);
writeFileSync(resolve(out, 'browser-semantic-report.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(Object.fromEntries(Object.entries(report.models).map(([name, result]) => [name, { artifact_sha256: result.artifact_sha256, summary: result.summary, mixed: { rows: result.mixed.rows, correct_label_sequences: result.mixed.correct_label_sequences, correct_segment_counts: result.mixed.correct_segment_counts } }])), null, 2));
