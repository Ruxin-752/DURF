"""Secondary decontamination analysis using already frozen predictions only.

The root's post-evaluation overlap audit found one exact single-sentence match
and one exact oracle-clause match in training. No training text is read here.
The primary 150-sentence report is retained unchanged.
"""
from pathlib import Path
import importlib.util
import hashlib
import json

ROOT = Path(__file__).resolve().parent
DATASET_HASH = 'ddbdc0ed38517865427c01d8ae24b72400fc163034acb857945045a064f258de'
PREDICTION_HASH = '367cfd7bc42d1986310e4b4cc9deca6ec2e3a7f69a76f65c6e17f5ac57f41ba4'
EXCLUDED_SINGLE = 'R3E019'
EXCLUDED_CLAUSE = 'R3M012:C3'
EXCLUDED_MESSAGE = 'R3M012'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(ROOT / 'frozen-round3.json') == DATASET_HASH
assert sha(ROOT / 'round3-predictions.json') == PREDICTION_HASH
spec = importlib.util.spec_from_file_location('acceptance_score', ROOT / 'score_acceptance.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
dataset = json.loads((ROOT / 'frozen-round3.json').read_text(encoding='utf-8'))
dump = json.loads((ROOT / 'round3-predictions.json').read_text(encoding='utf-8'))
predictions = {r['id']: r for r in dump['records']}
singles = [r for r in dataset['singles'] if r['id'] != EXCLUDED_SINGLE]
oracle = [{**clause, 'id': f"{row['id']}:C{n}"}
          for row in dataset['mixed'] for n, clause in enumerate(row['clauses'], 1)
          if f"{row['id']}:C{n}" != EXCLUDED_CLAUSE]
mixed = [r for r in dataset['mixed'] if r['id'] != EXCLUDED_MESSAGE]

def score(rows, key, labels):
    return scorer.metrics([r[key] for r in rows], [predictions[r['id']].get(key) for r in rows], labels)

speech = score(singles, 'speech_act', scorer.SPEECH)
grounding = score(singles, 'grounding', scorer.GROUNDING)
mixed_results = []
for row in mixed:
    predicted = sorted({c.get('speech_act') for c in predictions[row['id']].get('clauses', [])
                        if c.get('speech_act') in scorer.SPEECH})
    mixed_results.append({'id': row['id'], 'correct': predicted == row['expected_speech_acts']})
mixed_correct = sum(r['correct'] for r in mixed_results)
mixed_accuracy = mixed_correct / len(mixed_results)
gates = {
    'speech_act': speech['macro_f1'] >= 0.87 and all(c['recall'] >= 0.85 for c in speech['per_class'].values()),
    'grounding': grounding['macro_f1'] >= 0.87 and all(c['recall'] >= 0.85 for c in grounding['per_class'].values()),
    'mixed_components': mixed_accuracy >= 0.85,
}
report = {
    'analysis_kind': 'secondary_clean_subset_from_frozen_predictions',
    'source': 'Synthetic only; independently authored but the original corpus had incidental exact training overlaps.',
    'not_a_new_inference_or_blind_evaluation': True,
    'primary_report_unchanged': 'round3-report.json',
    'primary_report_sha256': sha(ROOT / 'round3-report.json'),
    'dataset_sha256': DATASET_HASH,
    'predictions_sha256': PREDICTION_HASH,
    'overlap_audit': {'path': 'round3-training-overlap.json',
                      'sha256': sha(ROOT / 'round3-training-overlap.json'),
                      'training_exact_matches': 2, 'development_exact_matches': 0,
                      'audit_performed_by': 'root agent; this script does not read training text'},
    'exclusions': {'single': [EXCLUDED_SINGLE], 'oracle_clause': [EXCLUDED_CLAUSE],
                   'whole_mixed_message': [EXCLUDED_MESSAGE],
                   'whole_message_reason': 'Conservatively remove the complete mixed message containing the overlapping clause from the secondary component-set metric.'},
    'speech_act': speech,
    'grounding': grounding,
    'mixed_oracle_clause_speech': score(oracle, 'speech_act', scorer.SPEECH),
    'mixed_oracle_clause_grounding': score(oracle, 'grounding', scorer.GROUNDING),
    'mixed_component_n': len(mixed), 'mixed_component_correct': mixed_correct,
    'mixed_component_accuracy': mixed_accuracy, 'mixed': mixed_results,
    'predeclared_gates': gates, 'aggregate_predeclared_pass': all(gates.values()),
    'limitations': [
        'Removing exact matches does not prove template or distribution independence.',
        'Labels are AI-authored, not human-validated.',
        'The known evaluative/feature weakness remains; this analysis excludes a correctly classified trajectory evaluation.',
    ],
}
output = ROOT / 'round3-clean-subset-report.json'
assert not output.exists(), 'Refusing to overwrite an existing clean-subset analysis'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ['speech_act', 'grounding', 'mixed_component_n',
                  'mixed_component_correct', 'mixed_component_accuracy', 'predeclared_gates']}, indent=2))
