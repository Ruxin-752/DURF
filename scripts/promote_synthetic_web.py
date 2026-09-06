"""Promote only complete, hash-bound synthetic and stateful acceptance evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADS = {'speech_act': {'evaluative', 'imperative', 'descriptive'},
         'grounding': {'action', 'feature', 'trajectory'}}
INFERENCE_SOURCES = ('web/lib/synthetic-browser-models.ts', 'web/lib/browser-models.ts',
                     'web/lib/route-inputs.ts', 'web/lib/game.ts', 'web/lib/subgoal-policy.ts')
STATEFUL_SOURCES = ('web/lib/route1-grounding.ts', 'web/lib/pragmatic-route1.ts',
                    'web/lib/vader-sentiment.ts', 'web/lib/vendor/vader-data.json',
                    'web/lib/game.ts', 'web/lib/subgoal-policy.ts', 'web/lib/browser-models.ts')
CONTEXTS = {'fresh_game_no_recent_trajectory', 'synthetic_recent_onion_pickup'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def unit_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid bounded metric: ' + name)
    return value


def count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid count: ' + name)
    return value


def verify_hashes(hashes, required):
    if not isinstance(hashes, dict):
        raise ValueError('Missing frozen source hash inventory')
    normalized = {}
    for filename, expected in hashes.items():
        if not isinstance(filename, str) or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
            raise ValueError('Invalid frozen source hash entry')
        path = Path(filename).resolve()
        key = os.path.normcase(str(path))
        if key in normalized:
            raise ValueError('Duplicate normalized source path')
        if digest(path) != expected.lower():
            raise ValueError('Evaluated model or inference source changed: ' + str(path))
        normalized[key] = expected.lower()
    if any(os.path.normcase(str(path.resolve())) not in normalized for path in required):
        raise ValueError('Required evaluated model or inference source hash is missing')
    return normalized


def validate(args, root=ROOT):
    report, predictions, dataset = read(args.report), read(args.predictions), read(args.dataset)
    if report.get('aggregate_predeclared_pass') is not True:
        raise ValueError('Independent synthetic acceptance failed; promotion refused')
    gates = report.get('predeclared_gates')
    if not isinstance(gates, dict) or set(gates) != {'speech_act', 'grounding', 'mixed_components'} or any(value is not True for value in gates.values()):
        raise ValueError('Missing, malformed, or failed predeclared acceptance gate')
    for head, labels in HEADS.items():
        metrics = report[head]
        n = count(metrics.get('n'), head + '.n', 150)
        correct = count(metrics.get('correct'), head + '.correct')
        if correct > n or abs(unit_number(metrics.get('accuracy'), head + '.accuracy') - correct / n) > 1e-8:
            raise ValueError('Metric numerator/denominator mismatch')
        if unit_number(metrics.get('macro_f1'), head + '.macro_f1') < .87:
            raise ValueError(head + ' aggregate acceptance failed')
        classes = metrics.get('per_class')
        if not isinstance(classes, dict) or set(classes) != labels:
            raise ValueError('Missing required class metrics')
        if sum(count(value.get('support'), head + '.support', 1) for value in classes.values()) != n:
            raise ValueError('Class support count mismatch')
        if any(unit_number(value.get('recall'), head + '.recall') < .85 for value in classes.values()):
            raise ValueError(head + ' class recall acceptance failed')
    mixed = report.get('mixed')
    expected_mixed = {row['id'] for row in dataset['mixed']}
    if (not isinstance(mixed, list) or len(mixed) < 20 or len(mixed) != len(expected_mixed) or
            {row['id'] for row in mixed} != expected_mixed or
            any(type(row.get('component_set_correct')) is not bool for row in mixed)):
        raise ValueError('Incomplete mixed-component acceptance')
    mixed_accuracy = sum(row['component_set_correct'] for row in mixed) / len(mixed)
    if mixed_accuracy < .85 or abs(unit_number(report.get('mixed_component_accuracy'), 'mixed_accuracy') - mixed_accuracy) > 1e-8:
        raise ValueError('Mixed-component acceptance failed')
    dataset_sha, prediction_sha = digest(args.dataset), digest(args.predictions)
    if report.get('dataset_sha256') != dataset_sha or report.get('prediction_sha256') != prediction_sha:
        raise ValueError('Acceptance dataset or prediction hash mismatch')
    provenance = predictions['provenance']
    if provenance.get('candidate_frozen_before_inference') is not True or provenance.get('candidate_training_text_read_by_acceptance_agent') is not False:
        raise ValueError('Candidate freezing or acceptance-author independence missing')
    if provenance.get('dataset_sha256') != dataset_sha:
        raise ValueError('Prediction dataset binding mismatch')
    required = [args.dataset, *(args.models / (head + '.json') for head in HEADS),
                *(root / name for name in INFERENCE_SOURCES)]
    frozen_hashes = verify_hashes(provenance.get('hashes'), required)
    stateful = read(args.stateful_report)
    if stateful.get('dataset_sha256') != dataset_sha or stateful.get('predictions_sha256') != prediction_sha:
        raise ValueError('Stateful acceptance dataset/prediction binding mismatch')
    verify_hashes(stateful.get('source_hashes'), [root / name for name in STATEFUL_SOURCES])
    uncertain = dataset.get('uncertain')
    if not isinstance(uncertain, list) or len(uncertain) != 20 or len({row['id'] for row in uncertain}) != 20:
        raise ValueError('Expected twenty unique uncertain cases')
    expected_pairs = {(row['id'], context) for row in uncertain for context in CONTEXTS}
    records = stateful.get('records')
    if (not isinstance(records, list) or len(records) != len(expected_pairs) or
            {(row['id'], row['context']) for row in records} != expected_pairs or
            any(row.get('learning_applied') is not False or row.get('update_status') != 'rejected' for row in records)):
        raise ValueError('Missing contexts or unsafe learning in stateful acceptance')
    artifacts, entries, copies = {}, {}, []
    public = root / 'web/public/models'
    for head in HEADS:
        source = args.models / (head + '.json')
        content = source.read_bytes()
        artifact = json.loads(content.decode('utf-8-sig'))
        if artifact.get('training_scope') != 'synthetic_only' or artifact.get('target_semantics') != head:
            raise ValueError('Wrong model scope or semantics')
        key = 'speech_model_source_sha256' if head == 'speech_act' else 'grounding_model_source_sha256'
        if artifact['source']['model_sha256'] != provenance.get(key):
            raise ValueError('Model source identity mismatch')
        sha = hashlib.sha256(content).hexdigest()
        if sha != frozen_hashes[os.path.normcase(str(source.resolve()))]:
            raise ValueError('Model changed while preparing promotion')
        target = public / ('synthetic-' + head + '-' + sha + '.json')
        if target.exists() and digest(target) != sha:
            raise ValueError('Immutable target artifact collision')
        entries[head] = {'path': '/models/' + target.name, 'sha256': sha}
        artifacts[head] = artifact
        copies.append((target, content))
    route2 = public / 'route2-v5.json'
    if read(route2).get('claim_scope', {}).get('trained_on_synthetic_overcooked_feedback') is not True:
        raise ValueError('Retained Route2 artifact lacks synthetic training provenance')
    entries['route2'] = {'path': '/models/route2-v5.json', 'sha256': digest(route2)}
    manifest = {
        'schema_version': 'durf-synthetic-feedback-release-v1',
        'training_scope': 'synthetic_only', 'supported_language': 'en',
        'release_id': args.release_id, 'models': entries,
        'evidence': {
            'speech_model_sha256': artifacts['speech_act']['source']['model_sha256'],
            'grounding_model_sha256': artifacts['grounding']['source']['model_sha256'],
            'synthetic_test_sha256': dataset_sha, 'report_sha256': digest(args.report),
            'stateful_report_sha256': digest(args.stateful_report),
            'speech_accuracy': report['speech_act']['accuracy'],
            'speech_macro_f1': report['speech_act']['macro_f1'],
            'grounding_accuracy': report['grounding']['accuracy'],
            'rows': report['speech_act']['n'], 'human_accuracy': None, 'passed': True,
        },
    }
    return manifest, copies


def atomic_write(target, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def promote(args, root=ROOT):
    manifest, copies = validate(args, root)
    # Every acceptance/hash/collision check is complete before release writes.
    for target, content in copies:
        atomic_write(target, content)
    target = root / 'web/public/models/manifest-synthetic-v1.json'
    atomic_write(target, (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    return {'manifest': str(target), 'sha256': digest(target), 'release_id': args.release_id}


def main():
    parser = argparse.ArgumentParser()
    for name in ('models', 'report', 'dataset', 'predictions', 'stateful-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-id', required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args)))


if __name__ == '__main__':
    main()

