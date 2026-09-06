"""Read deployed research export; print aggregate diagnostics, never participant text."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/baseline_b_linguistic_feedback/adapted_overcooked'))
from scripts import prepare_web_retraining_data as pipeline

token = next(line.split('=', 1)[1].strip().strip('\"\'') for line in (ROOT / 'web/.env.local').read_text(encoding='utf-8-sig').splitlines() if line.strip().startswith('ADMIN_EXPORT_TOKEN='))
origin = 'https://durf-kitchen-lab-study.xc3083.chatgpt.site'
request = urllib.request.Request(origin + '/api/research/export', headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(request, timeout=45) as response:
    content = response.read()
    status = response.status
rows = [json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
groups = {kind: [r for r in rows if r['recordType'] == kind] for kind in ['session', 'event', 'feedback']}
sessions = {r['session_id']: r for r in groups['session']}
events = {r['event_id']: r for r in groups['event']}
consent = 'durf-anonymous-research-en-2026-08-27-v3'
eligible = {k for k, r in sessions.items() if r['consent_version'] == consent}
selected_rows = [r for r in rows if r['session_id'] in eligible]
feedback_events = [events[f['event_id']] for f in groups['feedback']]
count = lambda rs, key: dict(collections.Counter(str(r.get(key)) for r in rs))
iso = lambda ms: datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).isoformat()
def replay(items, selected_consent, filter_cohort=False):
    outputs = {}
    try:
        with patch.object(pipeline, '_read_jsonl', return_value=items), patch.object(pipeline, '_write_jsonl', side_effect=lambda p, v: outputs.update({p.name: list(v)})), patch.object(pipeline, '_write_json', side_effect=lambda p, v: outputs.update({p.name: v})), patch.object(pipeline, '_sha256_file', return_value=hashlib.sha256(content).hexdigest()), patch.object(Path, 'mkdir'):
            manifest = pipeline.prepare_export(Path('in-memory-web-export.jsonl'), Path('in-memory-only'), initialize_split_registry=True, consent_version=None if filter_cohort else selected_consent, filter_consent_version=selected_consent if filter_cohort else None)
        return {'ok': True, 'record_counts': manifest['record_counts'], 'split_counts': manifest['split_policy']['split_task_counts'], 'route2': manifest['route2'], 'trajectory_sources': count(outputs['route2_unlabeled.json'], 'trajectory_source'), 'consent_selection': manifest['consent_selection']}
    except ValueError as error:
        message = str(error)
        for r in rows:
            for key in ('session_id', 'feedback_id', 'event_id', 'anonymous_user_id'):
                if r.get(key): message = message.replace(str(r[key]), '[id]')
        return {'ok': False, 'error': message}
summary = {
    'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'authenticated_export_status': status, 'export_bytes': len(content),
    'export_sha256': hashlib.sha256(content).hexdigest(),
    'counts': {k: len(v) for k, v in groups.items()},
    'participant_count': len({r['anonymous_user_id'] for r in groups['session']}),
    'feedback_participant_count': len({sessions[f['session_id']]['anonymous_user_id'] for f in groups['feedback']}),
    'session_schema_counts': count(groups['session'], 'schema_version'),
    'event_type_counts': count(groups['event'], 'event_type'),
    'feedback_route_counts': count(groups['feedback'], 'route'),
    'feedback_online_prediction_counts': count(groups['feedback'], 'top_label'),
    'feedback_consent_counts': dict(collections.Counter(sessions[f['session_id']]['consent_version'] for f in groups['feedback'])),
    'feedback_phrase_count': sum(len(f['phrases']) for f in groups['feedback']),
    'current_consent_counts': dict(collections.Counter(r['recordType'] for r in selected_rows)),
    'current_consent_participant_count': len({sessions[k]['anonymous_user_id'] for k in eligible}),
    'event_time_range_utc': [iso(min(e['occurred_at'] for e in groups['event'])), iso(max(e['occurred_at'] for e in groups['event']))],
    'feedback_time_range_utc': [iso(min(f['created_at'] for f in groups['feedback'])), iso(max(f['created_at'] for f in groups['feedback']))],
    'event_feedback_model_hash_mismatches': sum(events[f['event_id']].get('model_hash') not in (None, f['model_hash']) for f in groups['feedback']),
    'feedback_payload_classifier_hash_count': sum('classifierModelHash' in e.get('payload', {}) for e in feedback_events),
    'feedback_payload_updater_hash_count': sum('updaterModelHash' in e.get('payload', {}) for e in feedback_events),
    'feedback_payload_recent_trajectory_count': sum('recentTrajectoryFeatures' in e.get('payload', {}).get('gameSnapshot', {}) for e in feedback_events),
    'feedback_payload_snapshot_feature_count': sum('features' in e.get('payload', {}).get('gameSnapshot', {}) for e in feedback_events),
    'tick_summary_receipt_count': sum('receiptSchemaVersion' in e.get('payload', {}) for e in groups['event'] if e['event_type'] == 'tick_summary'),
    'tick_summary_transition_count': sum('stateBefore' in e.get('payload', {}) and 'stateAfter' in e.get('payload', {}) for e in groups['event'] if e['event_type'] == 'tick_summary'),
    'orphan_events': sum(e['session_id'] not in sessions for e in groups['event']),
    'orphan_feedback': sum(f['event_id'] not in events or f['session_id'] not in sessions for f in groups['feedback']),
    'duplicate_session_sequence_count': len(groups['event']) - len({(e['session_id'], e['sequence_number']) for e in groups['event']}),
    'replay_full_export_with_current_consent': replay(rows, consent),
    'replay_full_export_without_consent_filter_for_structural_audit_only': replay(rows, None),
    'replay_current_consent_subset': replay(selected_rows, consent),
    'replay_explicit_consent_filter_full_export': replay(rows, consent, True),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
