"""Verify public model bytes and an isolated administrator DB roundtrip.

Does not download participant records, click consent, or create study sessions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://durf-kitchen-lab-study.xc3083.chatgpt.site'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(NoRedirect)

    def request(path, *, data=None, token=None):
        headers = {'Origin': ORIGIN, 'Cache-Control': 'no-cache',
                   'User-Agent': 'Mozilla/5.0 (compatible; DURF-Release-Verification/1.0)'}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(ORIGIN + path, data=data, headers=headers)
        try:
            with opener.open(req, timeout=40) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    local = ROOT / 'web/public/models/manifest-synthetic-v1.json'
    status, manifest_bytes = request('/models/manifest-synthetic-v1.json')
    if status != 200 or manifest_bytes != local.read_bytes():
        raise ValueError(f'Public manifest verification failed: HTTP {status}, '
                         f'bytes={len(manifest_bytes)}, sha256={hashlib.sha256(manifest_bytes).hexdigest()}')
    manifest = json.loads(manifest_bytes)
    model_checks = {}
    for head, entry in manifest['models'].items():
        if not re.fullmatch(r'/models/[A-Za-z0-9._-]+\.json', entry['path']):
            raise ValueError('Unexpected model path')
        status, content = request(entry['path'])
        sha = hashlib.sha256(content).hexdigest()
        model_checks[head] = {'status': status, 'sha256': sha, 'matches_release': sha == entry['sha256']}
        if status != 200 or sha != entry['sha256']:
            raise ValueError('Public model bytes differ from acceptance')
    page_status, page = request('/model-check')
    export_status, _ = request('/api/research/export')
    unauthorized_status, _ = request('/api/research/diagnostic', data=b'{}')
    token = None
    for line in (ROOT / 'web/.env.local').read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('ADMIN_EXPORT_TOKEN='):
            token = line.split('=', 1)[1].strip().strip('"\'')
    if not token or len(token) < 16:
        raise ValueError('Local administrator credential unavailable')
    diagnostic_status, content = request('/api/research/diagnostic', data=b'{}', token=token)
    body = json.loads(content)
    result = {
        'origin': ORIGIN, 'release_id': manifest['release_id'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'models': model_checks, 'model_check_status': page_status,
        'model_check_heading_present': b'Feedback classification check' in page,
        'unauthorized_export_status': export_status,
        'unauthorized_diagnostic_status': unauthorized_status,
        'diagnostic_status': diagnostic_status,
        'diagnostic': {k: body.get(k) for k in ('ok', 'diagnostic_version', 'schema_version', 'client_version', 'no_research_rows_written', 'data_roundtrip', 'scope')},
        'participant_data_downloaded': False, 'study_sessions_created': False,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    if (page_status != 200 or export_status != 401 or unauthorized_status != 401 or
            diagnostic_status != 200 or body.get('ok') is not True or
            body.get('no_research_rows_written') is not True or
            body.get('data_roundtrip') != {'inserted': True, 'read_back_matches': True, 'deleted': True}):
        raise ValueError('Public release verification failed; inspect the saved summary')


if __name__ == '__main__':
    main()
