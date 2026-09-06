import hashlib
import json
from pathlib import Path
import urllib.request

origin = 'https://durf-kitchen-lab-study.xc3083.chatgpt.site'
def fetch(path):
    with urllib.request.urlopen(urllib.request.Request(origin + path, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as response:
        return response.read()
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
manifest = json.loads(fetch('/models/manifest.json'))
entry = manifest['models']['feedback_form']
remote_bytes = fetch(entry['path'])
remote = json.loads(remote_bytes)
local_bytes = (Path(__file__).resolve().parents[1] / 'web/public/models/feedback-form-v3.json').read_bytes()
local = json.loads(local_bytes)
fields = ['classes', 'transformers', 'classifier', 'calibration', 'minimum_confidence']
projection = lambda obj: {key: obj[key] for key in fields}
print(json.dumps({
    'remote_artifact_sha256': hashlib.sha256(remote_bytes).hexdigest(),
    'remote_matches_manifest': hashlib.sha256(remote_bytes).hexdigest() == entry['sha256'],
    'local_artifact_sha256': hashlib.sha256(local_bytes).hexdigest(),
    'remote_inference_sha256': digest(projection(remote)),
    'local_inference_sha256': digest(projection(local)),
    'inference_fields_equal': projection(remote) == projection(local),
    'per_field_equal': {field: remote[field] == local[field] for field in fields},
    'other_equal_fields': {field: remote.get(field) == local.get(field) for field in ['canonical_labels', 'model_type', 'model_version', 'source', 'schema_version']},
    'remote_schema_version': remote['schema_version'],
    'local_schema_version': local['schema_version'],
}, indent=2))
