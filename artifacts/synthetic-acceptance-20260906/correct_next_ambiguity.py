"""Pre-inference authoring QA: replace one evaluative-sounding state adjective.

No candidate inference or training text was consulted. Retain the original seal.
"""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parent
old_path = ROOT / 'frozen-next.json'
old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
assert old_hash == 'de769f40f797d48b1ae5a86bfd47c9e5f83ee5341c8cafc55f23586296168ba2'
data = json.loads(old_path.read_text(encoding='utf-8'))
row = next(r for r in data['mixed'] if r['id'] == 'NM003')
row['text'] = 'This work area is two tiles wide, and I dislike its layout.'
row['clauses'][0]['text'] = 'This work area is two tiles wide'
data['revision'] = {
    'previous_sha256': old_hash,
    'reason': 'Independent authoring QA before any candidate inference found cramped could itself imply evaluation. Replaced that adjective with an objective width to preserve an unambiguous descriptive clause.',
    'changed_ids': ['NM003'], 'gold_labels_changed': False,
    'candidate_training_text_read': False, 'candidate_inference_before_revision': False,
}
target = ROOT / 'frozen-next-v2.json'
assert not target.exists()
content = (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
target.write_bytes(content)
sha = hashlib.sha256(content).hexdigest()
seal = json.loads((ROOT / 'seal-next.json').read_text(encoding='utf-8'))
seal['sha256'] = sha
seal['previous_sha256'] = old_hash
seal['revision'] = data['revision']
(ROOT / 'seal-next-v2.json').write_text(json.dumps(seal, indent=2)+'\n', encoding='utf-8')
(ROOT / 'frozen-next-v2.sha256').write_text(sha+'  frozen-next-v2.json\n', encoding='ascii')
print(json.dumps({'sha256': sha, 'single_count': 150, 'mixed_count': 20, 'uncertain_count': 20}))
