"""Read frozen synthetic data after model selection and record exact overlap."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def norm(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--sources', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = read(args.dataset)
    gold = [(r['id'], r['text']) for r in data['singles'] + data['mixed'] + data['uncertain']]
    gold += [(r['id'] + ':C' + str(i + 1), c['text']) for r in data['mixed'] for i, c in enumerate(r['clauses'])]
    result = {
        'dataset_sha256': sha(args.dataset),
        'scope': 'Exact normalized overlap only; independently authored AI synthetic labels are not human validation.',
        'sources': [],
    }
    for source in args.sources:
        rows = read(source)
        if not isinstance(rows, list):
            rows = rows['rows']
        if any(r.get('human_data') is not False for r in rows):
            raise ValueError('Missing explicit synthetic-only provenance')
        texts = {norm(r['text']) for r in rows}
        overlap_ids = [key for key, text in gold if norm(text) in texts]
        result['sources'].append({
            'path': str(source), 'sha256': sha(source), 'rows': len(rows),
            'all_rows_human_data_false': True,
            'overlap_count': len(overlap_ids), 'overlap_ids': overlap_ids,
        })
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
