"""Vendor the exact VADER 3.3.2 tables and synthetic differential test cases.

Run with VADER 3.3.2 on PYTHONPATH. No participant data is read.
"""
import ast
import hashlib
import json
from pathlib import Path

import vaderSentiment.vaderSentiment as vader

ROOT = Path(__file__).resolve().parents[1]
module_path = Path(vader.__file__)
analyzer = vader.SentimentIntensityAnalyzer()
source = module_path.read_text(encoding='utf-8')
tables = {
    'source': 'vaderSentiment 3.3.2',
    'source_sha256': hashlib.sha256(module_path.read_bytes()).hexdigest(),
    'lexicon': analyzer.lexicon,
    'emojis': analyzer.emojis,
    'boosters': vader.BOOSTER_DICT,
    'negators': vader.NEGATE,
    'specialCases': vader.SPECIAL_CASES,
}
vendor = ROOT / 'lib' / 'vendor'
vendor.mkdir(exist_ok=True)
(vendor / 'vader-data.json').write_text(json.dumps(tables, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
license_path = module_path.parent.parent / 'vaderSentiment-3.3.2.dist-info' / 'LICENSE.txt'
(vendor / 'VADER-LICENSE.txt').write_text(license_path.read_text(encoding='utf-8'), encoding='utf-8')

tree = ast.parse(source)
examples = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sentences' for t in node.targets):
        try:
            examples.extend(ast.literal_eval(node.value))
        except (ValueError, TypeError):
            pass
examples += ['', 'Not good.', 'NOT good.', 'not GOOD.', 'NOT GOOD.', 'No good.', 'No GOOD!', 'Never good.',
             'Pick an onion.', 'Do not pick onions.', 'Stop picking onions.', 'The pot is empty.',
             'You are blocking my path.', 'The path is blocked.', 'Onions are useful.', 'The onion was bad.',
             'You did a good job.', 'You did a BAD job!!!', 'good good but good good', 'Good but not good',
             'Good But not good', 'The soup is not very good.', 'There is no doubt it is good.',
             'This is without doubt good.', 'This is never so good.', 'This is least good.',
             'This is at least good.', 'This is kind of good.', 'That was the bomb.', 'I love this 😁']
for adjective in ['good', 'bad', 'excellent', 'terrible', 'helpful', 'annoying', 'correct', 'wrong', 'slow']:
    for prefix in ['', 'not ', 'Not ', 'NOT ', 'not very ', 'never so ', 'no ', 'barely ', 'really ']:
        examples += [prefix + adjective + ending for ending in ['.', '!', '???']]
examples = list(dict.fromkeys(examples))
fixture = {'source': 'official Python vaderSentiment 3.3.2', 'source_sha256': tables['source_sha256'],
           'provenance': 'Upstream public demonstration sentences and generated synthetic cases only.',
           'cases': [{'text': text, 'scores': analyzer.polarity_scores(text)} for text in examples]}
(ROOT / 'tests' / 'fixtures' / 'vader-python-parity.json').write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'lexicon_entries': len(tables['lexicon']), 'emoji_entries': len(tables['emojis']), 'parity_cases': len(examples)}))
