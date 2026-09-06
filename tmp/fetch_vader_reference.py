import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

metadata = json.load(urllib.request.urlopen('https://pypi.org/pypi/vaderSentiment/3.3.2/json'))
entry = next(row for row in metadata['urls'] if row['filename'].endswith('.whl'))
data = urllib.request.urlopen(entry['url']).read()
assert hashlib.sha256(data).hexdigest() == entry['digests']['sha256']
root = Path(__file__).parent / 'vader-reference'
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for item in archive.infolist():
        assert (root / item.filename).resolve().is_relative_to(root.resolve())
    archive.extractall(root)
print(json.dumps({'version': '3.3.2', 'sha256': entry['digests']['sha256'], 'filename': entry['filename']}))
