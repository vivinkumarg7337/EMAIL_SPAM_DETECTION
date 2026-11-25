"""Verify deployed single/batch decisions match the saved learned threshold."""
import io
import json
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8000'
threshold = json.loads((ROOT / 'model/decision.json').read_text())['threshold']
pipe = joblib.load(ROOT / 'model/pipeline.joblib')
df = pd.read_csv(ROOT / 'data/validation.csv')
p = pipe.predict_proba(df.text)[:, 1]
# Check emails nearest the boundary, where rounding and stale thresholds matter.
df = df.iloc[abs(p - threshold).argsort()[:12]].reset_index(drop=True)
expected = pipe.predict_proba(df.text)[:, 1] >= threshold
for i, row in df.iterrows():
    req = urllib.request.Request(BASE + '/api/predict',
        data=json.dumps(dict(subject=row.subject, body=row.body)).encode(),
        headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(req))
    assert result['threshold'] == threshold
    assert (result['label'] == 'spam') == expected[i]

boundary = 'mailguard-deployment-check'
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="check.csv"\r\n'
        'Content-Type: text/csv\r\n\r\n').encode()
body += df.to_csv(index=False).encode() + f'\r\n--{boundary}--\r\n'.encode()
req = urllib.request.Request(BASE + '/api/batch', data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
batch = pd.read_csv(io.BytesIO(urllib.request.urlopen(req).read()))
assert ((batch.prediction == 'spam').to_numpy() == expected).all()
assert urllib.request.urlopen(BASE + '/').status == 200
print(f'PASS: single and batch predictions agree with threshold {threshold:.8f} on 12 boundary cases')
