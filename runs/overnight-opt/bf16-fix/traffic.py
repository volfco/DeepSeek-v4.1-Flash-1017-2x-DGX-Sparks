"""Aggregate uncached suffix lengths in the recorded external interval. No request identifiers are retained."""
import hashlib
import json
from pathlib import Path
import statistics


root = Path(__file__).resolve().parents[2]
archive = root / 'overnight/break-1600Z.rank0.log'
rows = []
for line in archive.open():
    if line[:19] < '2026-11-13T07:20:00':
        continue
    if '127.0.0.1 "POST /v1/chat/completions' in line:
        raise ValueError('External cohort included local chat')
    if '[tensorfold] done ' not in line:
        continue
    fields = dict(word.split('=', 1) for word in line.split() if '=' in word)
    cached = int(fields['cached'])
    suffix = int(fields['prompt']) - cached
    if suffix < 0:
        raise ValueError('Cached length exceeded input length')
    rows.append((suffix, float(fields['prefill_s']), cached))

sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert len(rows) == 652 and sha == '3244da69e2fec01f386291552fea3246d256e5d1b069396cbf3c00d53fff998f'
cohorts = []
for low, high in [(0, 128), (129, 512), (513, 2048), (2049, 8192), (8193, None)]:
    selected = [r for r in rows if r[0] >= low and (high is None or r[0] <= high)]
    cohorts.append(dict(min_tokens=low, max_tokens=high, requests=len(selected),
                        positive_cuts=sum(r[2] > 0 for r in selected),
                        prefill_seconds=round(sum(r[1] for r in selected), 3)))
print(json.dumps(dict(archive_sha256=sha, requests=len(rows), positive_cuts=sum(r[2] > 0 for r in rows),
                      min_uncached_tokens=min(r[0] for r in rows), median_uncached_tokens=statistics.median(r[0] for r in rows),
                      max_uncached_tokens=max(r[0] for r in rows), cohorts=cohorts), indent=2))
