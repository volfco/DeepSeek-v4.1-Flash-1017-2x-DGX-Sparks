"""Is the per-transfer producer difference between the ranks (dp = p0 - p1, same sequence) systematic, content-
determined, or random? The mech arms replay identical tokens, so: a systematic part is the per-(position, rows) mean
of dp; a content-determined part repeats on the same sequence in another arm (correlation of the residual across
fx1L and fx1Lb); the rest is jitter. Cost to the pair of each part = mean |.| / 2 per transfer.
Also: the same for each rank's own producer time between two arms (how repeatable one rank's time is)."""
import sys, os, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate

def dps(name):
    g0, _, _ = annotate(name, 0); g1, _, _ = annotate(name, 1)
    m1 = {r['seq']: r for r in g1}
    out = {}
    for a in g0:
        b = m1.get(a['seq'])
        if not b or a['unit'] != 'F' or a['op'] not in ('attn_gather', 'moe_gather') or a['pos'] == 0: continue
        p0 = a['stage'] - a['prev']; p1 = b['stage'] - b['prev']
        if p0 > 5e6 or p1 > 5e6: continue
        out[a['seq']] = (a['op'], a['pos'], a['rows'], p0, p1)
    return out

def corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)

arms = sys.argv[1:] or ['fx1L', 'fx1Lb', 'np1L', 'ns1L']
D = {a: dps(a) for a in arms}
for a in arms:
    d = D[a]
    for op in ('attn_gather', 'moe_gather'):
        rows = [(q, v) for q, v in d.items() if v[0] == op]
        cell = collections.defaultdict(list)
        for q, v in rows: cell[(v[1], v[2])].append(v[3] - v[4])
        mean = {k: sum(v) / len(v) for k, v in cell.items()}
        raw = [v[3] - v[4] for q, v in rows]
        res = [(v[3] - v[4]) - mean[(v[1], v[2])] for q, v in rows]
        sysm = [mean[(v[1], v[2])] for q, v in rows]
        ma = lambda xs: sum(abs(x) for x in xs) / len(xs) / 1e3
        print(f"{a} {op}: n {len(rows)}; signed mean dp {sum(raw)/len(raw)/1e3:+.1f} us; mean|dp| {ma(raw):.1f}; systematic part (per position x rows mean) mean|.| {ma(sysm):.1f}; residual mean|.| {ma(res):.1f}; cost |dp|/2 = {ma(raw)/2:.1f} us a transfer; p50|dp| {pct(sorted(abs(x) for x in raw),50)/1e3:.1f} p90 {pct(sorted(abs(x) for x in raw),90)/1e3:.1f} p99 {pct(sorted(abs(x) for x in raw),99)/1e3:.1f}")
base = arms[0]
for other in arms[1:]:
    for op in ('attn_gather', 'moe_gather'):
        qs = [q for q in D[base] if q in D[other] and D[base][q][0] == op]
        dpa = [D[base][q][3] - D[base][q][4] for q in qs]; dpb = [D[other][q][3] - D[other][q][4] for q in qs]
        p0a = [D[base][q][3] for q in qs]; p0b = [D[other][q][3] for q in qs]
        p1a = [D[base][q][4] for q in qs]; p1b = [D[other][q][4] for q in qs]
        print(f"{base} vs {other} {op}: n {len(qs)}; corr of dp across arms {corr(dpa, dpb):+.3f}; corr of rank0 producer across arms {corr(p0a, p0b):+.3f}; rank1 {corr(p1a, p1b):+.3f}; corr rank0 vs rank1 within {base} {corr(p0a, p1a):+.3f}")
