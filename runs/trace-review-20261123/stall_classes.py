"""Stalls at the stretch boundaries (a forward is issued as three graph stretches: layer 0 | layers 1-13 | layers
14-39 + head, cut before each Engram layer for the host's Engram landing, round.zig forward()). The first transfer of
each stretch (attn L0, engram L1, engram L14) carries the host step in its producer interval. Classify by which rank
was late: both (same host-side event on both ranks: a graph capture on a new key is the candidate), rank 0 only,
rank 1 only (a rank-local host delay: the Engram read's landing is the candidate).
Pair cost of an event = max over ranks of the producer excess over that rank's median.
usage: stall_classes.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
THR = 0.5e6
for name in sys.argv[1:]:
    n = FILES[name][1]
    g0, u0, x0 = annotate(name, 0, greedy=False); g1, u1, x1 = annotate(name, 1, greedy=False)
    m1 = {r['seq']: r for r in g1}
    fw = len(x0['anchors'])
    print(f"{name}: forwards {fw}, sampling 1 in {n}")
    for lab, sel in (('engram L1 (stretch 2 start)', lambda r: r['op'] == 'engram' and r['layer'] == 1),
                     ('engram L14 (stretch 3 start)', lambda r: r['op'] == 'engram' and r['layer'] == 14),
                     ('all other forward transfers', lambda r: r['unit'] == 'F' and r['op'] != 'engram' and r['pos'] > 0)):
        rs = [(r, m1[r['seq']]) for r in g0 if sel(r) and r['seq'] in m1 and r['stage'] - r['prev'] < 50e6]
        if not rs: continue
        med0 = pct(sorted(a['stage'] - a['prev'] for a, b in rs), 50); med1 = pct(sorted(b['stage'] - b['prev'] for a, b in rs), 50)
        cls = collections.Counter(); cost = collections.Counter()
        for a, b in rs:
            if 'engram' not in lab:
                # producer medians differ by position: use the ring wait instead (a late peer shows as this rank's wait)
                e0 = b['flag'] - b['collect']; e1 = a['flag'] - a['collect']   # rank 0 late <=> rank 1 waits
            else:
                e0 = (a['stage'] - a['prev']) - med0; e1 = (b['stage'] - b['prev']) - med1
            k = 'both late' if e0 > THR and e1 > THR else 'rank 0 late' if e0 > THR else 'rank 1 late' if e1 > THR else None
            if k: cls[k] += 1; cost[k] += max(e0, e1)
        tot = sum(cost.values())
        print(f"   {lab}: sampled {len(rs)}; events > 0.5 ms: {dict(cls)}; pair cost {tot/1e6:.1f} ms sampled -> {tot*n/1e6:.0f} ms over the arm = {tot*n/fw/1e3:.0f} us a forward; by class (us a forward): " + ', '.join(f"{k} {v*n/fw/1e3:.0f}" for k, v in cost.items()))
