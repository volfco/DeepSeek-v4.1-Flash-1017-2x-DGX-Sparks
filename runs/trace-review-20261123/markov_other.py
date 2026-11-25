"""Lead 2, second files: the Markov-step producer interval in the other traces (no step index needed: step 0 is the
interval near 1.1-1.25 ms because it holds the vocabulary head; steps 1+ are below 0.5 ms). One-stream steps
(rows == 1) and four-stream steps (rows == 4, where a step is slow if any of its four rows misses).
usage: markov_other.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
def hist(vals, edges):
    c = [0] * (len(edges) + 1)
    for v in vals:
        i = 0
        while i < len(edges) and v >= edges[i]: i += 1
        c[i] += 1
    return c
E = [8, 12, 20, 40, 60, 100, 130, 145, 160, 180, 220, 300, 500]
print('# histogram edges (us):', E)
for name in sys.argv[1:]:
    for rank in (0, 1):
        g = load(name, rank)
        for rows in (1, 4):
            v = [r['stage'] - r['prev'] for r in g if r['op'] == 'other' and r['layer'] == -1 and r['rows'] == rows and r['bytes'] == 16 * rows]
            v = [t for t in v if t < 500e3]
            if len(v) < 30: continue
            fast = sorted(t for t in v if t < 60e3); slow = sorted(t for t in v if t >= 60e3)
            print(f"{name} r{rank} rows {rows}: steps 1+ n {len(v)} hist {hist([t/1e3 for t in v], E)}; fast p50 {pct(fast,50)/1e3:.1f} us (n {len(fast)}); slow p50 {pct(slow,50)/1e3:.1f} us (n {len(slow)}, {100*len(slow)/len(v):.1f}%); gap {(pct(slow,50)-pct(fast,50))/1e3:.1f} us; mean {sum(v)/len(v)/1e3:.1f} us")
