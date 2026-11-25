"""Split a trace into busy segments at idle gaps (the exact elapsed time between consecutive sampled records is
known even though only one tile in N is recorded), and describe each: forwards, streams (max rows), duration."""
import sys, os, collections, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate

def segments(name, rank, idle=60e6):
    g, units, x = annotate(name, rank, greedy=name in ('fx1L', 'fx1Lb', 'np1L', 'ns1L'))
    segs = []; cur = [g[0]]
    for a, b in zip(g, g[1:]):
        if b['copy'] - a['copy'] > idle:
            segs.append(cur); cur = []
        cur.append(b)
    segs.append(cur)
    return g, x, segs

if __name__ == '__main__':
    name = sys.argv[1]; rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    g, x, segs = segments(name, rank)
    Fl = [a for a, _ in x['anchors']]
    print(name, 'rank', rank, 'segments', len(segs))
    for i, s in enumerate(segs):
        fw = {r['F'] for r in s if r['unit'] == 'F'}
        rows = collections.Counter(r['rows'] for r in s if r['op'] == 'attn_gather')
        head = sum(1 for r in s if r['op'] == 'head')
        kvp = sorted(r['stage'] - r['prev'] for r in s if r['op'] == 'kv')
        dur = (s[-1]['copy'] - s[0]['prev']) / 1e9
        if len(segs) - i <= 14 or len(fw) > 90:
            print(f"  seg {i}: seq {s[0]['seq']}..{s[-1]['seq']} forwards {len(fw)} dur {dur:.2f}s maxrows {max(rows) if rows else 0} head recs {head} kv producer p50 {kvp[len(kvp)//2]/1e3 if kvp else 0:.0f} us; first tile {(s[0]['copy']-s[0]['prev'])/1e6:.1f} ms")
