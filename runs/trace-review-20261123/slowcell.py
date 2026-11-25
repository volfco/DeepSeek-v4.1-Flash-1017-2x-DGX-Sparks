"""Lead 5: where a 'slow cell' is slow. The k272 arms tr16a and tr16b ran the same deterministic one-stream workload
cells (same rounds, same tokens); the saved GPU-clock forward per round differs with no code change (32K x 1: 33,523
vs 32,556 us; 128K x 1: 33,056 vs 32,462 us; 4K x 1: 33,503 vs 33,522 us as the control). The cells are located in the
traces by their forward counts (97 / 114 / 126, one stream, at the end of the arm) and compared class by class.
Per class: mean producer and ring per transfer, times the transfers a round = ms a round. Layer classes: attention at
plain layers, the KV layers' three parts, Engram layers, MoE (layer-standardized: mean over the 40 layers of the
per-layer mean, so a different sampled layer mix does not move it).
usage: slowcell.py"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, KV, ENGRAM
from segments import segments

def cell(name, rank, nfwd):
    g, x, segs = segments(name, rank)
    cands = []
    for s in segs:
        fw = {r['F'] for r in s if r['unit'] == 'F'}
        rows = [r['rows'] for r in s if r['op'] == 'attn_gather']
        if len(fw) == nfwd and rows and max(rows) <= 7: cands.append(s)
    return cands[-1] if cands else None   # the workload runs last in the arm

def classes(s):
    out = collections.defaultdict(list)
    lay = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in s:
        if r['unit'] != 'F': 
            if r['unit'] == 'P':
                out[('pass', 'tile')].append(r['copy'] - r['prev'])
            continue
        p = r['stage'] - r['prev']; rg = r['copy'] - r['stage']
        if p > 20e6: continue
        op, l = r['op'], r['layer']
        if op == 'attn_gather':
            k = 'attn L0' if l == 0 else 'attn after kv' if l in KV else 'attn plain'
        elif op == 'moe_gather': k = 'moe'
        else: k = op
        out[(k, 'prod')].append(p); out[(k, 'ring')].append(rg)
        out[(k, 'wait')].append(r['flag'] - r['collect'])
        if op in ('moe_gather', 'attn_gather', 'kv'): lay[(op, 'prod')][l].append(p)
    return out, lay

PER_ROUND = {'attn L0': 1, 'attn plain': 31, 'attn after kv': 8, 'kv': 8, 'engram': 2, 'moe': 40, 'head': 1}
for label, nfwd in (('4K x 1 (control)', 97), ('32K x 1', 114), ('128K x 1', 126)):
    print(f"## {label}: {nfwd} forwards")
    for rank in (1, 0):
        A = cell('tr16a', rank, nfwd); B = cell('tr16b', rank, nfwd)
        if not A or not B: print('  cell not found', rank); continue
        da = (A[-1]['copy'] - A[0]['copy']) / 1e6; db = (B[-1]['copy'] - B[0]['copy']) / 1e6
        ca, la = classes(A); cb, lb = classes(B)
        print(f"  rank {rank}: tr16a {da:.1f} ms over the cell, tr16b {db:.1f} ms; per forward {da/nfwd:.3f} vs {db/nfwd:.3f} ms (difference {1e3*(da-db)/nfwd:+.0f} us a round)")
        tot = 0
        for k, n in PER_ROUND.items():
            for f in ('prod', 'ring'):
                va, vb = ca.get((k, f), []), cb.get((k, f), [])
                if not va or not vb: continue
                ma, mb = sum(va) / len(va), sum(vb) / len(vb)
                if k == 'moe' and f == 'prod':
                    # layer-standardized
                    ls = [l for l in range(40) if la[('moe_gather', 'prod')].get(l) and lb[('moe_gather', 'prod')].get(l)]
                    ma = sum(sum(la[('moe_gather', 'prod')][l]) / len(la[('moe_gather', 'prod')][l]) for l in ls) / len(ls)
                    mb = sum(sum(lb[('moe_gather', 'prod')][l]) / len(lb[('moe_gather', 'prod')][l]) for l in ls) / len(ls)
                sa = stats(va); sb = stats(vb)
                d = (ma - mb) * n
                tot += d
                print(f"      {k:<14}{f:<5} n {len(va):>4}/{len(vb):<4} mean {ma/1e3:>7.1f} vs {mb/1e3:>7.1f} us (p50 {sa['p50']/1e3:>7.1f} vs {sb['p50']/1e3:>7.1f}) x{n:<2} = {d/1e3:>+7.0f} us a round")
        print(f"      sum of class differences {tot/1e3:+.0f} us a round (forward only)")
