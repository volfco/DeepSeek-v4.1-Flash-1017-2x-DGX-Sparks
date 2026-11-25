"""Lead 4 (continued): per-layer least-squares fit producer = a + b*rows for the MoE and attention intervals.
A routed expert place costs 27.2 us (runs/w4/RESULTS.md), so b/27.2 reads as distinct new expert places per added row.
For KV layers the attention work is kv.producer + kv.ring + attn.producer and for Engram layers the engram tile is
added; those sums are formed from per-(position, rows) means because the parts are never sampled together.
usage: layer_fit.py arm [arm ...]   (ranks pooled per arm unless --rank r)"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, TEMPLATE, OFF, KV, ENGRAM
from drains import drain_starts

def cells(name, rank):
    g, units, x = annotate(name, rank)
    ds = drain_starts(name, rank)
    urows = {}
    for r in g:
        if r['unit'] == 'F' and r['op'] in ('attn_gather', 'moe_gather'): urows[r['round']] = r['rows']
    c = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in g:
        if r['unit'] != 'F' or r['round'] not in urows: continue
        if r['pos'] == 0 and r['seq'] in ds: continue
        if r['copy'] - r['prev'] > 20e6: continue
        rows = urows[r['round']]
        c[(r['pos'], 'prod')][rows].append(r['stage'] - r['prev'])
        c[(r['pos'], 'ring')][rows].append(r['copy'] - r['stage'])
    return c

def fit(pairs):
    """pairs: list of (rows, value, weight). Weighted least squares; returns a, b."""
    sw = sum(w for _, _, w in pairs); sx = sum(w * x for x, _, w in pairs); sy = sum(w * y for _, y, w in pairs)
    sxx = sum(w * x * x for x, _, w in pairs); sxy = sum(w * x * y for x, y, w in pairs)
    b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx); a = (sy - b * sx) / sw
    return a, b

def layer_table(name, ranks=(0, 1)):
    out = {}
    cs = [cells(name, r) for r in ranks]
    for l in range(40):
        for kind in ('attn', 'moe'):
            pairs = []
            for rows in range(2, 8):
                tot = 0; n = 1e9; ok = True
                for c in cs:
                    if kind == 'moe':
                        parts = [(OFF[('moe_gather', l)], 'prod')]
                    else:
                        parts = [(OFF[('attn_gather', l)], 'prod')]
                        if l in KV: parts += [(OFF[('kv', l)], 'prod'), (OFF[('kv', l)], 'ring')]
                        if l in ENGRAM: parts += [(OFF[('engram', l)], 'prod'), (OFF[('engram', l)], 'ring')]
                    for p in parts:
                        v = c[p].get(rows)
                        if not v: ok = False; break
                        tot += sum(v) / len(v) / len(cs); n = min(n, len(v))
                if ok: pairs.append((rows, tot, n))
            a, b = fit(pairs)
            out[(l, kind)] = (a, b, a + 6 * b, sum(w for _, _, w in pairs))
    return out

if __name__ == '__main__':
    arms = sys.argv[1:]
    T = {a: layer_table(a) for a in arms}
    print('# per layer: value at 6 rows (us), slope (us/row); attention includes kv/engram tiles at those layers; ranks pooled')
    print('layer ' + ' '.join(f'{a + " attn@6":>14}{"slope":>7}{a + " moe@6":>14}{"slope":>7}{"places/row":>11}' for a in arms))
    for l in range(40):
        s = ''
        for a in arms:
            ta = T[a][(l, 'attn')]; tm = T[a][(l, 'moe')]
            s += f'{ta[2] / 1e3:>14.1f}{ta[1] / 1e3:>7.1f}{tm[2] / 1e3:>14.1f}{tm[1] / 1e3:>7.1f}{tm[1] / 27.2e3:>11.2f}'
        tag = ' engram' if l in ENGRAM else ''
        tag += ' kv' if l in KV else ''
        print(f'{l:>5} {s}{tag}')
    for a in arms:
        print(f"# {a}: sum over layers at 6 rows: attention {sum(T[a][(l, 'attn')][2] for l in range(40)) / 1e6:.3f} ms, MoE {sum(T[a][(l, 'moe')][2] for l in range(40)) / 1e6:.3f} ms; MoE slope sum {sum(T[a][(l, 'moe')][1] for l in range(40)) / 1e3:.1f} us/row, attention slope sum {sum(T[a][(l, 'attn')][1] for l in range(40)) / 1e3:.1f} us/row")
