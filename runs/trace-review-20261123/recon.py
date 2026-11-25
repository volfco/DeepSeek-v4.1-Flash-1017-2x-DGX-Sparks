"""Position reconstruction for the 0272 ring traces.

Every ring transfer takes one ring sequence (seq). A decode forward (Model.verify) issues a fixed template of
transfers (round.zig stretch(): per layer [engram at layers 1,14] [kv at 2,8,14,20,24,28,32,36] attn_gather
moe_gather; then, on a full-logit round, the head gather). A drafter pass (draft.zig body()) issues 6 hidden gathers
(stage j attention then stage j MoE, both keyed `other`, layer j) and then one gather per Markov step (`other`,
layer -1). Only every Nth seq is recorded, so a record's position is recovered from its seq relative to the forward
anchor F (the seq of attn_gather layer 0) of its own / the neighbouring work units.
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import load

ENGRAM = (1, 14)
KV = (2, 8, 14, 20, 24, 28, 32, 36)

def template(head=True):
    t = []
    for l in range(40):
        if l in ENGRAM: t.append(('engram', l))
        if l in KV: t.append(('kv', l))
        t.append(('attn_gather', l))
        t.append(('moe_gather', l))
    if head: t.append(('head', -1))
    return t

# arms whose workload is greedy only (b2/load.py: temperature 0): every forward ends with the ring head gather
GREEDY = ('fx1L', 'fx1Lb', 'np1L', 'ns1L', 'k0x1', 'k0x4', 'k6x1', 'k6x4', 'fx1', 'fx4')
TEMPLATE = template()
OFF = {k: i for i, k in enumerate(TEMPLATE)}
FWD_OPS = ('attn_gather', 'moe_gather', 'kv', 'engram', 'head')

def annotate(name, rank, greedy=None):
    """Returns (records, units, info). Each record gains:
      unit: 'F' or 'P' or '?'; pos: index inside its unit; lab: label tuple;
      for 'P': steps (Markov steps of that pass, or None), kind 'A'/'M'/'K', idx (stage j or step i)
      F: the unit's forward anchor (verify) / P: pass start seq."""
    if greedy is None: greedy = name in GREEDY
    g = load(name, rank)
    by_round = collections.defaultdict(list)
    for r in g: by_round[r['round']].append(r)
    units = {}   # round id -> dict(kind, F, L, bad)
    info = collections.Counter()
    for u, rs in by_round.items():
        f = [r for r in rs if r['op'] in FWD_OPS]
        o = [r for r in rs if r['op'] == 'other']
        if f and o:
            info['mixed_unit'] += 1
        if f:
            Fs = collections.Counter(r['seq'] - OFF[(r['op'], r['layer'])] for r in f)
            F = Fs.most_common(1)[0][0]
            if len(Fs) > 1: info['fwd_anchor_conflict'] += 1
            has_head = any(r['op'] == 'head' for r in f)
            units[u] = dict(kind='F', F=F, head=has_head, n=len(f), conflict=len(Fs) > 1, rows=f[0]['rows'])
        elif o:
            units[u] = dict(kind='P', n=len(o))
    order = sorted(units)
    fwd = [u for u in order if units[u]['kind'] == 'F']
    # forward length: 91 with a ring head, 90 without (NCCL head exchange). Decide per forward from the next anchor
    # when the gap is only explained one way; default 91 when the head record is seen.
    nxt = {}; prv = {}
    for a, b in zip(fwd, fwd[1:]):
        nxt[a] = b; prv[b] = a
    for r in g:
        u = r['round']; un = units.get(u)
        r['unit'] = '?'
        if un is None: continue
        if r['op'] in FWD_OPS:
            if un['kind'] != 'F': continue
            pos = r['seq'] - un['F']
            if 0 <= pos < len(TEMPLATE) and TEMPLATE[pos] == (r['op'], r['layer']):
                r['unit'] = 'F'; r['pos'] = pos; r['F'] = un['F']
            else:
                info['fwd_pos_mismatch'] += 1
    # passes: locate by seq between forwards
    fa = sorted((units[u]['F'], u) for u in fwd)
    import bisect
    Fl = [x[0] for x in fa]
    for r in g:
        if r['op'] != 'other': continue
        i = bisect.bisect_right(Fl, r['seq']) - 1
        if i < 0 or i + 1 >= len(Fl):
            info['pass_unanchored_edge'] += 1; continue
        Fp, Fn = Fl[i], Fl[i + 1]
        gap = Fn - Fp
        r['gap'] = gap
        # gap = L + 6 + steps, L in (90, 91)
        cands = []
        for L in ((91,) if greedy else (91, 90)):
            steps = gap - L - 6
            if 1 <= steps <= 6:
                P = Fp + L
                idx = r['seq'] - P
                if idx < 0 or idx >= 6 + steps: continue
                if idx < 6:
                    ok = r['layer'] == idx // 2 and r['bytes'] > 16
                else:
                    ok = r['layer'] == -1 and r['bytes'] == 16 * r['rows']
                if ok: cands.append((L, steps, P, idx))
        r['cands'] = cands
        if not cands:
            info['pass_no_candidate'] += 1; continue
        # resolve: prefer L=91 when the previous forward showed a head record; else if one candidate, take it
        up = fa[i][1]
        if len(cands) > 1:
            if units[up]['head']:
                cands = [c for c in cands if c[0] == 91]
            else:
                info['pass_ambiguous'] += 1
                r['amb'] = True
        L, steps, P, idx = cands[0]
        r['unit'] = 'P'; r['P'] = P; r['steps'] = steps; r['pos'] = idx; r['L'] = L
        if idx < 6:
            r['kind'] = 'A' if idx % 2 == 0 else 'M'; r['idx'] = idx // 2
        else:
            r['kind'] = 'K'; r['idx'] = idx - 6
        r['Fnext'] = Fn; r['Fprev'] = Fp
    gaps = collections.Counter()
    for a, b in zip(Fl, Fl[1:]): gaps[b - a] += 1
    info['units_F'] = len(fwd); info['units_P'] = len(order) - len(fwd)
    return g, units, dict(info=info, gaps=gaps, anchors=fa)

if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        for rank in (0, 1):
            g, units, x = annotate(name, rank)
            c = collections.Counter(r['unit'] for r in g)
            print(name, rank, 'records', len(g), dict(c), dict(x['info']))
            print('   forward-anchor gaps:', sorted(x['gaps'].items()))
            st = collections.Counter((r['steps'], r['rows'] if r['kind'] != 'K' else None) for r in g if r['unit'] == 'P')
            print('   pass (steps, rows of hidden recs):', sorted(st.items(), key=str))
