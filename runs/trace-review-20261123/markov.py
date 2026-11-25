"""Lead 2: the cost of a Markov bias miss, from the per-step producer interval of the drafter's Markov gathers.

Step i>0 producer = _pick(i-1) + _bias(i) + _score(i) + _finish(i) (tri_markov.zig:120-165); _bias recomputes a row
against the 64,640 x 256 fp16 Markov head (33.1 MB a rank) when the step's token has no cached row. One stream only
(rows == 1). Checks: (1) bimodality and the gap between the modes; (2) the same ring sequence is slow on both ranks
and in every arm (the arms replay the same greedy tokens, so a token-determined miss must coincide; noise would not);
(3) the slow share against the served miss counter (51.7-53.3%).
usage: markov.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate

def steps(name, rank):
    g, units, x = annotate(name, rank)
    return {r['seq']: r for r in g if r['unit'] == 'P' and r['kind'] == 'K' and r['rows'] == 1 and r['stage'] - r['prev'] < 20e6}

def hist(vals, edges):
    c = [0] * (len(edges) + 1)
    for v in vals:
        i = 0
        while i < len(edges) and v >= edges[i]: i += 1
        c[i] += 1
    return c

if __name__ == '__main__':
    arms = sys.argv[1:]
    S = {(a, r): steps(a, r) for a in arms for r in (0, 1)}
    edges = [5, 8, 10, 12, 15, 20, 30, 50, 80, 110, 125, 135, 145, 155, 165, 180, 220, 300]
    print('# producer interval of Markov steps 1+ (us): histogram edges', edges)
    cut = 60e3
    for (a, r), d in S.items():
        v = [(x['stage'] - x['prev']) for x in d.values() if x['idx'] > 0]
        h = hist([t / 1e3 for t in v], edges)
        fast = sorted(t for t in v if t < cut); slow = sorted(t for t in v if t >= cut)
        print(f"{a} r{r}: n {len(v)} hist {h}")
        print(f"     fast mode n {len(fast)} p10/p50/p90 {pct(fast,10)/1e3:.1f}/{pct(fast,50)/1e3:.1f}/{pct(fast,90)/1e3:.1f} us; slow mode n {len(slow)} ({100*len(slow)/len(v):.1f}%) p10/p50/p90 {pct(slow,10)/1e3:.1f}/{pct(slow,50)/1e3:.1f}/{pct(slow,90)/1e3:.1f} us; gap of medians {(pct(slow,50)-pct(fast,50))/1e3:.1f} us; mean step {sum(v)/len(v)/1e3:.1f} us")
        # by step index
        bi = collections.defaultdict(list)
        for x in d.values():
            if x['idx'] > 0: bi[x['idx']].append(x['stage'] - x['prev'])
        print('     slow share by step:', ' '.join(f"K{i}:{100*sum(1 for t in bi[i] if t>=cut)/len(bi[i]):.0f}%(n{len(bi[i])})" for i in sorted(bi)))
        # step 0: head + step; split at the midpoint between its two modes
        v0 = sorted((x['stage'] - x['prev']) for x in d.values() if x['idx'] == 0 and x['steps'] >= 5)
        if v0:
            h0 = hist([t / 1e3 for t in v0], [900, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400])
            lo = [t for t in v0 if t < 1.16e6]; hi = [t for t in v0 if t >= 1.16e6]
            print(f"     step 0 (head + step) n {len(v0)} hist@[900,1000,1050,1100,1150,1200,1250,1300,1400] {h0}; low mode p50 {pct(lo,50)/1e3:.1f} (n {len(lo)}), high mode p50 {pct(hi,50)/1e3:.1f} (n {len(hi)}, {100*len(hi)/len(v0):.0f}%), gap {(pct(hi,50)-pct(lo,50))/1e3:.1f} us")
    # coincidence across ranks and arms on the same sequence
    print('# coincidence of the slow mode on the same ring sequence (steps 1+)')
    keys = list(S)
    base = keys[0]
    for k in keys[1:]:
        both = [q for q in S[base] if q in S[k] and S[base][q]['idx'] > 0 and S[k][q]['idx'] == S[base][q]['idx']]
        a = [(S[base][q]['stage'] - S[base][q]['prev']) >= cut for q in both]
        b = [(S[k][q]['stage'] - S[k][q]['prev']) >= cut for q in both]
        agree = sum(1 for x, y in zip(a, b) if x == y)
        print(f"  {base} vs {k}: common steps {len(both)}, same mode {agree} ({100*agree/len(both):.2f}%); slow in base {sum(a)}, slow in other {sum(b)}")
