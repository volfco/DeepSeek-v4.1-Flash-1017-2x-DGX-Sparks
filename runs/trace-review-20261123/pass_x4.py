"""Lead 6: the drafter pass at four streams (rowsplit arms, greedy, 1-in-16 sampling, so few samples): producer
interval per pass position, pooled over the arms given, for passes whose hidden gathers carry 20 or 24 block rows.
usage: pass_x4.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
LAB = ['A0', 'M0', 'A1', 'M1', 'A2', 'M2'] + ['K%d' % i for i in range(6)]
for rank in (0, 1):
    P = collections.defaultdict(list); R = collections.defaultdict(list); rows = collections.Counter(); steps = collections.Counter()
    for name in sys.argv[1:]:
        g, units, x = annotate(name, rank)
        # a pass's stream count from any of its Markov records (rows = streams) or hidden records (rows = streams * block)
        for r in g:
            if r['unit'] != 'P' or r['stage'] - r['prev'] > 20e6: continue
            n = r['rows'] if r['kind'] == 'K' else r['rows'] // max(5, r['steps'])
            if n != 4: continue
            P[r['pos']].append(r['stage'] - r['prev']); R[r['pos']].append(r['copy'] - r['stage'])
            steps[r['steps']] += 1
            if r['kind'] != 'K': rows[r['rows']] += 1
    print(f"rank {rank}: four-stream passes; steps mix {dict(steps)}; hidden rows {dict(rows)}")
    tot = 0
    for pos in sorted(P):
        s = stats(P[pos]); t = stats(R[pos])
        tot += s['mean'] + t['mean']
        print(f"   {LAB[pos]:<3} n {s['n']:>3} producer p50 {s['p50']/1e3:>7.1f} mean {s['mean']/1e3:>7.1f} us | ring p50 {t['p50']/1e3:>6.1f} mean {t['mean']/1e3:>6.1f}")
    print(f"   sum of tile means over the positions seen: {tot/1e3:.0f} us")
