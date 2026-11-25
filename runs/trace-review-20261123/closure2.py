"""Closure on steady decode only: inside runs of consecutive 103-sequence cycles (6-step pass + 91-transfer forward),
compare the exact elapsed time between consecutive sampled records (= four consecutive tiles) with N x the sampled
tile, and report the mean cycle both ways."""
import sys, os, collections, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
for name in sys.argv[1:]:
    n = FILES[name][1]
    for rank in (0, 1):
        g, units, x = annotate(name, rank)
        Fl = [a for a, _ in x['anchors']]
        steady = set()   # anchors F whose cycle [F, F+103) is a 103 cycle and whose neighbours are too
        for i in range(1, len(Fl) - 2):
            if Fl[i] - Fl[i - 1] == 103 and Fl[i + 1] - Fl[i] == 103 and Fl[i + 2] - Fl[i + 1] == 103: steady.add(Fl[i])
        def cyc(seq):
            i = bisect.bisect_right(Fl, seq) - 1
            return Fl[i] if i >= 0 else None
        exact = 0; est = 0; k = 0
        for a, b in zip(g, g[1:]):
            if b['seq'] - a['seq'] != n: continue
            if cyc(a['seq'] + 1) in steady and cyc(b['seq']) in steady:
                t = b['copy'] - b['prev']
                if b['copy'] - a['copy'] > 20e6: continue
                exact += b['copy'] - a['copy']; est += n * t; k += 1
        print(f"{name} r{rank}: steady pairs {k}: exact {exact/1e9:.4f} s, N*tiles {est/1e9:.4f} s, ratio {est/exact:.4f}; mean cycle exact {exact/k/n*103/1e6:.3f} ms, from tiles {est/k/n*103/1e6:.3f} ms")
