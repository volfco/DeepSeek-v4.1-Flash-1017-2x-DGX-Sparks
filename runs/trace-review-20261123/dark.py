"""Where the time the sampled tiles do not show sits. Consecutive sampled records are N sequences apart and their
exact elapsed time (b.copy - a.copy) is the sum of N consecutive tiles, N-1 of them unsampled. Predict those from
the per-position median tile and list the residuals: a large one pins hidden time on one of N-1 positions (request
boundaries, idle gaps). Also prints the largest sampled tiles and the whole-trace closure.
usage: dark.py arm rank   (greedy arms: forward length 91)"""
import sys, os, collections, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
name = sys.argv[1]; rank = int(sys.argv[2])
n = FILES[name][1]
g, units, x = annotate(name, rank)
Fl = [a for a, _ in x['anchors']]
def where(seq):
    i = bisect.bisect_right(Fl, seq) - 1
    if i < 0: return None
    F = Fl[i]
    if seq - F < 91: return ('F', seq - F)
    if i + 1 >= len(Fl): return None
    return ('P', seq - F - 91)
tiles = sorted(((r['copy'] - r['prev']), r['seq'], r['unit'], r.get('pos')) for r in g)
span = g[-1]['copy'] - g[0]['prev']
print(f"{name} r{rank}: span {span/1e9:.3f} s; N x sum of sampled tiles {n*sum(t[0] for t in tiles)/1e9:.3f} s")
print('largest sampled tiles (ms, seq, unit, position):', [(round(t[0] / 1e6, 2), t[1], t[2], t[3]) for t in tiles[-8:]])
inreq = [t for t in tiles if t[0] < 50e6]
print('largest sampled tile inside a request (ms):', round(inreq[-1][0] / 1e6, 2), 'at', inreq[-1][1:])
med = collections.defaultdict(list)
for r in g:
    if r['unit'] in 'FP': med[(r['unit'], r['pos'])].append(r['copy'] - r['prev'])
med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
res = []
for a, b in zip(g, g[1:]):
    if b['seq'] - a['seq'] != n: continue
    ws = [where(s) for s in range(a['seq'] + 1, b['seq'])]
    if any(w is None or w not in med for w in ws): continue
    res.append((b['copy'] - a['copy'] - (b['copy'] - b['prev']) - sum(med[w] for w in ws), a['seq'], ws))
res.sort(reverse=True)
big = [r for r in res if r[0] > 2e6]
print(f"pairs {len(res)}; residuals > 2 ms: {len(big)}, sum {sum(r[0] for r in big)/1e6:.1f} ms; the rest average {sum(r[0] for r in res if r[0] <= 2e6)/max(1,len(res)-len(big))/1e3:.1f} us per {n-1} unsampled tiles")
for r in big: print(f"   {r[0]/1e6:9.2f} ms after seq {r[1]}: unsampled positions {r[2]}")
