"""Leads 3 and 4: per-position producer profile of the forward, row-standardized, per arm and rank.

For every forward position (recon.TEMPLATE) the producer interval (stage - prev) and the ring time (copy - stage),
standardized to the arm's overall row mix (direct standardization over rows 2..7) so layers with different sampled
row mixes compare. Position 0's interval also holds the pass tail and the host hand-off; it is reported with and
without a trace drain on the boundary (drains.py).
usage: layer_profile.py [--csv out.csv] arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, TEMPLATE
from drains import drain_starts

def profile(name, rank, field='prod'):
    g, units, x = annotate(name, rank)
    ds = drain_starts(name, rank)
    cell = collections.defaultdict(list)   # (pos, rows) -> values
    roww = collections.Counter()
    for r in g:
        if r['unit'] != 'F': continue
        if r['pos'] == 0 and r['seq'] in ds: continue
        v = {'prod': r['stage'] - r['prev'], 'ring': r['copy'] - r['stage'], 'tile': r['copy'] - r['prev'], 'wait': r['flag'] - r['collect'], 's2d': r['doorbell'] - r['stage']}[field]
        if v > 20e6: continue
        cell[(r['pos'], r['rows'])].append(v)
        roww[r['rows']] += 1
    tot = sum(roww.values())
    out = {}
    for pos in range(len(TEMPLATE)):
        num = 0; den = 0; n = 0; allv = []
        for rows, w in roww.items():
            v = cell.get((pos, rows))
            if not v: continue
            num += w * sum(v) / len(v); den += w; n += len(v); allv += v
        allv.sort()
        out[pos] = dict(std=num / den if den else float('nan'), n=n, p50=allv[len(allv) // 2] if allv else float('nan'))
    return out, roww

if __name__ == '__main__':
    args = sys.argv[1:]
    csv = None
    if args[0] == '--csv': csv = args[1]; args = args[2:]
    P = {}
    for name in args:
        for rank in (0, 1):
            for f in ('prod', 'ring'):
                P[(name, rank, f)], roww = profile(name, rank, f)
        print('#', name, 'row mix', sorted(roww.items()))
    hdr = 'pos op          layer ' + ' '.join(f'{n[:6]:>6}.r{r}' for n in args for r in (0, 1)) + ' | ring ' + ' '.join(f'{n[:6]:>6}.r{r}' for n in args for r in (0, 1))
    print('# row-standardized mean producer interval (us), then ring time (us)')
    print(hdr)
    lines = []
    for pos, (op, layer) in enumerate(TEMPLATE):
        a = ' '.join(f"{P[(n, r, 'prod')][pos]['std'] / 1e3:>9.1f}" for n in args for r in (0, 1))
        b = ' '.join(f"{P[(n, r, 'ring')][pos]['std'] / 1e3:>9.1f}" for n in args for r in (0, 1))
        print(f'{pos:>3} {op:<12}{layer:>4}  {a} | {b}')
        lines.append([pos, op, layer] + [round(P[(n, r, f)][pos]['std'] / 1e3, 2) for f in ('prod', 'ring') for n in args for r in (0, 1)])
    if csv:
        with open(csv, 'w') as fh:
            fh.write('pos,op,layer,' + ','.join(f'{f}_{n}_r{r}' for f in ('prod', 'ring') for n in args for r in (0, 1)) + '\n')
            for l in lines: fh.write(','.join(str(v) for v in l) + '\n')
