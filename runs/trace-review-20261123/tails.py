"""Lead 5: tails and periodicity. For every record the excess of its tile parts over the median of its
(unit, position, rows) cell; then the percentiles of the excess, where the large ones sit, whether they coincide
on both ranks (same seq), and whether they recur on a period in work units.
usage: tails.py arm [arm ...]"""
import sys, os, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, TEMPLATE
from drains import drain_starts

def excess(name, rank):
    g, units, x = annotate(name, rank)
    ds = drain_starts(name, rank)
    cell = collections.defaultdict(list)
    def key(r):
        if r['unit'] == 'F': return ('F', r['pos'], r['rows'], r['pos'] == 0 and r['seq'] in ds)
        if r['unit'] == 'P':
            k = r['kind']
            # Markov steps: the mode (hit/miss) is content, not a tail: split at 60 us
            mode = (r['stage'] - r['prev'] >= (1.16e6 if r['idx'] == 0 else 60e3)) if k == 'K' else None
            return ('P', r['pos'], r['rows'], mode)
        return None
    for r in g:
        k = key(r)
        if k is None: continue
        cell[k].append(r)
    out = []
    for k, rs in cell.items():
        if len(rs) < 8: continue
        mp = sorted(r['stage'] - r['prev'] for r in rs)[len(rs) // 2]
        mr = sorted(r['copy'] - r['stage'] for r in rs)[len(rs) // 2]
        for r in rs:
            out.append(dict(seq=r['seq'], round=r['round'], unit=r['unit'], pos=r['pos'], rows=r['rows'], op=r['op'], layer=r['layer'],
                            ep=(r['stage'] - r['prev']) - mp, er=(r['copy'] - r['stage']) - mr, mp=mp, mr=mr,
                            wait=r['flag'] - r['collect']))
    out.sort(key=lambda d: d['seq'])
    return out

if __name__ == '__main__':
    for name in sys.argv[1:]:
        E = [excess(name, r) for r in (0, 1)]
        for rank in (0, 1):
            e = E[rank]
            for fld, lab in (('ep', 'producer excess'), ('er', 'ring excess')):
                v = sorted(d[fld] for d in e)
                n = len(v)
                print(f"{name} r{rank} {lab}: n {n} p50 {pct(v,50)/1e3:.1f} p90 {pct(v,90)/1e3:.1f} p99 {pct(v,99)/1e3:.1f} p99.9 {pct(v,99.9)/1e3:.1f} max {v[-1]/1e3:.1f} us; mean of positive part {sum(x for x in v if x>0)/n/1e3:.2f} us/transfer; >100us {sum(1 for x in v if x>100e3)} >500us {sum(1 for x in v if x>500e3)} >2ms {sum(1 for x in v if x>2e6)}")
            # by op: p99 of producer excess and of ring excess
            byop = collections.defaultdict(list)
            for d in e: byop[d['op'] if d['unit'] == 'F' else 'pass'].append(d)
            for op, ds_ in sorted(byop.items()):
                vp = sorted(d['ep'] for d in ds_); vr = sorted(d['er'] for d in ds_)
                print(f"    {op:<12} n {len(ds_):>6} producer excess p90/p99/p99.9 {pct(vp,90)/1e3:>6.1f}/{pct(vp,99)/1e3:>6.1f}/{pct(vp,99.9)/1e3:>7.1f} us | ring excess p90/p99/p99.9 {pct(vr,90)/1e3:>6.1f}/{pct(vr,99)/1e3:>6.1f}/{pct(vr,99.9)/1e3:>7.1f} us | mean ring-excess>0 {sum(max(0,x) for x in vr)/len(vr)/1e3:.2f}")
        # large producer excesses: coincide on both ranks?
        m0 = {d['seq']: d for d in E[0]}; m1 = {d['seq']: d for d in E[1]}
        big0 = {q for q, d in m0.items() if d['ep'] > 100e3}; big1 = {q for q, d in m1.items() if d['ep'] > 100e3}
        print(f"  producer excess > 100 us: r0 {len(big0)}, r1 {len(big1)}, both {len(big0 & big1)}")
        both = sorted(big0 | big1)
        c = collections.Counter((m0[q]['unit'], m0[q]['op'], m0[q]['layer']) if q in m0 else ('?',) for q in both)
        print('  where:', c.most_common(14))
        for q in both[:40]:
            a = m0.get(q); b = m1.get(q)
            print(f"     seq {q} unit {a['round'] if a else '?'} {a['unit'] if a else ''} {a['op'] if a else ''} L{a['layer'] if a else ''} rows {a['rows'] if a else ''}: excess r0 {a['ep']/1e3 if a else float('nan'):.0f} r1 {b['ep']/1e3 if b else float('nan'):.0f} us (median {a['mp']/1e3 if a else 0:.0f})")
        # periodicity: per work unit, mean producer excess over its sampled records (rank 0), autocorrelation by lag in units of forwards
        per = collections.defaultdict(list)
        for d in E[0]:
            if d['unit'] == 'F' and d['pos'] > 0 and d['op'] in ('attn_gather', 'moe_gather'): per[d['round']].append(d['ep'])
        us = sorted(per)
        series = [sum(per[u]) / len(per[u]) for u in us]
        mu = sum(series) / len(series); var = sum((s - mu) ** 2 for s in series) / len(series)
        ac = []
        for lag in (1, 2, 3, 4, 8, 16, 31, 32, 33, 64):
            c_ = sum((series[i] - mu) * (series[i + lag] - mu) for i in range(len(series) - lag)) / (len(series) - lag) / var
            ac.append((lag, round(c_, 3)))
        print(f"  per-forward mean producer excess (r0): n {len(series)} sd {math.sqrt(var)/1e3:.2f} us; autocorrelation by lag (forwards): {ac}")
