"""Where the trace's own drain lands. traceTick drains every 4 work units at Model.pass / Model.verify entry
(rdma.zig:853-866); the drain record carries the ring sequence at that moment (= the last seq issued before the unit
that is about to start). So the unit that starts at seq+1 has the drain's host time inside its first tile."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate

def drain_starts(name, rank):
    """Set of sequences s such that a drain ran immediately before the transfer with sequence s."""
    return {d['seq'] + 1 for d in load(name, rank, kinds=('drain', 'gpu'))['drain']}

if __name__ == '__main__':
    for name in sys.argv[1:]:
        for rank in (0, 1):
            g, units, x = annotate(name, rank)
            ds = drain_starts(name, rank)
            Fset = {a for a, _ in x['anchors']}
            Pset = {r['P'] for r in g if r['unit'] == 'P'}
            kinds = collections.Counter('F' if s in Fset else 'P' if s in Pset else '?' for s in ds)
            n = FILES[name][1]
            samp = collections.Counter(('F' if s in Fset else 'P' if s in Pset else '?', s % n == 0) for s in ds)
            print(name, rank, 'drains', len(ds), 'landing before:', dict(kinds), 'sampled start tile:', dict(samp))
            # effect on the first tile of the unit
            byseq = {r['seq']: r for r in g}
            for kind in 'FP':
                a = [byseq[s]['stage'] - byseq[s]['prev'] for s in ds if s in byseq and byseq[s]['unit'] == kind and byseq[s]['pos'] == 0]
                b = [r['stage'] - r['prev'] for r in g if r['unit'] == kind and r['pos'] == 0 and r['seq'] not in ds and (kind == 'F' or r['steps'] == 6)]
                a = [v for v in a if v < 50e6]; b = [v for v in b if v < 50e6]
                if a or b:
                    sa, sb = stats(a), stats(b)
                    print(f"   {kind} pos0 producer (stage-prev): with drain n={sa.get('n',0)} p50={sa.get('p50',0)/1e3:.1f} mean={sa.get('mean',0)/1e3:.1f} us | without n={sb.get('n',0)} p50={sb.get('p50',0)/1e3:.1f} mean={sb.get('mean',0)/1e3:.1f} us")
