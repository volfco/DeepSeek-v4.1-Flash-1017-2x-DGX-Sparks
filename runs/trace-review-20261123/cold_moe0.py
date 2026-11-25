"""Lead 3, MoE at layer 0: the interval after a sampled Markov miss against after a sampled hit, within the same row
count (a miss token goes with fewer verified rows, so the unstratified comparison in handoff.py is confounded).
With 1-in-4 sampling the step visible before forward position 1 is K3 of a 6-step pass.
usage: cold_moe0.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
for name in sys.argv[1:]:
    for rank in (0, 1):
        g, units, x = annotate(name, rank)
        byseq = {r['seq']: r for r in g}
        cell = collections.defaultdict(lambda: ([], []))
        for r in g:
            if r['unit'] != 'F' or r['pos'] != 1: continue
            k = byseq.get(r['seq'] - 4)
            if not k or k['unit'] != 'P' or k['kind'] != 'K' or k['steps'] != 6 or k['idx'] == 0: continue
            cell[r['rows']][1 if k['stage'] - k['prev'] >= 60e3 else 0].append(r['stage'] - r['prev'])
        num = 0; den = 0; parts = []
        for rows, (hit, miss) in sorted(cell.items()):
            if len(hit) < 4 or len(miss) < 4: continue
            d = sum(miss) / len(miss) - sum(hit) / len(hit); w = min(len(hit), len(miss))
            num += d * w; den += w; parts.append(f"rows {rows}: {d/1e3:+.0f} us (n {len(hit)}/{len(miss)})")
        print(f"{name} r{rank}: MoE L0 after a miss minus after a hit, within rows: " + '; '.join(parts) + (f" | weighted {num/den/1e3:+.1f} us" if den else ''))
