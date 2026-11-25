"""Second check of the slow-cell finding, using every round of the cell, not only the sampled engram records:
consecutive sampled records are 16 sequences apart and their exact elapsed time covers 16 consecutive tiles. Spans
that contain forward position 2 (the Engram layer-1 gather, first transfer of the second graph stretch) are compared
with spans that do not, as excess over the span's expected time (sum of arm-wide per-(position, rows) median tiles of
one-stream forwards / 6-step passes).
Also: the noise tap. Markov-step fast mode after a forward known to be greedy (gap 103 = 91 + 12) against after a
forward that is TOPK_MERGE in most cases (gap 101)."""
import sys, os, collections, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
from segments import segments

def run(name, rank, nfwd):
    g, x, segs = segments(name, rank)
    Fl = [a for a, _ in x['anchors']]
    rows_of = {}
    for r in g:
        if r['unit'] == 'F' and r['op'] in ('attn_gather', 'moe_gather'): rows_of[r['F']] = r['rows']
    med = collections.defaultdict(list)
    for r in g:
        if r['unit'] == 'F' and r['rows'] <= 7 and r['op'] != 'kv': med[('F', r['pos'], r['rows'])].append(r['copy'] - r['prev'])
        if r['unit'] == 'F' and r['op'] == 'kv' and rows_of.get(r['F'], 99) <= 7: med[('F', r['pos'], rows_of[r['F']])].append(r['copy'] - r['prev'])
    med = {k: sorted(v)[len(v) // 2] for k, v in med.items() if len(v) >= 5}
    cell = None
    for s in segs:
        fw = {r['F'] for r in s if r['unit'] == 'F'}
        rr = [r['rows'] for r in s if r['op'] == 'attn_gather']
        if len(fw) == nfwd and rr and max(rr) <= 7: cell = s
    lo, hi = cell[0]['seq'], cell[-1]['seq']
    byseq = {r['seq']: r for r in cell}
    w = []; wo = []
    for a, b in zip(cell, cell[1:]):
        if b['seq'] - a['seq'] != 16: continue
        exp = 0; has = False; ok = True; npass = 0
        for q in range(a['seq'] + 1, b['seq'] + 1):
            i = bisect.bisect_right(Fl, q) - 1
            F = Fl[i]; pos = q - F
            if pos < 91 and i + 1 < len(Fl) and Fl[i + 1] - F >= 97:
                L = 91 if (Fl[i + 1] - F) in (103, 98) or pos < 90 else 91
                if pos >= 90: npass += 1; continue          # head or pass position: handled as a lump below
                k = ('F', pos, rows_of.get(F))
                if k not in med: ok = False; break
                exp += med[k]
                if pos == 2: has = True
            else:
                npass += 1
        if not ok or npass: continue                        # keep spans wholly inside one forward's layers
        (w if has else wo).append((b['copy'] - a['copy']) - exp)
    return w, wo

for label, nfwd in (('4K x 1 (control)', 97), ('32K x 1', 114), ('128K x 1', 126)):
    for name in ('tr16a', 'tr16b'):
        for rank in (1, 0):
            w, wo = run(name, rank, nfwd)
            sw, so = stats(w), stats(wo)
            big_w = sum(1 for v in w if v > 0.8e6); big_o = sum(1 for v in wo if v > 0.8e6)
            print(f"{label} {name} r{rank}: spans with the Engram L1 gather n {sw['n']}: excess p50 {sw['p50']/1e3:.0f} mean {sw['mean']/1e3:.0f} us, > 0.8 ms: {big_w} ({100*big_w/max(1,sw['n']):.0f}%), sum of those {sum(v for v in w if v>0.8e6)/1e6:.1f} ms | spans without n {so['n']}: p50 {so['p50']/1e3:.0f} mean {so['mean']/1e3:.0f} us, > 0.8 ms: {big_o}")

print('# noise tap: Markov steps 1+ fast mode by the forward gap around the pass (rank 1)')
for name in ('tr16a', 'tr16b'):
    g, units, x = annotate(name, 1, greedy=False)
    Fl = [a for a, _ in x['anchors']]
    by = collections.defaultdict(list)
    for r in g:
        if r['op'] != 'other' or r['layer'] != -1 or r['rows'] != 1: continue
        i = bisect.bisect_right(Fl, r['seq']) - 1
        if i < 0 or i + 1 >= len(Fl): continue
        v = r['stage'] - r['prev']
        if v < 60e3: by[Fl[i + 1] - Fl[i]].append(v)
    for gap in (103, 102, 101, 97):
        v = sorted(by[gap])
        if v: print(f"  {name} gap {gap}: fast-mode steps n {len(v)} p10 {pct(v,10)/1e3:.1f} p50 {pct(v,50)/1e3:.1f} p90 {pct(v,90)/1e3:.1f} us")
