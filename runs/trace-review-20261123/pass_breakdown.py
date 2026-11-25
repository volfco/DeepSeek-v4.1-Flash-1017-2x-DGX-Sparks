"""Lead 1: the drafter pass, segment by segment, from the `other` records.

Tile of a transfer = [prev, copy]; producer = stage - prev (GPU compute, plus any host gap, since the previous
transfer's copy-out); ring = copy - stage (stage copy-in, doorbell, wait for the peer's data, copy-out).
Pass positions (draft.zig body(), 646-714): 0 A0, 1 M0, 2 A1, 3 M1, 4 A2, 5 M2 (hidden gathers of the three stages),
6.. K0.. (one gather per Markov step). What sits in each producer interval:
  A0: forward tail after its last ring transfer, host hand-off (sync, pick, absorb, uploads, graph launch), embedInit,
      hcPre2 mix, stage-0 attention
  Aj (j>0): hcPre2 mix (with the pending MoE post) + stage-j attention
  Mj: hcPre2 mix (ffn) + stage-j MoE (top 3 of 128 + shared)
  K0: hcPost, collapse, RMSNorm, the vocabulary head on all block rows, then _bias, _score, _finish of step 0
  Ki (i>0): _pick of step i-1, _bias, _score, _finish of step i
  (forward position 0, reported by handoff.py: _pick of the last step, conf pack, conf GEMV, sync, 2 DtoH, host,
   uploads, graph launch, embedInit, mix, layer-0 attention)
usage: pass_breakdown.py <arm> [steps]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
from drains import drain_starts

LAB = ['A0', 'M0', 'A1', 'M1', 'A2', 'M2'] + ['K%d' % i for i in range(6)]

def rows_for(name, rank, steps):
    g, units, x = annotate(name, rank)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in g:
        if r['unit'] != 'P' or r['steps'] != steps: continue
        if r['stage'] - r['prev'] > 20e6: continue   # request boundary (prefill) tiles
        d = out[r['pos']]
        d['prod'].append(r['stage'] - r['prev'])
        d['s2d'].append(r['doorbell'] - r['stage'])
        d['d2f'].append(r['flag'] - r['doorbell'])
        d['c2f'].append(r['flag'] - r['collect'])
        d['f2c'].append(r['copy'] - r['flag'])
        d['ring'].append(r['copy'] - r['stage'])
        d['tile'].append(r['copy'] - r['prev'])
    return out

if __name__ == '__main__':
    name = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    res = [rows_for(name, rank, steps) for rank in (0, 1)]
    print(f"# {name}: passes with {steps} Markov steps; microseconds; rank0 | rank1")
    print(f"{'pos':<4}{'n':>5} | {'prod p50':>9}{'mean':>8}{'p10':>8}{'p90':>8} | {'ring p50':>9}{'mean':>7} {'s2d':>5}{'d2f':>6}{'f2c':>5} | {'tile mean':>9} || {'prod p50':>9}{'mean':>8} | {'ring p50':>9}{'mean':>7} | {'tile mean':>9}")
    tot = [0, 0]; totp = [0, 0]; totr = [0, 0]
    for pos in range(6 + steps):
        a, b = res[0][pos], res[1][pos]
        sa = {k: stats(v) for k, v in a.items()}; sb = {k: stats(v) for k, v in b.items()}
        u = lambda s, k, f: s[k][f] / 1e3
        print(f"{LAB[pos]:<4}{sa['prod']['n']:>5} | {u(sa,'prod','p50'):>9.1f}{u(sa,'prod','mean'):>8.1f}{u(sa,'prod','p10'):>8.1f}{u(sa,'prod','p90'):>8.1f} | {u(sa,'ring','p50'):>9.1f}{u(sa,'ring','mean'):>7.1f} {u(sa,'s2d','p50'):>5.1f}{u(sa,'d2f','p50'):>6.1f}{u(sa,'f2c','p50'):>5.1f} | {u(sa,'tile','mean'):>9.1f} || {u(sb,'prod','p50'):>9.1f}{u(sb,'prod','mean'):>8.1f} | {u(sb,'ring','p50'):>9.1f}{u(sb,'ring','mean'):>7.1f} | {u(sb,'tile','mean'):>9.1f}")
        for i, s in enumerate((sa, sb)):
            tot[i] += s['tile']['mean']; totp[i] += s['prod']['mean']; totr[i] += s['ring']['mean']
    print(f"sum of tile means: r0 {tot[0]/1e3:.1f} us (producer {totp[0]/1e3:.1f} + ring {totr[0]/1e3:.1f}); r1 {tot[1]/1e3:.1f} us (producer {totp[1]/1e3:.1f} + ring {totr[1]/1e3:.1f})")
