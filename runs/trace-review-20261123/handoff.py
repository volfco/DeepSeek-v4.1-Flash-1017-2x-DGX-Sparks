"""Lead 3: the pass-to-forward and forward-to-pass hand-offs, and whether the forward starts cold.

(a) hand-off sizes: A0 producer minus the mean of A1/A2 producers (same work minus the boundary), and forward
    position 0 producer (no trace drain on the boundary) minus the plain-layer attention producer at the same rows.
(b) cold start: the first forward intervals conditioned on whether the Markov step sampled in the preceding pass was
    a miss (33 MB read) or a hit. With 1-in-4 sampling the step that is visible depends on the phase:
    forward pos 0 sees K2, pos 1 (MoE L0) sees K3, pos 2 (engram L1) sees K4, pos 3 (attn L1) sees K5 (6-step passes).
(c) the same intervals with and without prefetch (np1L against fx1L), against later layers of the same kind.
usage: handoff.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, OFF, KV, ENGRAM
from drains import drain_starts

for name in sys.argv[1:]:
    for rank in (0, 1):
        g, units, x = annotate(name, rank)
        ds = drain_starts(name, rank)
        byseq = {r['seq']: r for r in g}
        P = collections.defaultdict(list)
        for r in g:
            if r['unit'] == 'P' and r['steps'] == 6 and r['kind'] in 'AM' and r['stage'] - r['prev'] < 20e6: P[(r['kind'], r['idx'])].append(r['stage'] - r['prev'])
        a0 = stats(P[('A', 0)]); a12 = stats(P[('A', 1)] + P[('A', 2)])
        plain = [l for l in range(3, 40) if l not in KV and l not in ENGRAM]
        F = collections.defaultdict(list)
        for r in g:
            if r['unit'] != 'F' or r['stage'] - r['prev'] > 20e6: continue
            if r['pos'] == 0: F['f0d' if r['seq'] in ds else 'f0'].append(r['stage'] - r['prev'])
            elif r['op'] == 'attn_gather' and r['layer'] in plain: F['plain'].append(r['stage'] - r['prev'])
        f0 = stats(F['f0']); f0d = stats(F['f0d']); pl = stats(F['plain'])
        print(f"{name} r{rank}: into the pass: A0 {a0['p50']/1e3:.0f} - A1/A2 {a12['p50']/1e3:.0f} = {(a0['p50']-a12['p50'])/1e3:.0f} us (medians; n {a0['n']}/{a12['n']}) | into the forward: pos0 {f0['p50']/1e3:.0f} (with drain {f0d['p50']/1e3:.0f}) - plain attention {pl['p50']/1e3:.0f} = {(f0['p50']-pl['p50'])/1e3:.0f} us (n {f0['n']}/{pl['n']})")
        # (b) conditioned on the sampled Markov step of the preceding 6-step pass
        for pos, lab, back in ((0, 'attn L0', 4), (1, 'MoE L0', 4), (2, 'engram L1', 4), (3, 'attn L1', 4)):
            hit = []; miss = []
            for r in g:
                if r['unit'] != 'F' or r['pos'] != pos or (pos == 0 and r['seq'] in ds): continue
                k = byseq.get(r['seq'] - back)
                if not k or k['unit'] != 'P' or k['kind'] != 'K' or k['steps'] != 6 or k['idx'] == 0: continue
                (miss if k['stage'] - k['prev'] >= 60e3 else hit).append(r['stage'] - r['prev'])
            if len(hit) > 5 and len(miss) > 5:
                sh, sm = stats(hit), stats(miss)
                print(f"      {lab}: after a sampled hit step p50 {sh['p50']/1e3:.1f} mean {sh['mean']/1e3:.1f} (n {sh['n']}); after a sampled miss step p50 {sm['p50']/1e3:.1f} mean {sm['mean']/1e3:.1f} (n {sm['n']}); difference of medians {(sm['p50']-sh['p50'])/1e3:+.1f} us")
