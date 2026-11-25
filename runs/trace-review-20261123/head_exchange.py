"""Lead 6/7: what the end of a forward costs on a full-logit (ring head) round against a TOPK_MERGE round (the top-k
pairs travel over NCCL, so the round has no ring head record), from the mixed tr16 traces.

For a one-stream 6-step pass (hidden rows == 6) the forward before it has L = gap - 12 ring transfers (91: ring head;
90: no ring head), where gap is the distance between the neighbouring forward anchors. The first hidden gather of the
pass (A0) then has producer = [forward tail after its last ring transfer + host hand-off + stage-0 attention].
  L=91: tail-to-A0 = head.producer + head.ring + A0.producer   (head GEMM, ring exchange of the logits, pick, absorb..)
  L=90: tail-to-A0 = A0.producer                                (head GEMM, top-k select, NCCL exchange, draw, absorb..)
Rank 0 of tr16 carries the round profile's host syncs (B2), so read rank 1 first.
usage: head_exchange.py arm [arm ...]"""
import sys, os, collections, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, OFF, FWD_OPS

for name in sys.argv[1:]:
    for rank in (1, 0):
        g, units, x = annotate(name, rank, greedy=False)
        Fl = [a for a, _ in x['anchors']]
        rows_of = {}
        for r in g:
            if r['op'] in ('attn_gather', 'moe_gather') and r['unit'] == 'F': rows_of[r['F']] = r['rows']
        A0 = collections.defaultdict(list); M0 = collections.defaultdict(list)
        for r in g:
            if r['op'] != 'other' or r['layer'] != 0 or r['rows'] != 6: continue
            i = bisect.bisect_right(Fl, r['seq']) - 1
            if i < 0 or i + 1 >= len(Fl): continue
            gap = Fl[i + 1] - Fl[i]; L = gap - 12
            if L not in (90, 91): continue
            pos = r['seq'] - Fl[i] - L
            v = r['stage'] - r['prev']
            if v > 20e6: continue
            if pos == 0: A0[L].append(v)
            elif pos == 1: M0[L].append(v)
        head_p = [r['stage'] - r['prev'] for r in g if r['op'] == 'head' and r['rows'] <= 7 and r['stage'] - r['prev'] < 20e6]
        head_r = [r['copy'] - r['stage'] for r in g if r['op'] == 'head' and r['rows'] <= 7]
        s = lambda v: stats(v) if v else dict(n=0, p50=0, mean=0)
        a91, a90, hp, hr = s(A0[91]), s(A0[90]), s(head_p), s(head_r)
        print(f"{name} r{rank}: A0 producer after a ring-head forward (L=91): n {a91['n']} p50 {a91['p50']/1e3:.1f} mean {a91['mean']/1e3:.1f} us | after a TOPK_MERGE forward (L=90): n {a90['n']} p50 {a90['p50']/1e3:.1f} mean {a90['mean']/1e3:.1f} us")
        print(f"     head producer n {hp['n']} p50 {hp['p50']/1e3:.1f} mean {hp['mean']/1e3:.1f} us; head ring p50 {hr['p50']/1e3:.1f} mean {hr['mean']/1e3:.1f} us")
        if a91['n'] and a90['n'] and hp['n']:
            t91 = hp['p50'] + hr['p50'] + a91['p50']
            print(f"     tail-to-A0 (medians): ring-head round {t91/1e3:.1f} us; TOPK_MERGE round {a90['p50']/1e3:.1f} us; difference {(a90['p50']-t91)/1e3:+.1f} us")
        m91, m90 = s(M0[91]), s(M0[90])
        print(f"     control (M0 producer, should not depend on L): L=91 n {m91['n']} p50 {m91['p50']/1e3:.1f} | L=90 n {m90['n']} p50 {m90['p50']/1e3:.1f} us")
