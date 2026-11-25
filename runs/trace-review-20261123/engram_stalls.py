"""Engram stalls: the engram gather's producer interval (previous transfer's copy-done -> engram stage entry) spans a
host step (round.zig forward(): engramLand between graph stretches). List the long ones per rank, and what the
peer's ring wait was on the same sequence.
usage: engram_stalls.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
from segments import segments

for name in sys.argv[1:]:
    G = {}
    for rank in (0, 1):
        g, x, segs = segments(name, rank)
        segof = {}
        for i, s in enumerate(segs):
            for r in s: segof[r['seq']] = i
        G[rank] = ({r['seq']: r for r in g}, segof, segs)
    m0, so, segs = G[0]; m1 = G[1][0]
    eng = [q for q, r in m0.items() if r['op'] == 'engram' and q in m1]
    n = FILES[name][1]
    for thr in (0.5e6, 1e6, 3e6, 10e6):
        c0 = sum(1 for q in eng if m0[q]['stage'] - m0[q]['prev'] > thr); c1 = sum(1 for q in eng if m1[q]['stage'] - m1[q]['prev'] > thr)
        print(f"{name}: engram records {len(eng)} (1 in {n} sampled); producer > {thr/1e6:.1f} ms: rank0 {c0}, rank1 {c1}")
    p0 = sorted(m0[q]['stage'] - m0[q]['prev'] for q in eng); p1 = sorted(m1[q]['stage'] - m1[q]['prev'] for q in eng)
    print(f"   producer rank0 p50 {pct(p0,50)/1e3:.0f} p90 {pct(p0,90)/1e3:.0f} p99 {pct(p0,99)/1e3:.0f} max {p0[-1]/1e3:.0f} mean {sum(p0)/len(p0)/1e3:.0f} us | rank1 p50 {pct(p1,50)/1e3:.0f} p90 {pct(p1,90)/1e3:.0f} p99 {pct(p1,99)/1e3:.0f} max {p1[-1]/1e3:.0f} mean {sum(p1)/len(p1)/1e3:.0f} us")
    # excess over the median, summed, scaled by the sampling: stall time a forward
    fw = len({r['F'] for r in m0.values() if r['unit'] == 'F'})
    for rank, m in ((0, m0), (1, m1)):
        med = pct(sorted(m[q]['stage'] - m[q]['prev'] for q in eng), 50)
        exc = sum(max(0, (m[q]['stage'] - m[q]['prev']) - med - 200e3) for q in eng)
        print(f"   rank{rank}: sum of producer excess beyond median+200us over sampled engram records {exc/1e6:.1f} ms -> x{n} = {exc*n/1e6:.0f} ms over {fw} forwards = {exc*n/fw/1e3:.0f} us a forward")
    big = sorted((q for q in eng if max(m0[q]['stage'] - m0[q]['prev'], m1[q]['stage'] - m1[q]['prev']) > 1e6))
    print('   stalls > 1 ms (seq, segment, layer, rows | rank0 producer, rank0 wait | rank1 producer, rank1 wait) ms:')
    for q in big[:60]:
        a, b = m0[q], m1[q]
        print(f"     {q:>7} seg {so[q]:>3} L{a['layer']:<2} rows {a['rows']:>2} | r0 prod {(a['stage']-a['prev'])/1e6:7.2f} wait {(a['flag']-a['collect'])/1e6:7.2f} | r1 prod {(b['stage']-b['prev'])/1e6:7.2f} wait {(b['flag']-b['collect'])/1e6:7.2f}")
    # by layer
    for L in (1, 14):
        q_ = [q for q in eng if m0[q]['layer'] == L]
        c1 = sum(1 for q in q_ if m1[q]['stage'] - m1[q]['prev'] > 1e6); c0 = sum(1 for q in q_ if m0[q]['stage'] - m0[q]['prev'] > 1e6)
        print(f"   layer {L}: n {len(q_)}, >1 ms rank0 {c0} rank1 {c1}")
