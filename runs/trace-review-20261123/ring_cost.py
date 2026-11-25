"""Leads 6/7: what each kind of transfer costs the pair, and how much of it is waiting for the other rank.

Pairs both ranks' records by sequence. Per transfer and rank: producer p (stage - prev), stage s (doorbell - stage),
wait w (flag - collect: time in the collect kernel until the peer's data is flagged), copy c (copy - flag), ring
(copy - stage). The two ranks leave a transfer almost together, so a transfer's cost to the pair is the later rank's
(p + ring); what an exactly balanced pair would pay instead is the mean of the two ranks' p plus the ring time with
no skew. skew cost := mean over transfers of |p0 - p1| / 2 (half the absolute producer difference; B2 used the signed
median). floor := the standalone ring bench (runs/sched-20261122/RESULTS-A.md): 9.4 us + 1.16 us a row of 20,480 B.
usage: ring_cost.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate, KV
from drains import drain_starts

def fit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx); return (sy - b * sx) / n, b

for name in sys.argv[1:]:
    g0, _, x0 = annotate(name, 0, greedy=name in ('fx1L', 'fx1Lb', 'np1L', 'ns1L')); g1, _, _ = annotate(name, 1, greedy=name in ('fx1L', 'fx1Lb', 'np1L', 'ns1L'))
    ds = drain_starts(name, 0)
    m1 = {r['seq']: r for r in g1}
    cls = collections.defaultdict(list)
    for a in g0:
        b = m1.get(a['seq'])
        if not b or a['unit'] not in 'FP' or a['stage'] - a['prev'] > 20e6 or b['stage'] - b['prev'] > 20e6: continue
        if a['unit'] == 'F':
            k = a['op']
            if a['op'] == 'attn_gather' and a['layer'] in KV: k = 'attn after kv'
            if a['pos'] == 0: k = 'attn L0 (hand-off)'
        else:
            k = 'pass hidden' if a['kind'] in 'AM' else 'pass markov'
        cls[k].append((a, b))
    print(f"# {name}: per transfer, microseconds (rank0 / rank1)")
    print(f"{'class':<20}{'n':>6} {'bytes':>9} | {'stage':>11} {'wait p50':>11} {'wait mean':>11} {'copy':>9} {'ring mean':>11} | {'|dp| mean':>9} {'p50':>6} {'skew cost':>9} | {'bal.ring':>8} {'floor':>6}")
    tot = collections.Counter()
    per_cycle = {'attn_gather': 31, 'attn after kv': 8, 'attn L0 (hand-off)': 1, 'moe_gather': 40, 'kv': 8, 'engram': 2, 'head': 1, 'pass hidden': 6, 'pass markov': 6}
    for k in ('attn L0 (hand-off)', 'attn_gather', 'attn after kv', 'moe_gather', 'kv', 'engram', 'head', 'pass hidden', 'pass markov'):
        ps = cls.get(k)
        if not ps: continue
        f = lambda fn: (sum(fn(a) for a, b in ps) / len(ps) / 1e3, sum(fn(b) for a, b in ps) / len(ps) / 1e3)
        med = lambda fn: (pct(sorted(fn(a) for a, b in ps), 50) / 1e3, pct(sorted(fn(b) for a, b in ps), 50) / 1e3)
        st = med(lambda r: r['doorbell'] - r['stage']); wm = f(lambda r: r['flag'] - r['collect']); w5 = med(lambda r: r['flag'] - r['collect'])
        cp = med(lambda r: r['copy'] - r['flag']); rg = f(lambda r: r['copy'] - r['stage'])
        dp = sorted(abs((a['stage'] - a['prev']) - (b['stage'] - b['prev'])) for a, b in ps)
        dpm = sum(dp) / len(dp) / 1e3
        by = sum(max(a['bytes'], b['bytes']) for a, b in ps) / len(ps)
        floor = 9.4 + 1.16 * by / 20480
        bal = (rg[0] + rg[1]) / 2
        print(f"{k:<20}{len(ps):>6} {by:>9.0f} | {st[0]:>5.1f}/{st[1]:<5.1f} {w5[0]:>5.1f}/{w5[1]:<5.1f} {wm[0]:>5.1f}/{wm[1]:<5.1f} {cp[0]:>4.1f}/{cp[1]:<4.1f} {rg[0]:>5.1f}/{rg[1]:<5.1f} | {dpm:>9.1f} {pct(dp,50)/1e3:>6.1f} {dpm/2:>9.1f} | {bal:>8.1f} {floor:>6.1f}")
        c = per_cycle[k]
        tot['ring'] += bal * c; tot['floor'] += floor * c; tot['skew'] += dpm / 2 * c
        # random skew only: plain-layer attention and MoE gathers of the forward (the gather after a KV exchange waits
        # for the exchange, the first gather for the hand-off, kv/engram/head carry payload and host steps), and the pass
        if k in ('attn_gather', 'moe_gather'):
            tot['f_ring'] += bal * c; tot['f_floor'] += floor * c; tot['f_skew'] += dpm / 2 * c
        if k in ('pass hidden', 'pass markov'):
            tot['p_ring'] += bal * c; tot['p_floor'] += floor * c; tot['p_skew'] += dpm / 2 * c
    print(f"  per cycle (6-step pass + forward): balanced ring time {tot['ring']/1e3:.2f} ms, standalone floor {tot['floor']/1e3:.2f} ms, excess {(tot['ring']-tot['floor'])/1e3:.2f} ms; skew cost by |dp|/2: {tot['skew']/1e3:.2f} ms")
    print(f"  random skew only, forward (31 plain attention + 40 MoE gathers): ring {tot['f_ring']/1e3:.2f} ms, floor {tot['f_floor']/1e3:.2f} ms, excess {(tot['f_ring']-tot['f_floor'])/1e3:.2f} ms; by |dp|/2: {tot['f_skew']/1e3:.2f} ms | pass (12 gathers): excess {(tot['p_ring']-tot['p_floor'])/1e3:.2f} ms; by |dp|/2: {tot['p_skew']/1e3:.2f} ms")
    # kv: the receiving rank's ring time against bytes
    ps = cls.get('kv', [])
    if ps:
        for rank in (0, 1):
            xs = [max(a['bytes'], b['bytes']) / 147456 for a, b in ps]; ys = [((a, b)[rank]['copy'] - (a, b)[rank]['stage']) / 1e3 for a, b in ps]
            a_, b_ = fit(xs, ys)
            sent = sum(1 for a, b in ps if (a, b)[rank]['bytes'] > 0)
            print(f"  kv rank {rank}: sends bytes in {sent}/{len(ps)} records; ring time = {a_:.1f} + {b_:.2f} us per decode row (147,456 B a row)")
