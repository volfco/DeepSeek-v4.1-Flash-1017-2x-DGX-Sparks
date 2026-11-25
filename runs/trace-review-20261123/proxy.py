"""Lead 6: the proxy's records (host CLOCK_MONOTONIC, rdma.zig proxy loop) by transfer class, including the drafter's
`other` transfers and the head, which k272's table did not list: doorbell observed -> last RDMA write posted, last
NIC part's completion -> flag published, doorbell -> flag, and how many records carry `woke` (first after a nap).
usage: proxy.py arm [arm ...]"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import *
from recon import annotate
for name in sys.argv[1:]:
    for rank in (0, 1):
        g, units, x = annotate(name, rank, greedy=name in ('fx1L', 'fx1Lb', 'np1L', 'ns1L'))
        px = {d['seq']: d for d in load(name, rank, kinds=('proxy', 'gpu'))['proxy']}
        cls = collections.defaultdict(list)
        for r in g:
            p = px.get(r['seq'])
            if not p or r['unit'] not in 'FP': continue
            k = r['op'] if r['unit'] == 'F' else ('pass hidden' if r['kind'] in 'AM' else 'pass markov')
            parts = [t for t in p['parts'] if t]
            cls[k].append((p['last_post'] - p['doorbell'] if p['last_post'] and p['doorbell'] else None,
                           p['flag'] - max(parts) if parts and p['flag'] else None,
                           p['flag'] - p['doorbell'] if p['flag'] and p['doorbell'] else None, p['woke']))
        print(f"# {name} rank {rank}: proxy records matched {sum(len(v) for v in cls.values())} of {len(px)}; woke flags {sum(1 for d in px.values() if d['woke'])}")
        for k in ('attn_gather', 'moe_gather', 'kv', 'engram', 'head', 'pass hidden', 'pass markov'):
            v = cls.get(k)
            if not v: continue
            col = lambda i: sorted(t[i] for t in v if t[i] is not None)
            a, b, c = col(0), col(1), col(2)
            print(f"   {k:<12} n {len(v):>5} doorbell->last post p50 {pct(a,50)/1e3:6.2f} p99 {pct(a,99)/1e3:7.2f} us | last completion->flag p50 {pct(b,50):5.0f} p99 {pct(b,99):6.0f} ns | doorbell->flag (host clock; includes waiting for the peer's write) p50 {pct(c,50)/1e3:6.1f} p99 {pct(c,99)/1e3:7.1f} us | woke {sum(t[3] for t in v)}")
