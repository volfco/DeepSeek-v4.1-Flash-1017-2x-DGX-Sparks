"""B2 row/concurrency split: per arm, paired by ring sequence, the producer gap (r0 - r1) on hidden gathers, its share of
rank 0's producer time, the rows seen, and the imbalance cost per round (gap/2 x gathers in a round).
  python3 analyze.py <dir> arm [arm ...]"""
import json, sys, collections, statistics as st
def load(p):
    d = {}
    for l in open(p):
        if '"gpu"' in l:
            r = json.loads(l)
            if r.get("prev"): d[r["seq"]] = r
    return d
D = sys.argv[1]
print(f"{'arm':<6}{'op':<6}{'n':>7}{'rows(p50)':>10}{'gap mean':>10}{'gap p50':>9}{'gap %':>7}{'cost ms/rnd':>12}{'wait r0':>9}{'wait r1':>9}  (µs)")
for arm in sys.argv[2:]:
    a, b = load(f"{D}/{arm}.trace.r0.jsonl"), load(f"{D}/{arm}.trace.r1.jsonl")
    g = collections.defaultdict(list)
    for q, x in a.items():
        y = b.get(q)
        if not y or x["op"] not in ("attn_gather", "moe_gather"): continue
        p0, p1 = x["stage"] - x["prev"], y["stage"] - y["prev"]
        if not (0 < p0 < 5e6 and 0 < p1 < 5e6): continue
        g[x["op"][:4]].append((p0 - p1, p0, x["rows"], x["flag"] - x["doorbell"], y["flag"] - y["doorbell"]))
    for op, v in sorted(g.items()):
        gap = [t[0] / 1000 for t in v]; p0 = [t[1] / 1000 for t in v]
        print(f"{arm:<6}{op:<6}{len(v):>7}{st.median([t[2] for t in v]):>10}{st.mean(gap):>+10.2f}{st.median(gap):>+9.2f}"
              f"{100 * st.mean(gap) / st.mean(p0):>+7.2f}{40 * st.mean(gap) / 2 / 1000:>+12.3f}"
              f"{st.median([t[3] for t in v]) / 1000:>9.1f}{st.median([t[4] for t in v]) / 1000:>9.1f}")
