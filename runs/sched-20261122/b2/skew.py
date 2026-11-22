"""Round B2: where rank 0's extra producer time sits. Pairs both ranks' TF_DS_RING_TRACE gpu records by ring sequence.
Per paired gather: producer = stage - prev (GPU time from the previous gather's completion on that stream to this
gather's stage entry: the compute between collectives, plus any GPU idle), stage = doorbell - stage, wait = flag -
doorbell. Each is a same-clock, same-rank difference; only the *differences* are compared across ranks.
  python3 skew.py r0.jsonl r1.jsonl [--by op|layer|rows|graph|round_pos]"""
import json, sys, collections, statistics as st

def load(p):
    d = {}
    for line in open(p):
        if '"gpu"' not in line: continue
        r = json.loads(line)
        if r.get("kind") != "gpu" or not r.get("prev"): continue
        d[r["seq"]] = r
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else "op"
groups = collections.defaultdict(list)
for q, x in a.items():
    y = b.get(q)
    if not y or x["op"] != y["op"]: continue
    p0, p1 = x["stage"] - x["prev"], y["stage"] - y["prev"]
    if not (0 < p0 < 5e6 and 0 < p1 < 5e6): continue
    key = {"op": x["op"], "layer": (x["op"], x["layer"]), "rows": (x["op"], x["rows"]), "graph": (x["op"], x["graph"]),
           "opl": (x["op"], "L%02d" % x["layer"])}[by]
    groups[key].append((p0 - p1, p0, p1, x["doorbell"] - x["stage"], y["doorbell"] - y["stage"],
                        x["flag"] - x["doorbell"], y["flag"] - y["doorbell"]))
print(f"{'group':<34}{'n':>7}{'dprod p50':>11}{'mean':>9}{'prod0':>9}{'prod1':>9}{'stg0':>7}{'stg1':>7}{'wait0':>8}{'wait1':>8}  (µs)")
for k in sorted(groups, key=str):
    g = groups[k]
    if len(g) < 30: continue
    col = lambda i: [t[i] / 1000 for t in g]
    print(f"{str(k):<34}{len(g):>7}{st.median(col(0)):>11.2f}{st.mean(col(0)):>9.2f}{st.median(col(1)):>9.1f}{st.median(col(2)):>9.1f}"
          f"{st.median(col(3)):>7.2f}{st.median(col(4)):>7.2f}{st.median(col(5)):>8.1f}{st.median(col(6)):>8.1f}")
