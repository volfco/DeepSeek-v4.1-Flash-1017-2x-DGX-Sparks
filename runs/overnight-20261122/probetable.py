"""Decode-probe table for an arms.sh suite run with PROBE=1.

  python3 runs/overnight-20261122/probetable.py <OUT> --control a,b --test c,d [--metric M]

Per probe row (label, in order): control mean, test mean, delta of the metric. Default metric: exact_tok_s for solo
rows, agg_tok_s_wall for bursts (both quantized by the engine's 0.1 s decode_s). Other metrics, from probes that
recorded client timing: client_tok_s (first-to-last chunk, finer), chunk_gap_ms_p50, chunk_gap_ms_p95 (lower is
better: the sign is flipped so positive is always an improvement). A "rounds" column flags solo rows whose round
count differs between arms (the doc's "matched round counts": a mismatch means the comparison mixes in a different
token path).
"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--control", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--metric", default="tok_s")
a = ap.parse_args(); ctl, tst = a.control.split(","), a.test.split(",")
R = {n: json.load(open(os.path.join(a.out, f"{n}.probe.json"))) for n in ctl + tst}
lower_better = a.metric.startswith("chunk_gap")


def val(r):
    if a.metric == "tok_s": return r.get("exact_tok_s") or r.get("agg_tok_s_wall") or r.get("tok_per_s")
    return r.get(a.metric)


print(f"metric: {a.metric}")
print("| Probe row | " + " | ".join(ctl + tst) + " | control | test | test vs control | rounds |")
print("|---|" + "---:|" * (len(ctl) + len(tst) + 3) + "---|")
ds = []
for i, r0 in enumerate(R[ctl[0]]):
    v = {n: val(R[n][i]) for n in R}
    if any(x is None for x in v.values()): continue
    v = {n: float(x) for n, x in v.items()}
    cm = sum(v[n] for n in ctl) / len(ctl); tm = sum(v[n] for n in tst) / len(tst)
    d = 100 * (tm - cm) / cm * (-1 if lower_better else 1); ds.append(d)
    rs = {R[n][i].get("rounds") for n in R}
    rounds = "" if rs == {None} else ("match" if len(rs) == 1 else "DIFFER " + "/".join(str(R[n][i].get("rounds")) for n in ctl + tst))
    print(f"| {r0['label']} | " + " | ".join(f"{v[n]:.2f}" for n in ctl + tst) + f" | {cm:.2f} | {tm:.2f} | **{d:+.2f}%** | {rounds} |")
if ds: print(f"\nmean change: {sum(ds) / len(ds):+.2f}% over {len(ds)} rows (positive = better)")
