"""Decode-probe table for an arms.sh suite run with PROBE=1.

  python3 runs/overnight-20261122/probetable.py <OUT> --control a,b --test c,d

Per probe row (label, in order): exact_tok_s for solo rows, agg_tok_s_wall for bursts; control mean, test mean, delta.
"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--control", required=True); ap.add_argument("--test", required=True)
a = ap.parse_args(); ctl, tst = a.control.split(","), a.test.split(",")
R = {n: json.load(open(os.path.join(a.out, f"{n}.probe.json"))) for n in ctl + tst}
def val(r): return r.get("exact_tok_s") or r.get("agg_tok_s_wall") or r.get("tok_per_s")
print("| Probe row | " + " | ".join(ctl + tst) + " | control | test | test vs control |")
print("|---|" + "---:|" * (len(ctl) + len(tst) + 3))
ds = []
for i, r0 in enumerate(R[ctl[0]]):
    v = {n: val(R[n][i]) for n in R}
    if any(x is None for x in v.values()): continue
    cm = sum(v[n] for n in ctl) / len(ctl); tm = sum(v[n] for n in tst) / len(tst); d = 100 * (tm - cm) / cm; ds.append(d)
    print(f"| {r0['label']} | " + " | ".join(f"{float(v[n]):.1f}" for n in ctl + tst) + f" | {cm:.1f} | {tm:.1f} | **{d:+.2f}%** |")
print(f"\nmean tok/s change: {sum(ds) / len(ds):+.2f}% over {len(ds)} rows (positive = faster)")
