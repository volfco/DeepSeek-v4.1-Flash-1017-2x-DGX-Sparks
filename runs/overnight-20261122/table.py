"""Round-time table for an arms.sh suite.

  python3 runs/overnight-20261122/table.py <OUT dir> --control a,b --test c,d [--md]

Reads <OUT>/<arm>.json (kvreuse-size workload output) and prints, per cell, each arm's graphs-on forward_ms a round,
the control mean, the test mean, the test-vs-control delta, the control-vs-control drift, and the count of
test arms below every control arm. Exact-panel results come from <OUT>/<arm>.cmp.txt (last line).
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--control", required=True)
ap.add_argument("--test", required=True)
a = ap.parse_args()
ctl, tst = a.control.split(","), a.test.split(",")
arms = {n: json.load(open(os.path.join(a.out, f"{n}.json"))) for n in ctl + tst}
cells = arms[ctl[0]]["cells"]
print("| Depth × streams | " + " | ".join(ctl + tst) + " | control mean | test mean | test vs control | control drift | test below all controls |")
print("|---|" + "---:|" * (len(ctl) + len(tst) + 5))
deltas = []
for i, c in enumerate(cells):
    v = {n: arms[n]["cells"][i]["per_round"]["forward_ms_per_round"] for n in arms}
    cm = sum(v[n] for n in ctl) / len(ctl)
    tm = sum(v[n] for n in tst) / len(tst)
    d = 100 * (tm - cm) / cm
    deltas.append(d)
    drift = 100 * (v[ctl[-1]] - v[ctl[0]]) / v[ctl[0]] if len(ctl) > 1 else 0.0
    below = sum(all(v[t] < v[k] for k in ctl) for t in tst)
    print(f"| {c['depth'] // 1024}K × {c['streams']} | " + " | ".join(f"{v[n]:.2f}" for n in ctl + tst)
          + f" | {cm:.2f} | {tm:.2f} | **{d:+.2f}%** | {drift:+.2f}% | {below}/{len(tst)} |")
print(f"\nmean test vs control: {sum(deltas) / len(deltas):+.2f}%")
for n in ctl + tst:
    p = os.path.join(a.out, f"{n}.cmp.txt")
    if os.path.exists(p):
        print(f"exact {n}: {open(p).read().strip().splitlines()[-1]}")
