"""Fit a TF_DS_ROUND_MS table (forward ms a round by rows 1..24) from the rowcost arms.

  python3 runs/overnight-20261122/rowcost/fit.py [--depth 4096|all]

Points: every cell's (rows_a_round, forward_ms) from fk*.json. The table is a least-squares line through the points
plus the piecewise-linear residual where points exist (isotonic: never decreasing), the line past the last point.
Prints the points, the served table, the fit, and the TF_DS_ROUND_MS string.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVED = [30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6,
          93.7, 96.8, 99.9, 103.0, 106.1, 109.2]
depth = sys.argv[sys.argv.index("--depth") + 1] if "--depth" in sys.argv else "4096"
pts = []
for f in sorted(glob.glob(os.path.join(HERE, "fk*.json"))):
    r = json.load(open(f))
    for c in r["cells"]:
        if (depth == "all" or c["depth"] == int(depth)) and c.get("rounds", 0) >= 20 and c.get("rows_a_round"):
            pts.append((c["rows_a_round"], c["forward_ms"], os.path.basename(f)[:-5], c["depth"], c["streams"], c["rounds"]))
pts.sort()
print("| arm | depth | streams | rounds | rows a round | forward ms | served table at rows |")
print("|---|---:|---:|---:|---:|---:|---:|")
for x, y, a, d, s, n in pts:
    i = min(len(SERVED) - 1, max(0, round(x) - 1))
    print(f"| {a} | {d // 1024}K | {s} | {n} | {x:.2f} | {y:.2f} | {SERVED[i]:.1f} |")
n = len(pts)
mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
a = my - b * mx
print(f"\nline: {a:.2f} + {b:.3f} x rows")


def interp(r):
    # piecewise-linear through the points (averaging points closer than 0.25 rows), the line outside them
    xs = []
    for x, y, *_ in pts:
        if xs and x - xs[-1][0] < 0.25:
            px, py, k = xs[-1]; xs[-1] = ((px * k + x) / (k + 1), (py * k + y) / (k + 1), k + 1)
        else:
            xs.append((x, y, 1))
    if r <= xs[0][0]:
        return xs[0][1] - b * (xs[0][0] - r)
    if r >= xs[-1][0]:
        return xs[-1][1] + b * (r - xs[-1][0])
    for (x0, y0, _), (x1, y1, _) in zip(xs, xs[1:]):
        if x0 <= r <= x1:
            return y0 + (y1 - y0) * (r - x0) / (x1 - x0)


tab = []
for r in range(1, 25):
    v = interp(r)
    tab.append(round(max(v, tab[-1] if tab else 0.0), 1))
print("\n| rows | served | fit | fit / served |")
print("|---:|---:|---:|---:|")
for r, (s, f) in enumerate(zip(SERVED, tab), 1):
    print(f"| {r} | {s:.1f} | {f:.1f} | {f / s:.2f} |")
print("\nTF_DS_ROUND_MS=" + ",".join(f"{v:.1f}" for v in tab))
