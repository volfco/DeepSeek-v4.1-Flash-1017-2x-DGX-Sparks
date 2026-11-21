#!/usr/bin/env python3
"""1-stream decode cells: per-cell fixed part (ms/round - 4.93*rows), and a joint least-squares fit
ms/round = a + b*rows + c*(prompt/100K). python3 -I fit1.py depth.json"""
import json, sys
doc = json.load(open(sys.argv[1]))
pts = []
for c in doc["cells"]:
    if c["kind"] not in ("dec1", "end1", "low1") or c["streams"] != 1:
        continue
    r = c["attempts"][-1]["results"][0]
    if r.get("ms_per_round") is None:
        continue
    pts.append((c["name"], r["prompt_tokens"], r["rows_per_round"], r["ms_per_round"], r.get("ms_per_round_client")))
print("| cell | prompt | rows | ms/round | ms - 4.93*rows | model (11.4+4.93r+4) | vs model |")
print("|---|---|---|---|---|---|---|")
for n, p, rw, ms, msc in pts:
    m = 15.4 + 4.93 * rw
    print(f"| {n} | {p} | {rw:.2f} | {ms:.1f} | {ms - 4.93*rw:.1f} | {m:.1f} | {100*(ms/m-1):+.1f}% |")
# joint fit by normal equations (3 params)
X = [(1.0, rw, p / 1e5) for _, p, rw, _, _ in pts]
y = [ms for *_, ms, _ in pts]
import itertools
A = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
B = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(3)]
# solve 3x3
def solve(A, B):
    M = [row[:] + [b] for row, b in zip(A, B)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda k: abs(M[k][i])); M[i], M[piv] = M[piv], M[i]
        for k in range(3):
            if k != i:
                f = M[k][i] / M[i][i]
                M[k] = [a - f * b for a, b in zip(M[k], M[i])]
    return [M[i][3] / M[i][i] for i in range(3)]
a, b, c = solve(A, B)
res = [yy - (a + b * x[1] + c * x[2]) for x, yy in zip(X, y)]
import math
print(f"\njoint fit: ms/round = {a:.2f} + {b:.2f}*rows + {c:.2f}*(prompt/100K); n={len(pts)}, rms resid {math.sqrt(sum(r*r for r in res)/len(res)):.2f} ms")

# short line from the short cells only (prompt <= 2.1K), then each depth cell against it at its own rows
sh = [(rw, ms) for _, p, rw, ms, _ in pts if p <= 2100]
mx = sum(r for r, _ in sh) / len(sh); my = sum(m for _, m in sh) / len(sh)
bs = sum((r - mx) * (m - my) for r, m in sh) / sum((r - mx) ** 2 for r, _ in sh)
as_ = my - bs * mx
rs = [m - (as_ + bs * r) for r, m in sh]
print(f"\nshort line (n={len(sh)}, prompt<=2.1K, rows {min(r for r,_ in sh):.2f}-{max(r for r,_ in sh):.2f}): ms/round = {as_:.2f} + {bs:.2f}*rows, rms resid {math.sqrt(sum(x*x for x in rs)/len(rs)):.2f}")
print("\n| cell | prompt | rows | ms/round | short line at these rows | excess ms | excess % |")
print("|---|---|---|---|---|---|---|")
for n, p, rw, ms, _ in pts:
    if p > 2100:
        s = as_ + bs * rw
        print(f"| {n} | {p} | {rw:.2f} | {ms:.1f} | {s:.1f} | {ms - s:+.1f} | {100 * (ms / s - 1):+.1f}% |")
