#!/usr/bin/env python3
"""Sum the shards' STAT lines of a harness (stdin) into its summary. Usage: sum_stats.py <title> <digits-of>"""
import sys
title = sys.argv[1]
tot = None
cls = {}
order = []
for line in sys.stdin:
    if not line.startswith("STAT "):
        sys.stdout.write(line)
        continue
    f = line.split()[1:]
    head = [float(x) for x in f[:10]]
    tot = head if tot is None else [a + b for a, b in zip(tot, head)]
    rest = f[10:]
    for i in range(0, len(rest), 5):
        name = rest[i]
        v = [float(x) for x in rest[i + 1:i + 5]]
        if name not in cls:
            cls[name] = [0.0] * 4
            order.append(name)
        cls[name] = [a + b for a, b in zip(cls[name], v)]
cases, sel, draw, launches, nbytes, fail, d0, d1, d2, b = tot
print("%s: %d cases (%d select, %d draw), %d launches, each case under 3 lane orders (asc, desc, shuffle)" % (title, cases, sel, draw, launches))
print("compared %d output bytes old-vs-off and old-vs-cut: %d failures" % (nbytes, fail))
if b:
    print("select digit passes a block (of 8), small widths: old %.2f  off %.2f  cut %.2f  (%d blocks)" % (d0 / b, d1 / b, d2 / b, b))
print("select digit passes a block at 64,640 columns, by row class (old / off / cut, blocks):")
for name in order:
    v = cls[name]
    if v[3]:
        print("  %-16s %.2f / %.2f / %.2f   (%d)" % (name, v[0] / v[3], v[1] / v[3], v[2] / v[3], v[3]))
print("PASS" if fail == 0 else "FAIL")
sys.exit(0 if fail == 0 else 1)
