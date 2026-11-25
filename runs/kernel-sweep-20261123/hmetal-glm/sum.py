#!/usr/bin/env python3
"""Sum the shards' STAT lines (stdin)."""
import sys
tot = [0.0] * 9
for line in sys.stdin:
    if not line.startswith("STAT "):
        sys.stdout.write(line)
        continue
    tot = [a + float(b) for a, b in zip(tot, line.split()[1:])]
cases, launches, nbytes, fail, d_old, d_off, d_cut, blocks, skipped = tot
print("hmetal-glm GLM_INDEX_CUT: %d cases, %d launches, each case under 3 lane orders (asc, desc, shuffle)" % (cases, launches))
print("compared %d output bytes old-vs-cut0 and old-vs-cut1: %d failures" % (nbytes, fail))
if blocks:
    print("digit passes a threadgroup that runs the select (of 4): old %.2f  cut 0 %.2f  cut 1 %.2f  (%d threadgroup runs; %d more took every block and ran no select)"
          % (d_old / blocks, d_off / blocks, d_cut / blocks, blocks, skipped))
print("PASS" if fail == 0 else "FAIL")
sys.exit(0 if fail == 0 else 1)
