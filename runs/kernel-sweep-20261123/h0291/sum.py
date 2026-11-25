#!/usr/bin/env python3
"""Sum the shards' STAT lines (stdin): cases, launches, bytes compared, failures, the encodings seen, and the warp
collectives each arm's source executed."""
import sys
tot = [0] * 8
seen = 0
for line in sys.stdin:
    if line.startswith("SEEN "):
        seen |= int(line.split()[1], 16)
        continue
    if not line.startswith("STAT "):
        sys.stdout.write(line)
        continue
    tot = [a + int(b) for a, b in zip(tot, line.split()[1:])]
cases, launches, nbytes, fail, red_old, shf_old, red_new, shf_new = tot
print("h0291 TF_DS_QD_PAIR_SHFL: %d cases, %d launches, each case under 3 lane orders (asc, desc, shuffle)" % (cases, launches))
print("compared %d output bytes (pair old vs pair off, pair old vs pair shfl, one-element old vs pair shfl): %d failures" % (nbytes, fail))
print("warp collectives the source executed, summed over lanes: old pair %d masked reductions, %d shuffles; shfl pair %d masked reductions, %d shuffles" % (red_old, shf_old, red_new, shf_new))
print("PASS" if fail == 0 else "FAIL")
sys.exit(0 if fail == 0 else 1)
