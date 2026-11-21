#!/usr/bin/env python3
"""Phase-2 stack comparison for runs/combo (scripts/bench_overnight.py blocks s*).

    python3 runs/combo/stack_report.py [runs/combo] [block prefix, default s; q = phase 4] [reference arm]

Per metric and arm: the block median normalised to the mean of the base blocks of the same pass (the bases that
bracket it: the nearest base before and after in time), then the median and range of those ratios over passes.
Arm vs arm: per pass, the ratio of two arms' block medians (decode paired per rep index first). Uses
tools/overnight_report.py's metric extraction (clean attempts only).
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import overnight_report as orpt  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "runs/combo"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "s"
PAIRED = orpt.PAIRED
KEYS = ["pd.2000", "pd.8000", "pd.32000", "pd.100000", "ttft.8000", "pc.8000", "pp.8000", "dec1", "dec4", "dec8", "t1", "t4"]


def load():
    docs = []
    ignore_clock = os.environ.get("REPORT_IGNORE_CLOCK") == "1"
    for f in glob.glob(os.path.join(D, "blocks", PREFIX + "[0-9]*.json")):
        d = json.load(open(f))
        if isinstance(d, dict) and "cells" in d and orpt.evidence_status(d) == "ok":
            if d.get("clock_valid") or ignore_clock:
                docs.append(d)
    docs.sort(key=lambda d: d["t_start"])
    return [orpt.block_summary(d) for d in docs]


def val(s, k):
    return orpt.med(s["m"].get(k) or [])


def ratio(c, b, k):
    """candidate block over base block: paired per rep for decode, medians otherwise."""
    xc, xb = c["m"].get(k), b["m"].get(k)
    if not xc or not xb:
        return None
    if k in PAIRED:
        p = orpt.paired(xc, xb)
        return orpt.med(p) if p else None
    mc, mb = orpt.med(xc), orpt.med(xb)
    return mc / mb if mc and mb else None


def main():
    S = load()
    observed = {s["arm"] for s in S}
    reference = sys.argv[3] if len(sys.argv) > 3 else ("base" if "base" in observed else "best")
    arms = sorted(observed - {reference})
    if reference not in observed:
        print(f"No valid {reference} reference block; only per-block comparisons with data are reported.", file=sys.stderr)
    print(f"blocks: {' '.join(s['id'] + ':' + s['arm'] for s in S)}\n")
    # vs base: average of the ratios to the nearest base before and after
    print(f"## Each arm vs {reference} controls (median over passes [min..max]; ttft: base/cand so >1 is better)\n")
    print("| metric | " + " | ".join(arms) + " | base-to-base gap |\n|---|" + "---|" * (len(arms) + 1))
    bases = [i for i, s in enumerate(S) if s["arm"] == reference]
    for k in KEYS:
        row = []
        for a in arms:
            rs = []
            for i, s in enumerate(S):
                if s["arm"] != a:
                    continue
                L = next((S[j] for j in reversed(bases) if j < i), None)
                R = next((S[j] for j in bases if j > i), None)
                vals = [ratio(s, b, k) for b in (L, R) if b]
                vals = [v for v in vals if v]
                if vals:
                    r = statistics.mean(vals)
                    rs.append(1 / r if k.startswith("ttft") else r)
            row.append(f"{(orpt.med(rs) - 1) * 100:+.1f}% [{(min(rs) - 1) * 100:+.1f}..{(max(rs) - 1) * 100:+.1f}] n{len(rs)}" if rs else "-")
        gaps = [ratio(S[bases[j + 1]], S[bases[j]], k) for j in range(len(bases) - 1)]
        gaps = [abs(g - 1) * 100 for g in gaps if g]
        print(f"| {k} | " + " | ".join(row) + f" | {max(gaps):.1f}% |" if gaps else f"| {k} | " + " | ".join(row) + " | - |")
    # arm vs arm within a pass (a pass = the blocks between two bases)
    print("\n## Arm vs arm, per pass (ratio of the two arms' blocks in the same pass; median [min..max])\n")
    passes, cur = [], []
    for s in S:
        if s["arm"] == reference:
            if cur:
                passes.append(cur)
            cur = []
        else:
            cur.append(s)
    if cur:
        passes.append(cur)
    pairs = [(x, y) for x in arms for y in arms if x < y]
    print("| metric | " + " | ".join(f"{x}/{y}" for x, y in pairs) + " |\n|---|" + "---|" * len(pairs))
    for k in KEYS:
        row = []
        for x, y in pairs:
            rs = []
            for p in passes:
                bx = next((s for s in p if s["arm"] == x), None)
                by = next((s for s in p if s["arm"] == y), None)
                if bx and by:
                    r = ratio(bx, by, k)
                    if r:
                        rs.append(1 / r if k.startswith("ttft") else r)
            row.append(f"{(orpt.med(rs) - 1) * 100:+.1f}% [{(min(rs) - 1) * 100:+.1f}..{(max(rs) - 1) * 100:+.1f}] n{len(rs)}" if rs else "-")
        print(f"| {k} | " + " | ".join(row) + " |")
    print("\n## Per block (medians)\n")
    print("| block | arm | " + " | ".join(KEYS) + " | mem low H/W GiB |\n|---|---|" + "---|" * (len(KEYS) + 1))
    for s in S:
        print(f"| {s['id']} | {s['arm']} | " + " | ".join("-" if val(s, k) is None else f"{val(s, k):.1f}" for k in KEYS)
              + f" | {s['mem_low_gib'][0]}/{s['mem_low_gib'][1]} |")


if __name__ == "__main__":
    main()
