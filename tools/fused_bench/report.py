#!/usr/bin/env python3
"""Turn bench.py's JSON, bw.py's peak and sass.py's occupancy into the ranked baseline table.

  python3 report.py wp0.json bw.json sass.json [--md BASELINE.md]

For each timed fused launch it prints time, launch floor, bytes moved, the copy roofline (bytes / measured peak
bandwidth), the achieved bandwidth, the bound it sits at, and the headroom over max(floor, roofline).
"""

import argparse
import json

BOUND_FLOOR = 1.10   # within 10% of the empty-kernel floor: launch-bound
BOUND_MEM = 0.85     # roofline within 15% of the measured time: bandwidth-bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("bw")
    ap.add_argument("sass")
    ap.add_argument("--md")
    a = ap.parse_args()
    bench = json.loads(open(a.bench).read())
    bw = json.loads(open(a.bw).read())
    sass = json.loads(open(a.sass).read())
    cache = bw.get("cache_GBps", bw["peak_GBps"])
    dram = bw.get("dram_GBps", bw["peak_GBps"])
    rows = []
    for r in bench["results"]:
        t = r.get("timing")
        if not t or "new" not in t or "floor" not in t:
            continue
        med = t["new"]["median"]
        floor = t["floor"]["median"]
        moved = r.get("bytes", 0)
        # a working set past ~32 MB cannot stay in L2 across replays: use the measured DRAM copy rate for it
        peak = dram if moved > 32e6 else cache
        basis = "DRAM" if moved > 32e6 else "cache"
        roof = moved / (peak * 1e9) * 1e6 if moved else 0.0
        floor_bound = med <= floor * BOUND_FLOOR
        mem_bound = roof >= med * BOUND_MEM
        bound = "launch floor" if floor_bound else ("bandwidth" if mem_bound else "issue/latency")
        head = med - max(floor, roof)
        ach = (moved / (med * 1e-6) / 1e9) if med else 0.0
        rows.append({"kernel": r["kernel"], "shape": r["shape"], "us": med, "iqr": t["new"]["iqr"], "basis": basis,
                     "floor": floor, "MB": moved / 1e6, "roof": roof, "GBps": ach,
                     "bound": bound, "headroom": head, "ref": t.get("ref", {}).get("median"),
                     "old": t.get("old", {}).get("median")})
    rows.sort(key=lambda x: -x["headroom"])
    print(f"copy peaks: cache {cache:.1f} GB/s, DRAM {dram:.1f} GB/s; ranked by headroom over max(floor, roofline)")
    print(f"{'kernel':<24} {'shape':<48} {'us':>9} {'floor':>7} {'MB':>7} {'roof':>8} {'GB/s':>7} "
          f"{'bound':<14} {'head':>8}")
    lines = ["| kernel | shape | us | floor | MB | roof us | GB/s | basis | bound | headroom us |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for x in rows:
        print(f"{x['kernel']:<24} {x['shape']:<48} {x['us']:>9.3f} {x['floor']:>7.3f} {x['MB']:>7.2f} "
              f"{x['roof']:>8.2f} {x['GBps']:>7.1f} {x['bound']:<14} {x['headroom']:>8.3f}")
        lines.append(f"| {x['kernel']} | {x['shape']} | {x['us']:.3f} | {x['floor']:.3f} | {x['MB']:.2f} | "
                     f"{x['roof']:.2f} | {x['GBps']:.1f} | {x['basis']} | {x['bound']} | {x['headroom']:.3f} |")
    if a.md:
        open(a.md, "w").write("\n".join(lines) + "\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
