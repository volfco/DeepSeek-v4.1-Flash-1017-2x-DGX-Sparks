#!/usr/bin/env python3
"""W4's numbers from the run's logs: the probe's GB/s per mode and each lanes cell's round profile, places and bytes.
usage: w4_analyze.py [DIR]   (default runs/w4) -> stdout (summary.txt)
Bytes per place (routed and shared) come from the CUDA_LAUNCH_BLOCKING=1 cells only: without it the counter's read-back (a
synchronous copy on the legacy stream) is not ordered after the experts on the engine's non-blocking stream, and its
places disagree with the device-side expert_prune counts (see RESULTS.md)."""
import glob, json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/w4"


def lines(path):
    out = {}
    for ln in open(path, errors="replace"):
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        for k in ("profile", "expert_prune", "expert_bytes", "phases_ms", "prefill_phases_ms", "follow_profile",
                  "prefill_profile_ms", "all_equal"):
            if k in d:
                out[k] = d if k == "all_equal" else d[k]
        for k in d:
            if k.startswith("prefill_profile"):
                out[k] = d[k]
    return out


print("== probe (median GB/s; bytes = sum of launched ranges)")
rx = re.compile(r"kernel=(\w+) case=([\w-]+) prefetch=(\w+) experts=(\d+) tiles=(\d+) bytes=(\d+) reps=(\d+) "
                r"median_gbs=([\d.]+) best_gbs=([\d.]+) worst_gbs=([\d.]+)")
for f in sorted(glob.glob(f"{D}/probe.*.log")):
    lab = f.split("probe.")[1][:-4]
    for ln in open(f):
        m = rx.search(ln)
        if m:
            k, case, pf, e, t, b, reps, med, best, worst = m.groups()
            print(f"{lab:10s} {k:7s} {case:17s} pf={pf:3s} e={e:2s} t={t:2s} MB={int(b)/1e6:7.1f} med={med:>7s} "
                  f"best={best:>7s} worst={worst:>7s}")

print("\n== lanes cells (rank 0 unless noted)")
cells = sorted({os.path.basename(f).rsplit(".rank0.log", 1)[0] for f in glob.glob(f"{D}/*.rank0.log")})
bpp = {}
rows = []
for c in cells:
    r0 = lines(f"{D}/{c}.rank0.log")
    r1 = lines(f"{D}/{c}.rank1.log") if os.path.exists(f"{D}/{c}.rank1.log") else {}
    p, ep, xb, ph = r0.get("profile"), r0.get("expert_prune"), r0.get("expert_bytes"), r0.get("phases_ms")
    if not p or not p["rounds"]:
        print(f"{c}: no profile line")
        continue
    rounds = p["rounds"]
    wpr = ep["windows"] / rounds if ep else None
    line = {"cell": c, "rounds": rounds, "rows": p["rows_a_round"], "forward_ms": p["forward_ms"],
            "moe_ms": ph.get("moe") if ph else None, "windows_a_round": wpr,
            "routed_places": ep["distinct_after"] if ep else None, "before_prune": ep["distinct_before"] if ep else None,
            "all_equal": r0.get("all_equal", {}).get("all_equal")}
    if xb:
        line["xb_places_r0"] = xb["places_a_window"]
        line["xb_mb_window_r0"] = xb["kb_a_window"] * 1024 / 1e6
        if r1.get("expert_bytes"):
            line["xb_places_r1"] = r1["expert_bytes"]["places_a_window"]
        if "-lb" in c and ep:
            # fresh read-back: places = routed + 1 (the shared expert's place, with its own weights)
            line["mb_per_place"] = line["xb_mb_window_r0"] / line["xb_places_r0"]
            bpp[c] = line["mb_per_place"]
    rows.append(line)
    print(json.dumps(line))

print("\n== Estimated eager MoE rate from blocking bytes/place and nonblocking times (routed + shared)")
for line in rows:
    if line["moe_ms"] is None or line["routed_places"] is None or "-lb" in line["cell"]:
        continue
    src = "e-r4-lb" if line["rows"] > 6 and "e-r4-lb" in bpp else "e-r1-lb"
    if src not in bpp:
        continue
    gb = (line["routed_places"] + 1) * line["windows_a_round"] * bpp[src] / 1e3
    gb_avg = (line["routed_places"] + 1) * line["windows_a_round"] * 6.47 / 1e3
    print(f"{line['cell']:12s} rows={line['rows']:5.2f} places/window={line['routed_places']:6.3f} "
          f"MB/place({src})={bpp[src]:.3f} GB/round={gb:.3f} moe={line['moe_ms']:.3f} ms -> {gb / line['moe_ms'] * 1e3:.0f} GB/s "
          f"(6.47 MB average: {gb_avg / line['moe_ms'] * 1e3:.0f} GB/s)")

pf = {c: lines(f"{D}/{c}.rank0.log") for c in cells if c.startswith("pf")}
for c, d in pf.items():
    print(f"\n== {c} prefill profile")
    for k, v in d.items():
        if k.startswith("prefill"):
            print(k, json.dumps(v))
