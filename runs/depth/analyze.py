#!/usr/bin/env python3
"""Tables from runs/depth/depth.json: per decode cell and per warm turn. python3 -I analyze.py depth.json"""
import json
import statistics
import sys

doc = json.load(open(sys.argv[1]))


def f(x, n=1):
    return "-" if x is None else f"{x:.{n}f}"


def model1(rows):
    return 11.4 + 4.93 * rows + 4.0


print("## decode cells (last attempt; one row per request)\n")
print("| cell | clean | prompt | cached | ct | finish | ms/round | ms/round (client) | rows/round | model ms | +% vs model | tok/round | tok/s (done) | prefill_s | queue_s | restore_s | mem low H/W |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in doc["cells"]:
    if c["kind"] not in ("dec1", "dec4", "end1", "end4", "warmup"):
        continue
    a = c["attempts"][-1]
    rs = a.get("results", [])
    tot_rows = sum(r.get("rows_per_round") or 0 for r in rs)
    for r in rs:
        dn = r.get("done", {})
        rows = r.get("rows_per_round")
        if c["streams"] == 1:
            m = model1(rows) if rows else None
        else:
            m = 11.4 + 4.93 * tot_rows  # forward only, summed rows; no 1-stream step overhead
        msr = r.get("ms_per_round")
        print(f"| {c['name']} | {a.get('clean')} | {r.get('prompt_tokens')} | {r.get('cached_tokens')} | {r.get('completion_tokens')} | {r.get('finish')} | "
              f"{f(msr)} | {f(r.get('ms_per_round_client'))} | {f(rows, 2)} | {f(m)} | {f(100 * (msr / m - 1) if msr and m else None)} | "
              f"{f(r.get('tok_per_round'), 2)} | {dn.get('tok_per_s', '-')} | {dn.get('prefill_s', '-')} | {dn.get('queue_s', '-')} | {dn.get('restore_s', '-')} | "
              f"{c['mem_low_gib'].get('head')}/{c['mem_low_gib'].get('worker')} |")

print("\n## primes (incremental prefill per rung)\n")
print("| cell | prompt | cached | uncached | prefill_s | tok/s uncached | restore_s | mem low H/W |")
print("|---|---|---|---|---|---|---|---|")
for c in doc["cells"]:
    if c["kind"] not in ("prime", "hist"):
        continue
    for r in c["attempts"][-1].get("results", []):
        dn = r.get("done", {})
        pt, ca = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
        pf = float(dn["prefill_s"]) if "prefill_s" in dn else None
        print(f"| {c['name']} | {pt} | {ca} | {pt - ca} | {f(pf, 3)} | {f((pt - ca) / pf if pf else None, 0)} | {dn.get('restore_s', '-')} | "
              f"{c['mem_low_gib'].get('head')}/{c['mem_low_gib'].get('worker')} |")

print("\n## warm turns\n")
print("| cell | prompt | cached | uncached | prefill_s | ttft_s (server) | ttft client | queue_s | restore_s | tok/s uncached |")
print("|---|---|---|---|---|---|---|---|---|---|")
fit = {}
for c in doc["cells"]:
    if c["kind"] != "warm":
        continue
    for r in c["attempts"][-1].get("results", []):
        dn = r.get("done", {})
        pt, ca = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
        pf = float(dn["prefill_s"]) if "prefill_s" in dn else None
        print(f"| {c['name']} | {pt} | {ca} | {pt - ca} | {f(pf, 3)} | {dn.get('ttft_s', '-')} | {f(r.get('ttft_s'), 3)} | {dn.get('queue_s', '-')} | "
              f"{dn.get('restore_s', '-')} | {f((pt - ca) / pf if pf else None, 0)} |")
        if pf is not None and float(dn.get("restore_s", 0)) == 0:
            fit.setdefault(c["depth_target"], []).append((pt - ca, pf))

print("\n## fit prefill_s = a + b * uncached per history (restore_s == 0 rows)\n")
for H, pts in sorted(fit.items()):
    if len(pts) < 3:
        continue
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    print(f"- history {H}: a = {a:.3f} s fixed, b = {b * 1000:.3f} ms/token (= {1 / b:.0f} tok/s marginal), n={len(pts)}")
