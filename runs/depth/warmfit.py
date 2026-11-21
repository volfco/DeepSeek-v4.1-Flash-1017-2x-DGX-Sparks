#!/usr/bin/env python3
"""Warm-turn fits: prefill_s = a + b*uncached per history, warm rows with cached > 0 plus the history prime
(zero append) as an extra point; and the ttft_s - prefill_s gap. python3 -I warmfit.py depth.json"""
import json, statistics, sys
doc = json.load(open(sys.argv[1]))
rows = {}
for c in doc["cells"]:
    if c["kind"] not in ("warm", "hist"):
        continue
    r = c["attempts"][-1]["results"][0]
    dn = r.get("done", {})
    pt, ca = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
    if ca == 0 or "prefill_s" not in dn:
        continue
    rows.setdefault(c["depth_target"], []).append((c["name"], pt - ca, float(dn["prefill_s"]), float(dn["ttft_s"])))
for H, pts in sorted(rows.items()):
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    gap = [p[3] - p[2] for p in pts if p[0].startswith("warm")]
    pred22 = a + b * 2200
    print(f"history {H}: n={len(pts)} uncached {min(xs)}-{max(xs)}: prefill_s = {a:.3f} + {b*1000:.3f} ms * uncached "
          f"(marginal {1/b:.0f} tok/s); ttft-prefill gap median {statistics.median(gap):.3f} s; "
          f"a 2,200-token re-prefill -> {pred22:.2f} s prefill, {pred22 + statistics.median(gap):.2f} s ttft")
