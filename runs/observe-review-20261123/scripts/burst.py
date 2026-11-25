#!/usr/bin/env python3
"""The saturated stretch of the current production epoch (E7): throughput by wall clock, queueing, disconnects,
and decode rounds while a long prompt pass was in flight. Reads data/scrapes.json and data/records.json.

    python3 -I scripts/burst.py
"""
import datetime as dt
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "data/scrapes.json")))
R = json.load(open(os.path.join(ROOT, "data/records.json")))
f = lambda u: dt.datetime.fromtimestamp(u / 1e6, dt.timezone.utc).strftime("%H:%M:%S")
last = S[-1]["life"]
cur = [s for s in S if s["life"] == last]
print("current lifetime:", f(cur[0]["t"]), "..", f(cur[-1]["t"]), "UTC,", len(cur), "scrapes")
# stretches with a queue (waiting > 0)
sat = [s for s in cur if s.get("waiting", 0) > 0]
if sat:
    a, b = sat[0]["t"], sat[-1]["t"]
    full = [s for s in cur if a <= s["t"] <= b]
    g0 = next(s for s in cur if s["t"] >= a)
    g1 = [s for s in cur if s["t"] <= b][-1]
    wall = (g1["t"] - g0["t"]) / 1e6
    print(f"queue present {f(a)}..{f(b)} ({wall/60:.1f} min): running mean {sum(s['running'] for s in full)/len(full):.2f}, "
          f"waiting mean {sum(s['waiting'] for s in full)/len(full):.1f}, max {max(s['waiting'] for s in full):.0f}")
    rr = [r for r in R if r["life"] == last and a < r["t"] <= b]
    n = sum(r["n"] for r in rr)
    print(f"  finished in the stretch: {n} requests, generated {sum(r['gen'] for r in rr):,.0f} tokens -> {sum(r['gen'] for r in rr)/wall:.1f} tok/s by wall "
          f"(tokens are counted when a request ends); prompt tokens {sum(r['prompt'] for r in rr):,.0f} ({100*sum(r['cached'] for r in rr)/sum(r['prompt'] for r in rr):.1f}% cached)")
    print(f"  mean time to first token {sum(r['ttft_s'] for r in rr)/max(1,sum(r['ttft_n'] for r in rr)):.0f} s, mean prompt pass {sum(r['pre_s'] for r in rr)/max(1,sum(r['pre_n'] for r in rr)):.2f} s, "
          f"mean latency {sum(r['lat_s'] for r in rr)/n:.0f} s; client disconnects {sum(r['disc'] for r in rr):.0f} of {n}")
    one = [r for r in rr if r["n"] == 1]
    cut = [r for r in one if 599.5 <= r["lat_s"] <= 601.5]
    print(f"  single-request records: {len(one)}; of them {len(cut)} ended at 600.0-601.5 s after arrival")
    # the stretch's decode rounds while a cold prompt pass of >= 100K uncached tokens finished nearby
    big = [r for r in rr if (r["prompt"] - r["cached"]) >= 100000]
    for r in big:
        t1 = r["t"]
        near = [x for x in rr if x is not r and t1 - 150e6 <= x["t"] <= t1 and x["rounds"] > x["dec_n"]]
        ms = [1000 * x["dec_s"] / (x["rounds"] - x["dec_n"]) for x in near]
        print(f"  cold prompt pass ending by {f(t1)}: {r['prompt']-r['cached']:,.0f} uncached tokens in {r['pre_s']:.1f} s of prompt pass "
              f"({(r['prompt']-r['cached'])/r['pre_s']:.0f} tok/s with {r['run_max']:.0f} streams live); decode ms/round of requests finishing in the 150 s before: "
              + ", ".join(f"{m:.0f}" for m in ms))
# idle and single-stream time
print("scrapes by running count:", {int(k): sum(1 for s in cur if s.get('running', 0) == k) for k in sorted(set(s.get('running', 0) for s in cur))})
