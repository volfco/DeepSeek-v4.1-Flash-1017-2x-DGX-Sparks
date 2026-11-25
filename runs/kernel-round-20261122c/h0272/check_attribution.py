#!/usr/bin/env python3
"""The harness's attribution check (BRIEF-0272: "known injected delays (producer-late on rank 0, proxy-late on
rank 1) showing that it attributes them correctly"): reads summarize.py --json on stdin and asserts each injected
delay landed in exactly its (metric, rank) bucket, with no false attribution elsewhere. The expected values are
gen_trace.py's constants read from its source (not imported: the check states its own expectations)."""
import json, sys

s = json.load(sys.stdin)
g = s["groups"]
fails = []


def want(what, cond):
    print(f"  {'ok' if cond else 'FAIL'}: {what}")
    if not cond:
        fails.append(what)


def q(key, metric):
    return g.get(key, {}).get(metric, {})


# producer-late on rank 0: the injected +50us lands in (attn_gather, rank 0) and nowhere else
p0 = q("attn_gather|0", "producer")
p1 = q("attn_gather|1", "producer")
want(f"rank 0 attn_gather producer p95 is the injected one ({p0.get('p95')} >= 45000 ns)",
     p0.get("p95") is not None and p0["p95"] >= 45000)
want(f"rank 1 attn_gather producer p95 stays baseline ({p1.get('p95')} < 8000 ns)",
     p1.get("p95") is not None and p1["p95"] < 8000)
for op in ("moe_gather", "kv", "engram", "head"):
    r0 = q(f"{op}|0", "producer")
    want(f"rank 0 {op} producer p95 stays baseline ({r0.get('p95')} < 8000 ns)",
         r0.get("p95") is not None and r0["p95"] < 8000)

# proxy-late on rank 1: the injected +30us lands in its proxy doorbell->last-post on every op, rank 0 stays baseline
for op in ("attn_gather", "moe_gather", "kv", "engram", "head"):
    a = q(f"{op}|1", "proxy_doorbell_post")
    b = q(f"{op}|0", "proxy_doorbell_post")
    want(f"rank 1 {op} proxy doorbell->last-post p95 is the injected one ({a.get('p95')} >= 30000 ns)",
         a.get("p95") is not None and a["p95"] >= 30000)
    want(f"rank 0 {op} proxy doorbell->last-post p95 stays baseline ({b.get('p95')} < 5000 ns)",
         b.get("p95") is not None and b["p95"] < 5000)

# no false attribution in the untouched metrics: copy and doorbell->flag are baseline on both ranks
for rank in ("0", "1"):
    c = q(f"attn_gather|{rank}", "copy")
    d = q(f"attn_gather|{rank}", "doorbell_flag")
    want(f"rank {rank} copy p95 stays baseline ({c.get('p95')})", c.get("p95") is not None and 2000 <= c["p95"] <= 5000)
    want(f"rank {rank} doorbell->flag p95 stays baseline ({d.get('p95')})",
         d.get("p95") is not None and 14000 <= d["p95"] <= 28000)

# drops and sample counts survive the summary
t0, t1 = s["ranks"]["0"], s["ranks"]["1"]
want(f"rank 0 reports 400 gpu samples, no drops ({t0})", t0["gpu_samples"] == 400 and t0["gpu_dropped"] == 0)
want(f"rank 1 reports its drops ({t1})", t1["gpu_samples"] == 400 and t1["gpu_dropped"] == 3
     and t1["proxy_dropped"] == 2)

print("check_attribution: FAIL" if fails else "check_attribution: ok")
sys.exit(1 if fails else 0)
