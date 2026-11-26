#!/usr/bin/env python3
"""Every OpenObserve pull behind REPORT.md. Credentials come from the environment, never a file:

    . runs/observe-review-20261123/scripts/o2env.sh
    python3 -I runs/observe-20261126/scripts/pulls.py OUTDIR

Writes OUTDIR/engine_logs.json (holds the client address: do not commit), OUTDIR/gauges.json, and prints the host /
counter / epoch / restore tables. Then run parse.py, score.py, churn.py, overlap.py, deeper.py, fixes.py in OUTDIR.
"""
import datetime as dt
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../observe-review-20261123/scripts"))
import o2  # noqa: E402

OUT = sys.argv[1]
LIFE = ("2026-11-25T19:29:00Z", "2026-11-26T04:25:00Z")
H = 'host_name="gx10-eb0e"'


def s(iso):
    return o2.iso_us(iso) / 1e6


def stamp(t):
    return dt.datetime.fromtimestamp(float(t), dt.timezone.utc).strftime("%H%M")


# 1. engine log lines (done, access, warnings, banners), shipped since runs/observe-20261125
r = o2.sql("logs", "SELECT _timestamp, host_name, body FROM \"default\" WHERE body LIKE '%[tensorfold]%' ORDER BY _timestamp",
           o2.iso_us("2026-11-25T04:40:00Z"), o2.iso_us(LIFE[1]), size=200000)
json.dump(r["hits"], open(os.path.join(OUT, "engine_logs.json"), "w"))

# 2. streams running / waiting at 30 s
g = {}
for e in ["tensorfold:requests_running", "tensorfold:requests_waiting"]:
    res = o2.prom_range(e, s(LIFE[0]), s(LIFE[1]), step=30)["data"]["result"]
    g[e] = res[0]["values"] if res else []
json.dump(g, open(os.path.join(OUT, "gauges.json"), "w"))

# 3. GPU / host over the lifetime (15-min step)
HOSTQ = {
    "sm_clock": 'avg by (host_name)(dcgm_fi_dev_sm_clock{host_name=~"gx10.*"})',
    "power": 'avg by (host_name)(dcgm_fi_dev_power_usage{host_name=~"gx10.*"})',
    "temp": 'avg by (host_name)(dcgm_fi_dev_gpu_temp{host_name=~"gx10.*"})',
    "mem_free_cached_buf_GiB": 'sum by (host_name)(system_memory_usage{host_name=~"gx10.*",state=~"free|cached|buffered"})/2^30',
    "major_faults/s": 'sum by (host_name)(rate(system_paging_faults{host_name=~"gx10.*",type="major"}[5m]))',
    "paging_ops/s (file page-in, not swap)": 'sum by (host_name)(rate(system_paging_operations{host_name=~"gx10.*"}[5m]))',
    "swap_used_GiB": 'sum by (host_name)(system_paging_usage{host_name=~"gx10.*",state="used"})/2^30',
    "disk_read_MB/s": 'sum by (host_name)(rate(system_disk_io{host_name=~"gx10.*",direction="read"}[5m]))/1e6',
}
# 4. engine counters over the lifetime (20-min step)
ENGQ = {
    "uncached_tok/s": "rate(tensorfold:prompt_tokens_total[10m])-rate(tensorfold:prompt_tokens_cached_total[10m])",
    "spill_read_MB/s": "rate(tensorfold:spill_read_bytes[10m])/1e6",
    "pc_evicted/min": "60*rate(tensorfold:prompt_cache_evicted[10m])",
    "pc_entries": "tensorfold:prompt_cache_entries",
    "pc_held_GiB": "tensorfold:prompt_cache_held_bytes/2^30",
    "pc_disk_hits/min": "60*rate(tensorfold:prompt_cache_disk_hits[10m])",
    "engram_ms/lookup": "1000*rate(tensorfold:engram_lookup_seconds_sum[10m])/rate(tensorfold:engram_lookup_seconds_count[10m])",
    "page_cache_GiB": "sum(system_memory_usage{" + H + ',state="cached"})/2^30',
}
for title, Q, a, b, step in [("host", HOSTQ, LIFE[0], LIFE[1], 900), ("engine", ENGQ, "2026-11-26T01:00:00Z", "2026-11-26T04:20:00Z", 1200)]:
    print(f"### {title}")
    for k, e in Q.items():
        for sr in o2.prom_range(e, s(a), s(b), step=step)["data"]["result"]:
            print(f"{k:40s} {sr['metric'].get('host_name', '')[-4:]} " + " ".join(f"{stamp(t)}:{float(v):.3g}" for t, v in sr["values"]))

# 5. disk MB per uncached prefill token, by epoch (Engram page-ins vs workload)
print("### epochs")
EPQ = {"unc": ENGQ["uncached_tok/s"], "disk": "sum(rate(system_disk_io{" + H + ',direction="read"}[10m]))/1e6',
       "eng": ENGQ["engram_ms/lookup"]}
for lab, a, b in [("E7 2bb052f", "2026-11-23T05:30:00Z", "2026-11-23T07:10:00Z"), ("E2 fused", "2026-11-21T08:00:00Z", "2026-11-22T05:00:00Z"),
                  ("E1 ea52", "2026-11-19T00:00:00Z", "2026-11-21T07:00:00Z"), ("now 1421187", "2026-11-26T01:00:00Z", "2026-11-26T04:20:00Z")]:
    D = {}
    for k, e in EPQ.items():
        res = o2.prom_range(e, s(a), s(b), step=600)["data"]["result"]
        D[k] = {t: float(v) for t, v in res[0]["values"]} if res else {}
    rows = [(D["unc"][t], D["disk"][t], D["eng"].get(t)) for t in D["unc"] if D["unc"][t] > 100 and t in D["disk"]]
    eng = [x for *_, x in rows if x is not None and x == x]
    print(f"{lab:12s} buckets {len(rows)} disk MB/uncached tok med {st.median(d / u for u, d, _ in rows):.3f} "
          f"engram ms/lookup med {st.median(eng) if eng else float('nan'):.1f}")

# 6. restore counters over the saturated window
print("### restore counters 03:10-04:20 (delta)")
a, b = s("2026-11-26T03:10:00Z"), s("2026-11-26T04:20:00Z")
for n in ["restores_begun", "restore_waits", "restore_wait_ns", "restores_deferred", "restores_room_waits", "restores_refused",
          "restores_shared", "restored_tokens", "disk_hits", "evicted_lru", "evicted_superseded", "held_bytes", "entries"]:
    res = o2.prom_range(f"tensorfold:prompt_cache_{n}", a, b, step=4200)["data"]["result"]
    if res:
        v = [float(x[1]) for x in res[0]["values"]]
        print(f"{n:22s} start {v[0]:.6g} end {v[-1]:.6g} delta {v[-1] - v[0]:.6g}")
