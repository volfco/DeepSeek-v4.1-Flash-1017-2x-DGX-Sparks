#!/usr/bin/env python3
"""Inventory of every metric stream: which hosts send it, how many label sets, first/last sample, interval.

    . scripts/o2env.sh && python3 -I scripts/inventory.py [hours=24]

Reads data/streams-metrics.json (the stream listing saved first) and writes data/inventory-metrics.json.
One aggregate query per stream over the window; streams whose last sample is older use their own last 24 h.
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
NOW = int(time.time() * 1e6)


def one(s):
    name = s["name"]
    fields = set(f["name"] for f in (s.get("schema") or []))
    hk = "host_name" if "host_name" in fields else ("host" if "host" in fields else None)
    sel = [f"{hk} AS h" if hk else "'' AS h"]
    grp = ["h"]
    if "service_name" in fields:
        sel.append("service_name AS svc")
        grp.append("svc")
    st = s["stats"]
    end = NOW
    if st["doc_time_max"] and st["doc_time_max"] < NOW - int(HOURS * 3600e6):
        end = st["doc_time_max"] + 1_000_000
    start = end - int(HOURS * 3600e6)
    q = (f'SELECT {", ".join(sel)}, count(*) AS n, count(DISTINCT __hash__) AS series, min(_timestamp) AS t0, '
         f'max(_timestamp) AS t1 FROM "{name}" GROUP BY {", ".join(grp)}')
    out = {"name": name, "type": (s.get("metrics_meta") or {}).get("metric_type"),
           "labels": sorted(f for f in fields if not f.startswith("_") and f != "value"),
           "doc_num": st["doc_num"], "first": st["doc_time_min"], "last": st["doc_time_max"],
           "window_end": end, "hosts": []}
    if not st["doc_num"]:
        return out
    try:
        r = o2.sql("metrics", q, start, end, timeout=120)
        for h in r.get("hits", []):
            span = (h["t1"] - h["t0"]) / 1e6
            per = h["n"] / max(1, h["series"])
            h["interval_s"] = round(span / (per - 1), 1) if per > 1 else None
            out["hosts"].append(h)
    except Exception as e:  # keep going; the failure is recorded
        out["error"] = str(e)[:300]
    return out


def main():
    streams = json.load(open(os.path.join(ROOT, "data/streams-metrics.json")))["list"]
    res = []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for i, r in enumerate(ex.map(one, streams)):
            res.append(r)
            if i % 400 == 0:
                print(i, len(streams), file=sys.stderr, flush=True)
    json.dump({"generated_us": NOW, "hours": HOURS, "streams": res},
              open(os.path.join(ROOT, "data/inventory-metrics.json"), "w"))
    print("streams", len(res), "errors", sum(1 for r in res if "error" in r))


if __name__ == "__main__":
    main()
