#!/usr/bin/env python3
"""Pull raw samples of the engine's series from OpenObserve into data/raw/ (one JSON per stream).

    . scripts/o2env.sh && python3 -I scripts/pull.py START_ISO [END_ISO] [--only regex] [--buckets]

Each file: {"stream":..., "labels":[...], "rows":[[t_us, value, label values...], ...]} sorted by time.
Only samples scraped on the head (host_name='gx10-eb0e', service_name='tensorfold') are kept; the pre-11-10
telegraf-style samples (label `host`) are left out. Pulled a day at a time so no reply is huge.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = ("le", "rank", "pool", "part", "stream", "key", "status")
DAY = 86400 * 10**6


def pull(stream, fields, start, end):
    labs = [l for l in EXTRA if l in fields]
    where = ["host_name='gx10-eb0e'"] if "host_name" in fields else []
    cols = ", ".join(["_timestamp AS t", "value AS v"] + labs)
    rows = []
    t = start
    while t < end:
        t2 = min(end, t + DAY)
        q = f'SELECT {cols} FROM "{stream}"' + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY _timestamp"
        r = o2.sql("metrics", q, t, t2, size=200000, timeout=300)  # raises on a partial reply
        hits = r.get("hits", [])
        for h in hits:
            rows.append([h["t"], h["v"]] + [h.get(l) for l in labs])
        t = t2
    rows.sort(key=lambda x: x[0])
    return labs, rows


def main():
    a = sys.argv[1:]
    only = re.compile(a[a.index("--only") + 1]) if "--only" in a else None
    buckets = "--buckets" in a
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--only")]
    start = o2.iso_us(pos[0])
    end = o2.iso_us(pos[1]) if len(pos) > 1 else int(time.time() * 1e6)
    streams = json.load(open(os.path.join(ROOT, "data/streams-metrics.json")))["list"]
    os.makedirs(os.path.join(ROOT, "data/raw"), exist_ok=True)
    for s in streams:
        n = s["name"]
        if not (n.startswith("tensorfold:") or n in ("up", "scrape_samples_scraped", "scrape_duration_seconds",
                                                     "scrape_samples_post_metric_relabeling", "scrape_series_added")):
            continue
        if n.endswith("_bucket") != buckets:
            continue
        if only and not only.search(n):
            continue
        if not s["stats"]["doc_num"]:
            continue
        fields = set(f["name"] for f in (s.get("schema") or []))
        t0 = time.time()
        if n.startswith("tensorfold:"):
            labs, rows = pull(n, fields, start, end)
        else:  # collector's own series: keep both hosts' tensorfold job
            q = (f'SELECT _timestamp AS t, value AS v, host_name AS host FROM "{n}" '
                 f"WHERE service_name='tensorfold' ORDER BY _timestamp")
            rows, t = [], start
            while t < end:
                t2 = min(end, t + DAY)
                r = o2.sql("metrics", q, t, t2, size=200000, timeout=300)
                rows += [[h["t"], h["v"], h.get("host")] for h in r.get("hits", [])]
                t = t2
            labs = ["host"]
        fn = os.path.join(ROOT, "data/raw", n.replace(":", "__") + ".json")
        first = start
        if "--append" in a and os.path.exists(fn):
            old = json.load(open(fn))
            if old.get("labels") == labs:
                rows = [r for r in old["rows"] if r[0] < start] + rows
                first = old.get("start", start)
        json.dump({"stream": n, "labels": labs, "start": first, "end": end, "rows": rows}, open(fn, "w"))
        print(f"{n} rows={len(rows)} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
