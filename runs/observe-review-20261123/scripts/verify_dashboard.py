#!/usr/bin/env python3
"""Run every panel query of a dashboard through the API and report what comes back.

    . scripts/o2env.sh && python3 -I scripts/verify_dashboard.py <dashboard.json> [hours=24] [out.json]

<dashboard.json> is either the API's GET reply ({"v8": {...}, "version": 8, ...}) or a bare definition. PromQL panels
go to the Prometheus-compatible query_range (step 300 s); variables are replaced by their defaults below. A panel is
"data" when a query returns a finite sample, "empty" when every query returns none, "error" when the API refuses one.
"""
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2  # noqa: E402

DEFAULTS = {"host": ".*", "host_name": ".*", "rank": ".*", "part": ".*", "device": ".*", "mountpoint": ".*",
            "series": "tensorfold:requests_running", "doca_series": "rx_bytes_phy", "host_series": "system_cpu_load_average_1m",
            "__interval": "5m", "__rate_interval": "5m", "__range": "24h"}


def subst(qs, variables):
    def rep(m):
        name = m.group(1) or m.group(2)
        return str(variables.get(name, DEFAULTS.get(name, ".*")))
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", rep, qs)


def main():
    path = sys.argv[1]
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
    d = json.load(open(path))
    v = d.get(f"v{d.get('version')}") if "version" in d and f"v{d.get('version')}" in d else d
    variables = {}
    for var in (v.get("variables") or {}).get("list", []):
        val = var.get("value")
        if isinstance(val, list):
            val = "|".join(map(str, val)) if val else None
        if val not in (None, ""):
            variables[var["name"]] = val
    end = time.time()
    start = end - hours * 3600
    rep, counts = [], {"data": 0, "empty": 0, "error": 0, "static": 0}
    for tab in v.get("tabs", []):
        for p in tab.get("panels", []):
            row = {"tab": tab.get("name"), "id": p.get("id"), "title": p.get("title"), "type": p.get("type"), "queries": []}
            if p.get("type") in ("markdown", "html"):
                row["status"] = "static"
                counts["static"] += 1
                rep.append(row)
                continue
            status = "empty"
            for qd in p.get("queries", []):
                expr = subst(qd.get("query") or "", variables)
                r = {"query": qd.get("query"), "sent": expr}
                try:
                    if p.get("queryType") == "promql":
                        res = o2.prom_range(expr, start, end, step=300)
                        ser = res.get("data", {}).get("result", [])
                        pts = sum(1 for s in ser for _, y in s.get("values", []) if y not in ("NaN", "+Inf", "-Inf") and not math.isnan(float(y)))
                        r.update(series=len(ser), points=pts)
                        if pts:
                            status = "data"
                    else:
                        res = o2.sql(qd.get("fields", {}).get("stream_type", "logs"), expr, start * 1e6, end * 1e6, size=1000)
                        r.update(rows=len(res.get("hits", [])))
                        if res.get("hits"):
                            status = "data"
                except Exception as e:
                    r["error"] = str(e)[:300]
                    status = "error" if status != "data" else status
                row["queries"].append(r)
            row["status"] = status
            counts[status] += 1
            rep.append(row)
    out = {"file": path, "title": v.get("title"), "hours": hours, "checked_at": o2.us_iso(end * 1e6), "counts": counts, "panels": rep}
    if len(sys.argv) > 3:
        json.dump(out, open(sys.argv[3], "w"), indent=1)
    print(v.get("title"), "| panels", len(rep), "|", counts)
    for row in rep:
        if row["status"] in ("empty", "error"):
            print(f'  {row["status"].upper()}: [{row["tab"]}] {row["title"]} :: ' + " ;; ".join((q.get("error") or q["sent"])[:150] for q in row["queries"]))


if __name__ == "__main__":
    main()
