#!/usr/bin/env python3
"""Small OpenObserve client for the 2026-11-23 review. Stdlib only.

Credentials come from the environment (OBS_USER / OBS_PASS); `. scripts/o2env.sh` exports them from the
operator's own helper at run time. Nothing secret is stored in this file.

    . scripts/o2env.sh
    python3 -I scripts/o2.py sql metrics 'SELECT ... FROM "stream" ...' [--hours 24 | --start ISO --end ISO]
    python3 -I scripts/o2.py prom 'expr' [--hours 24] [--step 60]
    python3 -I scripts/o2.py get /api/default/dashboards
"""
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("OBS_URL", "https://openobserve1.lab.volf.co")
ORG = os.environ.get("OBS_ORG", "default")


def _auth():
    cred = base64.b64encode(f"{os.environ['OBS_USER']}:{os.environ['OBS_PASS']}".encode()).decode()
    return "Basic " + cred


def http(method, path, body=None, params=None, timeout=120):
    url = BASE + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", _auth())
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {msg}") from None


def iso_us(s):
    """ISO UTC string -> microseconds since the epoch."""
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1e6)


def us_iso(us):
    return dt.datetime.fromtimestamp(us / 1e6, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sql(stream_type, query, start_us, end_us, size=100000, timeout=180):
    """Run SQL. size=-1 silently caps a reply at 1,000 rows (is_partial), so an explicit size is the default and a
    partial or full-to-the-brim reply raises instead of passing for the whole answer."""
    body = {"query": {"sql": query, "start_time": int(start_us), "end_time": int(end_us), "size": size}}
    r = http("POST", f"/api/{ORG}/_search", body, {"type": stream_type}, timeout=timeout)
    if r.get("is_partial") or (size > 0 and len(r.get("hits", [])) >= size):
        raise RuntimeError(f"partial reply ({len(r.get('hits', []))} rows, size {size}): {query[:120]}")
    return r


def prom_range(expr, start_s, end_s, step=60, timeout=180):
    return http("GET", f"/api/{ORG}/prometheus/api/v1/query_range", None,
                {"query": expr, "start": str(int(start_s)), "end": str(int(end_s)), "step": str(step)}, timeout)


def prom_instant(expr, t_s=None, timeout=120):
    p = {"query": expr}
    if t_s:
        p["time"] = str(int(t_s))
    return http("GET", f"/api/{ORG}/prometheus/api/v1/query", None, p, timeout)


def _window(args):
    now = time.time()
    if "--start" in args:
        s = iso_us(args[args.index("--start") + 1])
        e = iso_us(args[args.index("--end") + 1]) if "--end" in args else int(now * 1e6)
    else:
        h = float(args[args.index("--hours") + 1]) if "--hours" in args else 24.0
        s, e = int((now - h * 3600) * 1e6), int(now * 1e6)
    return s, e


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "sql":
        s, e = _window(a)
        r = sql(a[1], a[2], s, e)
        for h in r.get("hits", []):
            print(json.dumps(h, sort_keys=True))
        print("# total", r.get("total"), "took_ms", r.get("took"), file=sys.stderr)
    elif a[0] == "prom":
        s, e = _window(a)
        step = int(a[a.index("--step") + 1]) if "--step" in a else 60
        r = prom_range(a[1], s / 1e6, e / 1e6, step)
        for res in r.get("data", {}).get("result", []):
            v = res["values"]
            print(json.dumps(res["metric"], sort_keys=True), len(v), v[:2], v[-2:])
        print("# series", len(r.get("data", {}).get("result", [])), file=sys.stderr)
    elif a[0] == "get":
        print(json.dumps(http("GET", a[1]), indent=1))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
