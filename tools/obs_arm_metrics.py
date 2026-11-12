#!/usr/bin/env python3
"""Per-arm metric deltas from the lab's OpenObserve (Prometheus-compatible) store, for a bench_matrix run.

    OBS_USER=... OBS_PASS=... python3 -I tools/obs_arm_metrics.py runs/matrix/status.log [start end]

Arm windows come from status.log ("arm X start" -> "X DONE"); counter deltas are reset-aware (each arm's
engine restart zeroes the counters). Series of interest are listed in SERIES below. start/end (ISO UTC) bound
the log lines to one pass of the matrix (the log is cumulative across passes). Stdlib only.
"""
import json
import os
import re
import sys
import urllib.request
import base64
import datetime as dt

BASE = os.environ.get("OBS_URL", "https://openobserve1.lab.volf.co")
ORG = os.environ.get("OBS_ORG", "default")
SERIES = [
    "tensorfold:engram_lookup_seconds_count",
    "tensorfold:engram_lookup_seconds_sum",
    "tensorfold:pair_broken",
    "tensorfold:pair_failed",
    "tensorfold:mtp_accepted_total",
    "tensorfold:mtp_drafted_total",
    "tensorfold:request_prefill_seconds_sum",
    "tensorfold:request_prefill_seconds_count",
]


def api(path, **params):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    cred = base64.b64encode(f"{os.environ['OBS_USER']}:{os.environ['OBS_PASS']}".encode()).decode()
    req.add_header("Authorization", "Basic " + cred)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def series_points(name, start, end, step=30):
    d = api(f"/api/{ORG}/prometheus/api/v1/query_range", query=name,
            start=str(start), end=str(end), step=str(step))
    pts = []
    for r in d.get("data", {}).get("result", []):
        for t, v in r["values"]:
            pts.append((int(t), float(v)))
    return sorted(pts)


def delta(pts, w0, w1):
    """Reset-aware counter delta over [w0, w1]. A segment after a reset counts from 0 (the engine's counters
    restart with the container); the window's first segment is cut by the window start and counts its tail only."""
    total, prev, baseline = 0.0, None, None
    for t, v in pts:
        if not (w0 <= t <= w1):
            continue
        if prev is not None and v < prev:
            if baseline is not None:
                total += prev - baseline
            baseline = 0.0
        elif baseline is None:
            baseline = v
        prev = v
    if baseline is not None and prev is not None:
        total += prev - baseline
    return total


def arm_windows(path, start_iso=None, end_iso=None):
    arms, cur = [], None
    for line in open(path):
        m = re.match(r"(\S+)Z arm (\S+) start", line)
        if m:
            cur = (m.group(2), dt.datetime.fromisoformat(m.group(1)))
            continue
        m = re.match(r"(\S+)Z (\S+) DONE ", line)
        if m and cur and cur[0] == m.group(2):
            t1 = dt.datetime.fromisoformat(m.group(1))
            if (not start_iso or cur[1] >= dt.datetime.fromisoformat(start_iso)) and \
               (not end_iso or t1 <= dt.datetime.fromisoformat(end_iso)):
                arms.append((cur[0], cur[1], t1))
            cur = None
    return arms


def main():
    path = sys.argv[1]
    start_iso = sys.argv[2] if len(sys.argv) > 2 else None
    end_iso = sys.argv[3] if len(sys.argv) > 3 else None
    arms = arm_windows(path, start_iso, end_iso)
    if not arms:
        sys.exit("no arm windows matched")
    w0 = min(a[1] for a in arms)
    w1 = max(a[2] for a in arms)
    epoch0 = int(w0.replace(tzinfo=dt.timezone.utc).timestamp())
    epoch1 = int(w1.replace(tzinfo=dt.timezone.utc).timestamp())
    data = {s: series_points(s, epoch0, epoch1) for s in SERIES}

    hdr = ["arm", "window_utc", "engram_reads", "engram_read_s", "pair_broken", "pair_failed",
           "mtp_acc", "mtp_draft", "accept%", "prefill_s"]
    print(" | ".join(hdr))
    print(" | ".join("---" for _ in hdr))
    for name, t0, t1 in arms:
        e0 = int(t0.replace(tzinfo=dt.timezone.utc).timestamp())
        e1 = int(t1.replace(tzinfo=dt.timezone.utc).timestamp())
        d = {s.split(":")[-1]: delta(data[s], e0, e1) for s in SERIES}
        acc = 100 * d["mtp_accepted_total"] / d["mtp_drafted_total"] if d["mtp_drafted_total"] else None
        print(" | ".join([
            name, f"{t0:%H:%M:%S}-{t1:%H:%M:%S}",
            f"{d['engram_lookup_seconds_count']:.0f}", f"{d['engram_lookup_seconds_sum']:.1f}",
            f"{d['pair_broken']:.0f}", f"{d['pair_failed']:.0f}",
            f"{d['mtp_accepted_total']:.0f}", f"{d['mtp_drafted_total']:.0f}",
            f"{acc:.1f}" if acc is not None else "-", f"{d['request_prefill_seconds_sum']:.1f}",
        ]))


if __name__ == "__main__":
    main()
