#!/usr/bin/env python3
"""4-stream per-step cost inside the window where all four streams decode (no staggered prefill, no tail), from
/metrics counters polled at 5 Hz: decode_rounds_total and mtp_drafted_total. Validates the counters on a 1-stream
cell first (delta must equal the done line's rounds/drafted). 3 reps at 2K and 3 at 300K.
  python3 -I runs/depth/fourwin.py
"""
import json
import os
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth as D  # noqa: E402

# rebuild the rung texts exactly as four.py does
primes = {c["depth_target"]: c for c in D.DOC["cells"] if c["kind"] == "prime"}
texts = {}
for d in [D.SHORT] + D.LADDER:
    if d not in primes:
        break
    h = D.hay(d)
    texts[d] = h
    D.learn(D.SALT + h, primes[d]["attempts"][-1]["results"][0])

SAMPLES = []
STOP = threading.Event()


def poll():
    while not STOP.is_set():
        m = D.srv.metrics()
        if "_error" not in m:
            SAMPLES.append((time.monotonic(), m.get("tensorfold:decode_rounds_total"), m.get("tensorfold:mtp_drafted_total"),
                            m.get("tensorfold:requests_running")))
        time.sleep(0.2)


def window(t0, t1):
    s = [x for x in SAMPLES if t0 <= x[0] <= t1]
    if len(s) < 2:
        return None
    a, b = s[0], s[-1]
    return {"dt": b[0] - a[0], "d_rounds": b[1] - a[1], "d_drafted": b[2] - a[2], "n_samples": len(s)}


def full_delta(t0, t1):
    pre = [x for x in SAMPLES if x[0] <= t0]
    post = [x for x in SAMPLES if x[0] >= t1]
    if not pre or not post:
        return None
    return {"d_rounds": post[0][1] - pre[-1][1], "d_drafted": post[0][2] - pre[-1][2]}


th = threading.Thread(target=poll, daemon=True)
th.start()
out = {"cells": []}
D.log("fourwin: start")
# 1-stream validation at 2K
t0 = time.monotonic()
rec = D.cell("win-val-s1", D.SHORT, [D.SALT + texts[D.SHORT] + D.task("win-val")], 512, True, "winval")
time.sleep(1.0)
r = rec["attempts"][-1]["results"][0]
fd = full_delta(t0 + 0.0, time.monotonic())
out["validation"] = {"done_rounds": r.get("rounds"), "done_drafted": r.get("drafted"), "counter": fd}
D.log(f"fourwin: validation done rounds={r.get('rounds')} drafted={r.get('drafted')} counter={fd}")
for d in (D.SHORT, 300_000):
    if d != D.SHORT:
        D.cell(f"win-ref-{d}", d, [D.SALT + texts[d] + D.task(f"win-ref-{d}")], 64, True, "ref1")
    for rep in range(3):
        rec = D.cell(f"win-d{d}-s4-r{rep}", d, [D.SALT + texts[d] + D.task(f"win-d{d}-s4-r{rep}-{s}") for s in range(4)],
                     D.DEC_TOKENS, True, "win4")
        rs = rec["attempts"][-1]["results"]
        # results carry time.monotonic() stamps (bench_div chat): all four decoding in [max t_first, min t_last]
        w0 = max(x["t_first"] for x in rs) + 0.3
        w1 = min(x["t_last"] for x in rs) - 0.3
        w = window(w0, w1)
        row = {"depth": d, "rep": rep, "prompt": rs[0].get("prompt_tokens"), "cached": [x.get("cached_tokens") for x in rs],
               "window_s": w1 - w0, "w": w}
        out["cells"].append(row)
        D.log(f"fourwin: {d} rep{rep} window {w1 - w0:.1f}s {w}")
        with open(f"{D.OUT}/fourwin.json", "w") as fh:
            json.dump(out, fh, indent=1)
STOP.set()
D.log("fourwin: end")
