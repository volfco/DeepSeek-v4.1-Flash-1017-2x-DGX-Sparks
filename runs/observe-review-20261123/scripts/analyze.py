#!/usr/bin/env python3
"""Classify every scrape interval (benchmark / deploy check / unattributed) and build per-interval request records.

    python3 -I scripts/analyze.py     # reads data/scrapes.json, data/windows.json; writes data/records.json, data/epochs.json

A record is the difference between two consecutive scrapes of one engine lifetime in which at least one request
finished. n = requests finished. With n == 1 every field is that request's own value (measured); with n > 1 the
fields are exact totals over the n requests. Concurrency and overlap with prompt passes come from the 30 s gauge
samples around the record, so they are inferred, not measured.
"""
import bisect
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = 524288  # the served context window (README: CONTEXT=524288); pool ratio x CTX = tokens a live stream holds
K = ["lat_n", "lat_s", "ttft_n", "ttft_s", "dec_n", "dec_s", "pre_n", "pre_s", "tpot_n", "tpot_s", "prompt", "cached",
     "gen", "drafted", "accepted", "rounds", "disc", "preempt", "pc_mem_hits", "pc_mem_miss", "pc_disk_hits",
     "pc_disk_committed", "pc_restores_begun", "pc_restored_tokens", "pc_restore_wait_ns", "pc_restore_waits",
     "pc_kept", "pc_evicted", "pc_miss_no_prefix", "pc_miss_stale", "pc_miss_tail", "chats_continued",
     "chats_continued_tokens", "eng_n", "eng_s", "pair_failed", "service_refused"]


def main():
    S = json.load(open(os.path.join(ROOT, "data/scrapes.json")))
    W = json.load(open(os.path.join(ROOT, "data/windows.json")))
    wins = []
    for w in W["windows"]:
        if w["kind"] == "deploy":
            wins.append((w["end"] - 360e6, w["end"] + 60e6, w["name"], w["kind"]))
        elif w["kind"] == "suite":
            wins.append((w["start"] - 60e6, w["end"] + 180e6, w["name"], w["kind"]))
        else:
            wins.append((w["start"], w["end"] + 30e6, w["name"], w["kind"]))
    epochs = W["epochs"]
    estarts = [e["start"] for e in epochs]

    def window_of(t):
        for a, b, name, kind in wins:
            if a <= t <= b:
                return name, kind
        return None, None

    ts = [s["t"] for s in S]
    recs, prev = [], None
    life_first = {}
    for i, s in enumerate(S):
        first = prev is None or prev["life"] != s["life"]
        base = {} if first else prev
        d = {k: s.get(k, 0) - base.get(k, 0) for k in K}
        if first:
            life_first[s["life"]] = i
        s["_i"] = i
        if d["lat_n"] > 0:
            name, kind = window_of(s["t"])
            r = {"t": s["t"], "t_prev": (S[i - 1]["t"] if i else s["t"]) if not first else None, "life": s["life"],
                 "first_of_life": first, "n": int(d["lat_n"]), "cls": kind or "unattributed", "window": name,
                 "epoch": bisect.bisect_right(estarts, s["t"]) - 1, "i": i}
            r.update({k: d[k] for k in K if k != "lat_n"})
            recs.append(r)
        prev = s
    # validate signature at the head of a lifetime: 23 (or 24) requests, 22,9xx prompt tokens, ~1,31x generated
    by_life = {}
    for r in recs:
        by_life.setdefault(r["life"], []).append(r)
    for life, rs in by_life.items():
        n = p = 0
        for j, r in enumerate(rs):
            n += r["n"]
            p += r["prompt"]
            if n in (23, 24) and 22900 <= p <= 23000:
                for q in rs[:j + 1]:
                    if q["cls"] == "unattributed":
                        q["cls"], q["window"] = "validate", "validate signature (23 requests, 22,9xx prompt tokens)"
                break
            if n > 24:
                break
    # a lifetime shorter than 15 minutes (not the current one): the engine was restarted around it, which is
    # the operator or a harness, never a client; its otherwise unattributed requests are set aside
    last_life = S[-1]["life"]
    span = {}
    for s in S:
        a = span.setdefault(s["life"], [s["t"], s["t"]])
        a[1] = s["t"]
    for r in recs:
        if r["cls"] == "unattributed" and r["life"] != last_life and span[r["life"]][1] - span[r["life"]][0] < 900e6:
            r["cls"], r["window"] = "short-lifetime", "engine lifetime under 15 minutes (harness, inferred)"
    # context from the gauges around each record
    for r in recs:
        i = r["i"]
        span = r["lat_s"] / r["n"] if r["n"] else 0  # mean latency (exact for n == 1)
        t0 = r["t"] - (span + 30) * 1e6
        j = bisect.bisect_left(ts, t0)
        samp = [S[x] for x in range(j, i) if S[x]["life"] == r["life"]]  # scrapes during the request(s), before the finish
        runs = [x.get("running", 0) for x in samp] or [S[i - 1].get("running", 0) if i and S[i - 1]["life"] == r["life"] else 0]
        waits = [x.get("waiting", 0) for x in samp] or [0]
        r["run_min"], r["run_max"], r["wait_max"] = min(runs), max(runs), max(waits)
        r["run_mean"] = sum(runs) / len(runs)
        r["n_samp"] = len(samp)
        pools = [max([v * CTX for v in x.get("pools", {}).values()] or [0]) for x in samp]
        r["pool_max"] = max(pools) if pools else 0
        r["span_s"] = span
    json.dump(recs, open(os.path.join(ROOT, "data/records.json"), "w"))
    # epoch inventory
    out = []
    for e, ep in enumerate(epochs):
        a = ep["start"]
        b = epochs[e + 1]["start"] if e + 1 < len(epochs) else S[-1]["t"]
        sc = [s for s in S if a <= s["t"] < b]
        rr = [r for r in recs if a <= r["t"] < b]
        row = {"epoch": e, "label": ep["label"], "start": a, "end": b, "wall_h": (b - a) / 3600e6,
               "scrapes": len(sc), "up_h": len(sc) * 30 / 3600, "lifetimes": len(set(s["life"] for s in sc)),
               "run_max": max([s.get("running", 0) for s in sc] or [0]), "wait_max": max([s.get("waiting", 0) for s in sc] or [0])}
        for cls in ("suite", "deploy", "probe-on-production", "validate", "short-lifetime", "unattributed"):
            x = [r for r in rr if r["cls"] == cls]
            row[cls] = {"requests": sum(r["n"] for r in x), "records": len(x), "single": sum(1 for r in x if r["n"] == 1),
                        "prompt": sum(r["prompt"] for r in x), "cached": sum(r["cached"] for r in x),
                        "gen": sum(r["gen"] for r in x), "dec_s": sum(r["dec_s"] for r in x), "rounds": sum(r["rounds"] for r in x)}
        busy = [s for s in sc if s.get("running", 0) > 0 and window_of(s["t"])[1] is None]
        row["unattributed_busy_h"] = len(busy) * 30 / 3600
        out.append(row)
    json.dump(out, open(os.path.join(ROOT, "data/epochs.json"), "w"), indent=1)
    import datetime as dt
    f = lambda u: dt.datetime.fromtimestamp(u / 1e6, dt.timezone.utc).strftime("%m-%d %H:%M")
    print("epoch            window(UTC)        wall_h up_h lifetimes | suite+deploy+probe+validate+short-lifetime req | unattributed: req (single-req records) prompt_tok cached% gen_tok busy_h | max running/waiting")
    for r in out:
        b = sum(r[c]["requests"] for c in ("suite", "deploy", "probe-on-production", "validate", "short-lifetime"))
        u = r["unattributed"]
        print(f'E{r["epoch"]} {f(r["start"])}..{f(r["end"])} {r["wall_h"]:6.1f} {r["up_h"]:5.1f} {r["lifetimes"]:4d} | {b:6d} | '
              f'{u["requests"]:6d} ({u["single"]:5d}) {int(u["prompt"]):>13,d} {100 * u["cached"] / max(1, u["prompt"]):5.1f}% {int(u["gen"]):>10,d} '
              f'{r["unattributed_busy_h"]:5.1f} | {int(r["run_max"])}/{int(r["wait_max"])}')


if __name__ == "__main__":
    main()
