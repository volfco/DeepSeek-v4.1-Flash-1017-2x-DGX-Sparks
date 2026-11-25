#!/usr/bin/env python3
"""Join the pulled series on their scrape timestamp and cut the result into engine lifetimes.

    python3 -I scripts/table.py            # reads data/raw/*.json, writes data/scrapes.json and data/lifetimes.json

All `tensorfold:*` samples of one scrape carry the same timestamp, so one row a scrape is exact. The counters of
finished requests (tokens, rounds, drafts and the five histograms' _sum/_count) are updated together when a request
finishes and read under one lock at scrape time (server/metrics.zig `note` / `render`), so the difference between
two consecutive scrapes is the exact total over the requests that finished in between; where the request count
rises by one, it is that request's own values.

A lifetime ends when any monotonic counter falls (the engine restarted: its counters start from zero) or when the
collector's `up` for the head's tensorfold job reads 0 / scrapes are missing and a counter fell across the gap.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/raw")

# short name -> stream (scalar, no extra label)
SCALARS = {
    "lat_n": "request_latency_seconds_count", "lat_s": "request_latency_seconds_sum",
    "ttft_n": "time_to_first_token_seconds_count", "ttft_s": "time_to_first_token_seconds_sum",
    "dec_n": "request_decode_seconds_count", "dec_s": "request_decode_seconds_sum",
    "pre_n": "request_prefill_seconds_count", "pre_s": "request_prefill_seconds_sum",
    "tpot_n": "request_time_per_output_token_seconds_count", "tpot_s": "request_time_per_output_token_seconds_sum",
    "prompt": "prompt_tokens_total", "cached": "prompt_tokens_cached_total", "gen": "generation_tokens_total",
    "drafted": "mtp_drafted_total", "accepted": "mtp_accepted_total", "rounds": "decode_rounds_total",
    "disc": "client_disconnections_total", "preempt": "preemptions_total",
    "running": "requests_running", "waiting": "requests_waiting", "gen_running": "generation_tokens_running",
    "mem": "device_memory_bytes", "mem_peak": "device_memory_peak_bytes",
    "pc_mem_hits": "prompt_cache_memory_hits", "pc_mem_miss": "prompt_cache_memory_misses",
    "pc_disk_lookups": "prompt_cache_disk_lookups", "pc_disk_hits": "prompt_cache_disk_hits",
    "pc_disk_committed": "prompt_cache_disk_hits_committed", "pc_restores_begun": "prompt_cache_restores_begun",
    "pc_restored_tokens": "prompt_cache_restored_tokens", "pc_cached_tokens": "prompt_cache_cached_tokens",
    "pc_computed_tokens": "prompt_cache_computed_tokens", "pc_kept": "prompt_cache_kept",
    "pc_evicted": "prompt_cache_evicted", "pc_entries": "prompt_cache_entries",
    "pc_restore_wait_ns": "prompt_cache_restore_wait_ns", "pc_restore_waits": "prompt_cache_restore_waits",
    "pc_miss_no_prefix": "prompt_cache_misses_no_prefix", "pc_miss_stale": "prompt_cache_misses_stale",
    "pc_miss_tail": "prompt_cache_misses_tail_window",
    "chats_continued": "chats_continued", "chats_continued_tokens": "chats_continued_prefill_tokens",
    "eng_n": "engram_lookup_seconds_count", "eng_s": "engram_lookup_seconds_sum",
    "pair_failed": "pair_failed", "pair_broken": "pair_broken", "pair_stalled": "pair_stalled",
    "pair_busy_ms": "pair_busy_ms", "service_refused": "service_refused",
    "kv_fetch_rounds": "kv_shard_fetch_rounds", "kv_owner_rounds": "kv_shard_owner_rounds",
}
# counters that only rise within a lifetime: a fall in any of them is a restart
MONO = ["lat_n", "lat_s", "ttft_n", "dec_n", "pre_n", "tpot_n", "prompt", "cached", "gen", "drafted", "accepted",
        "rounds", "disc", "pc_mem_hits", "pc_mem_miss", "pc_kept", "pc_computed_tokens", "pc_cached_tokens",
        "eng_n", "kv_fetch_rounds", "kv_owner_rounds", "chats_continued"]


def load(name):
    fn = os.path.join(RAW, "tensorfold__" + name + ".json")
    if not os.path.exists(fn):
        return None
    return json.load(open(fn))


def main():
    rows = {}
    for short, stream in SCALARS.items():
        d = load(stream)
        if d is None:
            print("missing", stream, file=sys.stderr)
            continue
        for r in d["rows"]:
            rows.setdefault(r[0], {})[short] = r[1]
    # per-stream context occupancy (ratio of the context window), as a list ordered by pool index
    d = load("kv_cache_usage_ratio")
    if d:
        li = d["labels"].index("pool") + 2
        for r in d["rows"]:
            rows.setdefault(r[0], {}).setdefault("pools", {})[str(r[li])] = r[1]
    ts = sorted(rows)
    up = {}
    fn = os.path.join(RAW, "up.json")
    if os.path.exists(fn):
        for t, v, host in json.load(open(fn))["rows"]:
            if host == "gx10-eb0e":
                up[t] = v
    scrapes = []
    for t in ts:
        r = rows[t]
        r["t"] = t
        scrapes.append(r)
    # lifetimes
    lifetimes, cur, prev = [], None, None
    for r in scrapes:
        new = prev is None
        why = "first"
        if prev is not None:
            fell = [k for k in MONO if k in r and k in prev and r[k] < prev[k]]
            gap = (r["t"] - prev["t"]) / 1e6
            if fell:
                new, why = True, "counters fell: " + ",".join(fell[:4])
            elif gap > 45:
                # scrapes missing but nothing fell: the same process only if it had served something before
                # the gap; an idle engine (all zero) that restarts into all zero cannot be told apart, so cut.
                if all(prev.get(k, 0) == 0 for k in ("lat_n", "prompt", "eng_n")):
                    new, why = True, f"gap {gap:.0f}s with zero counters"
                else:
                    r["gap_s"] = gap
        if new:
            cur = {"id": len(lifetimes), "start": r["t"], "why": why, "i0": len(lifetimes) and None}
            cur["prev_end"] = prev["t"] if prev else None
            lifetimes.append(cur)
        r["life"] = cur["id"]
        cur["end"] = r["t"]
        cur["n_scrapes"] = cur.get("n_scrapes", 0) + 1
        for k in ("lat_n", "prompt", "cached", "gen", "rounds", "dec_s", "pre_s", "drafted", "accepted", "pair_failed"):
            if k in r:
                cur[k] = r[k]
        prev = r
    for lf in lifetimes:
        lf.pop("i0", None)
    json.dump({"up": sorted(up.items())}, open(os.path.join(ROOT, "data/up-head.json"), "w"))
    json.dump(scrapes, open(os.path.join(ROOT, "data/scrapes.json"), "w"))
    json.dump(lifetimes, open(os.path.join(ROOT, "data/lifetimes.json"), "w"), indent=0)
    print("scrapes", len(scrapes), "lifetimes", len(lifetimes))


if __name__ == "__main__":
    main()
