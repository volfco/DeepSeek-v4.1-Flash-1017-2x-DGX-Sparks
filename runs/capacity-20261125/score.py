#!/usr/bin/env python3
"""Score the capacity arms: aggregate decode throughput and the wait, which are the numbers a capacity arm is about.

    python3 runs/capacity-20261125/score.py runs/capacity-20261125/s8 [runs/capacity-20261125/s16 ...]

Per arm it prints the workload cells (sum completion tokens / wall = aggregate tok/s, per stream), the per-request
records from the `done` lines (ttft_s, queue_s, rounds, accepted/drafted, tokens_per_round) and the exact/sampled
gate results. Round latency is printed but must not be read as the result: a wider round is longer by construction.
"""
import glob
import json
import os
import re
import statistics
import sys

KV = re.compile(r"(\w+)=([^\s]+)")


def done_records(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        i = line.find("done chatcmpl-")
        if i < 0:
            continue
        kv = dict(KV.findall(line[i:]))
        rec = {}
        for k in ("ttft_s", "queue_s", "prefill_s", "decode_s", "tok_per_s", "tokens_per_round", "restore_s"):
            if k in kv:
                try:
                    rec[k] = float(kv[k].rstrip("%"))
                except ValueError:
                    pass
        for k in ("rounds", "tokens"):
            if k in kv:
                try:
                    rec[k] = int(kv[k])
                except ValueError:
                    pass
        if "accepted" in kv and "/" in kv["accepted"]:
            a, b = kv["accepted"].split("/", 1)
            try:
                rec["accepted"], rec["drafted"] = int(a), int(b)
            except ValueError:
                pass
        if "finish" in kv:
            rec["finish"] = kv["finish"]
        out.append(rec)
    return out


def stat(vals, f=statistics.median):
    return f(vals) if vals else None


def main(dirs):
    for d in dirs:
        print(f"\n## {d}")
        arms = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.json"))
                      if not p.endswith((".exact.json", ".sampled.json", ".probe.json", ".validate.json")))
        print("\n| arm | exact | sampled | cell (depth/streams) | wall s | aggregate tok/s | a stream tok/s "
              "| req tok/s med | ttft med s | ttft max s | queue med s | tokens/round med | accept% |")
        print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in arms:
            gate = []
            for kind, name in (("exact", "cmp"), ("sampled", "sampled")):
                f = os.path.join(d, f"{a}.{kind}.json")
                g = "—"
                if os.path.exists(f):
                    g = "ok" if kind == "sampled" else "recorded"
                c = os.path.join(d, f"{a}.cmp.txt")
                if kind == "exact" and os.path.exists(c):
                    g = "IDENTICAL" if "IDENTICAL" in open(c, errors="replace").read() else "DIFFERS"
                if kind == "sampled":
                    g = "see log"
                gate.append(g)
            try:
                w = json.load(open(os.path.join(d, f"{a}.json")))
            except (OSError, ValueError):
                w = {"cells": []}
            recs = done_records(os.path.join(d, f"{a}.done.txt"))
            tok_s = [r["tok_per_s"] for r in recs if "tok_per_s" in r]
            ttft = [r["ttft_s"] for r in recs if "ttft_s" in r]
            q = [r["queue_s"] for r in recs if "queue_s" in r]
            tpr = [r["tokens_per_round"] for r in recs if "tokens_per_round" in r]
            acc = [100.0 * r["accepted"] / r["drafted"] for r in recs
                   if r.get("drafted") and "accepted" in r]
            for cell in w.get("cells", []):
                gen = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in cell.get("requests", []))
                agg = gen / cell["wall_s"] if cell.get("wall_s") else 0.0
                print(f"| {a} | {gate[0]} | {gate[1]} | {cell['depth']}/{cell['streams']} | {cell['wall_s']:.1f} "
                      f"| {agg:.1f} | {agg / max(1, cell['streams']):.1f} | {stat(tok_s) or 0:.1f} "
                      f"| {stat(ttft) or 0:.2f} | {max(ttft) if ttft else 0:.2f} | {stat(q) or 0:.3f} "
                      f"| {stat(tpr) or 0:.1f} | {stat(acc) or 0:.0f} |")
            if not w.get("cells"):
                print(f"| {a} | {gate[0]} | {gate[1]} | (no workload) | | | | {stat(tok_s) or 0:.1f} "
                      f"| {stat(ttft) or 0:.2f} | {max(ttft) if ttft else 0:.2f} | {stat(q) or 0:.3f} "
                      f"| {stat(tpr) or 0:.1f} | {stat(acc) or 0:.0f} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/capacity-20261125/s8", "runs/capacity-20261125/s16"])
