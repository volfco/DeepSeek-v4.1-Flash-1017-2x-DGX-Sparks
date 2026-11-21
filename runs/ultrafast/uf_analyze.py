#!/usr/bin/env python3
"""Summarise runs/ultrafast/full: bench medians per arm, the resumed 100K turn's done-line timings, the Tetris row,
sampled tok/s. usage: uf_analyze.py DIR"""
import glob, json, os, re, statistics as st, sys

D = sys.argv[1]
ARMS = ["x-base", "x-tok", "x-k0", "x-rbase", "x-ron"]


def done_lines(arm_pass):
    out = {}
    try:
        for line in open(os.path.join(D, arm_pass + ".done.log")):
            m = re.search(r"done (\S+) (.*)", line)
            if m:
                out[m.group(1)] = dict(re.findall(r"(\w+)=([^\s]+)", m.group(2)))
    except FileNotFoundError:
        pass
    return out


def num(x):
    try:
        return float(str(x).rstrip("%"))
    except Exception:
        return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return (st.median(xs), min(xs), max(xs), len(xs)) if xs else None


def fmt(m, nd=1):
    return "n/a" if not m else f"{m[0]:.{nd}f} [{m[1]:.{nd}f}-{m[2]:.{nd}f}] n={m[3]}"


passes = sorted({int(p.split("-")[-1].split(".")[0]) for p in glob.glob(os.path.join(D, "x-*-[0-9].restart.log"))})
print("passes", passes)
print("\n## bench (median [min-max] over passes)")
keys = [("prefill.2000.tok_s", "p2k"), ("prefill.8000.tok_s", "p8k"), ("prefill.32000.tok_s", "p32k"),
        ("decode.tok_s", "decode"), ("multi.1.aggregate_tok_s", "multi1"), ("multi.4.aggregate_tok_s", "multi4")]
for arm in ARMS:
    vals = {l: [] for _, l in keys}
    for p in passes:
        try:
            s = json.load(open(os.path.join(D, f"{arm}-{p}.bench.json")))["summary"]
        except Exception:
            continue
        for k, l in keys:
            vals[l].append(s.get(k))
    if any(vals.values()):
        print(arm, " | ".join(f"{l} {fmt(med(v))}" for l, v in vals.items()))

print("\n## longctx 100K (done lines): cold and resumed turn")
for arm in ARMS:
    rows = {"cold": [], "resume": []}
    for p in passes:
        try:
            lc = json.load(open(os.path.join(D, f"{arm}-{p}.longctx.json")))["rows"]
        except Exception:
            continue
        dl = done_lines(f"{arm}-{p}")
        for r in lc:
            d = dl.get(r["id"])
            if d:
                rows[r["turn"]].append(d)
    for turn, ds in rows.items():
        if ds:
            print(arm, turn, " | ".join(f"{k} {fmt(med([num(d.get(k)) for d in ds]), 3)}"
                                        for k in ("render_s", "encode_s", "prefill_s", "ttft_s", "cached")))

print("\n## tetris (done lines)")
for arm in ARMS:
    for mode in ("greedy", "sampled"):
        ds = []
        for p in passes:
            try:
                tr = json.load(open(os.path.join(D, f"{arm}-{p}.tetris.json")))["rows"]
            except Exception:
                continue
            dl = done_lines(f"{arm}-{p}")
            ds += [dl[r["id"]] for r in tr if r["mode"] == mode and r["id"] in dl]
        if ds:
            print(arm, mode, " | ".join(f"{k} {fmt(med([num(d.get(k)) for d in ds]), 2)}"
                                        for k in ("tokens", "tokens_per_round", "accept", "tok_per_s")))

print("\n## sampled probe agg tok/s")
for arm in ARMS:
    cells = {}
    for p in passes:
        try:
            rows = json.load(open(os.path.join(D, f"{arm}-{p}.sampled.json")))["rows"]
        except Exception:
            continue
        for r in rows:
            cells.setdefault((r["mode"], r["streams"]), []).append(r["agg_tok_s"])
    if cells:
        print(arm, " | ".join(f"{m}x{s} {fmt(med(v))}" for (m, s), v in sorted(cells.items())))

print("\n## prompt_reuse (last lines)")
for f in sorted(glob.glob(os.path.join(D, "x-r*-[0-9].reuse.log"))):
    lines = open(f).read().strip().splitlines()
    print(os.path.basename(f), "|", " / ".join(lines[-2:])[:300])
