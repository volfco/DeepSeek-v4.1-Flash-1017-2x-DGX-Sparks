#!/usr/bin/env python3
"""Summarize runs/matrix/ (scripts/bench_matrix.sh) as the two to-benchmark.md Results tables.

    python3 -I tools/matrix_report.py [runs/matrix]

Arm order and flags come from the ARMS block in scripts/bench_matrix.sh; numbers come from each arm's
bench.json summary, the engram histogram delta in its metrics snapshots, its done lines and mem.txt.
Stdlib only.
"""
import json
import re
import sys
import statistics
import pathlib

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/matrix")
matrix_sh = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "bench_matrix.sh"


def arms():
    order = []
    text = matrix_sh.read_text()
    for name, flags, thinking in re.findall(r'^\s*"(\S+?)\|([^|]*)\|([01])"', text, re.M):
        order.append((name, flags, thinking))
    return order


def eng_metrics(name):
    """(count delta, sum seconds delta) of tensorfold:engram_lookup_seconds around the block."""
    def side(when):
        p = root / f"{name}.engram-{when}.txt"
        text = p.read_text() if p.exists() else ""
        count = sum_ = None
        for line in text.splitlines():
            m = re.match(r"tensorfold:engram_lookup_seconds_count\s+(\S+)", line)
            if m:
                count = float(m.group(1))
            m = re.match(r"tensorfold:engram_lookup_seconds_sum\s+(\S+)", line)
            if m:
                sum_ = float(m.group(1))
        return count, sum_
    (c0, s0), (c1, s1) = side("before"), side("after")
    if c0 is None or c1 is None:
        return None, None
    return int(c1 - c0), (s1 - s0)


def done_stats(path):
    """(accept mean, accept median, tokens_per_round mean, s_per_token mean, n) from a done.log.
    Field order on the 0070 done line: s_per_token ... accept ... tokens_per_round (last)."""
    acc, tpr, spt = [], [], []
    for line in path.read_text().splitlines():
        m = re.search(r"s_per_token=([0-9.]+).*accept=([0-9.]+)%.*tokens_per_round=([0-9.]+)", line)
        if m:
            spt.append(float(m.group(1)))
            acc.append(float(m.group(2)))
            tpr.append(float(m.group(3)))
    if not acc:
        return (None,) * 5
    return (sum(acc) / len(acc), statistics.median(acc), sum(tpr) / len(tpr), sum(spt) / len(spt), len(acc))


def mem(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1]:
                continue
            out[parts[0]] = float(parts[1]) / 1024 / 1024  # kB -> GiB
    return out


def summary(name):
    p = root / f"{name}.bench.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get("summary", {})


rows = []
for name, flags, thinking in arms():
    s = summary(name)
    if not s and not (root / f"{name}.mem.txt").exists():
        continue
    doc = json.load(open(root / f"{name}.bench.json")) if s else {}
    rd, rs = eng_metrics(name)
    am, amed, tpr, spt, n = done_stats(root / f"{name}.done.log")
    m = mem(root / f"{name}.mem.txt")
    rows.append(dict(name=name, flags=flags or "\u2014", thinking=thinking, s=s, reads=rd, readsum=rs,
                     accept=am, accept_med=amed, tpr=tpr, spt=spt, n=n,
                     mem_head=m.get("head_kb"), mem_worker=m.get("worker_kb")))

fmt = lambda v, f="{:.1f}": "\u2014" if v is None else f.format(v)

print("### Engram (prefill/decode blocks, thinking off)\n")
print("| Run | Flags | Prefill 8K tok/s | Decode tok/s | multi 4 agg tok/s | engram reads (count) | engram read sum (s) | MemAvailable low H/W (GiB) |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for r in rows:
    if r["thinking"] == "1":
        continue
    s = r["s"]
    print(f"| {r['name']} | {r['flags']} | {fmt(s.get('prefill.8000.tok_s'))} | {fmt(s.get('decode.tok_s'))} "
          f"| {fmt(s.get('multi.4.aggregate_tok_s'))} | {r['reads'] if r['reads'] is not None else '\u2014'} "
          f"| {fmt(r['readsum'], '{:.2f}')} | {fmt(r['mem_head'], '{:.1f}')}/{fmt(r['mem_worker'], '{:.1f}')} |")

print("\n### Draft depth (thinking on; accept/tokens_per_round/s_per_token are done-line means, median accept in parens)\n")
print("| Run | Flags | accept % | tokens/round | s_per_token | decode tok/s | multi 4 agg tok/s | MemAvailable low H/W (GiB) |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for r in rows:
    if r["thinking"] != "1":
        continue
    s = r["s"]
    acc = "\u2014" if r["accept"] is None else f"{r['accept']:.1f} ({r['accept_med']:.1f})"
    print(f"| {r['name']} | {r['flags']} | {acc} | {fmt(r['tpr'], '{:.2f}')} | {fmt(r['spt'], '{:.4f}')} "
          f"| {fmt(s.get('decode.tok_s'))} | {fmt(s.get('multi.4.aggregate_tok_s'))} "
          f"| {fmt(r['mem_head'], '{:.1f}')}/{fmt(r['mem_worker'], '{:.1f}')} |")
