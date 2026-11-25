#!/usr/bin/env python3
"""TF_DS_RING_TRACE's summarizer (BRIEF-0272): reads both ranks' JSONL files (rdma.zig's drain output) and prints
p50/p95 per (op, rank) of the same-clock intervals the brief names:
  - producer interval: the previous gather's completion to this doorbell (GPU, per stream)
  - doorbell -> flag wait and collect-entry -> flag (GPU)
  - copy (GPU, flag to copy end)
  - proxy doorbell -> last write posted and last completion -> flag published (host)
plus the drop counters and sample counts. Proxy records carry no op: they pair with their gpu record's op by "seq"
(records pair by sequence). --json prints a machine-readable summary for the harness's attribution check."""
import json, sys


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    rank = (p * len(xs) + 99) // 100
    return xs[rank - 1 if rank else 0]


def load(path):
    gpu, proxy, drains = [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        k = rec.get("kind")
        if k == "gpu":
            gpu.append(rec)
        elif k == "proxy":
            proxy.append(rec)
        elif k == "drain":
            drains.append(rec)
    return gpu, proxy, drains


def summarize(paths):
    groups = {}  # (op, rank) -> metric -> [ns]
    totals = {}  # rank -> counters
    for path in paths:
        gpu, proxy, drains = load(path)
        rank = drains[0]["rank"] if drains else 0
        op_of = {g["seq"]: g["op"] for g in gpu}
        t = totals.setdefault(rank, {"gpu_samples": 0, "proxy_samples": 0,
                                     "gpu_dropped": 0, "proxy_dropped": 0})
        for g in gpu:
            m = groups.setdefault((g["op"], rank), {})

            def add(name, lo, hi):  # both endpoints present (0 = not recorded), no underflow: gap()'s rule
                if lo > 0 and hi >= lo:
                    m.setdefault(name, []).append(hi - lo)
            add("producer", g["prev"], g["doorbell"])
            add("doorbell_flag", g["doorbell"], g["flag"])
            add("collect_flag", g["collect"], g["flag"])
            add("copy", g["flag"], g["copy"])
            t["gpu_samples"] += 1
        for p in proxy:
            op = op_of.get(p["seq"], "unpaired")
            m = groups.setdefault((op, rank), {})
            if p["last_post"] > p["doorbell"] > 0:
                m.setdefault("proxy_doorbell_post", []).append(p["last_post"] - p["doorbell"])
            last = max(p["parts"])
            if p["flag"] > last > 0:
                m.setdefault("proxy_completion_flag", []).append(p["flag"] - last)
            t["proxy_samples"] += 1
        for d in drains:
            t["gpu_dropped"] += d["gpu_dropped"]
            t["proxy_dropped"] += d["proxy_dropped"]
    out = {"groups": {}, "ranks": totals}
    for (op, rank), metrics in sorted(groups.items()):
        cell = {}
        for name, xs in sorted(metrics.items()):
            cell[name] = {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95)}
        out["groups"][f"{op}|{rank}"] = cell
    return out


def print_table(s):
    for key, cell in s["groups"].items():
        op, rank = key.rsplit("|", 1)
        print(f"op={op} rank={rank}")
        for name, q in cell.items():
            print(f"  {name:24s} n={q['n']:5d}  p50={q['p50']} ns  p95={q['p95']} ns")
    for rank, t in sorted(s["ranks"].items(), key=lambda kv: int(kv[0])):
        print(f"rank {rank}: samples gpu={t['gpu_samples']} proxy={t['proxy_samples']} "
              f"dropped gpu={t['gpu_dropped']} proxy={t['proxy_dropped']}")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    as_json = "--json" in argv
    if not args:
        print("usage: summarize.py [--json] RANK0.jsonl [RANK1.jsonl ...]", file=sys.stderr)
        return 2
    s = summarize(args)
    if as_json:
        print(json.dumps(s, indent=1, sort_keys=True))
    else:
        print_table(s)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
