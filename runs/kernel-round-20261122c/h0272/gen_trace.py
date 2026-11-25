#!/usr/bin/env python3
"""Synthetic two-rank TF_DS_RING_TRACE files with known injected delays (BRIEF-0272's harness section):
producer-late on rank 0 (the producer interval of its attn_gather records), proxy-late on rank 1 (its
doorbell->last-post). Everything else is a shared baseline, so the summarizer's attribution can be checked: the
injected delays must land in exactly one (metric, rank) bucket each, and drop counters must survive. The line shapes
are the ones rdma.zig's drain writes (ring_trace.zig's gpuJson/proxyJson/drainJson/metaJson)."""
import json, random, sys, os

BASE = {
    "producer": 2000,        # previous gather's completion -> doorbell (GPU ns)
    "doorbell_flag": 20000,  # doorbell -> flag observed (GPU ns)
    "collect_flag": 15000,   # collect entry -> flag observed (GPU ns)
    "copy": 3000,            # flag -> copy end (GPU ns)
    "doorbell_post": 400,    # proxy doorbell -> last write posted (host ns)
    "completion_flag": 1200, # proxy last completion -> flag published (host ns)
}
OPS = ["attn_gather", "moe_gather", "kv", "engram", "head"]
N = 400          # sampled sequences a rank
N_INT = 16       # the run's sampling interval (meta)
CAP = 4096       # the record ring's size (meta)
INJECT = {
    0: {"producer_attn": 50000},   # rank 0 is producer-late on its attention gathers
    1: {"proxy_all": 30000},       # rank 1 is proxy-late everywhere
}
DROPS = {0: (0, 0), 1: (3, 2)}  # rank -> (gpu_dropped, proxy_dropped) in the drain line


def jitter(rng, v, frac=0.25):
    return int(v * (1 + rng.uniform(-frac, frac)))


def write_rank(path, rank, seed):
    rng = random.Random(seed)
    lines = [json.dumps({"kind": "meta", "rank": rank, "n": N_INT, "cap": CAP})]
    t_gpu = 1_000_000
    t_host = 500_000
    for i in range(1, N + 1):
        seq = i * N_INT
        op = OPS[i % len(OPS)]
        layer = -1 if op == "head" else (i * 7) % 40
        rows = 1 + (i % 7)
        bytes = rows * 5120 * 4
        producer = jitter(rng, BASE["producer"])
        if rank == 0 and op == "attn_gather":
            producer += INJECT[0]["producer_attn"]
        doorbell_flag = jitter(rng, BASE["doorbell_flag"])
        collect_flag = jitter(rng, BASE["collect_flag"])
        copy = jitter(rng, BASE["copy"])
        t_gpu += 1000
        prev = t_gpu
        doorbell = prev + producer
        stage = doorbell - 500
        flag = doorbell + doorbell_flag
        collect = flag - collect_flag
        copy_done = flag + copy
        t_gpu = copy_done
        lines.append(json.dumps({
            "kind": "gpu", "seq": seq, "round": 1 + i // 10, "op": op, "layer": layer,
            "rows": rows, "bytes": bytes, "stream": i % 2, "graph": i % 3 != 0,
            "prev": prev, "stage": stage, "doorbell": doorbell, "collect": collect,
            "flag": flag, "copy": copy_done,
        }))
        doorbell_post = jitter(rng, BASE["doorbell_post"])
        if rank == 1:
            doorbell_post += INJECT[1]["proxy_all"]
        completion_flag = jitter(rng, BASE["completion_flag"])
        t_host += 700
        h_doorbell = t_host
        h_flag = h_doorbell + doorbell_post + completion_flag + 200
        parts = [h_flag - completion_flag, 0, 0, 0]
        if i % 3 == 0:
            parts[1] = parts[0] - 50
        lines.append(json.dumps({
            "kind": "proxy", "seq": seq, "doorbell": h_doorbell, "last_post": h_doorbell + doorbell_post,
            "parts": parts, "flag": h_flag, "woke": int(i % 25 == 0),
        }))
        t_host = h_flag
    gd, pd = DROPS[rank]
    lines.append(json.dumps({
        "kind": "drain", "rank": rank, "seq": N * N_INT, "gpu_dropped": gd,
        "proxy_dropped": pd, "gpu_kept": N, "proxy_kept": N,
    }))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    a = write_rank(os.path.join(out, "synth_rank0.jsonl"), 0, 12270)
    b = write_rank(os.path.join(out, "synth_rank1.jsonl"), 1, 12271)
    print(f"wrote {a} and {b} ({N} sampled sequences a rank, n={N_INT}, injected: rank0 producer-late "
          f"+{INJECT[0]['producer_attn']}ns on attn_gather, rank1 proxy-late +{INJECT[1]['proxy_all']}ns everywhere)")
