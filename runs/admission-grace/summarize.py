"""Emit paired burst metrics from the four finish-grace blocks."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ARMS = (("control-a", 5000), ("zero", 0), ("half-second", 500), ("control-b", 5000))


def load(path, ms, deep=False):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if events[0]["kind"] != "start" or events[-1]["kind"] != "complete" or events[0]["ranks"] != events[-1]["ranks"]:
        raise ValueError(f"Incomplete or restarted arm: {path}")
    ranks = events[0]["ranks"]
    if any(r["env"]["TF_DS_FINISH_GRACE_MS"] != str(ms) for r in ranks):
        raise ValueError(f"Grace differs on a rank: {path}")
    cells = [c for c in events if c["kind"] == "cell"]
    for c in cells:
        if not c["clean"] or not c["full_length"] or c["own"] != c["completed_delta"] or c["own"] != len(c["requests"]):
            raise ValueError(f"Incomplete request cell in {path}")
    bursts = {(c["sharing"], c["rep"]): c for c in cells if c["cell"] == "admission"}
    shapes = ("tail-384", "tail-1990") if deep else ("burst", "short-lead", "long-lead")
    if set(bursts) != {(shape, rep) for shape in shapes for rep in range(2)}:
        raise ValueError(f"Missing or invalid workload in {path}")
    return ranks, bursts


def fingerprint(r):
    return (r["input_sha256"], r["seed"], r["prompt_tokens"], r["cached"], r["tokens"])


def metric(c, baseline):
    req = c["requests"]
    control = baseline["requests"]
    if [fingerprint(r) for r in req] != [fingerprint(r) for r in control]:
        raise ValueError("Request payload, work or cached prefix differs across arms")
    if c["sharing"] == "burst":
        lead = min(req, key=lambda r: r["first"])
    else:
        lead = req[0]
        for r, planned in zip(req[1:], (0.20, 0.55, 0.90)):
            if abs((r["start"] - lead["first"]) - planned) > .1:
                raise ValueError("Mixed-arrival offsets missed the timing window")
    followers = [r for r in req if r is not lead]
    wall = max(r["start"] + r["total"] for r in req) - min(r["start"] for r in req)
    return dict(throughput=sum(r["tokens"] for r in req) / wall, wall_s=wall,
                mean_request_total_s=statistics.mean(r["total"] for r in req), lead_total_s=lead["total"],
                lead_max_gap_s=lead["max_gap_s"], lead_p95_gap_s=lead["p95_gap_s"],
                worst_stream_gap_s=max(r["max_gap_s"] for r in req),
                follower_ttft_median_s=statistics.median(r["ttft"] for r in followers),
                follower_ttft_max_s=max(r["ttft"] for r in followers),
                overlap_s=max(0,c["overlap"]), queue_max_s=max(float(r["done"]["queue_s"]) for r in req),
                sha_matches=sum(r["done"]["sha"] == b["done"]["sha"] for r,b in zip(req, control)),
                min_prefill_s=min(float(r["done"]["prefill_s"]) for r in req),
                mean_tokens_per_round=statistics.mean(float(r["done"]["tokens_per_round"]) for r in req),
                no_target_delta=c["health_after"]["refused_no_target"] - c["health_before"]["refused_no_target"])


def summarize(out, deep_grace_ms=None):
    baseline = images = None
    writer = None
    for arm, ms in ARMS:
        if deep_grace_ms is not None and ms not in (deep_grace_ms, 5000):
            continue
        ranks, bursts = load(out / f"{arm}.jsonl", ms, deep=deep_grace_ms is not None)
        ids = [r["image_id"] for r in ranks]
        if baseline is None:
            baseline, images = bursts, ids
        if ids != images:
            raise ValueError("Experiment image changed")
        for (shape, rep), c in bursts.items():
            metrics = metric(c, baseline[(shape, rep)])
            row = dict(arm=arm, grace_ms=ms, shape=shape, repeat=rep,
                       **{k:round(v, 4) for k,v in metrics.items()})
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row, lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--depth-grace-ms", type=int, choices=(0, 500))
    args = parser.parse_args()
    summarize(args.run_dir, args.depth_grace_ms)
