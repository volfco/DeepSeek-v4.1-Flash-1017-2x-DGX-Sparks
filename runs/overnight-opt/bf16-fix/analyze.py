"""Check same-image rank settings, shape probes, frozen scores and cold-prefill pairs."""
import argparse
import collections
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys


here = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(here))
scores = importlib.import_module("odds-results")
prefills = importlib.import_module("screen-results")


def read(path):
    return json.loads(path.read_text())


def frozen_scores(control, candidate):
    a, b = [read(p / "score.json") for p in (control, candidate)]
    source_file = here.parent / "engine-followup/runtime-score01/items.jsonl"
    lines = [s.strip() for s in source_file.read_text().splitlines() if s.strip()]
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    delta_by_source = collections.defaultdict(list)
    for result in (a, b):
        if result["meta"]["set_sha"] != digest or result["meta"]["errors"] or result["meta"]["scored"] != 40:
            raise ValueError("Missing items or wrong frozen panel")
    for line in lines:
        item = json.loads(line)
        left, right = [r["items"][item["id"]] for r in (a, b)]
        for observed in (left, right):
            if observed["tokens"] != len(item["continuation_ids"]) or observed["prompt_tokens"] != len(item["prompt_ids"]):
                raise ValueError("Frozen target or prompt length mismatch")
        delta_by_source[(item["kind"], item["provenance"]["sha256"])].append(right["nll_mean"] - left["nll_mean"])
    if len(delta_by_source) != 20 or {len(ds) for ds in delta_by_source.values()} != {2}:
        raise ValueError("Expected 20 distinct source hashes, two excerpts each")
    quality = scores.quality(control, candidate)
    delta = [statistics.mean(v) for v in delta_by_source.values()]
    if not math.isclose(quality["delta_nats"], statistics.mean(delta), abs_tol=1e-12):
        raise ValueError("Provenance and predeclared grouping disagree")
    quality["passes_020_margin"] = quality["ci95"][1] <= 0.020
    quality["frozen_set_sha"] = digest
    quality["targets"] = a["overall"]["tokens"]
    quality["argmax_match_fp32_bf16"] = [x["overall"]["argmax_match"] for x in (a, b)]
    quality["source_mean_delta_by_kind"] = {
        name: statistics.mean(statistics.mean(v) for (kind, _), v in delta_by_source.items() if name == kind)
        for name in sorted({kind for kind, _ in delta_by_source})}
    return quality


def compare(control, candidate):
    dirs = (control, candidate)
    cfg = [read(p / "requested.json") for p in dirs]
    armed = [read(p / "armed.json") for p in dirs]
    allowed = {"SPILL_DIR", "WORKER_SPILL_DIR", "TF_DS_BF16_PARTIALS"}
    if {k for k in set(cfg[0]) | set(cfg[1]) if cfg[0].get(k) != cfg[1].get(k)} != allowed:
        raise ValueError("Unmatched experiment configuration")
    if any(len(ranks) != 2 for ranks in armed):
        raise ValueError("Missing rank")
    for left, right in zip(*armed, strict=True):
        if left["id"] != right["id"] or left["cpuset"] != right["cpuset"]:
            raise ValueError("Image or CPU placement changed between controls")
        if [r["switches"]["TF_DS_BF16_PARTIALS"] for r in (left, right)] != ["0", "1"]:
            raise ValueError("Precision mode was not explicitly observed on both ranks")
    result = {"image": cfg[0]["IMAGE"], "rank_image_ids": [r["id"] for r in armed[0]]}
    probes = [read(p / "bf16-probes.json") for p in dirs]
    for rows in probes:
        if [r["prompt_tokens"] for r in rows] != [128, 129, 2048] or any(
                r["cached"] or r["score"]["tokens"] != 8 or not math.isfinite(r["score"]["nll_sum"]) for r in rows):
            raise ValueError("Shape or score mismatch")
    result["eight_target_probe_nll_delta"] = {str(a["prompt_tokens"]): (b["score"]["nll_sum"] - a["score"]["nll_sum"]) / 8
                                              for a, b in zip(*probes, strict=True)}
    result["quality"] = frozen_scores(*dirs)
    if (candidate / "prefill/clock.json").exists():
        rows = list(prefills.compare(control / "prefill", candidate / "prefill"))
        if {(r["depth"], r["rep"]) for r in rows} != {(d, i) for d in (2094, 8192, 32000) for i in (0, 1)} or len(rows) != 6:
            raise ValueError("Incomplete cold timing matrix")
        if any(r["base_cut"] or r["arm_cut"] for r in rows):
            raise ValueError("Expected all timing probes to be cold")
        result["prefill_pairs"] = rows
        result["cold_rate_delta_by_depth"] = {
            str(d): [float(r["base_prefill_s"]) / float(r["arm_prefill_s"]) - 1 for r in rows if r["depth"] == d]
            for d in (2094, 8192, 32000)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate), indent=2))
