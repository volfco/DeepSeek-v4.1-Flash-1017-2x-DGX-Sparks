"""Compare warm appends against a fresh cold process on the same unsharded engine and inputs."""
import argparse
import csv
import importlib
from pathlib import Path
import sys

records = importlib.import_module("screen-results").requests


def rows(root):
    cold, grid, exact = (records(root / name) for name in ("append-cold", "grid", "exact"))
    for (depth, shape, rep, index), fresh in cold.items():
        if shape != "append-cold" or fresh["cached"] != 0:
            raise ValueError("Expected independently uncached appended IDs")
        shared = [g[depth,"append",rep,index] for g in (grid,exact)]
        if any(fresh["input_sha256"] != q["input_sha256"] for q in shared):
            raise ValueError("Append inputs differ")
        yield dict(depth=depth,rep=rep,input_ids=fresh["prompt_tokens"],cold_cut=fresh["cached"],
                   grid_cut=shared[0]["cached"],exact_cut=shared[1]["cached"],
                   cold_ttft_s=fresh["ttft"],grid_ttft_s=shared[0]["ttft"],exact_ttft_s=shared[1]["ttft"],
                   cold_hash=fresh["done"]["sha"],grid_hash=shared[0]["done"]["sha"],exact_hash=shared[1]["done"]["sha"],
                   grid_matches_cold=fresh["done"]["sha"]==shared[0]["done"]["sha"],
                   exact_matches_cold=fresh["done"]["sha"]==shared[1]["done"]["sha"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    panel=list(rows(parser.parse_args().directory))
    if len(panel)!=4:raise ValueError("Expected a two-depth, two-source crossover")
    writer=csv.DictWriter(sys.stdout,fieldnames=list(panel[0]))
    writer.writeheader()
    writer.writerows(panel)
