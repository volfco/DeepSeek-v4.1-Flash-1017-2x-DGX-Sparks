"""Summarize matched attribution cells and bracketing control variability."""
import argparse
import csv
import importlib
from pathlib import Path
import sys

import paired

compare_prefill = importlib.import_module("screen-results").compare


def rows(root, labels, first, last):
    controls = [root / first, root / last]
    for label in labels:
        arm = root / label
        for row in paired.compare(arm, controls):
            rate = row["arm_tok_s"]
            yield dict(arm=label, workload="warm", depth=row["depth"], width=row["width"], seed=row["rep"],
                       unit="output_tok_s", control_low=row["control_min_tok_s"], control_high=row["control_max_tok_s"],
                       candidate=rate, pct_vs_midpoint=row["tok_s_change_pct"],
                       pct_vs_faster=100*(rate/row["control_max_tok_s"]-1), matched=row["equal_hashes"],
                       responses=row["outputs"], placement_matched=row["same_start_occupancy"],
                       max_refusals=row["highest_placement_refusals"])
        left, right = (compare_prefill(c / "prefill", arm / "prefill") for c in controls)
        for a, b in zip(left, right, strict=True):
            key = ("depth", "rep", "request", "prompt", "arm_prefill_s")
            if any(a[k] != b[k] for k in key) or any(x["base_cut"] or x["arm_cut"] for x in (a, b)):
                raise ValueError("Cold prompts differ across comparisons")
            low, high = sorted(x["prompt"]/float(x["base_prefill_s"]) for x in (a, b))
            rate = a["prompt"]/float(a["arm_prefill_s"])
            yield dict(arm=label, workload="cold", depth=a["depth"], width=1, seed=a["rep"],
                       unit="input_tok_s", control_low=low, control_high=high, candidate=rate,
                       pct_vs_midpoint=100*(2*rate/(low+high)-1), pct_vs_faster=100*(rate/high-1),
                       matched=int(a["same_output"] and b["same_output"]), responses=1,
                       placement_matched="", max_refusals="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    options = (["full-candidate"], "full-a", "full-return") if args.expanded else (
        ["short-token", "short-host", "short-eight", "short-unkeyed"], "short-control-a", "short-control-return")
    panel = list(rows(args.root, *options))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(panel[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(panel)
