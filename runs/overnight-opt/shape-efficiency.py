"""Compare each rank's prefill work with its independently timed same-shape dense GEMMs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def dimensions(row):
    return tuple(int(row[k]) for k in ("m", "k", "n"))


def summarize(directory, rank, label, group_rows=None):
    measurements = defaultdict(list)
    for row in rows(directory / f"{label}-math.csv"):
        measurements[dimensions(row)].append(float(row["ms_gemm"]))
    if any(len(samples) != 3 for samples in measurements.values()):
        raise ValueError("Expected three isolated references for each measured shape")
    stages = defaultdict(list)
    for row in group_rows if group_rows is not None else rows(directory / f"rank{rank}-shapes.csv"):
        stages[row["stage"]].append(row)
    for stage, group in stages.items():
        elapsed = sum(float(r["elapsed_ms"]) for r in group)
        flop = sum(int(r["useful_flop"]) for r in group)
        matched = [r for r in group if stage == "dense" and dimensions(r) in measurements]
        reference = sum(int(r["calls"]) * statistics.median(measurements[dimensions(r)]) for r in matched)
        matched_ms = sum(float(r["elapsed_ms"]) for r in matched)
        yield dict(rank=label, stage=stage, shapes=len(group), calls=sum(int(r["calls"]) for r in group),
                   elapsed_ms=elapsed, useful_TFLOP_s=flop / elapsed / 1e9,
                   matched_shapes=len(matched), matched_elapsed_ms=matched_ms, reference_ms=reference or None,
                   matched_reference_time_pct=100 * reference / matched_ms if matched_ms else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cold-window",action="store_true")
    args=parser.parse_args()
    directory=args.directory
    results=[]
    for rank,label in enumerate(("head","worker")):
        group_rows=None
        if args.cold_window:
            import importlib
            import json
            import shapes
            stages=directory/"stages"
            bounds=next((lo,hi) for cell,lo,hi in importlib.import_module("stage-windows").measured_windows(stages,label)
                        if cell=="chat-prime")
            with (stages/"meter"/(label+"-io.jsonl")).open() as stream:
                offsets=[q["mono_offset_ns"] for line in stream if "mono_offset_ns" in (q:=json.loads(line))]
            offset=statistics.median(offsets)
            group_rows=shapes.collect(stages/f"rank{rank}.stages.bin",bounds[0]-offset,bounds[1]-offset)
        results.extend(summarize(directory,rank,label,group_rows))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
