"""Summarize each rank's calibrated read samples inside the measured prefill and decode windows."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import stages


def measured_windows(directory, rank):
    offset = statistics.mean(json.loads((directory / "clock.json").read_text()))
    if rank == "worker":
        sync = json.loads((directory / "meter/sync.json").read_text())
        estimates = [min(samples, key=lambda s: s["uncertainty_ns"]) for samples in sync.values()]
        offset += statistics.mean(e["worker_minus_head_ns"] for e in estimates) * 1e-9
    for line in (directory / "workload.jsonl").read_text().splitlines():
        cell = json.loads(line)
        if cell["kind"] != "cell":
            continue
        requests = cell["requests"]
        if cell["cell"] == "chat-prime":
            end = requests[0]["first"] - .25
            start = requests[0]["first"] - float(requests[0]["done"]["prefill_s"]) + .25
        else:
            start = max(r["first"] for r in requests) + .25
            end = min(r["last"] for r in requests) - .25
        if end > start:
            yield cell["cell"], (start + offset) * 1e9, (end + offset) * 1e9


def run(directory):
    for rank in ("head", "worker"):
        prepared = stages.prepare(directory, rank)
        for cell, lo, hi in measured_windows(directory, rank):
            for row in stages.summarize(directory, rank, 5000, lo, hi, prepared):
                span=(hi-lo)*1e-9
                roof=253.19 if rank=="head" else 252.11
                gb=row["sample_GB_s"]
                yield {"workload":cell,"wall_window_s":span,"event_share_of_window_pct":100*row["phase_ms"]/1000/span,
                       "interior_pct_of_contiguous_read_reference":100*gb/roof if gb is not None else None,**row}
        del prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    results = iter(run(parser.parse_args().directory))
    first = next(results)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(first))
    writer.writeheader()
    writer.writerow(first)
    writer.writerows(results)
