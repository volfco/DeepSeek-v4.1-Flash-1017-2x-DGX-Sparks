"""Join time-based NVML activity with continuous measured prefill/decode windows."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import statistics
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "engine-followup"))
import analyze


def telemetry(path):
    records = []
    with path.open() as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) != 5:
                continue
            when = datetime.datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
            records.append((when.timestamp(), *map(int, row[1:4])))
    return records


def intervals(directory):
    clocks = json.loads((directory / "clock.json").read_text())
    if abs(clocks[-1] - clocks[0]) > .05:
        raise ValueError("Unstable head timebase")
    offset = statistics.mean(clocks)
    for c in analyze.cells(directory / "workload.jsonl"):
        requests = c["requests"]
        if c["cell"] == "chat-prime":
            r = requests[0]
            seconds = float(r["done"]["prefill_s"])
            if seconds <= 2 or r["cached"] == r["prompt_tokens"]:
                continue
            # The first event follows the first generation step. Discard both boundaries.
            start, end = r["first"] - seconds, r["first"]
            phase = "prefill"
        else:
            active = analyze.active_segment(c)
            if len(active) < 2:
                continue
            start, end = active[0]["t"], active[-1]["t"]
            phase = "decode"
        yield c, phase, start + offset, end + offset


def summarize(directory, series):
    for c, phase, start, end in intervals(directory):
        for rank, records in series.items():
            samples = [r for r in records if start + 1.1 < r[0] < end - .25]
            if len(samples) < 3:
                continue
            yield dict(rank=rank, phase=phase, depth=c["depth"], width=len(c["requests"]), rep=c["rep"],
                       workload=c["sharing"], samples=len(samples), span_s=samples[-1][0]-samples[0][0],
                       gpu_busy_pct=statistics.mean(r[1] for r in samples),
                       memory_busy_pct=statistics.mean(r[2] for r in samples),
                       sm_MHz=statistics.median(r[3] for r in samples))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("telemetry", type=Path)
    a = p.parse_args()
    data = {rank: telemetry(a.telemetry / f"{rank}-nvml.csv") for rank in ("head", "worker")}
    values = list(summarize(a.directory, data))
    if not values:
        raise SystemExit("No overlapping windows with at least three interior NVML observations")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
