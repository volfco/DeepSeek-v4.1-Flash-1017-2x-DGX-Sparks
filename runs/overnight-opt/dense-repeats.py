"""Separate first and repeated dense projection dimensions in the saved 9K prefill."""
import argparse
import collections
import csv
import gzip
import importlib
import json
import math
from pathlib import Path
import statistics
import struct


def read_activity(path, spans):
    """Integrate complete calibrated samples inside each occurrence of the selected shape."""
    rows = [dict(cohort=cohort, ns=b-a, covered=0, bytes=0) for a,b,cohort in spans]
    anchors = set()
    index = 0
    with gzip.open(path, "rt") as stream:
        for line in csv.reader(stream):
            if line and line[0] == "anchor":
                anchors.add(int(line[2]) - int(line[1]))
            elif len(line) == 4 and line[0].isdigit():
                if len(anchors) != 1 or line[2] != "lts__t_sectors_op_read_lookup_miss_realtime.sum":
                    raise ValueError("Unexpected CUPTI time base or metric")
                if index == len(spans):
                    continue
                offset = next(iter(anchors))
                a, b = int(line[0]) + offset, int(line[1]) + offset
                while index < len(spans) and a >= spans[index][1]:
                    index += 1
                count = float(line[3])
                if index < len(spans) and 0 < b-a < 10_000_000 and math.isfinite(count):
                    lo, hi, _ = spans[index]
                    if lo+5000 <= a and b <= hi-5000:
                        rows[index]["covered"] += b-a
                        rows[index]["bytes"] += 32*count
    result = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        group = [r for r in rows if r["cohort"] == cohort]
        covered = sum(r["covered"] for r in group)
        result[cohort] = dict(calls=len(group), enough_samples=sum(r["covered"]>0 for r in group),
                              coverage=covered/sum(r["ns"] for r in group),
                              interior_GB_s=sum(r["bytes"] for r in group)/covered if covered else None,
                              median_sampled_MB=statistics.median(r["bytes"] / 1e6 for r in group))
    return result


def events(path):
    pending = None
    data = []
    for a, b, kind, phase in struct.iter_unpack("<QQII", path.read_bytes()):
        if kind == 4:
            if pending or phase not in (18, 19, 20):
                raise ValueError("Unmatched shape metadata")
            pending = phase, (a >> 32, a & 0xffffffff, b)
        elif kind == 2 and phase in (18, 19, 20):
            if not pending or phase != pending[0] or a >= b:
                raise ValueError("Invalid shape event")
            data.append((a, b, phase, pending[1]))
            pending = None
    if pending:
        raise ValueError("Missing final event")
    return sorted(data)


def rank_results(directory, rank, label):
    stages = directory / "stages"
    lo, hi = next((a, b) for key, a, b in importlib.import_module("stage-windows").measured_windows(stages, label)
                  if key == "chat-prime")
    offsets = [v["mono_offset_ns"] for s in (stages / "meter" / (label + "-io.jsonl")).read_text().splitlines()
               if "mono_offset_ns" in (v := json.loads(s))]
    offset = round(statistics.median(offsets))
    lo -= offset
    hi -= offset
    reference = collections.defaultdict(list)
    with (directory / (label + "-math.csv")).open() as stream:
        for row in csv.DictReader(stream):
            reference[tuple(int(row[n]) for n in ("m", "k", "n"))].append(float(row["ms_gemm"]))
    if any(len(v) != 3 for v in reference.values()):
        raise ValueError("Incomplete dense reference")
    earlier = collections.Counter()
    inside = collections.defaultdict(list)
    experts = collections.Counter()
    selected_shape = (2048, 1280, 16384)
    selected_intervals = []
    for start, end, phase, shape in events(stages / f"rank{rank}.stages.bin"):
        if end <= lo and phase == 18:
            earlier[shape] += 1
        if lo <= start and end <= hi:
            if phase == 18:
                inside[shape].append((end - start) / 1e6)
                if shape == selected_shape:
                    selected_intervals.append((start+offset, end+offset))
            else:
                experts[(phase, shape)] += 1
    cohorts = collections.defaultdict(list)
    shapes = []
    for shape, durations in sorted(inside.items()):
        ref = statistics.median(reference[shape])
        repeated = durations[1:]
        med = statistics.median(repeated) if repeated else None
        long = [v for v in repeated if v > 3 * med]
        median_total = len(repeated) * med if repeated else 0
        shapes.append(dict(shape=shape, earlier_calls_in_trace=earlier[shape],
                           calls=len(durations), first_ms=durations[0], reference_ms=ref,
                           reference_repetitions_ms=reference[shape],
                           repeated_median_ms=med, repeated_sum_ms=sum(repeated), median_total_ms=median_total,
                           maximum_ms=max(repeated) if repeated else None,
                           above_3x_median=len(long), above_3x_median_ms=sum(long),
                           first_over_repeated_median=durations[0] / med if med else None))
        for i, duration in enumerate(durations):
            sample = (shape, duration, ref)
            cohorts["all"].append(sample)
            cohorts["first_in_interior" if i == 0 else "repeat_in_interior"].append(sample)
            cohorts["first_in_recorded_trace" if earlier[shape] + i == 0 else "repeat_in_recorded_trace"].append(sample)
    summary = {}
    for key, records in cohorts.items():
        elapsed = sum(v[1] for v in records)
        ref_ms = sum(v[2] for v in records)
        summary[key] = dict(calls=len(records), shapes=len({v[0] for v in records}), elapsed_ms=elapsed, reference_ms=ref_ms,
                            reference_pct=100 * ref_ms / elapsed, excess_ms=elapsed - ref_ms,
                            reference_min_max_pct=[100*sum(f(reference[v[0]]) for v in records)/elapsed
                                                   for f in (min,max)],
                            useful_TFLOP_s=sum(2*m*k*n for (m,k,n), _, _ in records) / (elapsed*1e9))
    with (directory / "cold-compute-efficiency.csv").open() as stream:
        expected = next(v for v in csv.DictReader(stream) if v["rank"] == label and v["stage"] == "dense")
    actual = summary["all"]
    if actual["calls"] != int(expected["calls"]) or abs(actual["elapsed_ms"] - float(expected["elapsed_ms"])) > 1e-6:
        raise ValueError("Cold window disagrees with the earlier published ledger")
    repeats = selected_intervals[1:]
    threshold = 3*statistics.median(b-a for a,b in repeats)
    spans = [(a,b,"above_3x_median" if b-a > threshold else "at_or_below_3x_median") for a,b in repeats]
    activity = read_activity(stages/"meter"/(label+"-pm.csv.gz"), spans)
    return dict(rank=label, interior_seconds=(hi-lo)/1e9, cohorts=summary, by_shape=shapes,
                selected_shape=selected_shape, repeat_read_activity=activity,
                routed_shapes=[dict(stage="gate_up" if phase == 19 else "down", shape=shape, calls=count)
                               for (phase,shape),count in sorted(experts.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps([rank_results(args.directory, i, name) for i, name in enumerate(("head", "worker"))], indent=2))
