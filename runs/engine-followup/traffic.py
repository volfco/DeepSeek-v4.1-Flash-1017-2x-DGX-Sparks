"""Estimate admission overlap from the November 13 archive using only numeric fields."""
from datetime import datetime
from pathlib import Path
import csv
import sys


def historical_completions(path):
    observations = []
    local_posts = 0
    for line in path.read_text().splitlines():
        if line[:19] < "2026-11-13T07:20:00":
            continue
        if '127.0.0.1 "POST /v1/chat/completions' in line:
            local_posts += 1
        if "[tensorfold] done " not in line:
            continue
        fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
        end = datetime.fromisoformat(line.split()[0].replace("Z", "+00:00")).timestamp()
        decode_start = end - float(fields["decode_s"])
        # The engine times decode from the prefilled event and TTFT to the first
        # emitted token. Using decode_start as first-token time is approximate.
        arrival = decode_start - float(fields["ttft_s"])
        observations.append((arrival, decode_start, end, float(fields["prefill_s"])))
    if local_posts:
        raise ValueError("Non-external client traffic in historical window")
    return observations


def counts(observations, margin):
    overlaps = set()
    within_grace = set()
    leads = set()
    for i, (arrival, _, _, _) in enumerate(observations):
        for j, (_, start, end, prefill) in enumerate(observations):
            if i == j or not start + margin < arrival < end - margin:
                continue
            overlaps.add(i)
            if prefill > 1 and arrival < start + 5 - margin:
                within_grace.add(i)
                leads.add(j)
    return (margin, len(observations), sum(p > 1 for _, _, _, p in observations),
            len(overlaps), len(within_grace), len(leads))


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "overnight/break-1600Z.rank0.log"
    observations = historical_completions(path)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("boundary_margin_s", "completions", "prefill_over_1s",
                     "arrived_during_other_decode", "arrival_in_grace", "leads_with_arrival_in_grace"))
    for margin in (0, 0.2, 0.5):
        writer.writerow(counts(observations, margin))
