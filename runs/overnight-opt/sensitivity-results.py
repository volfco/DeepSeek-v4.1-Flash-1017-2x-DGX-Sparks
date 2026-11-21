"""Join the CPU read intervention with both ranks' actual completed bytes over the active decode window."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "engine-followup"))
import analyze


def rows(directory):
    offset = statistics.mean(json.loads((directory / "clock.json").read_text()))
    for cell in analyze.cells(directory / "workload.jsonl"):
        if cell["cell"] == "chat-prime":
            continue
        samples = analyze.active_segment(cell)
        a, b = samples[0], samples[-1]
        span = b["t"] - a["t"]
        rounds = sum(b[k] - a[k] for k in ("kv_shard_fetch_rounds", "kv_shard_owner_rounds"))
        key = cell["sharing"]
        row = dict(depth=cell["depth"], width=len(cell["requests"]), rep=cell["rep"], arm=key,
                   active_s=span, rounds=rounds, ms_round=1000*span/rounds,
                   tok_s=(b["generation_tokens_running"]-a["generation_tokens_running"])/span)
        for rank in ("head", "worker"):
            # The pair clock offset is small compared with the 500 ms reader period.
            # Use only whole reader intervals and report their actual time coverage.
            path = directory / f"{cell['depth']}-{key}-{rank}.csv"
            with path.open() as f:
                counter = [(int(r["utc_ns"])/1e9, int(r["bytes"])) for r in csv.DictReader(f)]
            active = [r for r in counter if a["t"]+offset+.05 <= r[0] <= b["t"]+offset-.05]
            if len(active) < 2:
                raise ValueError(f"Insufficient {rank} samples inside active interval for {key}")
            seconds = active[-1][0]-active[0][0]
            row[rank+"_reader_GB_s"] = (active[-1][1]-active[0][1])/seconds/1e9
            row[rank+"_sample_s"] = seconds
        yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    records = list(rows(parser.parse_args().directory))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
