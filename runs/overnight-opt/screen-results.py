"""Pair engineering screen results by identical request body and report cache cuts and elapsed time."""
import argparse
import csv
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "engine-followup"))
import analyze


def requests(directory):
    records = {}
    for cell in analyze.cells(directory / "workload.jsonl"):
        if cell["sharing"]=="score-seed":
            continue
        for i, request in enumerate(cell["requests"]):
            key = (cell["depth"], cell["sharing"], cell["rep"], i)
            if key in records:
                raise ValueError(f"Duplicate {key} in {directory}")
            records[key] = request
    return records


def compare(control, arm):
    base, candidate = requests(control), requests(arm)
    if base.keys() != candidate.keys():
        raise ValueError("Incomplete or incompatible request matrix")
    for key, q in candidate.items():
        r = base[key]
        if q["input_sha256"] != r["input_sha256"] or q["prompt_tokens"] != r["prompt_tokens"]:
            raise ValueError(f"Input mismatch at {key}")
        a, b = r["done"], q["done"]
        yield dict(depth=key[0], workload=key[1], rep=key[2], request=key[3], prompt=q["prompt_tokens"],
                   base_cut=r["cached"], arm_cut=q["cached"], base_tokens=r["tokens"], arm_tokens=q["tokens"],
                   same_output=a["sha"]==b["sha"], base_ttft_s=r["ttft"], arm_ttft_s=q["ttft"],
                   base_prefill_s=a["prefill_s"], arm_prefill_s=b["prefill_s"],
                   base_decode_s=a["decode_s"], arm_decode_s=b["decode_s"],
                   base_rounds=a["rounds"], arm_rounds=b["rounds"],
                   base_accept=a.get("accepted"), arm_accept=b.get("accepted"),
                   base_engram_host_ms=a.get("engram_host_ms"), arm_engram_host_ms=b.get("engram_host_ms"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control", type=Path)
    p.add_argument("arm", type=Path)
    a = p.parse_args()
    rows = list(compare(a.control, a.arm))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
