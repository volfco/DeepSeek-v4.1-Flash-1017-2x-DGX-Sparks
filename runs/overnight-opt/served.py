"""Integrate both ranks' GPU and I/O counters over the same continuously active serving intervals."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent/"engine-followup")]
import analyze
from windows import bounds, report


def rows(directory):
    clock = json.loads((directory/"clock.json").read_text())
    if abs(clock[1]-clock[0]) > .05:
        raise ValueError("Head wall/monotonic offset moved")
    offset = statistics.mean(clock)
    for cell in analyze.cells(directory/"workload.jsonl"):
        if cell["cell"]=="chat-prime":
            continue
        req = cell["requests"]
        samples = analyze.active_segment(cell)
        if len(samples)<2:
            raise ValueError("No sustained all-active window")
        a,b = samples[0],samples[-1]
        duration = b["t"]-a["t"]
        rounds = sum(b[k]-a[k] for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds"))
        tokens = b["generation_tokens_running"]-a["generation_tokens_running"]
        if min(rounds,tokens)<=0 or duration<2:
            raise ValueError("Short or empty serving window")
        row = dict(depth=cell["depth"],width=len(req),shape=cell["sharing"],rep=cell["rep"],
                   active_s=duration,rounds=rounds,tokens=tokens,ms_round=1000*duration/rounds,
                   tok_s=tokens/duration,tok_stream_round=tokens/rounds/len(req),
                   output_hashes=" ".join(r["done"]["sha"] for r in req))
        if (directory/"meter").exists():
            meter=directory/"meter"
            lo,hi=bounds(meter)
            common=[p for p in samples if lo < (p["t"]+offset)*1e9 < hi]
            if len(common)<2 or common[-1]["t"]-common[0]["t"]<2:
                raise ValueError("Less than two seconds of overlap with the hardware counters")
            row["meter_active_s"]=common[-1]["t"]-common[0]["t"]
            meter_rounds=sum(common[-1][k]-common[0][k] for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds"))
            if meter_rounds<=0:
                raise ValueError("Hardware window has no physical rounds")
            row["meter_rounds"]=meter_rounds
            counts = report(meter,*(round((p["t"]+offset)*1e9) for p in (common[0],common[-1])))
            for rank,m in counts["ranks"].items():
                value, = m["pm"].values()
                io=m["device_GB_s"]
                row.update({rank+"_read_GB_s":value["GB_s"],rank+"_coverage":value["coverage"],
                            rank+"_read_GB_round":value["GB_s"]*row["meter_active_s"]/meter_rounds,
                            rank+"_RDMA_xmit":sum(v for k,v in io.items() if k.endswith(".xmit")),
                            rank+"_RDMA_rcv":sum(v for k,v in io.items() if k.endswith(".rcv")),
                            rank+"_NVMe_read":sum(v for k,v in io.items() if k.endswith(".read"))})
        yield row


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    records=list(rows(parser.parse_args().directory))
    out=csv.DictWriter(sys.stdout,fieldnames=list(records[0]))
    out.writeheader();out.writerows(records)
