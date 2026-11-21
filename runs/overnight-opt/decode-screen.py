"""Keep short and divergent cells visible alongside valid all-active comparisons."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"engine-followup"))
import analyze


def collect(path):
    result = {}
    for c in analyze.cells(path / "workload.jsonl"):
        if c["cell"] == "chat-prime":
            continue
        req = c["requests"]
        key = c["depth"], c["sharing"], len(req), c["rep"]
        samples = analyze.active_segment(c)
        span = samples[-1]["t"] - samples[0]["t"] if len(samples)>1 else 0
        def delta(k):
            return samples[-1][k] - samples[0][k] if samples else 0
        rounds = sum(delta(k) for k in ("kv_shard_owner_rounds","kv_shard_fetch_rounds"))
        metrics = dict(active_s=span, rounds=rounds, tok_s=None, ms_round=None)
        if c["clean"] and span >= 2 and rounds > 0:
            metrics.update(tok_s=delta("generation_tokens_running")/span, ms_round=1000*span/rounds)
        result[key] = req, metrics
    return result


def compare(control, treatment):
    a,b=collect(control),collect(treatment)
    if a.keys()!=b.keys():
        raise ValueError("Coverage differs")
    for key,(req,arm) in b.items():
        reference,base=a[key]
        if any((x["cached"],x["input_sha256"]) != (y["cached"],y["input_sha256"]) for x,y in zip(reference,req)):
            raise ValueError("Inputs or cuts differ")
        yield dict(depth=key[0],shape=key[1],width=key[2],rep=key[3],
                   equal_hashes=sum(x["done"]["sha"]==y["done"]["sha"] for x,y in zip(reference,req)),
                   equal_lengths=sum(x["tokens"]==y["tokens"] for x,y in zip(reference,req)),
                   **{"control_"+k:v for k,v in base.items()}, **{"arm_"+k:v for k,v in arm.items()},
                   delivered_change_pct=100*(arm["tok_s"]/base["tok_s"]-1) if arm["tok_s"] and base["tok_s"] else None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control",type=Path);parser.add_argument("arm",type=Path)
    args=parser.parse_args();rows=list(compare(args.control,args.arm))
    writer=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
