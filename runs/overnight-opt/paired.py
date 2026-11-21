"""Match serving A/B(/A) cells, flag changes in output or placement and compare physical rounds."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

here=Path(__file__).resolve().parent
sys.path[:0]=[str(here),str(here.parent/"engine-followup")]
import analyze
import served


def prepare(directory):
    cells=analyze.cells(directory/"workload.jsonl")
    windows={(r["depth"],r["shape"],r["width"],r["rep"]):r for r in served.rows(directory)}
    result={}
    for c in cells:
        if c["cell"]=="chat-prime":continue
        requests=c["requests"]
        key=(c["depth"],c["sharing"],len(requests),c["rep"])
        if key in result:raise ValueError(f"Duplicate cell {key}")
        result[key] = (windows[key],[(r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in requests],
                       [v["kept_rows"] for v in c["health_before"]["per_part"]],
                       c["health_after"]["refused_no_target"]-c["health_before"]["refused_no_target"])
    return result


def compare(treatment,controls):
    panel=prepare(treatment)
    refs=[prepare(p) for p in controls]
    if not all(r.keys()==panel.keys() for r in refs):raise ValueError("Arm coverage differs")
    for key,(observed,requests,parts,refused) in panel.items():
        paired=[r[key] for r in refs]
        if any([t[:2] for t in other]!=[t[:2] for t in requests] for _,other,_,_ in paired):
            raise ValueError(f"Inputs or cache cuts differ for {key}")
        base=[p[0] for p in paired]
        def med(k):return statistics.median(r[k] for r in base)
        matches=sum(all(r[3]==p[1][i][3] for p in paired) for i,r in enumerate(requests))
        lengths=sum(all(r[2]==p[1][i][2] for p in paired) for i,r in enumerate(requests))
        yield dict(depth=key[0],shape=key[1],width=key[2],rep=key[3],controls=len(refs),
                   control_tok_s=med("tok_s"),arm_tok_s=observed["tok_s"],
                   tok_s_change_pct=100*(observed["tok_s"]/med("tok_s")-1),
                   control_ms_round=med("ms_round"),arm_ms_round=observed["ms_round"],
                   control_tokens_per_stream_round=med("tok_stream_round"),arm_tokens_per_stream_round=observed["tok_stream_round"],
                   control_min_tok_s=min(r["tok_s"] for r in base),control_max_tok_s=max(r["tok_s"] for r in base),
                   equal_lengths=lengths,equal_hashes=matches,outputs=len(requests),
                   same_start_occupancy=all(v[2]==parts for v in paired),
                   highest_placement_refusals=max([refused]+[v[3] for v in paired]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("treatment",type=Path);p.add_argument("controls",type=Path,nargs="+")
    args=p.parse_args();records=list(compare(args.treatment,args.controls))
    out=csv.DictWriter(sys.stdout,fieldnames=list(records[0]));out.writeheader();out.writerows(records)
