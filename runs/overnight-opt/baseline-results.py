"""Summarize the two fresh-process deployed controls and the intervening warm-serving candidate."""
import argparse
import json
from pathlib import Path
import statistics

import paired


def summarize(directory):
    a,b,c=(directory/name for name in ("deployed-a","incumbent","deployed-return"))
    rows=list(paired.compare(b,[a,c]))
    result=[]
    for depth,width in sorted({(r["depth"],r["width"]) for r in rows}):
        subset=[r for r in rows if (r["depth"],r["width"])==(depth,width)]
        result.append(dict(depth=depth,width=width,repeats=len(subset),
                           control_range_tok_s=[min(r["control_min_tok_s"] for r in subset),max(r["control_max_tok_s"] for r in subset)],
                           arm_range_tok_s=[min(r["arm_tok_s"] for r in subset),max(r["arm_tok_s"] for r in subset)],
                           median_change_pct=statistics.median(r["tok_s_change_pct"] for r in subset),
                           changes_pct=[r["tok_s_change_pct"] for r in subset],
                           ms_round_change_pct=[100*(r["arm_ms_round"]/r["control_ms_round"]-1) for r in subset],
                           arm_beats_both_controls=[r["arm_tok_s"]>r["control_max_tok_s"] for r in subset],
                           matches=sum(r["equal_hashes"] for r in subset),same_length=sum(r["equal_lengths"] for r in subset),
                           count=sum(r["outputs"] for r in subset),same_placement=all(r["same_start_occupancy"] for r in subset),
                           refusals=max(r["highest_placement_refusals"] for r in subset)))
    def prime(d):
        return {cell["depth"]:cell["requests"][0] for cell in paired.analyze.cells(d/"workload.jsonl") if cell["cell"]=="chat-prime"}
    pa,pb,pc=map(prime,(a,b,c))
    seeds=[]
    for depth,q in pb.items():
        reqs=[pa[depth],q,pc[depth]]
        if len({(r["input_sha256"],r["prompt_tokens"],r["cached"]) for r in reqs})!=1:
            raise ValueError("Priming work differs")
        rates=[(r["prompt_tokens"]-r["cached"])/float(r["done"]["prefill_s"]) for r in reqs]
        seeds.append(dict(depth=depth,prompt=q["prompt_tokens"],cut=q["cached"],input_tok_s_A_B_A=rates,
                          change_pct=100*(rates[1]/statistics.mean([rates[0],rates[2]])-1)))
    def warm(d):
        return {(x["depth"],x["sharing"],len(x["requests"]),x["rep"]):x["requests"]
                for x in paired.analyze.cells(d/"workload.jsonl") if x["cell"]!="chat-prime"}
    left,middle,right=map(warm,(a,b,c))
    latency=[]
    for k,requests in middle.items():
        width=k[2]
        groups=[left[k],requests,right[k]]
        times=[[r["ttft"] for r in group] for group in groups]
        walls=[max(r["start"]+r["total"] for r in group)-min(r["start"] for r in group) for group in groups]
        control=statistics.mean(statistics.median(t) for t in (times[0],times[2]))
        latency.append(dict(depth=k[0],width=width,rep=k[3],
                        median_s_A_B_A=list(map(statistics.median,times)),max_s_A_B_A=list(map(max,times)),
                        batch_wall_s_A_B_A=walls,
                        batch_wall_change_pct=100*(walls[1]/statistics.mean([walls[0],walls[2]])-1),
                        median_change_pct=100*(statistics.median(times[1])/control-1)))
    return dict(decode=result,priming=seeds,warm_first_token=latency)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    print(json.dumps(summarize(p.parse_args().directory),indent=2))
