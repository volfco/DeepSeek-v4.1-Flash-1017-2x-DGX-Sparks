"""Summarize paired served odds screens, held-out scores and observed snapshot budget charges."""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics

ledger = re.compile(r"(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB")


def quality(base, arm):
    def read(path):
        return json.loads((path / "score.json").read_text())
    a,b = read(base),read(arm)
    if a["meta"]["set_sha"] != b["meta"]["set_sha"] or a["items"].keys() != b["items"].keys():
        raise ValueError("Score sets differ")
    groups = collections.defaultdict(list)
    for key,obs in b["items"].items():
        ref = a["items"][key]
        if obs["tokens"] != ref["tokens"]:
            raise ValueError("Score targets differ")
        groups[key.rsplit("-",1)[0]].append(obs["nll_mean"]-ref["nll_mean"])
    if len(groups) != 20 or any(len(vals)!=2 for vals in groups.values()):
        raise ValueError("Expected 20 source pairs")
    means = list(map(statistics.mean, groups.values()))
    mean = statistics.mean(means)
    spread = 2.093*statistics.stdev(means)/math.sqrt(len(means))
    return dict(items=len(a["items"]),sources=len(means),delta_nats=mean,ci95=[mean-spread,mean+spread],
                identical_scores=sum(a["items"][k]["nll_sum"]==v["nll_sum"] for k,v in b["items"].items()))


def trim(base, arm, geometry):
    c=json.loads(geometry.read_text())["head"]["fields"]
    ring=(c["sliding_window"]+16)*c["head_dim"]*2
    stages=sum(c["compress_ratios"][i]==2 for i in c["kv_source_layer_ids"])
    result={}
    for label,directory,count in (("off",base,c["num_hidden_layers"]),("on",arm,c["num_hidden_layers"]//2)):
        payload=count*ring+stages*2*64*c["head_dim"]*4
        rows=[tuple(map(int,m.groups())) for m in ledger.finditer((directory/"rank0.log").read_text())]
        if not rows:raise ValueError("Missing cache ledger")
        match=sum((n*payload)>>20==used for n,used,_,_ in rows)
        result[label]=dict(ledger_observations=len(rows),matching_geometry=match,bytes_per_entry=payload,
                           budget_MiB=sorted({r[3] for r in rows}),ideal_slots_at_512_MiB=(512<<20)//payload,
                           max_live_entries=max(r[0] for r in rows))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("quality","trim"))
    parser.add_argument("base",type=Path);parser.add_argument("arm",type=Path)
    args=parser.parse_args()
    geometry=Path(__file__).resolve().parent/"resend-fix/model-geometry.json"
    result=quality(args.base,args.arm) if args.mode=="quality" else trim(args.base,args.arm,geometry)
    print(json.dumps(result,indent=2))
