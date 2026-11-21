"""Compare identity, actual durable restores and residency when the 84-session crossover exceeds 80 slots."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics


def arm(directory):
    log=(directory/"rank0.log").read_text()
    done={m[1]:dict(re.findall(r"([\w]+)=([^\s]+)",m[2])) for m in re.finditer(r"done (\S+) ([^\n]+)",log)}
    held=[(int(a),int(b),int(c)) for a,b,c in re.findall(r"(\d+) states, (\d+) MiB .*?evicted (\d+), refused",log)]
    windows=json.loads((directory/"counter-windows.json").read_text())
    raw=[json.loads(l) for l in (directory/"requests.jsonl").open()]
    rows={(r["mode"],r["session"]):dict(r,done=done[r["request"]]) for r in raw}
    if len(rows)!=96 or len(held)<96:raise ValueError("Incomplete request or cache records")
    totals={}
    for mode in ("seed","revisit"):
        subset=[v for (shape,_),v in rows.items() if shape==mode]
        before,after=windows[mode+"_before"],windows[mode+"_after"]
        totals[mode]=dict(requests=len(subset),request_s=sum(r["wall_s"] for r in subset),
                          median_request_s=statistics.median(r["wall_s"] for r in subset),
                          observed_restores=sum(float(r["done"]["restore_s"])>0 for r in subset),
                          restore_s=sum(float(r["done"]["restore_s"]) for r in subset),
                          cuts=dict(collections.Counter(r["cut"] for r in subset)),
                          counter_delta={k:after[k]-v for k,v in before.items() if "hit" in k or "evict" in k or "restore" in k})
    return rows,dict(max_states=max(r[0] for r in held),max_charged_MiB=max(r[1] for r in held),
                     last_ledger_evicted=held[-1][2],**totals)


def compare(directory):
    a,off=arm(directory/"trim-off");b,on=arm(directory/"trim-on")
    if a.keys()!=b.keys():raise ValueError("Unmatched trim sessions")
    if any(a[k]["input_sha"]!=b[k]["input_sha"] for k in a):raise ValueError("Mismatched input")
    matches=sum(a[k]["done"]["sha"]==b[k]["done"]["sha"] and a[k]["output_tokens"]==b[k]["output_tokens"] for k in a)
    revisits=[]
    for key in a:
        if key[0]!="revisit":continue
        left,right=a[key],b[key]
        revisits.append(dict(session=key[1],cut_off_on=[left["cut"],right["cut"]],
                             wall_s_off_on=[left["wall_s"],right["wall_s"]],
                             restore_s_off_on=[float(r["done"]["restore_s"]) for r in (left,right)],
                             prefill_s_off_on=[float(r["done"]["prefill_s"]) for r in (left,right)]))
    return dict(inputs=len(a),matching_output=matches,off=off,on=on,revisits=revisits,
                revisit_seconds_saved=off["revisit"]["request_s"]-on["revisit"]["request_s"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    print(json.dumps(compare(p.parse_args().directory),indent=2))
