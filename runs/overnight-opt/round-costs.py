"""Recover verifier and drafter timings from cumulative coherent frames during continuous decode."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


def rows(directory):
    all_frames=[json.loads(l) for l in (directory/"frames.jsonl").open()]
    offset=statistics.mean(json.loads((directory/"clock.json").read_text()))
    for line in (directory/"workload.jsonl").open():
        c=json.loads(line)
        if c["kind"]!="cell" or c["cell"]=="chat-prime":continue
        req=c["requests"]
        lo=max(r["first"] for r in req)+offset+.4
        hi=min(r["last"] for r in req)+offset-.4
        frames=[f for f in all_frames if lo<f["at"]<hi and f["profile"]["rounds"]==f["clock_rounds"]]
        if len(frames)<2:continue
        a,b=frames[0],frames[-1];p,q=a["profile"],b["profile"]
        count=q["rounds"]-p["rounds"]
        if count<10:continue
        result=dict(width=len(req),rounds=count,passes=q["passes"]-p["passes"],
                    wall_ms_per_round=1000*(b["at"]-a["at"])/count)
        for k in ("rows_a_round","forward_ms","enqueue_ms","absorb_ms","logits_ms","sample_ms","pass_ms","pass_send_ms"):
            denom="passes" if k.startswith("pass_") else "rounds"
            result[k+"_per_round"]=(q[k]*q[denom]-p[k]*p[denom])/count
        yield result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    data=list(rows(p.parse_args().directory))
    writer=csv.DictWriter(sys.stdout,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
