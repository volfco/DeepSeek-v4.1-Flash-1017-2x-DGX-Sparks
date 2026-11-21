"""Aggregate time and work in the recorded external interval from completed-request fields only."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

here=Path(__file__).resolve().parent
archive=here.parent/"overnight/break-1600Z.rank0.log"
groups=defaultdict(list)
events=[]
for line in archive.open():
    if line[:19]<"2026-11-13T07:20:00":continue
    if '127.0.0.1 "POST /v1/chat/completions' in line:raise ValueError("Local chat traffic in cohort")
    if "[tensorfold] done " not in line:continue
    fields=dict(re.findall(r"\b(\w+)=([^\s]+)",line))
    q={k:float(fields[k]) for k in ("prompt","cached","tokens","prefill_s","ttft_s","decode_s","rounds")}
    done=datetime.fromisoformat(line.split()[0].replace("Z","+00:00")).timestamp()
    first=done-q["decode_s"]
    events.extend([(first,1),(done,-1)])
    groups["all"].append(q)
    groups["under_8k" if q["prompt"]<8192 else "8k_to_100k" if q["prompt"]<100000 else "100k_to_250k" if q["prompt"]<250000 else "250k_and_over"].append(q)
    groups["zero_cache_cut" if q["cached"]==0 else "positive_cache_cut"].append(q)

result={}
for label,rows in groups.items():
    totals={k:round(sum(r[k] for r in rows),3) for k in rows[0]}
    elapsed=totals["ttft_s"]+totals["decode_s"]
    result[label]=dict(requests=len(rows),totals=totals,request_seconds=elapsed,
                       decode_share_pct=100*totals["decode_s"]/elapsed,prefill_share_pct=100*totals["prefill_s"]/elapsed,
                       reported_ttft_minus_prefill_s=round(totals["ttft_s"]-totals["prefill_s"],3),
                       reused_input_token_pct=100*totals["cached"]/totals["prompt"])
active=0
previous=None
width_seconds=defaultdict(float)
for ts,delta in sorted(events):
    if previous is not None and active:
        width_seconds[active]+=ts-previous
    active+=delta
    previous=ts
if active or len(groups["all"])!=652:raise ValueError("Cohort mismatch")
print(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),cohorts=result,
                      estimated_clock_seconds_by_simultaneously_decoding=width_seconds),indent=2))
