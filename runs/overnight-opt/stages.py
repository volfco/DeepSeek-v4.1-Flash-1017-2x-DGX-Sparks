"""Aggregate only complete calibrated L2 sector samples inside instrumented phase boundaries."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import struct
import sys

from bandwidth import series

decode="start embed engram mix_attn attn gather_a mix_ffn moe gather_m post head a_in a_q a_idx a_core a_woa a_x m_rt m_gu m_x sh_fetch sh_rows sh_q sh_own sh_qwait sh_peer sh_out sh_owait sh_sel".split()
prefill="start embed engram mix_attn attn gather_a mix_ffn moe gather_m head a_in a_q a_idx a_core a_woa m_rt d_rot d_unpack d_gemm e_gu e_down".split()


def prepare(directory, rank):
    meter=directory/"meter"
    samples=series(meter/f"{rank}-pm.csv")
    samples=[r for r in samples if math.isfinite(r[3]) and 0<r[1]-r[0]<1e7]
    starts=[r[0] for r in samples]
    ends=[r[1] for r in samples]
    if starts!=sorted(starts) or ends!=sorted(ends):raise ValueError("Unordered counter frames")
    sectors=[0.];ns=[0]
    for a,b,_,value in samples:
        sectors.append(sectors[-1]+value);ns.append(ns[-1]+b-a)
    meta=[json.loads(l) for l in (meter/f"{rank}-io.jsonl").open()]
    clocks=[m["mono_offset_ns"] for m in meta if "mono_offset_ns" in m]
    if clocks:
        offset=round(statistics.median(clocks))
    elif rank=="head":
        offset=round(statistics.mean(json.loads((directory/"clock.json").read_text()))*1e9)
    else:raise ValueError("No worker monotonic offset")
    trace=(directory/f"rank{0 if rank=='head' else 1}.stages.bin").read_bytes()
    return starts,ends,sectors,ns,offset,trace


def summarize(directory, rank, guard, lo, hi, prepared=None):
    starts,ends,sectors,ns,offset,trace=prepared or prepare(directory,rank)
    lower=max(lo,starts[0]);upper=min(hi,ends[-1])
    agg=defaultdict(lambda:[0,0,0,0.,0])
    excursions=[]
    last_decode=None
    for s,e,kind,phase in struct.iter_unpack("<QQII",trace):
        if kind==4:continue  # Shape metadata are consumed by shapes.py.
        s+=offset;e+=offset
        if kind==3:
            if lower<s<e<upper and last_decode:
                excursions.append(e-last_decode)
            continue
        if kind==0:last_decode=e
        if not lower<s<e<upper:continue
        if kind not in (0,1,2):raise ValueError("Invalid phase kind")
        label=("decode","prefill","prefill-detail")[kind],(decode if kind==0 else prefill)[phase]
        row=agg[label]
        row[0]+=1;row[1]+=e-s
        start=bisect_left(starts,s+guard);stop=bisect_right(ends,e-guard)
        if stop>start:
            row[2]+=ns[stop]-ns[start];row[3]+=32*(sectors[stop]-sectors[start]);row[4]+=stop-start
    if excursions:
        print(f"{rank} decode anchor end residual p50={statistics.median(excursions)/1e3:.1f} us, p95={sorted(excursions)[int(.95*len(excursions))]/1e3:.1f} us",file=sys.stderr)
    return [dict(rank=rank,kind=k[0],stage=k[1],events=v[0],phase_ms=v[1]/1e6,sampled_ms=v[2]/1e6,samples=v[4],
                 coverage=v[2]/v[1],sample_GB_s=v[3]/v[2] if v[2] else None) for k,v in sorted(agg.items())]

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    p.add_argument("--rank",choices=("head","worker"),default="head");p.add_argument("--guard-us",type=float,default=10)
    p.add_argument("--start",type=float,default=0);p.add_argument("--end",type=float,default=1e12)
    a=p.parse_args();rows=summarize(a.directory,a.rank,a.guard_us*1e3,a.start*1e9,a.end*1e9)
    if rows:
        out=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
