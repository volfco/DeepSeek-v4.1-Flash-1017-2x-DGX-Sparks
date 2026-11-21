"""Summarize a known single- or multi-rank workload interval in head UTC nanoseconds."""
import argparse
from bisect import bisect_left
import datetime
import json
from pathlib import Path
from functools import lru_cache

from bandwidth import series, window

@lru_cache(maxsize=2)
def readings(directory,rank):
    pm=series(directory/f"{rank}-pm.csv")
    io=[json.loads(line) for line in (directory/f"{rank}-io.jsonl").open()]
    return pm,io

def bounds(directory):
    sync=json.loads((directory/"sync.json").read_text())
    skew=min(sync["before"]+sync["after"],key=lambda x:x["uncertainty_ns"])
    limits=[]
    for rank in ("head","worker"):
        pm,io=readings(directory,rank)
        shift=skew["worker_minus_head_ns"] if rank=="worker" else 0
        limits.append((max(pm[0][0],io[0]["t_ns"])-shift,min(pm[-1][1],io[-1]["t_ns"])-shift))
    return max(v[0] for v in limits),min(v[1] for v in limits)


def report(directory, start, stop):
    sync=json.loads((directory/"sync.json").read_text())
    skew=min(sync["before"]+sync["after"],key=lambda x:x["uncertainty_ns"])
    ret={}
    for rank in ("head","worker"):
        shift=skew["worker_minus_head_ns"] if rank=="worker" else 0
        lo,hi=start+shift,stop+shift
        counters,io=readings(directory,rank)
        pm=window(counters,lo,hi)
        times=[r["t_ns"] for r in io]
        if times[0]>lo or times[-1]<hi:
            raise ValueError("Counter capture does not cover this I/O window")
        first,last=(io[bisect_left(times,t)] for t in (lo,hi))
        rates={key:(last["bytes"][key]-first["bytes"][key])/(last["t_ns"]-first["t_ns"]) for key in first["bytes"]}
        if not pm or min(m["coverage"] for m in pm.values())<.98:
            raise ValueError("Insufficient sampled performance-counter coverage")
        ret[rank]={"pm":pm,"device_GB_s":rates,"IO_window_s":(last["t_ns"]-first["t_ns"])/1e9}
    return {"start":start,"end":stop,"seconds":(stop-start)/1e9,"clock":skew,"ranks":ret}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("start_utc")
    p.add_argument("end_utc")
    a=p.parse_args()
    def stamp(s):
        d=datetime.datetime.fromisoformat(s)
        if d.tzinfo is None:
            raise ValueError("Supply an explicit UTC offset")
        return round(d.timestamp()*1e9)
    print(json.dumps(report(a.directory,stamp(a.start_utc),stamp(a.end_utc)),indent=2,allow_nan=False))
