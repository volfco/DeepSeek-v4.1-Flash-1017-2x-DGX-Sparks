"""Align CUPTI ranges with UNIX-time windows and integrate complete or fractional sector counts."""
import csv
import gzip
import json
import math
from pathlib import Path
import sys


def series(path):
    rows=[]
    offset=None
    reader=path.open() if path.is_file() else gzip.open(str(path)+".gz","rt")
    with reader as f:
        for line in csv.reader(f):
            if not line:
                continue
            if line[0]=="anchor":
                offset=int(line[2])-int(line[1])
            elif line[0].isdigit() and len(line)==4:
                lo,hi,metric,count=line
                rows.append((int(lo),int(hi),metric,float(count)))
    if offset is None or not rows:
        raise ValueError(f"No counter data in {path}")
    return [(lo+offset,hi+offset,metric,count) for lo,hi,metric,count in rows]


def window(rows,start,end):
    totals={}
    durations={}
    for lo,hi,metric,count in rows:
        if hi<=lo or end<=lo or hi<=start or not math.isfinite(count):
            continue
        span=min(hi,end)-max(lo,start)
        totals[metric]=totals.get(metric,0)+32*count*span/(hi-lo)
        durations[metric]=durations.get(metric,0)+span
    return {key:{"bytes":b,"GB_s":b/durations[key],"coverage":durations[key]/(end-start)} for key,b in totals.items()}


def calibration(d):
    _,lo,hi,known=next(csv.reader((d/"scan.csv").open()))
    lo,hi,known=map(int,(lo,hi,known))
    rows=series(d/"counters.csv")
    measured=window(rows,lo,hi)
    return {"scan_GB_s":known/(hi-lo),"seconds":(hi-lo)*1e-9,"scan_bytes":known,"metrics":measured,
            "coverage_range_ns":[min(r[0] for r in rows),max(r[1] for r in rows)],"scan_range_ns":[lo,hi],
            "recovered_fraction":{k:m["bytes"]/known for k,m in measured.items()}}


if __name__=="__main__":
    print(json.dumps(calibration(Path(sys.argv[1])),indent=2))
