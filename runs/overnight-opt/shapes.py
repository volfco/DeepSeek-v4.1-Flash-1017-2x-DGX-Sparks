"""Join shape records to the next timed operation and accumulate useful matrix FLOPs."""
import argparse
import collections
import csv
from pathlib import Path
import struct
import sys

# Prefill phase indices from the recorded build's prefill_prof.zig.
shapes = {18:("dense",2),19:("expert_gate_up",4),20:("expert_down",2)}


def collect(path, lo=0, hi=2**64):
    pending = None
    rows = collections.defaultdict(lambda:[0,0,0])
    for a,b,kind,ph in struct.iter_unpack("<QQII", path.read_bytes()):
        if kind==4:
            if pending or ph not in shapes:
                raise ValueError("Unexpected shape metadata")
            pending = ph, a>>32, a&0xffffffff, b
        elif pending and kind==2:
            key,m,k,n = pending
            if ph != key:
                raise ValueError("Shape and phase disagree")
            name,factor = shapes[ph]
            if lo<=a<b<=hi:
                vals=rows[name,m,k,n]
                vals[0]+=1;vals[1]+=b-a;vals[2]+=factor*m*k*n
            pending=None
    if pending or not rows:
        raise ValueError("Incomplete shape/phase trace")
    return [dict(stage=s,m=m,k=k,n=n,calls=v[0],elapsed_ms=v[1]/1e6,
                 useful_flop=v[2],effective_TFLOP_s=v[2]/v[1]/1000)
            for (s,m,k,n),v in sorted(rows.items())]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    rows=collect(parser.parse_args().trace)
    out=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]))
    out.writeheader();out.writerows(rows)
