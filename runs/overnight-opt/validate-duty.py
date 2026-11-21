"""Check NVML memory activity against the timestamped stream scans on each board."""
import argparse
import csv
from pathlib import Path
import sys

from duty import telemetry


def validation(directory, rank):
    values = telemetry(directory/"serving-duty"/f"{rank}-nvml.csv")
    with (directory/f"{rank}-nvml-scan.csv").open() as f:
        scans = [(int(start)/1e9,int(end)/1e9,int(size)) for label,start,end,size in csv.reader(f) if label=="read"]
    if len(scans) != 6:
        raise ValueError(f"{rank} has {len(scans)} scans")
    for i, (a,b,size) in enumerate(scans):
        rows = [r for r in values if a+.10 < r[0] < b]
        if not rows:
            raise ValueError(f"NVML missed {rank} scan {i}")
        yield dict(rank=rank,scan=i+1,seconds=b-a,known_read_GB_s=size/(b-a)/1e9,observations=len(rows),
                   max_gpu_busy_pct=max(r[1] for r in rows),max_reported_memory_busy_pct=max(r[2] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--ranks",nargs="+",default=["head","worker"],choices=["head","worker"])
    args = p.parse_args()
    rows = [r for rank in args.ranks for r in validation(args.directory,rank)]
    writer = csv.DictWriter(sys.stdout,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
