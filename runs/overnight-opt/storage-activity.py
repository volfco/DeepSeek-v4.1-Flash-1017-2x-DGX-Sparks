"""Match Linux block request counters and two-rail bytes to completed engine intervals."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from pathlib import Path
import sys
import zlib

from duty import intervals


def load(path):
    # The last gzip block can still be in flight when the overnight checkpoint is taken.
    text = (zlib.decompressobj(31).decompress(path.read_bytes()).decode() if path.exists()
            else path.with_suffix("").read_text())
    rows = [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith("\n")]
    return [r["t_ns"]/1e9 for r in rows], rows


def summarize(directory, telemetry):
    for c,phase,start,end in intervals(directory):
        for rank,(times,rows) in telemetry.items():
            i = bisect_left(times, start+.25)
            j = bisect_right(times, end-.25)-1
            if j-i < 2:
                continue
            a,b = rows[i],rows[j]
            seconds = times[j]-times[i]
            for drive,stats in a["disk"].items():
                change = {k:b["disk"][drive][k]-v for k,v in stats.items()}
                diff = {k:b["bytes"][k]-v for k,v in a["bytes"].items()}
                count = change["read_ios"]
                yield dict(rank=rank,phase=phase,depth=c["depth"],workload=c["sharing"],width=len(c["requests"]),
                           rep=c["rep"],disk=drive,seconds=seconds,
                           read_MB_s=diff[drive+".read"]/seconds/1e6,read_IOPS=count/seconds,
                           read_KiB=diff[drive+".read"]/count/1024 if count else None,
                           mean_read_ms=change["read_ms"]/count if count else None,
                           mean_outstanding_reads=change["read_ms"]/seconds/1000,
                           all_io_busy_pct=change["active_ms"]/seconds/10,
                           read_write_depth=change["weighted_ms"]/seconds/1000,
                           RDMA_xmit_GB_s=sum(v for k,v in diff.items() if k.endswith(".xmit"))/seconds/1e9,
                           RDMA_recv_GB_s=sum(v for k,v in diff.items() if k.endswith(".rcv"))/seconds/1e9)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path);p.add_argument("telemetry",type=Path)
    args=p.parse_args()
    rank_data={r:load(args.telemetry/f"{r}.jsonl.gz") for r in ("head","worker")}
    result=list(summarize(args.directory,rank_data))
    if not result:raise SystemExit("No overlapping completed I/O windows")
    w=csv.DictWriter(sys.stdout,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
