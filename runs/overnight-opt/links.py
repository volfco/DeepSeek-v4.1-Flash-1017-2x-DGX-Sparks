"""Sample the rank's RDMA port counters and NVMe reads in Linux sysfs."""
import argparse
import json
from pathlib import Path
import time


def sensors():
    paths = {}
    for entry in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        if not (entry/"state").read_text().strip().endswith("ACTIVE"):
            continue
        device = entry.parents[1].name
        for direction in ("rcv", "xmit"):
            path = entry/"counters"/f"port_{direction}_data"
            if path.is_file():
                paths[f"{device}.{entry.name}.{direction}"] = path
    nvme = {p.parent.name: p for p in Path("/sys/block").glob("nvme*n*/stat")}
    return paths, nvme


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=2)
    parser.add_argument("--interval",type=float,default=.2)
    args=parser.parse_args()
    if args.interval <= 0:parser.error("A positive sampling interval is required")
    ib, disks=sensors()
    finish=time.monotonic()+args.seconds
    while time.monotonic()<finish:
        start=time.time_ns()
        values={key:int(path.read_text())*4 for key,path in ib.items()}
        disk_activity={}
        for device,path in disks.items():
            stats=path.read_text().split()
            values[device+".read"]=int(stats[2])*512
            values[device+".write"]=int(stats[6])*512
            disk_activity[device]=dict(read_ios=int(stats[0]),read_ms=int(stats[3]),write_ios=int(stats[4]),
                                       write_ms=int(stats[7]),active_ms=int(stats[9]),weighted_ms=int(stats[10]))
        print(json.dumps(dict(t_ns=(start+time.time_ns())//2,mono_offset_ns=time.time_ns()-time.monotonic_ns(),bytes=values,disk=disk_activity)),flush=True)
        time.sleep(args.interval)
