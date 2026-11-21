"""Collect calibrated L2 read misses, RDMA port bytes and NVMe sectors on both ranks."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
peer=["ssh","-n","-o","BatchMode=yes","10.50.0.11"]
metric="lts__t_sectors_op_read_lookup_miss_realtime.sum"


def offsets():
    samples=[]
    # Reuse the SSH connection. The first exchange includes setup; later exchanges bound skew.
    cmd = ["ssh","-o","BatchMode=yes",peer[-1],"python3 -u -c 'import sys,time; [print(time.time_ns(), flush=True) for line in sys.stdin]'"]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as p:
        for _ in range(8):
            a=time.time_ns()
            p.stdin.write("t\n")
            p.stdin.flush()
            other=int(p.stdout.readline())
            b=time.time_ns()
            samples.append({"worker_minus_head_ns":other-(b+a)//2,"uncertainty_ns":(b-a)//2})
        p.stdin.close()
    return samples


def record(command, dest):
    with dest.open("x") as out, dest.with_suffix(".err").open("x") as err:
        r=subprocess.run(command,stdout=out,stderr=err,timeout=args.seconds+60)
    if r.returncode:
        raise RuntimeError(f"{dest} exited {r.returncode}: {dest.with_suffix('.err').read_text()}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--seconds",type=int,default=90,choices=range(5,901),metavar="5..900")
    parser.add_argument("--interval-cycles",type=int,default=2400000)
    args=parser.parse_args()
    args.out.mkdir()
    before=offsets()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        jobs=[]
        for rank,prefix,binpath,links in [("head",[],"/tmp/dsv41-pmc-1118/pm_sampling",str(here/"links.py")),
                                        ("worker",peer,"/tmp/pm_sampling","/tmp/links.py")]:
            command=prefix+["sudo","-n",f"COUNTER_SECONDS={args.seconds}",binpath,"-i",str(args.interval_cycles),"-b","16777216","-s","10000","-m",metric]
            jobs.append(pool.submit(record,command,args.out/f"{rank}-pm.csv"))
            jobs.append(pool.submit(record,prefix+["python3",links,"--seconds",str(args.seconds+1)],args.out/f"{rank}-io.jsonl"))
        for job in jobs:
            job.result()
    (args.out/"sync.json").write_text(json.dumps({"before":before,"after":offsets()},indent=2)+"\n")
