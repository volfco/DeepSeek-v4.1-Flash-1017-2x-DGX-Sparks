"""Run the native AIO read reference on the same allocated checkpoint on each idle rank."""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import time


def local_rank():
    config=json.loads(subprocess.check_output(["docker","inspect","dsv41-1017-exl3-tf"]))[0]
    root=Path(next(m["Source"] for m in config["Mounts"] if m["Destination"]=="/engram"))
    model=max(root.glob("*.safetensors"),key=lambda f:f.stat().st_blocks)
    print(json.dumps(dict(file=model.name,allocated_bytes=model.stat().st_blocks*512)),file=sys.stderr,flush=True)
    subprocess.run(["/tmp/dsv41-storage-aio",str(model)],check=True,timeout=35)


def both(out):
    import buckets
    trial=buckets.driver.trial
    trial.idle()
    key="tensorfold:request_latency_seconds_count"
    baseline=trial.server.metrics().get(key)
    if baseline is None:raise ValueError("Missing serving request count")
    jobs=[]
    start=time.time()
    with contextlib.ExitStack() as resources:
        for rank,cmd in (("head",[sys.executable,str(Path(__file__).resolve())]),
                         ("worker",["ssh","-n","-o","BatchMode=yes","10.50.0.11","python3","/tmp/dsv41-drive-depth.py"])):
            output=resources.enter_context((out/f"{rank}-aio.csv").open("x"))
            errors=resources.enter_context((out/f"{rank}-aio.err").open("x"))
            jobs.append(subprocess.Popen([*cmd,"--rank"],stdout=output,stderr=errors))
        codes=[p.wait(timeout=50) for p in jobs]
    obs=trial.server.metrics()
    record=dict(start_utc=start,end_utc=time.time(),codes=codes,completed_before=baseline,completed_after=obs.get(key),
                running=obs.get("tensorfold:requests_running"),waiting=obs.get("tensorfold:requests_waiting"))
    (out/"aio-isolation.json").write_text(json.dumps(record,indent=2)+"\n")
    if any(codes) or record["completed_after"]!=baseline or record["running"]!=0 or record["waiting"]!=0:
        raise RuntimeError("Reference failed isolation or either rank exited")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument("--rank",action="store_true");g.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.rank:local_rank()
    else:both(args.out)
