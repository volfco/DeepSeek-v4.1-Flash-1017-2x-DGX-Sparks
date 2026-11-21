"""Run held-wait 10s / 500ms / 10s on separate serving processes and a private spill tier."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

here=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import pressure as p

switch="TF_DS_KV_HELD_WAIT_MS"
image="tensorfold-dsv41:bench-held-9a5e38c42edd"
name="dsv41-1017-exl3-tf"
ranks=([],["ssh","-n","-o","BatchMode=yes","10.50.0.11"])
pattern=re.compile(r"kv held trial ms=(\d+) elapsed_ms=(\d+) part=(\d+) at=(\d+) patient=(true|false) outcome=(full|resumed|dropped)")


def identity():
    result=[]
    for rank in ranks:
        d=json.loads(subprocess.check_output(rank+["docker","inspect",name],text=True,timeout=30))[0]
        if not d["State"]["Running"]:
            raise RuntimeError("Rank stopped")
        env=dict(s.split("=",1) for s in d["Config"]["Env"])
        mount=sorted((m["Source"],m["Destination"],m["RW"]) for m in d["Mounts"])
        spill,=[s for s,t,r in mount if t=="/spill"]
        skip={switch,"TF_DS_ENGINE_ID","TF_DS_RECIPE_PATCHES"}
        key=[d["Config"]["Cmd"],sorted((k,v) for k,v in env.items() if k not in skip),
             [(s if t!="/spill" else "SPILL",t,r) for s,t,r in mount],d["HostConfig"]["CpusetCpus"]]
        common=hashlib.sha256(json.dumps(key).encode()).hexdigest()
        full=hashlib.sha256(json.dumps([d["Image"],d["Config"]["Cmd"],sorted(env.items()),mount]).encode()).hexdigest()
        result.append({"image":d["Config"]["Image"],"image_id":d["Image"],"ms":env.get(switch),"spill":spill,
                       "common":common,"digest":full})
    return result


def policy_rows(prompts,cases,arm,partition):
    c=p.client
    c.args=argparse.Namespace(arm=arm,salt="5",occupancy=True,replay_depth=False,finish_grace=False)
    if sum(p.rows())>32768 or min(c.memory())<5.2:
        raise RuntimeError("Invalid initial occupancy or memory")
    target, fillers = ("f0", ("f3", "shared", "f1")) if partition else ("shared", ("f0", "f1", "f2"))
    for key in (target, *fillers):
        p.probe(key,"fill",0,cases[key])
        if partition and key == target:
            held = p.rows()
            if held[0] > 32768 or held[1] != p.grid:
                raise RuntimeError(f"Unexpected target partition {held}")
        if key!=target:
            p.probe(key,"pre-pressure",0,cases[key],p.grid)
    peak=p.rows()
    if min(peak)<p.grid or not 4*p.grid<=sum(peak)<=4*p.grid+32768:
        raise RuntimeError(f"Unexpected peak occupancy {peak}")
    if partition and not (3*p.grid <= peak[0] <= 3*p.grid+32768 and peak[1] == p.grid):
        raise RuntimeError(f"Could not establish three reservations on part 0: {peak}")
    c.emit("stage",stage="peak",rows=peak)
    since=str(time.time())
    reqs=c.cell(300000,"admission",prompts,list(range(4200,4204)),budget=1024,
                rep=0,sharing="peak")
    result=[(r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in reqs]
    log=subprocess.run(["docker","logs","--since",since,name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,check=True).stdout
    events=[dict(ms=int(ms),elapsed_ms=int(dt),part=int(part),at=int(at),patient=pat=="true",outcome=out)
            for ms,dt,part,at,pat,out in pattern.findall(log)]
    c.emit("patient",events=events)
    if not events:
        raise RuntimeError("No held-state placement records for measured burst")
    for key in fillers:
        p.probe(key,"after-pressure",0,cases[key])
    return result,peak


def main(dest, partition, reuse_spill):
    p.trial.out=dest
    dest.mkdir(parents=True,exist_ok=False)
    p.client.health=p.health
    p.client.args=argparse.Namespace(arm="held-trial",salt="5",occupancy=True,replay_depth=False,finish_grace=False)
    c=p.client
    baseline=identity()
    start=p.trial.identity()
    if any(flags for _,flags in start):
        raise RuntimeError("Unexpected original serving flags")
    ids=[subprocess.check_output(r+["docker","image","inspect","-f","{{.Id}}",image],text=True).strip() for r in ranks]
    prompts,cases=p.history()
    if partition:
        # The balanced run recorded f0 on part 1 and shared/f1 on part 0 in this spill namespace.
        # Allocate f3 after the sole part-1 restore to put a third independent state on part 0.
        cases["f3"] = "[review-partition-followup-3]\n" + cases["f0"].split("\n",1)[1]
        prompts = [cases["f0"].removesuffix("Summarize the preceding context in detail.")+task for task in c.tasks]
    c.emit("start",ranks=baseline,partition=partition)
    # This pair-only scratch namespace persists the four trial prefixes across arms.
    # An optional run-specific suffix lets the whole campaign be independently repeated.
    scratch=str(pathlib.Path.home()/".cache"/("dsv41-held-wait-"+(reuse_spill or dest.name)))
    result=None
    first_peak=None
    def stop(signum,frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        arms = (("earlier","500"),("control-b","10000")) if partition else (("control-a","10000"),("earlier","500"),("control-b","10000"))
        for arm,ms in arms:
            p.trial.idle()
            p.trial.restart(arm,{"IMAGE":image,"PREPARE":"0","SPILL_DIR":scratch,"WORKER_SPILL_DIR":scratch,switch:ms})
            records=identity()
            for rank,(actual,orig) in enumerate(zip(records,baseline)):
                if (actual["image_id"]!=ids[rank] or actual["common"]!=orig["common"] or actual["ms"]!=ms or
                    actual["spill"]!=scratch):
                    raise RuntimeError(f"Rank {rank} runtime mismatch")
            logs=subprocess.run(["docker","logs",name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout
            if "kv held trial configured ms="+ms not in logs:
                raise RuntimeError("Missing startup acknowledgement")
            c.emit("runtime",arm_name=arm,ranks=records)
            got,peak=policy_rows(prompts,cases,arm,partition)
            if result is None:
                result,first_peak=got,peak
            else:
                c.emit("matched",arm_name=arm,input_equal=all(a[0]==b[0] for a,b in zip(result,got)),
                       cached_equal=all(a[1]==b[1] for a,b in zip(result,got)),
                       length_equal=all(a[2]==b[2] for a,b in zip(result,got)),
                       output_equal=all(a[3]==b[3] for a,b in zip(result,got)),peak_equal=first_peak==peak)
                if first_peak!=peak or any(a[0]!=b[0] for a,b in zip(result,got)):
                    raise RuntimeError("Start occupancy or request inputs differ")
            if partition and arm=="earlier":
                t=p.trial
                record=dest/(arm+".exact.json")
                if t.run([sys.executable,"-I","tools/exact.py","record","--url",t.server.url,"--like",t.reference,"--label",arm,"--out",str(record)],arm+".exact-record.log"):
                    raise RuntimeError("Armed exact record failed")
                if t.run([sys.executable,"-I","tools/exact.py","compare",t.reference,str(record)],arm+".exact-compare.log"):
                    raise RuntimeError("Armed exact comparison failed")
                c.emit("exact",arm_name=arm,passed=True)
            p.trial.validate(arm)
    finally:
        p.trial.idle()
        p.trial.restart("production",{"PREPARE":"0"})
        if identity()!=baseline:
            raise RuntimeError("Restoration differs from original")
        p.trial.validate("production")
        p.trial.idle()
        c.emit("restored",ranks=identity(),health=p.health(),memory_gib=c.memory())
    c.emit("complete")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=pathlib.Path)
    parser.add_argument("--partition",action="store_true",help="Use the part-1 shared history, three part-0 fillers and a reversed wait comparison")
    parser.add_argument("--reuse-spill",help="Prior campaign directory name whose private spill contains the measured fixtures")
    args=parser.parse_args()
    if args.reuse_spill and pathlib.Path(args.reuse_spill).name != args.reuse_spill:
        parser.error("Reuse-spill expects a single campaign name")
    if args.partition and not args.reuse_spill:
        parser.error("The partition experiment needs the balanced campaign's observed spill assignments")
    # The other followup modes own this same head-local file during their run.
    lock=(here.parent/".pair.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    main(args.output.resolve(),args.partition,args.reuse_spill)
