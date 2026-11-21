"""Run one controlled campaign on the two Sparks and restore the pinned production recipe."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"review-round"))
import round as trial

sample_client=here/"client/sample.py"
if not sample_client.is_file():
    sample_client=trial.clients/"sample.py"

base=trial.image
adaptive_image="tensorfold-dsv41:bench-adaptive-f7e3ead1eb9f"
prefixes=([],["ssh","-n","-o","BatchMode=yes","10.50.0.11"])
switches={"TF_DS_T13_EXPERT_PRUNE","TF_DS_T13_EXPERT_CAP","TF_DS_SCORE","TF_DS_L2_PREFETCH","TF_DS_KV_SHARD_DECODE",
          "TF_DS_FINISH_ADAPTIVE","TF_DS_FINISH_GRACE_MS",*trial.candidate}
container="dsv41-1017-exl3-tf"

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def inspect():
    ranks=[]
    for prefix in prefixes:
        c=json.loads(subprocess.check_output(prefix+["docker","inspect",container],text=True,timeout=25))[0]
        if not c["State"]["Running"]:raise RuntimeError("Rank stopped")
        cfg=c["Config"];env=dict(s.split("=",1) for s in cfg["Env"])
        cmd=cfg["Cmd"][:]; prefetch=None
        if "--prefetch" in cmd:
            pos=cmd.index("--prefetch")+1;prefetch=cmd[pos];cmd[pos]="PREFETCH"
        mounts=sorted((m["Destination"],m["Source"],m["RW"]) for m in c["Mounts"])
        spill,=[s for d,s,rw in mounts if d=="/spill"]
        rest={k:v for k,v in env.items() if k not in switches|{"TF_DS_ENGINE_ID","TF_DS_RECIPE_PATCHES"}}
        ranks.append(dict(tag=cfg["Image"],id=c["Image"],flags={k:env.get(k) for k in sorted(switches)},
                          worker_prefetch=prefetch,spill=spill,namespace=env.get("TF_DS_ENGINE_ID"),
                          common=digest([rest,cmd,[(d,s if d!="/spill" else "SPILL",rw) for d,s,rw in mounts],c["HostConfig"]["CpusetCpus"]]),
                          complete=digest([c["Image"],cfg["Cmd"],sorted(env.items()),mounts,c["HostConfig"]["CpusetCpus"]])))
    return ranks

def plans(mode):
    if mode=="adaptive":
        return [(name,{"TF_DS_FINISH_GRACE_MS":ms,"TF_DS_FINISH_ADAPTIVE":ad},["shallow","deep","exact"] if name!="five-b" else ["shallow","deep"])
                for name,ms,ad in (("five-a","5000","0"),("adaptive","5000","1"),("half","500","0"),("five-b","5000","0"))]
    if mode=="owner":
        # The exact fixture leaves large retained KV states. Restart before
        # timing so the owner and fetch workloads begin with comparable space.
        return [("owner-gate",{"KV_SHARD_DECODE":"owner"},["exact"]),
                ("owner",{"KV_SHARD_DECODE":"owner"},["decode"]),
                ("fetch-return",{"KV_SHARD_DECODE":"fetch"},["decode"])]
    return [("current-a",{},["decode","cache","quality"]),("prune-off",{"TF_DS_T13_EXPERT_PRUNE":"0"},["quality"]),
            ("cap12",{"TF_DS_T13_EXPERT_CAP":"12,1"},["quality"]),("cache-off",{"TENSORFOLD_TOKEN_CACHE_MB":"0"},["cache"]),
            ("prefetch-off",{"PREFETCH":"0"},["decode","exact"]),("current-b",{},["decode","cache","quality"])]

def event(kind,**data):
    print(json.dumps({"kind":kind,"at":time.time(),**data}),flush=True)

def run(out,mode,only,workloads,reference_items):
    trial.out=out
    initial=inspect()
    if any(r["tag"]!=base or any(r["flags"][k] for k in trial.candidate) for r in initial):
        raise RuntimeError("Expected the pinned production pair")
    event("initial",ranks=initial)
    img=adaptive_image if mode=="adaptive" else base
    ids=[subprocess.check_output(pre+["docker","image","inspect","-f","{{.Id}}",img],text=True,timeout=20).strip() for pre in prefixes]
    failures=[]
    refs=out/"items.jsonl"
    if reference_items is not None:
        frozen=reference_items.read_bytes()
        items=[json.loads(line) for line in frozen.splitlines()]
        if len(items)!=40 or len({item["id"] for item in items})!=40 or any(
                not item["prompt_ids"] or not item["continuation_ids"] for item in items):
            raise ValueError("Incomplete frozen score set")
        refs.write_bytes(frozen)
        event("reference",sha256=hashlib.sha256(frozen).hexdigest(),items=len(items))
    def part(label,argv,log):
        started=time.time()
        event("begin",job=label)
        done=threading.Event()
        def sample_clocks():
            query=["nvidia-smi","--query-gpu=clocks.gr,temperature.gpu,clocks_event_reasons.active","--format=csv,noheader"]
            with (out/(label+".clocks.jsonl")).open("x") as f:
                while not done.is_set():
                    try:
                        clocks=[subprocess.check_output(pre+query,text=True,timeout=10).strip() for pre in prefixes]
                        print(json.dumps({"at":time.time(),"ranks":clocks}),file=f,flush=True)
                    except (OSError,subprocess.SubprocessError) as exc:
                        print(json.dumps({"at":time.time(),"error":str(exc)}),file=f,flush=True)
                    done.wait(5)
        timer=threading.Thread(target=sample_clocks)
        timer.start()
        try:
            rc=trial.run([sys.executable,"-I",*map(str,argv)],log)
        finally:
            done.set()
            timer.join()
        event("end",job=label,rc=rc,elapsed=round(time.time()-started,2))
        if rc:
            failures.append(label)
            if "_error" in trial.server.metrics():
                raise RuntimeError("Engine unavailable after "+label)
        return rc
    try:
        for arm,changes,jobs in plans(mode):
            if only and arm not in only:continue
            trial.idle()
            prune=changes.get("TF_DS_T13_EXPERT_PRUNE","1");cap=changes.get("TF_DS_T13_EXPERT_CAP","0")
            scratch=f"/home/colum/.cache/dsv41-followup-{mode}-p{prune}-cap{cap.split(',')[0]}"
            if mode=="owner":
                # An exact run can persist the shared target on a different partition. Give each
                # campaign a new spill tier, and isolate correctness from the owner timing tier.
                scratch+=f"-{changes['KV_SHARD_DECODE']}-{out.name}"
                if arm=="owner-gate":
                    scratch+="-gate"
            flags={**trial.candidate,"IMAGE":img,"PREPARE":"0","SPILL_DIR":scratch,"WORKER_SPILL_DIR":scratch,
                   **({"TF_DS_SCORE":"1"} if mode in ("runtime","owner") else {}),**changes}
            trial.restart(arm,flags)
            ranks=inspect()
            for i,r in enumerate(ranks):
                if r["tag"]!=img or r["id"]!=ids[i] or r["common"]!=initial[i]["common"] or r["spill"]!=scratch:raise RuntimeError(f"Rank {i} recipe mismatch")
                for k in switches:
                    expected=flags.get(k,initial[i]["flags"][k])
                    if k=="TF_DS_L2_PREFETCH" and i==0:expected=flags.get("PREFETCH","1")
                    if k=="TF_DS_KV_SHARD_DECODE" and mode=="owner":expected=changes["KV_SHARD_DECODE"]
                    if r["flags"][k]!=expected:raise RuntimeError(f"Rank {i} {k} expected {expected} got {r['flags'][k]}")
                if i==1 and r["worker_prefetch"]!=flags.get("PREFETCH","1"):raise RuntimeError("Worker prefetch mismatch")
            event("runtime",arm=arm,ranks=ranks)
            if mode=="adaptive":
                logs=subprocess.run(["docker","logs",container],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,text=True).stdout
                if f"lane host finish grace: {flags['TF_DS_FINISH_GRACE_MS']} ms" not in logs or ("lane host adaptive finish: on" in logs)!=(arm=="adaptive"):raise RuntimeError("Scheduler banner mismatch")
            for job in jobs:
                if job=="quality" and workloads and "score" in workloads and "quality" not in workloads:
                    job="score"
                if workloads and job not in workloads:continue
                key=arm+"-"+job
                if job in ("shallow","deep","decode","cache"):
                    options={"shallow":["--finish-grace","--adaptive-mix"],"deep":["--finish-grace","--adaptive-mix","--grace-depth"],
                             "decode":["--decode-window"],"cache":["--warm-chat","--chat-depth","500000"]}[job]
                    part(key,[sample_client,key,"9",*options],key+".jsonl")
                if job in ("quality","score"):
                    if not refs.exists():
                        if part("mint",[here/"references.py",refs,"--mint",here/"reference-text.jsonl"],"mint.log"):
                            raise RuntimeError("Could not mint held-out references")
                    part(arm+"-score",["tools/score.py","record","--url",trial.server.url,"--in",refs,"--out",out/f"{arm}.score.json","--label",arm],f"{arm}.score.log")
                    if job=="quality":
                        part(arm+"-tasks",[here/"tasks.py"],f"{arm}.tasks.jsonl")
                if job=="exact":
                    record=out/f"{arm}.exact.json"
                    if not part(key+"-record",["tools/exact.py","record","--url",trial.server.url,"--like",trial.reference,"--label",arm,"--out",record],key+".record.log"):
                        part(key+"-compare",["tools/exact.py","compare",trial.reference,record],key+".compare.log")
                    if mode=="owner":
                        counters=trial.server.metrics()
                        count=counters.get("tensorfold:kv_shard_owner_rounds",0)
                        event("owner_gate",rounds=count,fetch_rounds=counters.get("tensorfold:kv_shard_fetch_rounds"),
                              passed=count>0 and not failures)
                        if count==0 or failures:
                            if not count:failures.append("owner-path-unobserved")
                            break
            if mode=="adaptive" and arm=="adaptive":
                logs=subprocess.run(["docker","logs",container],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,text=True).stdout
                evidence=[l for l in logs.splitlines() if "adaptive finish extension:" in l]
                (out/"extensions.log").write_text("\n".join(evidence)+"\n")
                event("extensions",count=len(evidence))
            # An intentional lossy prune may change saved reference replies; retain outcomes as part of the comparison.
            part(arm+"-validate",["tools/validate.py",trial.server.url,"--model",trial.server.model,"--save",out/f"{arm}.validate.json","--compare","runs/matrix/base.validate.json"],f"{arm}.validate.log")
            if mode=="owner" and failures:
                break
    finally:
        if "_error" not in trial.server.metrics():trial.idle()
        trial.restart("production",{"PREPARE":"0"})
        final=inspect()
        if final!=initial:raise RuntimeError("Production restoration mismatch")
        trial.validate("production");event("restored",ranks=final)
    event("complete",failures=failures)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("runtime","adaptive","owner"));p.add_argument("out",type=Path)
    p.add_argument("--only",nargs="+");p.add_argument("--workloads",nargs="+")
    p.add_argument("--reference-items",type=Path,help="Reuse the exact tokenized references from a previous campaign")
    a=p.parse_args();out=a.out.resolve()
    with (here/".pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(exist_ok=False)
        def stop(sig,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        run(out,a.mode,a.only,a.workloads,a.reference_items)
