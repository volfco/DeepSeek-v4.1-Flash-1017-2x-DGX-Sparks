"""Compare exact-resume off/on on its supported, unsharded, unsliced, memory-only serving configuration."""
import argparse
import contextlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import buckets
import screen
from buckets import driver,write_json


def health(client):
    with client.bench.OPENER.open(client.srv.url+"/health",timeout=10) as reply:
        h=json.load(reply)
    return {"status":h["status"],"pair":h["pair"],"live":h.get("live"),"per_part":[],
            "refused_no_target":0,"kv_counts":{}}


def inspect(image, exact):
    result=[]
    for pre in driver.prefixes:
        d=json.loads(subprocess.check_output(pre+["docker","inspect",driver.container]))[0]
        env=dict(s.split("=",1) for s in d["Config"]["Env"])
        mounts=sorted(m["Destination"] for m in d["Mounts"])
        cmd=d["Config"]["Cmd"]
        expected=subprocess.check_output(pre+["docker","image","inspect","-f","{{.Id}}",image],text=True).strip()
        if (d["Image"]!=expected or "/spill" in mounts or any(arg.startswith("--spill-") for arg in cmd)
                or env.get("TF_DS_KV_SHARD","0") not in ("0","") or env.get("TF_DS_PREFILL_SLICE")!="0"
                or env.get("TF_DS_EXACT_RESUME")!=str(exact) or env.get("TF_DS_SCORE")!="1"):
            raise ValueError(f"Unsupported exact-resume rank settings {cmd}, mounts={mounts}")
        result.append(dict(id=d["Image"],image=image,cmd=cmd,mounts=mounts,
                           env={k:env.get(k) for k in ("TF_DS_WINDOW","TF_DS_POOL","TF_DS_KV_SHARD","TF_DS_PREFILL_SLICE","TF_DS_EXACT_RESUME","TF_DS_SCORE")}))
    return result


def installed_recipe(out, fallback):
    package=here/"bf16-fix/current-recipe-package.json"
    if not package.exists():return fallback
    proof=json.loads(package.read_text())
    digest=proof["digest"]
    if not re.fullmatch("[0-9a-f]{12}",digest):raise ValueError("Invalid recipe archive digest")
    driver.trial.idle()
    key=driver.trial.bench_div.DONE_KEY
    count=driver.trial.server.metrics()[key]
    with (out/"recipe-install.log").open("x") as log:
        subprocess.run(["bash",str(here/"pack-image.sh"),f"/tmp/gb10-opt-{digest}.tar.gz",digest],
                       check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
    for board in ("head","worker"):
        lines=(here/f"image-{digest}"/(board+".sha256")).read_text().splitlines()
        installed={p.removeprefix("/opt/tensorfold/"):h for h,p in map(str.split,lines)}
        if installed!=proof["sha256"]:raise RuntimeError("Current recipe executables differ on "+board)
    if driver.trial.server.metrics()[key]!=count:raise RuntimeError("Service requests during archive staging")
    write_json(out/"recipe-proof.json",proof)
    return "tensorfold-dsv41:bench-opt-"+digest


def run(out,image,baseline=None):
    out=out.resolve();out.mkdir()
    t=driver.trial;t.out=out;initial=driver.inspect() if baseline is None else baseline
    if any(r["tag"]!=driver.base or any(r["flags"][k] for k in t.candidate) for r in initial):
        raise RuntimeError("Expected original serving pair")
    write_json(out/"initial.json",initial)
    t.idle()
    try:
        image=installed_recipe(out,image)
        for exact,label in ((0,"grid"),(1,"exact"),(0,"append-cold")):
            arm=out/label;arm.mkdir()
            opts=dict(t.candidate, IMAGE=image, PREPARE="0", SPILL_GIB="0", KV_SHARD="0", POOL="65536",
                      CONTEXT="32768", TF_DS_PREFILL_SLICE="0",TF_DS_EXACT_RESUME=str(exact),TF_DS_SCORE="1")
            write_json(arm/"requested.json",opts)
            t.restart(label,opts)
            ranks=inspect(image,exact);write_json(arm/"armed.json",ranks)
            client=buckets.load_client()
            client.args.decode_window=False
            client.health=lambda:health(client)
            corpus=client.bench.Corpus("runs/combo4/corpus")
            args=argparse.Namespace(depths=[8192,8201],reps=2,budget=128,append_cold=label=="append-cold")
            offset=time.time()-time.monotonic()
            with (arm/"workload.jsonl").open("x",buffering=1) as f,contextlib.redirect_stdout(f):
                client.emit("start",ranks=ranks)
                screen.resume(client,corpus,args)
                if label!="append-cold":
                    screen.resume_scores(client,arm,cold=not exact)
                client.emit("complete",ranks=inspect(image,exact))
            write_json(arm/"clock.json",[offset,time.time()-time.monotonic()])
    finally:
        if baseline is None:
            buckets.restore(out,initial)

if __name__=="__main__":
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("out",type=Path);p.add_argument("--image",required=True)
    args=p.parse_args()
    with (driver.here/".pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        group=args.out.parent/"trim-pressure"
        outcome=group/"isolated-outcome.json"
        restored=group/"restored.json"
        if args.out.is_dir() and outcome.is_file() and restored.is_file():
            snapshots=json.loads(restored.read_text())
            if snapshots!=json.loads((group/"initial.json").read_text()):
                raise RuntimeError("Shared reservation did not restore its original fingerprints")
            proof=args.out/"restored.json"
            if not proof.exists():proof.symlink_to(Path("..")/group.name/restored.name)
            if not json.loads(outcome.read_text())["passed"]:
                raise RuntimeError("Earlier supported-mode trial failed; see "+str(outcome))
        else:
            run(args.out,args.image)
