"""After the main engineering queue, capture per-rank stage shapes and confirm only repeatable signals."""
import argparse
import csv
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
from engineering import idle,event,stage_image,odds_image
import shapes


def cells(directory):
    records=[json.loads(s) for s in (directory/"workload.jsonl").read_text().splitlines()]
    if records[-1]["kind"]!="complete" or not all(q["clean"] for q in records if q["kind"]=="cell"):
        raise ValueError("Incomplete comparison: "+str(directory))
    return [q for q in records if q["kind"]=="cell"]


def prefill_signal(control,arm):
    if not ((arm/"bench.validate.json").exists() and (control/"restored.json").exists()):
        return False
    def times(d):
        return {(q["depth"],q["rep"]):(q["requests"][0]["input_sha256"],float(q["requests"][0]["done"]["prefill_s"]))
                for q in cells(d) if q["depth"]>=8192}
    a,b=times(control),times(arm)
    if a.keys()!=b.keys() or any(a[k][0]!=b[k][0] for k in a):
        raise ValueError("Unmatched prefill fixtures")
    for depth in {k[0] for k in a}:
        gains=[a[k][1]/b[k][1]-1 for k in a if k[0]==depth]
        if min(gains)>.02:
            return True
    return False


def decode_signal(control,arm,shape=None):
    import analyze
    def timings(path):
        result={}
        for c in analyze.cells(path/"workload.jsonl"):
            if c["cell"]=="chat-prime" or (shape is not None and c["sharing"]!=shape):continue
            active=analyze.active_segment(c)
            if len(active)<2:return {}
            a,b=active[0],active[-1]
            rounds=sum(b[k]-a[k] for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds"))
            if rounds<=0 or b["t"]-a["t"]<2:return {}
            ids=[(r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in c["requests"]]
            result[(c["sharing"],c["rep"],len(ids))]=(ids,(b["t"]-a["t"])/rounds)
        return result
    a,b=timings(control),timings(arm)
    if not a or a.keys()!=b.keys() or any(a[k][0]!=b[k][0] for k in a):return 0
    gains=sorted(a[k][1]/b[k][1]-1 for k in a)
    return gains[len(gains)//2] if min(gains)>0 else 0


def record_shapes(out):
    # Match standalone dense GEMMs against the actually recorded shapes. Grouped expert operations have distinct weights.
    references=set()
    for rank in [0,1]:
        rows=shapes.collect(out/"stages"/f"rank{rank}.stages.bin")
        with (out/f"rank{rank}-shapes.csv").open("x") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        references.update((r["m"],r["k"],r["n"]) for r in rows if r["stage"]=="dense")
    shape_file=out/"dense-shapes.txt"
    shape_file.write_text("".join(f"{m} {k} {n}\n" for m,k,n in sorted(references)))
    subprocess.run(["scp",str(shape_file),"10.50.0.11:/tmp/dsv41-opt-shapes.txt"],check=True)
    with (here.parent/"engine-followup/.pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Engine is restored; check it remains idle before invoking the two small references.
        from buckets import driver
        driver.trial.idle()
        key="tensorfold:request_latency_seconds_count"
        base=driver.trial.server.metrics().get(key)
        with (out/"head-math.csv").open("x") as h,(out/"worker-math.csv").open("x") as w:
            a=subprocess.Popen(["/tmp/dsv41-opt-shape-math",str(shape_file)],stdout=h)
            b=subprocess.Popen(["ssh","-n","10.50.0.11","/tmp/dsv41-opt-shape-math","/tmp/dsv41-opt-shapes.txt"],stdout=w)
            codes=[a.wait(),b.wait()]
            if any(codes):raise RuntimeError("Shape calibration failed")
        after=driver.trial.server.metrics().get(key)
        event("math",before=base,after=after)
        if base is None or after!=base:raise RuntimeError("Service requests overlapped calibration")


def install_map_image():
    folder=here/"argmax-fix"
    pending=folder/"package.json"
    if not pending.exists() or (folder/"image.txt").exists():
        return
    import re
    from buckets import driver
    plan=json.loads(pending.read_text())
    digest=plan["digest"]
    if not re.fullmatch("[a-f0-9]{12}",digest):
        raise ValueError("Invalid fixed map package")
    with (here.parent/"engine-followup/.pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        driver.trial.idle()
        key=driver.trial.bench_div.DONE_KEY
        before=driver.trial.server.metrics()[key]
        with (folder/"package.log").open("x") as log:
            subprocess.run(["bash",str(here/"pack-image.sh"),f"/tmp/gb10-opt-{digest}.tar.gz",digest],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        for label in ("head","worker"):
            with (here/f"image-{digest}"/(label+".sha256")).open() as f:
                actual={path.removeprefix("/opt/tensorfold/"):sha for sha,path in map(str.split,f)}
            if actual!=plan["sha256"]:raise ValueError("Fixed map image has incorrect binaries")
        if driver.trial.server.metrics()[key]!=before:raise ValueError("Requests during image installation")
        (folder/"image.txt").write_text("tensorfold-dsv41:bench-opt-"+digest+"\n")


def run(args):
    handoff=args.gate
    before=datetime.datetime.fromisoformat(args.stop_by).timestamp()
    out=args.out.resolve();out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise ValueError("Follow-up directory already contains evidence")
    required=here/"kv/natural-handoff01-required"
    natural_release=here/"kv/natural-handoff01/released.json"
    while not(handoff.is_file() and (not required.exists() or natural_release.is_file()) and idle()):
        if time.time()>before:raise TimeoutError("Previous engineering work has not released the pair")
        time.sleep(5)
    try:
        install_map_image()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        (out/"map-image-error.txt").write_text(str(exc)+"\n")
    finished={}
    def measure(name,script,options,minutes=8):
        if time.time()+minutes*60>before or (out/"handoff-after-current").exists():
            finished[name]={"status":"unstarted","minutes_reserved":minutes,"utc":time.time()}
            event("unstarted",name=name,minutes_reserved=minutes,stop_by=args.stop_by)
            return False
        dest=out/name
        event("launch",name=name)
        with (out/(name+".log")).open("x") as f:
            rc=subprocess.run([sys.executable,"-I","-u",str(here/script),str(dest),*options],stdout=f,stderr=subprocess.STDOUT).returncode
        if not (dest/"restored.json").is_file():raise RuntimeError("Restore missing for "+str(dest))
        finished[name]=rc
        event("outcome",name=name,rc=rc)
        return rc==0
    stage=measure("stages", "buckets.py",["--image","tensorfold-dsv41:bench-opt-659adf5053d1",
            "--boundaries","--depths","9000","--reps","1","--widths","1","4","--serving-drafts","--stage-trace",
            "--solo-budget","256","--burst-budget","384","--interval-cycles","60000","--meter-seconds","65"])
    if stage:
        try:
            record_shapes(out)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            (out/"shape-error.txt").write_text(str(exc)+"\n")
            event("shape-reference-error",error=str(exc))
    recipe_image="tensorfold-dsv41:bench-opt-3ed889fd98ae"
    mode="prefill"
    extra=[]
    if (here/"resend-fix-image.txt").exists() and (args.engineering/"resend-repair/resend-gate.jsonl").exists():
        recipe_image=(here/"resend-fix-image.txt").read_text().strip()
        mode="grid-reference";extra=["--set","TF_DS_SCORE=1"]
    integrated_ok=measure("integrated-default","screen.py",[mode,"--image",recipe_image,
               "--depths","2094","8192",*extra,"--exact","--validate"],12)
    if integrated_ok:
        event("integrated-default-passed")
    # Reproduce the deployed baseline and bracket the incumbent on exactly the same long and short turns.
    opts=["--graphs","--serving-drafts","--boundaries","--depths","300032","3072","--widths","1","4",
          "--reps","2","--private-spill","--solo-budget","384","--burst-budget","512"]
    incumbent_ok=measure("baseline-series","buckets.py",[*opts,"--baseline-series"],24)
    from buckets import driver
    embedded=out/"baseline-series/incumbent/sensitivity"
    if not (embedded/"3072-comparison.json").exists() or (embedded/"error.json").exists():
        measure("read-sensitivity","screen.py",["sensitivity","--image",driver.base,"--depths","3072","--validate"],13)
    trim=args.engineering/"trim"
    exact_log=trim/"exact-compare.log"
    if (trim/"bench.validate.json").is_file() and exact_log.is_file() and "RESULT: IDENTICAL" in exact_log.read_text():
        measure("trim-pressure","trim-pressure.py",["--image",odds_image],12)
    outcomes={}
    if (here/"hot-plan-image.txt").is_file():
        image_record=here/"argmax-fix/image.txt"
        hot_image=(image_record if image_record.is_file() else here/"hot-plan-image.txt").read_text().strip()
        if not hot_image.startswith("tensorfold-dsv41:bench-opt-"):raise ValueError("Unknown hot plan image")
        decoded=["decode","--depths","3072","--budget","384","--image",hot_image]
        if measure("hot-series","screen.py",[*decoded,"--hot-series","--shared-diagnostics","--stop-at-utc",str(before)],15):
            series=out/"hot-series"
            outcomes=json.loads((series/"outcomes.json").read_text())
            promising=[]
            for label,k in (("two",2),("four",4)):
                if outcomes.get(label)=="passed":
                    signal=decode_signal(series/"control",series/label)
                    if signal>.03:promising.append((signal,k))
            if promising:
                _,k=max(promising)
                full=["decode","--image",hot_image,"--depths","3072","100000","--widths","1","4",
                      "--reps","2","--budget","512"]
                if measure("hot-full","screen.py",[*full,"--set",f"TF_DS_EXPERT_HOT_CACHE=hot:{k}",
                           "--set","TF_DS_HOT_PLAN_TRACE=1","--validate","--exact"],16):
                    measure("hot-full-return","screen.py",full,14)
            if outcomes.get("rank-map")=="passed" and decode_signal(series/"control",series/"rank-map","greedy")>.03:
                full=["decode","--image",hot_image,"--depths","3072","100000","--widths","1","4",
                      "--reps","2","--budget","512"]
                if measure("rank-map-full","screen.py",[*full,"--set","TF_DS_GREEDY_ARGMAX=1","--validate"],15):
                    measure("rank-map-return","screen.py",full,14)
    if "isolated-resume" not in outcomes:
        measure("isolated-resume","isolated-resume.py",["--image",stage_image],8)
    if "bf16-quality" not in outcomes:
        measure("bf16-quality","screen.py",["bf16-quality","--image",odds_image,"--depths","2094","8192","32000",
                "--set","TF_DS_SCORE=1","--set","TF_DS_BF16_PARTIALS=1"],6)
    for name,control,cfg,base_cfg in [
        ("aio64","prefill-control",["--set","TF_DS_ENGRAM_AIO_BATCH=64"],[]),
        ("engram-effective02","cpuset-effective02",["--set","CPUSET=10-19","--set","TF_DS_ENGRAM_CPUS=10-14"],["--set","CPUSET=10-19"])]:
        passed=(args.engineering/name/"restored.json").exists()
        if passed and prefill_signal(args.engineering/control,args.engineering/name):
            if name=="engram-effective02" and not prefill_signal(args.engineering/"prefill-control",args.engineering/name):
                event("screened",name=name,reason="within-mask gain did not also beat serving CPU placement")
                continue
            common=["prefill","--image",stage_image,"--depths","8192","32000","100000","--reps","3"]
            if measure(name+"-full","screen.py",[*common,*cfg,"--validate","--exact"],17):
                measure(name+"-return","screen.py",[*common,*base_cfg],14)
    # The explicit keyed-on incumbent already measured these fixtures on a fresh spill tier.
    # Reuse it as the opening control and bracket the keyed-off treatment with a new keyed-on return.
    if incumbent_ok:
        keyed_opts = opts.copy()
        if time.time()+12*60 > before:
            keyed_opts.remove("300032")
        if time.time()+8*60 > before:
            measure("keyed-off-pilot","buckets.py",[*keyed_opts,"--keyed","0"],5)
        else:
            measure("keyed-series","buckets.py",[*keyed_opts,"--keyed-series"],8 if "300032" not in keyed_opts else 12)
    (out/"released.json").write_text(json.dumps(finished,indent=2)+"\n")
    event("released",outcomes=finished)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("out",type=Path)
    p.add_argument("--gate",type=Path,required=True);p.add_argument("--engineering",type=Path,required=True)
    p.add_argument("--stop-by",default="2026-11-18T15:55:00+00:00");run(p.parse_args())
