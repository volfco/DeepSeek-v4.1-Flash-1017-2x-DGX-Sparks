"""Exercise one experimental configuration on the benchmark pair, saving workload evidence and restoring service."""
import argparse
import copy
import contextlib
import fcntl
import hashlib
import json
import re
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
import urllib.request

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import buckets
from buckets import driver, write_json

def exact(trial,out):
    record=out/"exact.json"
    for step,cmd in [("record",["record","--url",trial.server.url,"--like",trial.reference,"--label",out.name,"--out",record]),
                     ("compare",["compare",trial.reference,record])]:
        if trial.run([sys.executable,"-I","tools/exact.py",*cmd],f"exact-{step}.log"):
            raise RuntimeError("Exact "+step+" failed")

def prefills(client,corpus,args):
    for depth in args.depths:
        for rep in range(args.reps):
            salt=f"engram-work-{depth}-{rep}"
            body=corpus.text("div",depth+2048,salt,fixed=True)
            prompts,_=buckets.fixture(client,"["+salt+"]\n"+body,depth)
            client.cell(depth,"chat-prime",[prompts[0]],[4230],budget=16,temperature=0,rep=rep,sharing="uncached",expected_cached=0)


def decode(client,corpus,args):
    ordinary_request = client.request
    def request(prompt,seed,budget,gate,temperature,**kwargs):
        effective = (seed % 2) if temperature == -1 else temperature
        result = ordinary_request(prompt,seed,budget,gate,effective,**kwargs)
        return dict(result, temperature=effective)
    client.request = request
    for depth in args.depths:
        source = corpus.text("div",depth+4000,"opt-decode",fixed=True)
        prompts,context = buckets.fixture(client,source,depth)
        client.cell(depth,"chat-prime",[context+"Prime this history."],[7310],budget=16)
        for rep in range(args.reps):
            for width in args.widths:
                modes=((0,"greedy"),(1,"sampled")) if width==1 else ((0,"greedy"),(1,"sampled"),(-1,"mixed"))
                for temp,label in modes:
                    seed=7320+10*rep
                    client.cell(depth,"solo" if width==1 else "admission",prompts[:width],list(range(seed,seed+width)),
                                budget=args.budget,temperature=temp,rep=rep,sharing=label,
                                expected_cached=depth//2048*2048)


def score(trial,out):
    refs = driver.here/"runtime-score01/items.jsonl"
    cmd = [sys.executable,"-I","tools/score.py","record","--url",trial.server.url,
           "--in",refs,"--out",out/"score.json","--label",out.name]
    if trial.run(cmd,"score.log"):
        raise RuntimeError("Frozen continuation scoring failed")


def affinity():
    """Read the live engine threads in each container's PID namespace."""
    inspect = """for p in /proc/[0-9]*; do
        case $(readlink "$p/exe") in */tensorfold-native|*/tf-dsv41-lanes)
          awk '/^Cpus_allowed_list:/{print $2}' "$p"/task/*/status | sort | uniq -c;;
        esac
    done"""
    cmd = ["docker","exec",driver.container,"sh","-c",inspect]
    return [subprocess.check_output(pre+[shlex.join(cmd)] if pre else cmd,
                                    text=True).strip() for pre in driver.prefixes]

def post(client,path,body):
    req=urllib.request.Request(client.srv.url+path,json.dumps(body).encode(),{"Content-Type":"application/json"})
    return client.bench.OPENER.open(req,timeout=300)

def raw_request(client,ids,seed,budget,gate,temperature,**kwargs):
    body={"model":client.srv.model,"prompt":ids,"max_tokens":budget,"temperature":temperature,
          "seed":seed,"stream":True,"stream_options":{"include_usage":True}}
    gate.wait()
    start=time.monotonic()
    first=last=usage=finish=rid=None
    with post(client,"/v1/completions",body) as response:
        for line in response:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            item=json.loads(line[6:])
            if "error" in item:raise ValueError(str(item["error"]))
            rid=item.get("id",rid)
            usage=item.get("usage") or usage
            for choice in item.get("choices",[]):
                finish=choice.get("finish_reason") or finish
                if choice.get("text"):
                    last=time.monotonic()
                    if first is None:first=last
    if usage is None or first is None:raise ValueError("Missing usage or text")
    return {"id":rid,"seed":seed,"input_sha256":hashlib.sha256(json.dumps(body).encode()).hexdigest(),
            "prompt_tokens":usage["prompt_tokens"],"cached":usage["prompt_tokens_details"]["cached_tokens"],
            "tokens":usage["completion_tokens"],"finish":finish,"ttft":first-start,"total":time.monotonic()-start,
            "start":start,"first":first,"last":last}

def resume(client,corpus,args):
    client.request=lambda *a,**kw:raw_request(client,*a,**kw)
    for depth in args.depths:
        for rep in range(args.reps):
            salt=f"resume-study-{depth}-{rep}"
            text="["+salt+"]\n"+corpus.text("div",depth+6000,salt,fixed=True)
            with post(client,"/tokenize",{"messages":[{"role":"user","content":text}],
                      "chat_template_kwargs":{"enable_thinking":True}}) as response:
                ids=json.load(response)["tokens"]
            if len(ids)<depth+257:raise ValueError("Fixture is short")
            base=ids[:depth]
            extended=ids[:depth+257]
            cells=[("append-cold",extended,0)] if args.append_cold else [
                ("first",base,0),("same",base,None),("append",extended,None)]
            for tag,prompt,cache in cells:
                client.cell(depth,"chat-prime",[prompt],[4872],budget=args.budget,temperature=0,rep=rep,
                            sharing=tag,expected_prompt=len(prompt),expected_cached=cache)


def resume_scores(client,out,cold):
    """Score the same future 256 source tokens after an 8,201-token seed and a 128-token append."""
    files=["CHANGELOG.md","docs/baseten-ultrafast-plan.md","tools/fake_server.py","tools/spill_check.py"]
    client.request=lambda *a,**kw:raw_request(client,*a,**kw)
    with (out/"resume-score.jsonl").open("x",buffering=1) as records:
        for path in files:
            source=Path(path).read_bytes()
            with post(client,"/tokenize",{"prompt":source.decode(),"add_special_tokens":False}) as reply:
                ids=json.load(reply)["tokens"]
            cut=8201
            prompt=ids[:cut+128]
            future=ids[cut+128:cut+384]
            if len(future)!=256:
                raise ValueError("Short retained-state quality fixture: "+path)
            if not cold:
                client.cell(cut,"chat-prime",[ids[:cut]],[4931],budget=8,temperature=0,sharing="score-seed",
                            expected_prompt=cut,expected_cached=0)
            driver.trial.idle()
            before=client.srv.metrics()[client.bench.DONE_KEY]
            start=time.monotonic()
            with post(client,"/v1/completions",{"prompt":prompt,"model":client.srv.model,"stream":False,
                      "temperature":0,"max_tokens":len(future),"tensorfold_score":future}) as reply:
                q=json.load(reply)
            usage=q["usage"]
            result=q["tensorfold"]["score"]
            cached=usage["prompt_tokens_details"]["cached_tokens"]
            if (usage["prompt_tokens"]!=len(prompt) or result["tokens"]!=len(future)
                or client.srv.metrics()[client.bench.DONE_KEY]-before!=1 or (cold and cached)):
                raise ValueError("Unmatched retained-state scoring work")
            print(json.dumps(dict(source=path,file_sha=hashlib.sha256(source).hexdigest(),
                  ids_sha=hashlib.sha256(json.dumps(prompt+future).encode()).hexdigest(),
                  cached=cached,prompt=len(prompt),seconds=time.monotonic()-start,score=result)),file=records)

def trim_crossover(args):
    # Both arms use the odds image. Earlier resume runs used a different native/follower build.
    baseline = driver.inspect()
    if any(r["tag"] != driver.base or any(r["flags"][k] for k in driver.trial.candidate) for r in baseline):
        raise RuntimeError("Expected the original pair before the trim crossover")
    control = argparse.Namespace(**vars(args))
    control.out = args.out.with_name("trim-control")
    control.set = []
    control.exact = False
    control.validate = False
    try:
        run(control, baseline=baseline)
        run(args, baseline=baseline)
    finally:
        args.out.mkdir(exist_ok=True)
        buckets.restore(args.out, baseline)


def hot_series(args):
    out = args.out.resolve()
    out.mkdir()
    baseline = driver.inspect()
    if any(r["tag"] != driver.base or any(r["flags"][k] for k in driver.trial.candidate) for r in baseline):
        raise RuntimeError("Expected the original serving configuration before starting the hot-plan comparison")
    write_json(out / "initial.json", baseline)
    outcomes = {}
    try:
        import bf16_probe
        unified=bf16_probe.install(driver)
        if unified:args.image=unified
        plan=[("control", 0), ("two", 2), ("four", 4)]
        if unified:plan.insert(1,("bf16-quality",0))
        candidate=here/"argmax-fix/image.txt"
        if candidate.is_file() and candidate.read_text().strip()==args.image:
            plan.insert(1,("rank-map",0))
        for label, count in plan:
            if label=="bf16-quality":
                reference=out/"control"
                needed=("score.json","prefill/clock.json","bf16-probes.json")
                if any(not (reference/name).is_file() for name in needed):
                    outcomes[label]="Unmeasured: same-image precision control incomplete"
                    write_json(out/"outcomes.json",outcomes)
                    continue
            arm = argparse.Namespace(**vars(args))
            arm.out = out / label
            arm.set = [f"TF_DS_EXPERT_HOT_CACHE=hot:{count}", "TF_DS_HOT_PLAN_TRACE=1"] if count else []
            arm.validate = bool(count)
            if unified:
                arm.set.extend(["TF_DS_SCORE=1",f"TF_DS_BF16_PARTIALS={int(label=='bf16-quality')}"])
                if label=="control":arm.bf16_control=True
                if label=="bf16-quality":
                    arm.out=out.parent/label
                    arm.mode="bf16-probe"
                    arm.bf16_reference=out/"control"
            if label=="rank-map":
                arm.set.append("TF_DS_GREEDY_ARGMAX=1")
                arm.validate=True
                arm.exact=True
                arm.matched_reference=out/"control/workload.jsonl"
            try:
                run(arm, baseline=baseline)
                outcomes[label] = "passed"
            except (OSError, ValueError, RuntimeError) as exc:
                outcomes[label] = str(exc)
                if label=="control" or "_error" in driver.trial.server.metrics():
                    raise
            finally:
                write_json(out / "outcomes.json", outcomes)
        if (out.parent/"bf16-quality/full-prefill/clock.json").is_file() and time.time()+6*60 < args.stop_at_utc:
            arm=argparse.Namespace(**vars(args))
            arm.out=out.parent/"bf16-full-return";arm.mode="bf16-full-control"
            arm.set=["TF_DS_SCORE=1"];arm.validate=False;arm.exact=False
            try:
                run(arm,baseline=baseline)
                outcomes["bf16-full-return"]="passed"
            except (OSError,RuntimeError,ValueError) as exc:
                outcomes["bf16-full-return"]=str(exc)
            write_json(out/"outcomes.json",outcomes)
        if args.shared_diagnostics:
            early=out.parent/"trim-pressure/isolated-outcome.json"
            if early.exists() and (early.parent/"restored.json").exists():
                record=json.loads(early.read_text())
                outcomes["isolated-resume"]="passed earlier" if record["passed"] else record["error"]
                write_json(out/"outcomes.json",outcomes)
            import importlib
            from engineering import odds_image,stage_image
            if "bf16-quality" not in outcomes and time.time()+13*60 < args.stop_at_utc:
                arm=argparse.Namespace(**vars(args))
                arm.out=out.parent/"bf16-quality"
                arm.mode="bf16-quality";arm.image=odds_image;arm.depths=[2094,8192,32000]
                arm.set=["TF_DS_BF16_PARTIALS=1","TF_DS_SCORE=1"]
                arm.score=False;arm.exact=False;arm.validate=False
                try:
                    run(arm,baseline=baseline)
                    outcomes["bf16-quality"]="passed"
                except (ValueError,RuntimeError,OSError) as exc:
                    outcomes["bf16-quality"]=str(exc)
                write_json(out/"outcomes.json",outcomes)
            if "isolated-resume" not in outcomes and time.time()+7*60 < args.stop_at_utc:
                try:
                    importlib.import_module("isolated-resume").run(out.parent/"isolated-resume",stage_image,baseline=baseline)
                    outcomes["isolated-resume"]="passed"
                except (ValueError,RuntimeError,OSError) as exc:
                    outcomes["isolated-resume"]=str(exc)
                write_json(out/"outcomes.json",outcomes)
    finally:
        buckets.restore(out, baseline)


def run(args, baseline=None):
    out=args.out.resolve()
    out.mkdir()
    trial=driver.trial
    trial.out=out
    observed=driver.inspect()
    initial=observed if baseline is None else baseline
    if any(r["tag"]!=driver.base or any(r["flags"][k] for k in trial.candidate) for r in initial):
        raise ValueError("Expected restored production recipe")
    trial.idle()
    config=dict(p.split("=",1) for p in args.set)
    if args.score or args.mode=="resume":
        config["TF_DS_SCORE"]="1"
    spill="/home/colum/.cache/dsv41-opt-"+out.name+"-"+hashlib.sha256(str(out).encode()).hexdigest()[:12]
    flags=dict(trial.candidate,IMAGE=args.image,PREPARE="0",SPILL_DIR=spill,WORKER_SPILL_DIR=spill)
    flags.update(config)
    write_json(out/"initial.json",observed)
    write_json(out/"requested.json",flags)
    client=buckets.load_client()
    corpus=client.bench.Corpus("runs/combo4/corpus")
    try:
        trial.restart("bench",flags)
        ranks=driver.inspect()
        ids=[]
        for pre in driver.prefixes:
            ids.append(subprocess.check_output(pre+["docker","image","inspect","-f","{{.Id}}",args.image],text=True).strip())
        armed=[]
        for i,r in enumerate(ranks):
            if r["id"]!=ids[i] or r["spill"]!=spill or any(r["flags"][k]!=v for k,v in trial.candidate.items()):
                raise ValueError("Benchmark rank mismatch")
            c=json.loads(subprocess.check_output(driver.prefixes[i]+["docker","inspect",driver.container]))[0]
            env=dict(v.split("=",1) for v in c["Config"]["Env"])
            actual={k:env.get(k) for k in config if k.startswith("TF_DS_")}
            if any(config[k]!=v for k,v in actual.items()):
                raise ValueError(f"Missing experimental flag on rank {i}: {actual}")
            armed.append({"switches":actual,"cpuset":c["HostConfig"]["CpusetCpus"],"id":r["id"]})
            if "CPUSET" in config:
                cmd=["docker","exec",driver.container,"cat","/sys/fs/cgroup/cpuset.cpus.effective"]
                effective=subprocess.check_output(driver.prefixes[i]+cmd,text=True).strip()
                armed[-1]["effective_cpuset"]=effective
                if any(buckets.cpu_numbers(mask)!=buckets.cpu_numbers(config["CPUSET"]) for mask in (effective,armed[-1]["cpuset"])):
                    raise ValueError(f"CPU set differs on rank {i}: requested {config['CPUSET']}, effective {effective}")
        write_json(out/"armed.json",armed)
        since=time.time()
        clock_start=time.time()-time.monotonic()
        with buckets.sample_hosts(out), (out/"workload.jsonl").open("x",buffering=1) as f, contextlib.redirect_stdout(f):
            client.emit("start",ranks=ranks)
            if args.mode=="grid-reference":
                import resend_gate
                resend_gate.grid_reference(client,corpus,out)
            elif args.mode=="tail-gate":
                import resend_gate
                resend_gate.tail_measure(client,corpus,out)
            elif args.mode=="resend-gate":
                import resend_gate
                resend_gate.measure(client,corpus,out)
            elif args.mode=="sensitivity":
                import sensitivity
                sensitivity.measure(client,corpus,out)
            elif args.mode=="trim-pressure":
                import importlib
                importlib.import_module("trim-pressure").measure(client,corpus,out)
            elif args.mode=="bf16-full-control":
                import bf16_probe
                bf16_probe.panel(client,corpus,out/"full-prefill",[100000],3)
            elif args.mode=="bf16-probe":
                import bf16_probe
                bf16_probe.measure(client,corpus,out,args.bf16_reference,args.stop_at_utc)
            elif args.mode=="bf16-quality":
                # Score through the nonstreaming endpoint even when the small streaming pilot has no text.
                score(trial,out)
                prefills(client,corpus,args)
            elif args.mode=="prefill":
                prefills(client,corpus,args)
            elif args.mode=="decode":
                decode(client,corpus,args)
            else:
                resume(client,corpus,args)
                resume_scores(client,out,args.append_cold)
            client.emit("complete",ranks=driver.inspect())
        if getattr(args,"bf16_control",False):
            import bf16_probe
            try:
                bf16_probe.measure(client,corpus,out)
            except (OSError,RuntimeError,ValueError) as exc:
                write_json(out/"bf16-control-error.json",{"error":str(exc)})
        write_json(out/"clock.json", [clock_start,time.time()-time.monotonic()])
        if "CPUSET" in config:
            actual_masks=affinity()
            write_json(out/"thread-masks.json",actual_masks)
            if "TF_DS_ENGRAM_CPUS" in config:
                if any(config["TF_DS_ENGRAM_CPUS"] not in {line.split()[-1] for line in group.splitlines()}
                       for group in actual_masks):
                    raise ValueError("A rank did not retain the requested Engram thread affinity")
        for i,pre in enumerate(driver.prefixes):
            with (out/f"rank{i}.log").open("w") as f:
                subprocess.run(pre+["docker","logs","--since",str(since),driver.container],stdout=f,stderr=subprocess.STDOUT,check=True)
            if "TF_DS_HOT_PLAN_TRACE" in config:
                logs=subprocess.check_output(pre+["docker","logs",driver.container],stderr=subprocess.STDOUT,text=True)
                markers=[int(n) for n in re.findall(rf"hot plan rank={i} active_layers=(\d+)",logs)]
                if not any(markers):raise ValueError(f"Rank {i} did not publish a live expert plan")
                write_json(out/f"rank{i}-hot-plan.json",markers)
        if getattr(args,"matched_reference",None):
            import analyze
            def outputs(path):
                return [[(r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in c["requests"]]
                        for c in analyze.cells(path) if c["cell"]!="chat-prime"]
            if outputs(args.matched_reference)!=outputs(out/"workload.jsonl"):
                raise ValueError("Ranking changed an output versus the same-image control")
        if args.score: score(trial,out)
        if args.exact: exact(trial,out)
        if args.validate: trial.validate("bench")
    except Exception:
        if "since" in locals():
            for i,pre in enumerate(driver.prefixes):
                with (out/f"rank{i}-failure.log").open("w") as log:
                    subprocess.run(pre+["docker","logs","--since",str(since),driver.container],
                                   stdout=log,stderr=subprocess.STDOUT,timeout=20)
        raise
    finally:
        if baseline is None:
            buckets.restore(out,initial)
        # Run this once inside the existing pair lock after the CPU-affinity control has restored.
        if out.name=="cpuset-control" and not (here/"hardware01/aio-isolation.json").exists():
            status=subprocess.run([sys.executable,str(here/"drive-depth.py"),"--out",str(here/"hardware01")]).returncode
            write_json(out/"aio-reference.json",{"returncode":status})

def retry_affinity(args):
    folder=args.out.parent
    request=folder/"affinity-retry-requested"
    if args.out.name!="exact-resume" or not request.is_file():return
    # The engineering coordinator is already running. Use its next reserved idle boundary.
    result={}
    for name, extra in (("cpuset-effective02",[]),("engram-effective02",["TF_DS_ENGRAM_CPUS=10-14"])):
        arm=copy.copy(args)
        arm.out=folder/name
        arm.mode="prefill";arm.depths=[2094,8192,32000];arm.reps=2
        arm.append_cold=False;arm.validate=True;arm.exact=False;arm.set=["CPUSET=10-19",*extra]
        try:
            run(arm)
            result[name]="passed"
        except Exception as exc:
            result[name]=str(exc)
            write_json(folder/"affinity-retry02.json",result)
            if not (arm.out/"restored.json").is_file():raise
            break
        write_json(folder/"affinity-retry02.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path);p.add_argument("mode",choices=("prefill","resume","decode","sensitivity","resend-gate","grid-reference","tail-gate","bf16-quality"))
    p.add_argument("--set",action="append",default=[])
    p.add_argument("--image",default=driver.base);p.add_argument("--depths",nargs="+",type=int,required=True)
    p.add_argument("--budget",type=int,default=128);p.add_argument("--reps",type=int,default=1)
    p.add_argument("--widths",nargs="+",type=int,choices=[1,4],default=[4])
    p.add_argument("--shared-diagnostics",action="store_true")
    p.add_argument("--stop-at-utc",type=float,default=0)
    p.add_argument("--hot-series",action="store_true")
    p.add_argument("--append-cold",action="store_true")
    p.add_argument("--exact",action="store_true");p.add_argument("--validate",action="store_true")
    p.add_argument("--score",action="store_true")
    args=p.parse_args()
    def stop(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with (driver.here/".pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.out.parent.name=="engineering02" and args.out.name in ("bf16-partials","trim"):
            import importlib.util
            path=here/"map-kernel.py"
            if path.exists():
                spec=importlib.util.spec_from_file_location("map_kernel",path)
                mod=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                try:
                    mod.run(here,driver)
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                    write_json(here/"argmax-fix"/"kernel-error.json",{"error":str(exc)})
        retry_affinity(args)
        from resend_gate import queued
        queued(args)
        if args.out.parent.name == "engineering02" and args.out.name == "trim":
            trim_crossover(args)
        elif args.hot_series:
            hot_series(args)
        else:
            run(args)
