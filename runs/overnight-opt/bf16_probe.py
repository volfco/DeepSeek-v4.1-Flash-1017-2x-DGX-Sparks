"""Install the combined diagnostic image and gate prefill speed on a matched 40-item quality panel."""
import argparse
import contextlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

here=Path(__file__).resolve().parent


def install(driver):
    folder=here/"bf16-fix"
    ready=folder/"hot-image.txt"
    pending=folder/"hot-package.json"
    if ready.exists():return ready.read_text().strip()
    if not pending.is_file():return None
    plan=json.loads(pending.read_text());digest=plan["digest"]
    if not re.fullmatch("[a-f0-9]{12}",digest):raise ValueError("Invalid combined image identity")
    driver.trial.idle()
    key=driver.trial.bench_div.DONE_KEY
    before=driver.trial.server.metrics()[key]
    with (folder/"hot-package.log").open("w") as f:
        subprocess.run(["bash",str(here/"pack-image.sh"),f"/tmp/gb10-opt-{digest}.tar.gz",digest],
                       stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
    for rank in ("head","worker"):
        rows=(here/f"image-{digest}"/(rank+".sha256")).read_text().splitlines()
        actual={p.removeprefix("/opt/tensorfold/"):h for h,p in map(str.split,rows)}
        if actual!=plan["sha256"]:raise ValueError("Combined image binary mismatch")
    if driver.trial.server.metrics()[key]!=before:raise ValueError("Unrelated requests during staging")
    image="tensorfold-dsv41:bench-opt-"+digest
    ready.write_text(image+"\n")
    # This image retains the verified greedy CUDA and hot plans for their subsequent full arms.
    (here/"argmax-fix/image.txt").write_text(image+"\n")
    return image


def panel(client,corpus,path,depths,reps):
    import screen
    path.mkdir()
    offset=time.time()-time.monotonic()
    with (path/"workload.jsonl").open("x",buffering=1) as f,contextlib.redirect_stdout(f):
        ranks=screen.driver.inspect()
        client.emit("start",ranks=ranks)
        screen.prefills(client,corpus,argparse.Namespace(depths=depths,reps=reps))
        client.emit("complete",ranks=screen.driver.inspect())
    screen.write_json(path/"clock.json",[offset,time.time()-time.monotonic()])


def measure(client,corpus,out,reference=None,stop_at=0):
    import screen
    path=out/"prefill"
    source=corpus.text("div",8192,"bf16-128-threshold",fixed=True)
    with screen.post(client,"/tokenize",{"prompt":source,"add_special_tokens":False}) as response:
        ids=json.load(response)["tokens"]
    probes=[]
    for n in (128,129,2048):
        with screen.post(client,"/v1/completions",{"model":client.srv.model,"prompt":ids[:n],"temperature":0,
                          "stream":False,"max_tokens":8,"tensorfold_score":ids[n:n+8]}) as response:
            result=json.load(response)
        score=result["tensorfold"]["score"]
        cut=result["usage"]["prompt_tokens_details"]["cached_tokens"]
        if result["usage"]["prompt_tokens"]!=n or score["tokens"]!=8 or not math.isfinite(score["nll_sum"]) or cut:
            raise ValueError("Projection-shape pilot failed")
        probes.append(dict(prompt_tokens=n,cached=cut,score=score))
        screen.write_json(out/"bf16-probes.json",probes)
    screen.score(screen.driver.trial,out)
    if reference:
        quality=importlib.import_module("odds-results").quality(reference,out)
        screen.write_json(out/"quality-vs-control.json",quality)
        if quality["ci95"][1]>.020:
            raise ValueError("Source-paired NLL exceeds the predeclared quality margin")
    panel(client,corpus,path,[2094,8192,32000],2)
    if reference is not None:
        rows=list(importlib.import_module("screen-results").compare(reference/"prefill",path))
        gains={depth:[float(r["base_prefill_s"])/float(r["arm_prefill_s"])-1 for r in rows if r["depth"]==depth]
               for depth in (8192,32000)}
        signal=any(len(values)==2 and min(values)>.03 for values in gains.values())
        screen.write_json(out/"prefill-signal.json",{"by_depth":gains,"passed":signal})
        if signal and time.time()+7*60 < stop_at:
            panel(client,corpus,out/"full-prefill",[100000],3)
