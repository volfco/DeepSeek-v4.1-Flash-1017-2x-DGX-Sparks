"""Cross the real 512 MiB checkpoint budget with independent 2,305-token sessions and revisit victims."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import buckets
import screen

ledger=re.compile(r"(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB \(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)")


def measure(client,corpus,out):
    text=corpus.text("div",260000,"cache-pressure-1118",fixed=True)
    with screen.post(client,"/tokenize",{"prompt":text,"add_special_tokens":False}) as reply:
        ids=json.load(reply)["tokens"]
    size=2305
    prompts=[ids[i*size:(i+1)*size] for i in range(84)]
    if any(len(p)!=size for p in prompts):
        raise ValueError("Insufficient independent public prompt text")
    # A distinct first token prevents any two fixtures from sharing a grid state.
    for i,prompt in enumerate(prompts):prompt[0]=1000+i
    start_metrics=client.srv.metrics()
    def cache_counts():
        return {k:v for k,v in client.srv.metrics().items() if "prompt_cache" in k or "spill_" in k}
    windows={}
    with (out/"requests.jsonl").open("x",buffering=1) as f:
        for mode, indices in (("seed",range(84)),("revisit",[0,1,2,3,4,5,6,7,80,81,82,83])):
            windows[mode+"_before"]=cache_counts()
            for i in indices:
                prompt=prompts[i]
                begin=time.monotonic()
                with screen.post(client,"/v1/completions",{"model":client.srv.model,"prompt":prompt,
                                  "temperature":0,"max_tokens":1,"stream":False}) as reply:
                    result=json.load(reply)
                usage=result["usage"];cut=usage["prompt_tokens_details"]["cached_tokens"]
                if usage["prompt_tokens"]!=size or usage["completion_tokens"] not in (0,1) or (mode=="seed" and cut):
                    raise ValueError("Unexpected request work")
                row=dict(mode=mode,session=i,cut=cut,request=result["id"],wall_s=time.monotonic()-begin,
                         input_sha=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                         output_tokens=usage["completion_tokens"],finish=result["choices"][0]["finish_reason"],
                         text_sha=hashlib.sha256(result["choices"][0]["text"].encode()).hexdigest())
                print(json.dumps(row),file=f)
                if mode=="seed" and i==2:
                    log=subprocess.check_output(["docker","logs","--tail","120",buckets.driver.container],
                                                        stderr=subprocess.STDOUT,text=True)
                    observations=ledger.findall(log)
                    if not observations or int(observations[-1][0])<3 or int(observations[-1][3])!=512:
                        raise ValueError("Pilot did not retain three independent states in a 512 MiB cache")
            windows[mode+"_after"]=cache_counts()
    key=client.bench.DONE_KEY
    if client.srv.metrics()[key]-start_metrics[key]!=96:
        raise ValueError("Unexpected concurrent work")
    screen.write_json(out/"counter-windows.json",windows)


def run(out,image):
    out=out.resolve();out.mkdir()
    baseline=buckets.driver.inspect()
    if any(r["tag"]!=buckets.driver.base or any(r["flags"][k] for k in buckets.driver.trial.candidate) for r in baseline):
        raise ValueError("Expected restored pair")
    screen.write_json(out/"initial.json",baseline)
    try:
        for label,flag in (("trim-off",0),("trim-on",1)):
            args=argparse.Namespace(out=out/label,image=image,set=[f"TF_DS_SNAP_TRIM={flag}"],mode="trim-pressure",
                                   score=False,validate=False,exact=False)
            screen.run(args,baseline=baseline)
        # Reuse this lock and its single eventual restoration for two outstanding short screens.
        # A separate process and private tier for each switch keep the measured states independent.
        stop=datetime.fromisoformat("2026-11-18T15:55:00+00:00").timestamp()
        if time.time()+26*60 < stop:
            pilot=out.parent/"keyed-micro"
            options=buckets.options().parse_args([str(pilot),"--graphs","--serving-drafts","--boundaries","--depths","3072",
                    "--widths","1","4","--reps","1","--private-spill","--solo-budget","384","--burst-budget","512","--keyed","0"])
            try:
                buckets.run(options,baseline=baseline)
                screen.write_json(out/"keyed-outcome.json",{"passed":True})
            except (OSError,RuntimeError,ValueError) as exc:
                screen.write_json(out/"keyed-outcome.json",{"passed":False,"error":str(exc)})
        if time.time()+29*60 < stop:
            from engineering import stage_image
            try:
                importlib.import_module("isolated-resume").run(out.parent/"isolated-resume",stage_image,baseline=baseline)
                screen.write_json(out/"isolated-outcome.json",{"passed":True})
            except (OSError,RuntimeError,ValueError) as exc:
                screen.write_json(out/"isolated-outcome.json",{"passed":False,"error":str(exc)})
    finally:
        buckets.restore(out,baseline)


if __name__=="__main__":
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path);parser.add_argument("--image",required=True)
    args=parser.parse_args()
    with (buckets.driver.here/".pair.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.out,args.image)
