"""Run independent small serving experiments in the slot released by the policy campaign."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
stage_image = "tensorfold-dsv41:bench-opt-132c10d70cd0"
odds_image = "tensorfold-dsv41:bench-opt-f7ee965f8292"


def event(kind, **values):
    print(json.dumps(dict(kind=kind, at=time.time(), **values)), flush=True)


def idle():
    with (here.parent/"engine-followup/.pair.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def run(args):
    out = args.out.resolve()
    out.mkdir()
    stop = datetime.datetime.fromisoformat(args.stop_by).timestamp()
    event("waiting", gate=str(args.gate))
    while not (args.gate.is_file() and idle()):
        if time.time() > stop:
            raise TimeoutError("Handoff did not arrive")
        time.sleep(5)
    results = {}

    def trial(name, script, image, options):
        if time.time()+9*60 >= stop or (out/"handoff-after-current").exists():
            return False
        dest = out/name
        if args.prior:
            previous=args.prior.resolve()
            prior=json.loads((previous/"outcomes.json").read_text())
            if prior.get(name)=={"rc":0,"restored":True} and (previous/name/"restored.json").is_file():
                dest.symlink_to(previous/name,target_is_directory=True)
                results[name]=prior[name]
                event("reuse",name=name,source=str(previous/name))
                (out/"outcomes.json").write_text(json.dumps(results,indent=2)+"\n")
                return True
        command = [sys.executable, "-I", "-u", str(here/script), str(dest), "--image", image, *options]
        event("launch", name=name, command=command)
        with (out/(name+".log")).open("x") as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        restored = (dest/"restored.json").is_file()
        results[name] = dict(rc=code, restored=restored)
        event("finished", name=name, **results[name])
        (out/"outcomes.json").write_text(json.dumps(results, indent=2)+"\n")
        if not restored:
            raise RuntimeError("Missing restore proof from "+name)
        return code == 0

    def screen(name, image, mode, depths, *options):
        return trial(name,"screen.py",image,[mode,"--depths",*map(str,depths),*options])

    short = [2094,8192,32000]
    resume_depths = [8192,8201]
    if (here/"index-gate01/bench.validate.json").is_file():
        index = ["--graphs","--serving-drafts","--boundaries","--depths","132096","263168",
                 "--widths","4","--reps","2","--burst-budget","512"]
        if trial("index-a","buckets.py",stage_image,index):
            if trial("index-b","buckets.py",stage_image,[*index,"--finer"]):
                trial("index-return","buckets.py",stage_image,index)

    if screen("prefill-control",stage_image,"prefill",short,"--reps","2","--validate"):
        screen("aio64",stage_image,"prefill",short,"--reps","2","--set","TF_DS_ENGRAM_AIO_BATCH=64","--validate")
        if screen("cpuset-control",stage_image,"prefill",short,"--reps","2","--set","CPUSET=5-9,15-19"):
            screen("engram-cpus",stage_image,"prefill",short,"--reps","2","--set","CPUSET=5-9,15-19",
                   "--set","TF_DS_ENGRAM_CPUS=5-9","--validate")
    if screen("resume-control",stage_image,"resume",resume_depths,"--reps","2"):
        screen("append-cold",stage_image,"resume",resume_depths,"--append-cold","--reps","2")
        screen("exact-resume",stage_image,"resume",resume_depths,"--set","TF_DS_EXACT_RESUME=1","--reps","2","--validate")
        if screen("endstate",stage_image,"resume",resume_depths,"--set","TF_DS_PROMPT_ENDSTATE=2","--reps","2","--validate"):
            screen("tail-batch",stage_image,"resume",resume_depths,"--set","TF_DS_PROMPT_ENDSTATE=2",
                   "--set","TF_DS_TAIL_BATCH=2","--reps","2","--validate")
    if screen("odds-default",odds_image,"prefill",short,"--validate","--score"):
        screen("head-last",odds_image,"prefill",short,"--set","TF_DS_HEAD_LAST_CHUNK=1","--validate","--exact")
        if screen("odds-decode-control",odds_image,"decode",[3072],"--budget","384"):
            screen("greedy-map",odds_image,"decode",[3072],"--budget","384","--set","TF_DS_GREEDY_ARGMAX=1","--validate","--exact")
            screen("bf16-partials",odds_image,"decode",[3072],"--budget","384","--set","TF_DS_BF16_PARTIALS=1","--score","--validate")
        screen("trim",odds_image,"resume",resume_depths,"--set","TF_DS_SNAP_TRIM=1","--reps","2","--validate","--exact")
    (out/"released.json").write_text(json.dumps(results, indent=2)+"\n")
    event("released", outcomes=results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--prior",type=Path)
    p.add_argument("--stop-by",default="2026-11-18T15:48:00+00:00")
    run(p.parse_args())
