"""Compile the isolated greedy reduction and compare both GPU binaries while the pair is idle."""
import csv
import json
from pathlib import Path
import subprocess
import time

def run(here, driver):
    out=here/"argmax-fix"
    proof=out/"kernel.json"
    if proof.exists() or not (out/"compare.cpp").is_file():
        return
    driver.trial.idle()
    key=driver.trial.bench_div.DONE_KEY
    start=driver.trial.server.metrics()[key]
    then=time.time()
    fixed="/tmp/dsv41-hot-map-fixed.fatbin"
    with (out/"build.log").open("w") as f:
        subprocess.run(["bash","/tmp/build-hot-kernel.sh","/tmp/dsv41-hot-map-fixed.cu",fixed],
                       check=True,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        subprocess.run(["g++","-std=c++17","-O2","-I/usr/local/cuda/include", str(out/"compare.cpp"),
                        "-o","/tmp/dsv41-map-check","-lcuda"],check=True,stdout=f,stderr=subprocess.STDOUT)
    totals={}
    for name, image in (("original","/tmp/dsv41-hot-plan-ops.fatbin"), ("fixed",fixed)):
        with (out/(name+".csv")).open("w") as f:
            subprocess.run(["/tmp/dsv41-map-check",image],stdout=f,check=True)
        with (out/(name+".csv")).open() as f:
            rows=list(csv.DictReader(f))
            totals[name]=[sum(int(r[k]) for r in rows) for k in ("matched","total")]
    end=driver.trial.server.metrics()[key]
    data=dict(start=then,end=time.time(),requests_before=start,requests_after=end,matches=totals)
    data["passed"]=(start==end and totals["fixed"][0]==totals["fixed"][1] and totals["original"][0]<totals["original"][1])
    proof.write_text(json.dumps(data,indent=2)+"\n")
