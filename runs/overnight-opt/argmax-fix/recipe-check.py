"""Build the exact recipe CUDA module and check the fixed reduction on a quiescent GPU."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

def check(out,driver):
    source=Path("/tmp/dsv41-recipe-map.cu")
    if not source.is_file():
        return
    dest=Path("/tmp/dsv41-recipe-map.fatbin")
    driver.trial.idle()
    key=driver.trial.bench_div.DONE_KEY
    before=driver.trial.server.metrics()[key]
    args=["/usr/local/cuda/bin/nvcc","-fatbin","-D__CUDA_NO_HALF_OPERATORS__","-D__CUDA_NO_HALF_CONVERSIONS__",
          "-D__CUDA_NO_BFLOAT16_CONVERSIONS__","-D__CUDA_NO_HALF2_OPERATORS__","--expt-relaxed-constexpr",
          "-std=c++20","-O3","--fmad=false","--ftz=false","-gencode=arch=compute_121,code=sm_121",
          "-o",str(dest),str(source)]
    with (out/"recipe-build.log").open("x") as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    with (out/"recipe.csv").open("x") as data:
        subprocess.run(["/tmp/dsv41-map-check",str(dest)],stdout=data,check=True)
    with (out/"recipe.csv").open() as data:
        rows=list(csv.DictReader(data))
    after=driver.trial.server.metrics()[key]
    proof=dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               requests_before=before,requests_after=after,matched=sum(int(r["matched"]) for r in rows),
               total=sum(int(r["total"]) for r in rows))
    if before!=after or proof["matched"]!=proof["total"] or proof["total"]!=36:
        raise ValueError("Clean CUDA reduction did not pass the isolated 36-case gate")
    (out/"recipe.json").write_text(json.dumps(proof,indent=2)+"\n")
