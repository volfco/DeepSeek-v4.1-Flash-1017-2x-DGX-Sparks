"""Freeze authored repository continuations before scoring, then tokenize with the running model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd() / "tools"))
import score

def freeze(out):
    import score_human_text
    rev = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    files = subprocess.check_output(["git","ls-tree","-r","--name-only",rev],text=True).splitlines()
    for kind in ("prose","code"):
        paths = [p for p in files if ((p.startswith("docs/") or p=="README.md") and p.endswith(".md") if kind=="prose" else p.startswith("tools/") and p.endswith(".py"))]
        chosen = 0
        for filename in sorted(paths, key=lambda s: hashlib.sha256(s.encode()).digest()):
            blob = subprocess.check_output(["git","show",f"{rev}:{filename}"])
            # Skip the module docstring / heading to include internal code and prose.
            splits = list(score_human_text.splits(blob, 1280, 248, 4))
            if len(splits)<3:
                continue
            for j,(start,cut,end) in enumerate(splits[1:3]):
                s={"id":f"{kind}-{chosen:02d}-{j}","kind":kind,
                   "prompt_text":blob[start:cut].decode(), "continuation_text":blob[cut:end].decode(),
                   "provenance":{"source":filename,"commit":rev,"sha256":hashlib.sha256(blob).hexdigest(),
                                 "prompt_bytes":[start,cut],"continuation_bytes":[cut,end]}}
                print(json.dumps(s),file=out)
            chosen+=1
            if chosen==10:
                break
        if chosen!=10:
            raise ValueError(f"Only {chosen} independent {kind} sources")

def mint(src,out):
    def post(route,body):
        res=score.post("http://127.0.0.1:8000",route,body,40)
        if res.get("error"):raise RuntimeError(res["error"])
        return res
    for line in src:
        s=json.loads(line)
        msgs=[{"role":"user","content":"Continue the following "+s["kind"]+" excerpt exactly from where it ends.\n\n"+s["prompt_text"]}]
        prompt=post("/tokenize",{"messages":msgs,"chat_template_kwargs":{"enable_thinking":False}})["tokens"]
        continuation=post("/tokenize",{"prompt":s["continuation_text"],"add_special_tokens":False})["tokens"]
        if post("/detokenize",{"tokens":continuation})["prompt"]!=s["continuation_text"]:
            raise ValueError(f"Byte round trip failed for {s['id']}")
        print(json.dumps({k:s[k] for k in ("id","kind","provenance")} | {"prompt_ids":prompt,"continuation_ids":continuation}),file=out)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--mint",type=Path)
    args=parser.parse_args()
    with args.out.open("x") as out:
        if args.mint:
            with args.mint.open() as src:mint(src,out)
        else:freeze(out)
