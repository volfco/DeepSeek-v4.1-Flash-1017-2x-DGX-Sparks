"""Gate the repaired full-end-snapshot path on exact greedy outputs and teacher-forced scores."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import screen


def pair(client,corpus,depth,rep,label,grid=False):
    salt=f"resume-study-{depth}-{rep}"
    text="["+salt+"]\n"+corpus.text("div",depth+6000,salt,fixed=True)
    with screen.post(client,"/tokenize",{"messages":[{"role":"user","content":text}],
                     "chat_template_kwargs":{"enable_thinking":True}}) as response:
        ids=json.load(response)["tokens"][:depth]
    if len(ids)!=depth:raise ValueError("Short fixed-depth fixture")
    def request(key,cut):
        return client.cell(depth,"chat-prime",[ids],[4872],budget=128,temperature=0,
                           rep=rep,sharing=f"{label}-{key}",expected_prompt=depth,expected_cached=cut)[0]
    cold=request("cold",0);hot=request("same",None if grid else depth)
    return dict(depth=depth,rep=rep,request_sha=cold["input_sha256"],
                identical=cold["input_sha256"]==hot["input_sha256"] and cold["done"]["sha"]==hot["done"]["sha"]
                    and cold["tokens"]==hot["tokens"],
                cold_hash=cold["done"]["sha"],hot_hash=hot["done"]["sha"],cold_ttft_s=cold["ttft"],
                hot_ttft_s=hot["ttft"],cut=hot["cached"],output_tokens=hot["tokens"])


def grid_reference(client,corpus,out):
    client.request=lambda *a,**kw:screen.raw_request(client,*a,**kw)
    with (out/"resend-gate.jsonl").open("x",buffering=1) as f:
        for depth,rep,label in ((8201,0,"pilot"),(8192,0,"grid"),(8201,1,"off-grid"),
                                (8192,1,"grid"),(100009,0,"long")):
            row=pair(client,corpus,depth,rep,label,grid=True);print(json.dumps(row),file=f)
            if not (0<row["cut"]<depth and row["identical"]):
                raise ValueError("Ordinary grid reuse did not reproduce its cold reference")


def scores(client,out):
    # Public, frozen repository text. Compare the identical distribution after a cold and an exact-length prefill.
    for path in ("CHANGELOG.md","docs/baseten-ultrafast-plan.md","tools/fake_server.py","tools/spill_check.py"):
        raw=Path(path).read_bytes()
        with screen.post(client,"/tokenize",{"prompt":raw.decode(),"add_special_tokens":False}) as response:
            ids=json.load(response)["tokens"]
        cut=8201;prompt=ids[:cut];future=ids[cut:cut+256]
        if len(future)!=256:raise ValueError("Short continuation")
        def score():
            with screen.post(client,"/v1/completions",{"prompt":prompt,"model":client.srv.model,"stream":False,
                            "temperature":0,"max_tokens":256,"tensorfold_score":future}) as response:
                q=json.load(response)
            return q["usage"]["prompt_tokens_details"]["cached_tokens"],q["tensorfold"]["score"]
        before=client.srv.metrics()[client.bench.DONE_KEY]
        cold,base=score()
        client.cell(cut,"chat-prime",[prompt],[4931],budget=8,temperature=0,sharing="score-seed",expected_prompt=cut)
        hot,candidate=score()
        row=dict(source=path,file_sha=hashlib.sha256(raw).hexdigest(),cold_cut=cold,hot_cut=hot,
                 cold_score=base,hot_score=candidate,exact_score=base==candidate)
        print(json.dumps(row),file=out,flush=True)
        if (client.srv.metrics()[client.bench.DONE_KEY]-before!=3 or cold!=0 or hot!=cut
                or base["tokens"]!=256 or base!=candidate):
            raise ValueError("Exact-end distribution or expected work differed for "+path)


def measure(client,corpus,out):
    client.request=lambda *a,**kw:screen.raw_request(client,*a,**kw)
    # Run a single previously failing off-grid seed before expanding depth or running exact.py.
    with (out/"resend-gate.jsonl").open("x",buffering=1) as f:
        for depth,rep,label in ((8201,0,"pilot"),(8192,0,"grid"),(8201,1,"off-grid"),
                                (8192,1,"grid"),(100009,0,"long")):
            row=pair(client,corpus,depth,rep,label);print(json.dumps(row),file=f)
            if not row["identical"]:raise ValueError(f"Greedy resend changed at {depth}, repeat {rep}")
    with (out/"full-end-scores.jsonl").open("x",buffering=1) as f:
        scores(client,f)


def tail_measure(client,corpus,out):
    client.request=lambda *a,**kw:screen.raw_request(client,*a,**kw)
    odd_fixture=None
    with (out/"append-gate.jsonl").open("x",buffering=1) as results:
        for depth in (8201,8192,8202,100010):
            salt=f"tail-gate-{depth}"
            text="["+salt+"]\n"+corpus.text("div",depth+6000,salt,fixed=True)
            with screen.post(client,"/tokenize",{"messages":[{"role":"user","content":text}],
                         "chat_template_kwargs":{"enable_thinking":True}}) as response:
                ids=json.load(response)["tokens"][:depth+257]
            if len(ids)!=depth+257:raise ValueError("Short tail fixture")
            def request(prompt,tag,cut=None,budget=128):
                return client.cell(depth,"chat-prime",[prompt],[4872],budget=budget,temperature=0,
                                   rep=0,sharing=tag,expected_prompt=len(prompt),expected_cached=cut)[0]
            cold=request(ids,"cold-append",0)
            # Two newer end states displace the long fixture from the two-entry end-state capacity.
            with screen.post(client,"/tokenize",{"prompt":("decoy "+str(depth)+" ")*2100,"add_special_tokens":False}) as response:
                decoy=json.load(response)["tokens"][:2049]
            request(decoy,"displace",budget=1)
            first=request(ids[:depth],"seed")
            again=request(ids[:depth],"same",depth)
            if depth==8201:odd_fixture=(ids[:depth],first["done"]["sha"])
            warm=request(ids,"append")
            r=dict(depth=depth,append_cut=warm["cached"],match_resend=first["done"]["sha"]==again["done"]["sha"],
                   match_append=cold["done"]["sha"]==warm["done"]["sha"] and cold["input_sha256"]==warm["input_sha256"],
                   cold_append_ttft_s=cold["ttft"],warm_append_ttft_s=warm["ttft"],same_ttft_s=again["ttft"])
            print(json.dumps(r),file=results)
            correct_cut=(0 < warm["cached"] < depth and warm["cached"]%2048==0) if depth%2 else warm["cached"]==depth
            if not (correct_cut and r["match_resend"] and r["match_append"]):
                raise ValueError("Tail cache cut or output failed to match the independent cold request")
    ids,reference=odd_fixture
    client.cell(8201,"chat-prime",[ids],[4872],budget=128,temperature=0,sharing="concurrent-seed",expected_prompt=8201)
    same=client.cell(8201,"admission",[ids,ids],[4872,4872],budget=128,temperature=0,sharing="concurrent-same",
                     expected_prompt=8201)
    rows=[dict(cut=r["cached"],sha=r["done"]["sha"],matching=r["done"]["sha"]==reference,ttft_s=r["ttft"])
          for r in same]
    screen.write_json(out/"paired-odd-resends.json",rows)
    if not all(r["matching"] for r in rows):
        raise ValueError("Concurrent identical odd resends diverged")


def queued(options):
    folder=options.out.parent
    image_path=screen.here/"resend-fix-image.txt"
    if folder.name!="engineering02":return
    label="resend-repair"
    pending=screen.here/"resend-fix-pending.json"
    if (folder/label).exists():
        result=folder/"resend-repair-result.json"
        if not result.is_file() or not json.loads(result.read_text())["passed"]:return
        label="tail-repair"
        image_path=screen.here/"tail-fix-image.txt"
        pending=screen.here/"tail-fix-pending.json"
    dest=folder/label
    if dest.exists():return
    if not image_path.exists() and pending.is_file():
        plan=json.loads(pending.read_text())
        digest=plan["digest"]
        if not re.fullmatch(r"[0-9a-f]{12}",digest):raise ValueError("Invalid package identity")
        with (folder/(label+"-image.log")).open("w") as log:
            subprocess.run(["bash",str(screen.here/"pack-image.sh"),f"/tmp/gb10-opt-{digest}.tar.gz",digest],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        for rank in ("head","worker"):
            path=screen.here/f"image-{digest}"/f"{rank}.sha256"
            actual={k.removeprefix("/opt/tensorfold/"):v for line in path.read_text().splitlines()
                    for v,k in [line.split()]}
            if actual!=plan["sha256"]:raise ValueError("Installed binaries differ from the cross-build")
        image_path.write_text("tensorfold-dsv41:bench-opt-"+digest+"\n")
    if not image_path.is_file():return
    image=image_path.read_text().strip()
    if not image.startswith("tensorfold-dsv41:bench-opt-"):raise ValueError("Unexpected repair image")
    arm=argparse.Namespace(out=dest,mode="resend-gate",image=image,set=["TF_DS_PROMPT_ENDSTATE=2","TF_DS_SCORE=1"],
                           validate=True,exact=True,append_cold=False,score=False)
    if label=="tail-repair":
        arm.mode="tail-gate";arm.set.append("TF_DS_TAIL_BATCH=2")
    try:
        screen.run(arm)
        summary={"passed":True}
    except Exception as exc:
        summary={"passed":False,"error":str(exc)}
        if not (dest/"restored.json").is_file():raise
    screen.write_json(folder/(label+"-result.json"),summary)
