"""Recompute paired measurements from completed workload artifacts, including all-active windows."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"admission-grace"))
import summarize as grace

def events(path):
    with path.open() as f:return [json.loads(s) for s in f if s.strip()]

def cells(path):
    data=events(path)
    for event in (data[0], data[-1]):
        for rank in event["ranks"]:
            if isinstance(rank, dict) and isinstance(rank.get("mounts"), list) and all(isinstance(m, str) for m in rank["mounts"]):
                rank["mounts"] = sorted(rank["mounts"])
    assert data[0]["kind"]=="start" and data[-1]["kind"]=="complete" and data[-1]["ranks"]==data[0]["ranks"],path
    rows=[r for r in data if r["kind"]=="cell"]
    assert all(r["clean"] and r["own"]==r["completed_delta"]==len(r["requests"]) for r in rows),path
    return rows

def print_rows(rows):
    w=csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator="\n");w.writeheader()
    for r in rows:w.writerow({k:round(v,6) if isinstance(v,float) else v for k,v in r.items()})

def adaptive(dirs):
    results=[]
    for depth in ("shallow","deep"):
        base=None
        for arm in ("five-a","adaptive","half","five-b"):
            path=find(dirs,f"{arm}-{depth}.jsonl")
            cur={(c["sharing"],c["rep"]):c for c in cells(path) if c["cell"]=="admission"}
            if base is None:base=cur
            assert set(cur)==set(base)
            for k,c in cur.items():
                assert c["full_length"] and c["health_before"]["refused_no_target"]==c["health_after"]["refused_no_target"]
                results.append({"arm":arm,"depth":depth,"shape":k[0],"rep":k[1],**grace.metric(c,base[k])})
    return results

def find(dirs,name):
    paths=[d/name for d in dirs if (d/name).is_file()]
    if not paths:raise ValueError("Missing artifact "+name)
    return paths[-1]

def clocks(dirs):
    rows=[]
    for p in sorted({p.name for d in dirs for p in d.glob("*.clocks.jsonl")}):
        records=events(find(dirs,p))
        for i in (0,1):
            readings=[s["ranks"][i].split(",") for s in records if "ranks" in s]
            if not readings:continue
            mhz=[int(r[0].split()[0]) for r in readings]
            rows.append(dict(job=p.removesuffix(".clocks.jsonl"),rank=i,samples=len(readings),
                             median_mhz=st.median(mhz),min_mhz=min(mhz),max_temp_c=max(int(r[1]) for r in readings),
                             active_clock_reason_samples=sum(int(r[2].strip(),16)!=0 for r in readings)))
    return rows

def profile(dirs):
    frames=events(find(dirs,"frames.jsonl"))
    # The clock count on the pinned image starts at 0xaaaaaaaaaaaaaaaa.
    # Its increments track the zero-based completed-forward counter. Ignore
    # scrapes taken between updating those two counters in a live round.
    offsets=collections.Counter(f["clock_rounds"]-f["profile"]["rounds"] for f in frames)
    bias=offsets.most_common(1)[0][0]
    if bias not in (0,0xaaaaaaaaaaaaaaaa):raise ValueError("Unrecognized profile/clock count offset")
    frames=[f for f in frames if f["clock_rounds"]-f["profile"]["rounds"]==bias]
    offset=st.mean(json.loads(find(dirs,"clock.json").read_text()))
    rows=[]
    for cell in cells(find(dirs,"workload.jsonl")):
        if cell["cell"]=="chat-prime":continue
        reqs=cell["requests"]
        first=max(r["first"] for r in reqs)+.40
        last=min(r["last"] for r in reqs)-.40
        inside=[f for f in frames if first+offset<f["at"]<last+offset]
        if len(inside)<2:raise ValueError("No all-active phase interval")
        a,b=inside[0],inside[-1]
        r0,r1=a["profile"]["rounds"],b["profile"]["rounds"]
        rounds=r1-r0
        if rounds<10 or b["clock_rounds"]-a["clock_rounds"]!=rounds:raise ValueError("Inconsistent profile counters")
        # Both clocks and the eager phase timer count a completed synchronized forward.
        # Native phase means are printed to .001 ms and row means to .01 rows.
        measured={k:(m*r1-a["phases_ms"][k]*r0)/rounds
                  for k,m in b["phases_ms"].items()}
        measures=dict(depth=cell["depth"],width=len(reqs),shape=cell["sharing"],rep=cell["rep"],
                      rounds=rounds,active_s=b["at"]-a["at"],
                      rows_per_round=(b["profile"]["rows_a_round"]*r1-a["profile"]["rows_a_round"]*r0)/(r1-r0))
        rows.append({**measures,**measured})
    return rows

def active_segment(c):
    reqs=c["requests"];n=len(reqs)
    # Only use the longest contiguous run wholly between all first and last emitted text events.
    lo=max(r["first"] for r in reqs)+.25;hi=min(r["last"] for r in reqs)-.25
    segments=[[]]
    for p in c["trace"]:
        if (lo<p["t"]<hi and p["requests_running"]==n and p["requests_waiting"]==0
            and all(p[k] is not None for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds","generation_tokens_running"))):
            segments[-1].append(p)
        elif segments[-1]:segments.append([])
    return max(segments,key=len)

def decode(dirs,arms=("current-a","prefetch-off","current-b")):
    rows=[]
    reference={}
    for arm in arms:
        for c in cells(find(dirs,arm+"-decode.jsonl")):
            if c["cell"]=="chat-prime":continue
            reqs=c["requests"];n=len(reqs)
            segment=active_segment(c)
            if len(segment)<2:raise ValueError(f"No common window for {arm} {c['depth']} {c['rep']} {n}")
            a,b=segment[0],segment[-1]
            rounds=sum(b[k]-a[k] for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds"))
            tokens=b["generation_tokens_running"]-a["generation_tokens_running"];wall=b["t"]-a["t"]
            assert rounds>0 and tokens>0 and wall>=2,(arm,c["depth"],n,wall)
            if arm=="owner" and b["kv_shard_owner_rounds"]==a["kv_shard_owner_rounds"]:
                raise ValueError("Owner decode path inactive in this measured window")
            if n==1:
                cd=c["counter_delta"]
                assert int(cd["decode_rounds_total"])==int(reqs[0]["done"]["rounds"])
            fingerprint=tuple((r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in reqs)
            key=(c["depth"],n,c["sharing"],c["rep"])
            reference.setdefault(key,fingerprint)
            assert all(a[:3]==b[:3] for a,b in zip(reference[key],fingerprint))
            accepted=[list(map(int,r["done"]["accepted"].split("/"))) for r in reqs]
            part0,part1=c["health_before"]["per_part"]
            rows.append(dict(arm=arm,depth=c["depth"],width=n,shape=c["sharing"],rep=c["rep"],active_s=wall,rounds=rounds,
                start_kept_part0=part0["kept_rows"],start_kept_part1=part1["kept_rows"],
                new_placement_refusals=c["health_after"]["refused_no_target"]-c["health_before"]["refused_no_target"],
                cache_cut=reqs[0]["cached"],slowest_queue_s=max(float(r["done"]["queue_s"]) for r in reqs),
                physical_ms_per_round=1000*wall/rounds,committed_tokens_per_stream_round=tokens/(n*rounds),
                active_tok_s=tokens/wall,accepted=sum(v[0] for v in accepted),drafted=sum(v[1] for v in accepted),
                full_request_rounds=sum(int(r["done"]["rounds"]) for r in reqs),
                hashes_match=sum(a==b for a,b in zip(reference[key],fingerprint)),total_hashes=n))
    return rows

def owner(dirs):
    rows=decode(dirs,("current-a","current-b","owner","fetch-return"))
    for r in rows:
        if r["arm"] in ("owner","fetch-return") and r["new_placement_refusals"]:
            raise ValueError("KV pressure in clean owner/fetch comparison")
    return rows

def cache(dirs):
    all_rows={}
    for arm in ("current-a","cache-off","current-b"):
        cur=cells(find(dirs,arm+"-cache.jsonl"))
        all_rows[arm]={(r["append"],r["rep"]):r["requests"][0] for r in cur if r["cell"]=="warm-chat"}
    results=[]
    for (size,rep),a in all_rows["current-a"].items():
        off=all_rows["cache-off"][size,rep];b=all_rows["current-b"][size,rep]
        assert len({grace.fingerprint(r) for r in (a,off,b)})==1
        assert len({r["done"]["sha"] for r in (a,off,b)})==1
        results.append(dict(append=size,rep=rep,cached=a["cached"],on_a_ttft=a["ttft"],off_ttft=off["ttft"],on_b_ttft=b["ttft"],
            paired_saved_ms=1000*(off["ttft"]-(a["ttft"]+b["ttft"])/2),
            on_a_encode=float(a["done"]["encode_s"]),off_encode=float(off["done"]["encode_s"]),on_b_encode=float(b["done"]["encode_s"])))
    assert len(results)==9
    return results

def quality(dirs):
    arms=("current-a","prune-off","cap12","current-b")
    rec={a:json.loads(find(dirs,a+".score.json").read_text()) for a in arms}
    assert len({r["meta"]["set_sha"] for r in rec.values()})==1 and all(r["meta"]["scored"]==40 for r in rec.values())
    results=[]
    for ref in ("current-a","prune-off"):
        for a in arms:
            by_source=collections.defaultdict(list)
            for k,row in rec[a]["items"].items():
                delta=row["nll_mean"]-rec[ref]["items"][k]["nll_mean"]
                by_source[k.rsplit("-",1)[0]].append(delta)
            paired=[st.mean(vals) for vals in by_source.values()]
            assert len(paired)==20 and all(len(v)==2 for v in by_source.values())
            spread=2.093*st.stdev(paired)/math.sqrt(len(paired))
            results.append(dict(arm=a,reference=ref,items=40,sources=20,nll=rec[a]["overall"]["nll_mean"],delta=st.mean(paired),
                                ci95_low=st.mean(paired)-spread,ci95_high=st.mean(paired)+spread))
    return results

def tasks(dirs):
    results=[]
    for arm in ("current-a","prune-off","cap12","current-b"):
        rows=events(find(dirs,arm+".tasks.jsonl"))
        for n in (1,4,8):
            batches=[r for r in rows if r["width"]==n]
            req=[v for r in batches for v in r["results"]]
            assert len(req)==32 and len({(r["kind"],r["item"],r["seed"]) for r in req})==32
            all_active=[]
            active_tokens=active_rounds=active_seconds=0
            usable_windows=0
            for row in batches:
                rs=row["results"]
                end=min(r["engine_end"] for r in rs)
                start=max(r["engine_end"]-float(r["done"]["decode_s"]) for r in rs)
                intersection=end-start
                all_active.append(max(0,intersection))
                # Decode durations on done lines are rounded to 0.1 s. Also require that
                # the actual metrics snapshot saw every expected stream and zero waiters.
                epoch=row["start_epoch_s"]
                common=[p for p in row["trace"] if start+.10<epoch+p[0]<end-.10
                        and p[1]==n and p[5]==0 and all(v is not None for v in p[2:5])]
                if len(common)>=2:
                    a,b=common[0],common[-1]
                    rounds=sum(b[3:5])-sum(a[3:5])
                    if rounds>0 and b[2]>a[2]:
                        active_rounds+=rounds
                        active_tokens+=b[2]-a[2]
                        active_seconds+=b[0]-a[0]
                        usable_windows+=1
            results.append(dict(arm=arm,width=n,sql_pass=sum(r["success"] for r in req if r["kind"]=="sql"),sql_total=16,
                tool_pass=sum(r["success"] for r in req if r["kind"]=="tool"),tool_total=16,batches=len(batches),
                batches_all_decoding=sum(v>0 for v in all_active),median_common_s=st.median(all_active),
                measured_windows=usable_windows,common_rounds=active_rounds,common_tokens=active_tokens,
                common_ms_per_round=1000*active_seconds/active_rounds if active_rounds else None,
                common_tokens_per_stream_round=active_tokens/(n*active_rounds) if active_rounds else None,
                generated_tok_s=sum(r["tokens"] for r in req)/sum(r["wall_s"] for r in batches),
                median_wall_s=st.median(r["wall"] for r in req)))
    return results

if __name__=="__main__":
    funcs={"adaptive":adaptive,"decode":decode,"cache":cache,"quality":quality,"tasks":tasks,"profile":profile,"clocks":clocks,
           "owner":owner}
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=funcs);p.add_argument("dirs",nargs="+",type=Path)
    args=p.parse_args();print_rows(funcs[args.mode](args.dirs))
