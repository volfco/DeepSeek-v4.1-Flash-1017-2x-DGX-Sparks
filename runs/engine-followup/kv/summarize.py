"""Summarize captured KV cells, including the sequence of global refusals and per-part state changes."""
import argparse
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent


def load(path):
    rows=[]
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def occupancy(h):
    return [p["kept_rows"] for p in h["per_part"]]


def burst(c):
    req=c["requests"]
    start=min(r["start"] for r in req)
    wall=max(r["start"]+r["total"] for r in req)-start
    a,b=c["health_before"],c["health_after"]
    transitions=[]
    prev=a
    for snap in [v for v in c["trace"] if "per_part" in v]+[dict(b,t=start+wall)]:
        before=prev["kv_counts"]
        after=snap["kv_counts"]
        delta={k:after[k]-before[k] for k in ("refused_no_target","staled","twins_revoked","moves","no_part")
               if before[k]!=after[k]}
        if delta or occupancy(prev)!=occupancy(snap):
            transitions.append({"s":round(snap["t"]-start,2),"delta":delta,"kept":occupancy(snap),
                                "free":[p["free"] for p in snap["per_part"]],"live":[p["live"] for p in snap["per_part"]]})
        prev=snap
    return {"rep":c["rep"],"cell":c["sharing"],"kept":occupancy(a),"kept_after":occupancy(b),
            "queue_s":[float(r["done"]["queue_s"]) for r in req],"ttft_s":[round(r["ttft"],3) for r in req],
            "wall_s":round(wall,3),"aggregate_tok_s":round(sum(r["tokens"] for r in req)/wall,2),
            "common_s":round(max(0,c["overlap"]),3),"sha":[r["done"]["sha"] for r in req],
            "staled":b["kv_counts"]["staled"]-a["kv_counts"]["staled"],
            "no_target":b["refused_no_target"]-a["refused_no_target"],"transitions":transitions}


def analyze(rows, partial):
    if not partial and (rows[-1]["kind"]!="complete" or not any(r["kind"]=="restored" for r in rows)):
        raise ValueError("Incomplete campaign")
    cells=[r for r in rows if r["kind"]=="cell"]
    if any(not c["clean"] for c in cells):
        raise ValueError("Contaminated trial")
    bursts=[burst(c) for c in cells if c["cell"]=="admission"]
    bykey={}
    probes=[]
    for c in cells:
        if c["cell"]!="chat-prime":
            continue
        kind,key=c["sharing"].rsplit("-",1)
        r,=c["requests"]
        a,b=(h["kv_counts"]["cache"] for h in (c["health_before"],c["health_after"]))
        entry={"rep":c["rep"],"key":key,"phase":kind,"request_sha":r["input_sha256"],"reply_sha":r["done"]["sha"],
               "cached":r["cached"],"prompt":r["prompt_tokens"],"ttft_s":round(r["ttft"],4),
               "prefill_s":float(r["done"]["prefill_s"]),"restore_s":float(r["done"]["restore_s"]),
               "queue_s":float(r["done"]["queue_s"]), "kept_before":occupancy(c["health_before"]),
               "kept_after":occupancy(c["health_after"]), "mem_hits":b["memory_hits"]-a["memory_hits"],
               "disk_hits":b["disk_hits_committed"]-a["disk_hits_committed"],
               "misses_stale":b["misses_stale"]-a["misses_stale"]}
        if kind=="pre-pressure":
            bykey[(c["rep"],key)]=entry
        elif kind=="after-pressure":
            earlier=bykey[(c["rep"],key)]
            if (earlier["request_sha"],earlier["reply_sha"])!=(entry["request_sha"],entry["reply_sha"]):
                raise ValueError("Pre/post revisit token identity differs")
            entry["ttft_change_s"]=round(entry["ttft_s"]-earlier["ttft_s"],4)
        probes.append(entry)
    targets = (2,3) if rows[0]["arm"]=="kv-partition" else (4,9)
    if not partial and (len(bursts),sum(p["phase"]=="after-pressure" for p in probes))!=targets:
        raise ValueError("Missing required cells")
    return {"traffic":next((r for r in rows if r["kind"]=="traffic"),None),
            "bursts":bursts,"probes":probes,"resets":[r for r in rows if r["kind"]=="stage"],
            "restored":any(r["kind"]=="restored" for r in rows)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",nargs="?",type=pathlib.Path,default=root/"run.log")
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    print(json.dumps(analyze(load(args.path),args.partial),indent=2))
