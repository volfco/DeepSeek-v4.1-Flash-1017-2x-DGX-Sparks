"""Run from the recipe root on the Spark head. Each peak runs on a different serving process."""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

base = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path("runs/review-round").resolve()))
sys.path.insert(0, str(base))
import sample as client
import round as trial

starts = ["shared", "f0", "f1", "f2"]
grid = 299008


def health():
    with client.bench.OPENER.open(client.srv.url + "/health", timeout=10) as res:
        raw = json.load(res)
    shard = raw["kv_shard"]
    fields = ("plans","checks","staled","twins_made","twins_revoked","moves","moved_tokens","carries",
              "carry_pressure","refused_pinned","refused_twin","refused_loading","refused_partial","refused_no_target",
              "refused_small","no_part","transfers_failed")
    p = raw["prompt_cache"]
    counters = {k:shard[k] for k in fields}
    counters["cache"] = {k:p[k] for k in ("memory_hits","memory_misses","misses_stale","evicted_lru",
        "evicted_superseded","disk_lookups","disk_hits_committed","restores_begun","restored_tokens","restores_failed")}
    return {"status": raw["status"], "pair":raw["pair"], "live":raw["live"], "per_part":shard["per_part"],
            "refused_no_target":shard["refused_no_target"], "kv_counts":counters}


def traffic():
    # Last completed benchmark campaign's restore/validation boundary. No bodies, identifiers,
    # raw log lines, IPs or individual timestamps leave the head.
    boundary = "2026-11-18T01:27:19Z"
    stop = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lines = subprocess.run(["docker","logs","--since",boundary,"--until",stop,client.name],
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True).stdout.splitlines()
    counts = collections.Counter()
    for line in lines:
        if "done " not in line or "prompt=" not in line or "prefill_s=" not in line:
            continue
        d = dict(x.split("=",1) for x in line.split() if "=" in x)
        if not {"prompt","cached","prefill_s","queue_s","tokens"} <= d.keys():
            continue
        counts["completed"] += 1
        counts["deep_100k"] += int(d["prompt"]) >= 100000
        counts["queue_ge_1s"] += float(d["queue_s"]) >= 1
        counts["prefill_ge_1s"] += float(d["prefill_s"]) >= 1
    client.emit("traffic", start=boundary, end=stop, completed=counts["completed"], counts=dict(counts))


def history():
    src = client.bench.Corpus("runs/combo4/corpus").text("div",360000,"depth-1117-hay",fixed=True)
    cut = 4*300576
    for _ in range(4):
        ctx = "[nov17-deep-admission-5]\n" + src[:cut] + "\n\n"
        n = client.token_count([{"role":"user","content":ctx+client.tasks[0]}])
        if abs(n-300576) <= 32:
            break
        cut = round(cut*300576/n)
    else:
        raise RuntimeError("Fixture differs")
    prime = "Summarize the preceding context in detail."
    return [ctx+t for t in client.tasks], {"shared":ctx+prime, **{f"f{i}":
            f"[review-occupancy-5-{i}]\n"+src[:cut]+"\n\n"+prime for i in range(3)}}


def rows():
    return [p["kept_rows"] for p in health()["per_part"]]


def probe(key, tag, rep, prompt, check=None):
    seed = 4199 if key=="shared" else 4250+int(key[1])
    return client.cell(300000,"chat-prime",[prompt],[seed],budget=16,sharing=tag+"-"+key,rep=rep,
                       expected_cached=check)[0]


def burst(prompts,tag,rep):
    r = client.cell(300000,"admission",prompts,list(range(4200,4204)),budget=1024,rep=rep,sharing=tag,
                    expected_cached=grid)
    if any(q["tokens"]!=1024 for q in r):
        raise RuntimeError("Unequal generated burst lengths")
    return [(q["input_sha256"],q["done"]["sha"]) for q in r]


def run():
    client.args = argparse.Namespace(arm="kv-pressure",salt="5",occupancy=True,replay_depth=False,finish_grace=False)
    client.health = health
    trial.out = base
    baseline = trial.identity()
    if any(flags for _,flags in baseline):
        raise RuntimeError("Unexpected production overrides")
    trial.idle()
    health0 = health()
    client.emit("start", settings=baseline,health=health0,memory_gib=client.memory())
    traffic()
    if min(client.memory()) < 5.2:
        raise RuntimeError("Insufficient memory for the 300K fixture")
    prompts, cases = history()
    orders = (("shared","f0","f1","f2"),("f1","shared","f2","f0"),("f2","f0","shared","f1"))
    revisits = (("f0","f1","f2"),("f1","f2","f0"),("f2","f0","f1"))
    expected = None
    try:
        for rep, order in enumerate(orders):
            if rep:
                trial.idle()
                trial.restart("reset-"+str(rep),{})
                if trial.identity()!=baseline:
                    raise RuntimeError("Reset settings differ")
            beginning = rows()
            if sum(beginning)>32768:
                raise RuntimeError(f"Occupied before reset trial: {beginning}")
            client.emit("stage",stage="reset",rep=rep,rows=beginning,order=order,first_revisit=revisits[rep][0])
            for ix,key in enumerate(order):
                probe(key,"fill",rep,cases[key])
                if rep==0 and ix==0:
                    expected = burst(prompts,"low",rep)
                if key != "shared":
                    probe(key,"pre-pressure",rep,cases[key],grid)
            peak = rows()
            if not (4*grid<=sum(peak)<=4*grid+32768 and min(peak)>=grid):
                raise RuntimeError(f"Unexpected peak occupancy {peak}")
            client.emit("stage",stage="peak",rep=rep,rows=peak)
            if burst(prompts,"peak",rep)!=expected:
                raise RuntimeError("Peak replay differs from control")
            for key in revisits[rep]:
                probe(key,"after-pressure",rep,cases[key])
            client.emit("stage",stage="revisited",rep=rep,rows=rows())
    finally:
        trial.idle()
        trial.restart("restore",{})
        if trial.identity()!=baseline:
            raise RuntimeError("Restored settings differ")
        trial.validate("restore")
        trial.idle()
        client.emit("restored",settings=trial.identity(),health=health(),memory_gib=client.memory())
    client.emit("complete")


if __name__=="__main__":
    run()
