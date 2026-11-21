"""Compare matched warm-turn cells. Inputs are an A/B/A bracket of sample.py JSONL files.

A single input prints baseline or admission-control summaries. Two inputs compare
the --warm-chat 0119 trial (off, on) or the depth replay (best, same-input production).
Pair checks for the three-run
bracket compare corpus hashes, request seeds, cached lengths, output lengths,
and uninstrumented running image IDs. The salts intentionally change prompt bytes.
"""
import argparse
import collections
import json
import statistics


def checked(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if rows[0]["kind"] != "start" or rows[-1]["kind"] != "complete":
        raise ValueError(f"{path}: incomplete round")
    if any(not r["clean"] for r in rows if r["kind"] == "cell"):
        raise ValueError(f"{path}: unclean cell")
    return rows


def request_values(r):
    done = r["done"]
    return {"ttft_s": r["ttft"], "completion_s": r["total"], "output_tok_s": (r["tokens"]-1)/(r["last"]-r["first"]),
            "encode_ms": float(done["encode_s"])*1000, "prefill_s": float(done["prefill_s"]),
            "tokens_per_round": r["tokens"]/int(done["rounds"]), "queue_s": float(done["queue_s"])}


def cells(rows):
    out = {}
    for c in rows:
        if c["kind"] == "cell":
            key = (c["depth"], c["cell"], tuple(r["seed"] for r in c["requests"]))
            if key in out:
                raise ValueError(f"Duplicate cell key {key}")
            out[key] = c
    return out


def burst_values(c):
    req = c["requests"]
    wall = max(r["start"] + r["total"] for r in req) - min(r["start"] for r in req)
    return {"burst_wall_s": wall, "burst_tok_s": sum(r["tokens"] for r in req)/wall,
            "max_queue_s": max(float(r["done"]["queue_s"]) for r in req),
            "median_ttft_s": statistics.median(r["ttft"] for r in req), "all_decoding_s": max(0, c["overlap"])}


def compare(trials):
    histories = [{r["depth"]: r["unsalted_sha256"] for r in t if r["kind"] == "history"} for t in trials]
    if not (histories[0] == histories[1] == histories[2]) or not histories[0]:
        raise ValueError("History payload changed across arms")
    if len({t[0]["salt"] for t in trials}) != 3:
        raise ValueError("Cross-arm cache namespace reused")
    for rank in range(2):
        if len({t[0]["ranks"][rank]["image_id"] for t in trials}) != 1:
            raise ValueError(f"Image changed on rank {rank}")
        env_a, env_b, env_return = (t[0]["ranks"][rank]["env"] for t in trials)
        if env_a != env_return:
            raise ValueError(f"Return baseline differs on rank {rank}")
        candidate = {"TF_DS_ENGRAM_LEAD":"1", "TF_DS_ENGRAM_SCALES":"map", "TF_DS_ENGRAM_STRICT":"1",
                     "TENSORFOLD_TOKEN_CACHE_MB":"32", "TF_DS_DEVICE_ACCEPT":"1", "TF_DS_PDL":"1",
                     "TF_DS_COMPRESS_SIDE":"1", "TF_DS_FUSED_COMPRESS2":"1"}
        if env_b != dict(env_a, **candidate) or env_a == env_b:
            raise ValueError(f"Named B flags do not match the candidate on rank {rank}")
    blocks = [cells(t) for t in trials]
    if not (blocks[0].keys() == blocks[1].keys() == blocks[2].keys()):
        raise ValueError("Request schedule changed")
    print("depth,cell,metric,A_median,B_median,A_return_median,B_vs_mean_paired_refs_pct,ref_return_vs_A_pct")
    groups = collections.defaultdict(list)
    for key in blocks[0]:
        group = [b[key] for b in blocks]
        for req in zip(*(c["requests"] for c in group)):
            if len({r["cached"] for r in req}) > 1 or len({r["tokens"] for r in req}) > 1:
                raise ValueError(f"{key}: compare cache depth and output length before ranking")
            if max(r["prompt_tokens"] for r in req) - min(r["prompt_tokens"] for r in req) > 1:
                raise ValueError(f"{key}: prompt lengths differ by more than one token")
        if key[1] == "prime":
            continue
        if key[1] == "burst4":
            m = [burst_values(c) for c in group]
        else:
            m = [request_values(c["requests"][0]) for c in group]
        for name in m[0]:
            groups[(key[0], key[1], name)].append([x[name] for x in m])
    median = statistics.median
    for key, observations in groups.items():
        values = [median(xs) for xs in zip(*observations)]
        diffs = [100*(b/((a+c)/2)-1) for a,b,c in observations if a+c > 0]
        drift = [100*(c/a-1) for a,b,c in observations if a > 0]
        row = list(key) + [f"{x:.3f}" for x in values] + [f"{median(xs):+.2f}" if xs else "-" for xs in (diffs,drift)]
        print(",".join(map(str,row)))


def summary(rows):
    for c in rows:
        if c["kind"] != "cell" or c["cell"] == "prime":
            continue
        if c["cell"] in ("solo", "chat-prime", "warm-chat"):
            stats = request_values(c["requests"][0])
            if c["cell"] != "solo":
                r = c["requests"][0]
                stats.update(append=c["append"], cached=r["cached"], prompt=r["prompt_tokens"],
                             render_ms=1000*float(r["done"]["render_s"]), restore_ms=1000*float(r["done"]["restore_s"]))
        else:
            stats = burst_values(c)
            trace = [p for p in c.get("trace", []) if "per_part" in p]
            if trace:
                stats["max_live_lanes"] = max(sum(part["live"] for part in p["per_part"]) for p in trace)
                stats["min_free_rows_by_part"] = [min(p["per_part"][i]["free"] for p in trace) for i in range(2)]
                stats["peak_twin_rows"] = max(sum(part["twin_rows"] for part in p["per_part"]) for p in trace)
            stats.update(queue=[float(r["done"]["queue_s"]) for r in c["requests"]],
                         prefill=[float(r["done"]["prefill_s"]) for r in c["requests"]],
                         cached=[r["cached"] for r in c["requests"]],
                         no_target_increase=c["health_after"]["refused_no_target"]-c["health_before"]["refused_no_target"])
        for k in ("cell","depth","rep","sharing","temperature","budget","full_length"):
            if k in c:
                stats[k] = c[k]
        print(json.dumps(stats))


def compare_depth_replay(candidate, production):
    """Compare B against its same-salt production replay. Primes may have restored from spill."""
    for rank in range(2):
        b, a = (t[0]["ranks"][rank] for t in (candidate, production))
        if b["image_id"] != a["image_id"] or b["env"]["TF_DS_DEVICE_ACCEPT"] != "1" or a["env"]["TF_DS_DEVICE_ACCEPT"]:
            raise ValueError("Expected a best block and the production replay on the same image")
    signature = lambda t: (t[0]["salt"], t[0]["tasks"],
                           [(r["depth"], r["unsalted_sha256"]) for r in t if r["kind"] == "history"])
    if signature(candidate) != signature(production):
        raise ValueError("Depth fixtures differ")
    cb, ca = cells(candidate), cells(production)
    if cb.keys() != ca.keys():
        raise ValueError("Request schedule changed")
    for key, b in cb.items():
        a = ca[key]
        if b["cell"] == "prime":
            continue
        for rb, ra in zip(b["requests"], a["requests"]):
            if any(rb[k] != ra[k] for k in ("seed", "prompt_tokens", "cached", "tokens")):
                raise ValueError(f"{key}: cache, prompt or output length differs")
            match = rb["done"]["sha"] == ra["done"]["sha"]
            before, after = request_values(ra), request_values(rb)
            print(json.dumps({"depth": b["depth"], "cell": b["cell"], "seed": rb["seed"], "sha_match": match,
                              "production_sha": ra["done"]["sha"], "best_sha": rb["done"]["sha"],
                              "production": before, "best": after}))


def compare_chat(off, on):
    if off[0]["salt"] != on[0]["salt"]:
        raise ValueError("Use the same chat fixtures on both sides of the restart")
    flag = "TENSORFOLD_TOKEN_CACHE_MB"
    for rank in range(2):
        a, b = [t[0]["ranks"][rank] for t in (off, on)]
        if a["image_id"] != b["image_id"] or a["container_id"] == b["container_id"]:
            raise ValueError(f"Expected a restart into the same image on rank {rank}")
        left, right = dict(a["env"]), dict(b["env"])
        x, y = left.pop(flag), right.pop(flag)
        if left != right or x not in (None, "0") or y not in (("32",) if rank == 0 else (None, "0", "32")):
            raise ValueError(f"Expected an isolated 32 MiB token-cache trial on rank {rank}")
    sigs = [[r for r in trial if r["kind"] == "history"] for trial in (off, on)]
    for field in ("history_sha256", "assistant_sha256", "expected_warm_grid", "schedule"):
        if len(sigs[0]) != 1 or len(sigs[1]) != 1 or sigs[0][0][field] != sigs[1][0][field]:
            raise ValueError(f"Mismatched chat fixture {field}")
    blocks = [[c for c in t if c["kind"] == "cell"] for t in (off, on)]
    expected = {(s,r) for s in (100,500,2000) for r in range(3)}
    for cells_ in blocks:
        if len(cells_) != 10 or cells_[0]["cell"] != "chat-prime" or any(c["cell"] != "warm-chat" for c in cells_[1:]):
            raise ValueError("Incomplete chat measurements")
        if {(c["append"], c["rep"]) for c in cells_[1:]} != expected:
            raise ValueError("Warm appends or repeats missing")
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    for a, b in zip(*blocks):
        x, y = a["requests"][0], b["requests"][0]
        if ((a["cell"],a["append"],a["rep"]) != (b["cell"],b["append"],b["rep"])
                or x["input_sha256"] != y["input_sha256"] or x["prompt_tokens"] != y["prompt_tokens"]):
            raise ValueError("Request payloads differ")
        if a["cell"] == "warm-chat" and x["cached"] != y["cached"]:
            raise ValueError("Warm KV cuts differ")
        entry = {"cell": a["cell"], "append": a["append"], "rep": a["rep"], "seed": x["seed"],
                 "prompt": x["prompt_tokens"], "cached_off_on": [x["cached"], y["cached"]],
                 "token_hash_off_on": [x["done"]["sha"], y["done"]["sha"]], "tokens_off_on": [x["tokens"],y["tokens"]]}
        for field in ("render_s","encode_s","queue_s","restore_s","prefill_s","ttft_s","client_ttft","client_wall"):
            if field in ("client_ttft","client_wall"):
                pair = [1000*r["ttft" if field == "client_ttft" else "total"] for r in (x, y)]
            else:
                pair = [1000*float(r["done"][field]) for r in (x, y)]
            entry[field.removesuffix("_s") + "_ms_off_on"] = [round(val,2) for val in pair]
            if a["cell"] == "warm-chat":
                results[a["append"]][field].append(pair[1] - pair[0])
        print(json.dumps(entry))
    for append, fields in sorted(results.items()):
        diffs = {k: [round(f(v),2) for f in (min,statistics.median,max)] for k,v in fields.items()}
        print(json.dumps({"append": append, "paired_on_minus_off_ms_min_median_max": diffs}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    if len(args.files) not in (1,2,3):
        p.error("provide one run, an off/on chat pair, a best/production replay pair, or an A/B/A-return bracket")
    runs = [checked(path) for path in args.files]
    if len(runs) == 1:
        summary(runs[0])
    elif len(runs) == 2:
        if any(r.get("cell") == "warm-chat" for r in runs[0]):
            compare_chat(*runs)
        else:
            compare_depth_replay(*runs)
    else:
        compare(runs)
