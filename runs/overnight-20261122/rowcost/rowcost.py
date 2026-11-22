"""Forward ms a round by rows, on the served graphs-on build (for a measured TF_DS_ROUND_MS: M1 in
docs/benchmark-gaps-and-moe-review.md). Run under TF_DS_FIX_K=K (every drafted stream verifies K drafts a round) so a
cell's rows are about streams x (K + 1).

  python3 runs/overnight-20261122/rowcost/rowcost.py <label> <out.json>

Cells: 4K x 1/2/3/4 streams and 32K x 1/4 streams, greedy, 200 tokens each, prompts from the frozen corpus
(WORKLOAD_CORPUS). Each cell's rounds, rows a round, forward ms a round and the whole round's wall ms (enqueue +
forward + absorb + logits + sample, the profile's host-side stages) come from differenced /metrics profile lines (rank 0).
"""
import concurrent.futures as cf, json, os, subprocess, sys, time, urllib.request
from tokenizers import Tokenizer

URL = "http://127.0.0.1:8000"
C = "dsv41-1017-exl3-tf"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
CELLS = [(4096, 1), (4096, 2), (4096, 3), (4096, 4), (32768, 1), (32768, 4)]
MAX_TOKENS = 200
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."
corpus = json.load(open(os.environ.get("WORKLOAD_CORPUS", "runs/overnight-20261122/corpus.json")))
KEYS = ("forward_ms", "enqueue_ms", "absorb_ms", "logits_ms", "sample_ms", "rows_a_round", "pass_ms", "passes")


def prompt(depth, salt):
    off = (salt * 7919) % max(1, len(corpus) - depth)
    return TOK.decode(corpus[off:off + depth]) + ASK


def chat(p):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                                       {"Content-Type": "application/json"}), timeout=1800) as r:
        return json.load(r).get("usage")


def profile():
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 1))
    urllib.request.urlopen(URL + "/metrics", timeout=30).read()
    time.sleep(1.0)
    r = subprocess.run(["docker", "logs", "--since", since, C], capture_output=True, text=True)
    got = None
    for line in (r.stdout + r.stderr).splitlines():
        i = line.find('{"rank": 0, "profile"')
        if i >= 0:
            try:
                got = json.loads(line[i:])["profile"]
            except ValueError:
                pass
    return got


def tot(p):  # cumulative totals from per-round means
    n = p["rounds"]
    t = {"rounds": n, "passes": p.get("passes", 0)}
    for k in KEYS:
        if k != "passes" and k in p:
            t[k] = p[k] * (p["passes"] if k == "pass_ms" else n)
    return t


label, out = sys.argv[1], sys.argv[2]
res = {"label": label, "fix_k": os.environ.get("TF_DS_FIX_K"), "cells": []}
chat(prompt(512, 999))
for depth, streams in CELLS:
    ps = [prompt(depth, 31 * depth + s) for s in range(streams)]
    for p in ps:  # prefill first (one token each), so the timed window is decode only
        body_p = p
        b = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": body_p}], "max_tokens": 1,
             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(),
                                                      {"Content-Type": "application/json"}), timeout=1800).read()
    a = tot(profile())
    t0 = time.time()
    with cf.ThreadPoolExecutor(streams) as ex:
        us = list(ex.map(chat, ps))
    wall = time.time() - t0
    b = tot(profile())
    n = b["rounds"] - a["rounds"]
    cell = {"depth": depth, "streams": streams, "rounds": n, "wall_s": round(wall, 2),
            "tokens": sum((u or {}).get("completion_tokens", 0) for u in us)}
    for k in KEYS:
        if k in ("passes",):
            continue
        if k == "pass_ms":
            dp = b["passes"] - a["passes"]
            cell["pass_ms"] = round((b.get(k, 0) - a.get(k, 0)) / dp, 3) if dp > 0 else None
            cell["passes_a_round"] = round(dp / n, 3) if n > 0 else None
        elif n > 0:
            cell[k] = round((b.get(k, 0) - a.get(k, 0)) / n, 3)
    res["cells"].append(cell)
    print(json.dumps(cell), flush=True)
json.dump(res, open(out, "w"), indent=1)
