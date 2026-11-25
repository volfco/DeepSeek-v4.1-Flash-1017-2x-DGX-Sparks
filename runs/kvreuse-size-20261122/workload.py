"""Size the sharded-KV decode exchange (candidate 1: reuse fetched KV across a stretch) on the served, graphs-on build.

  python3 runs/kvreuse-size-20261122/workload.py <label> <out.json>

Runs on the head against a server started with TF_DS_ROUND_PROFILE=1 and TF_DS_CLOCK_MARKS=<n>. Each cell decodes real
rounds (a summary request that cannot finish in one token) at a fixed prompt depth and stream count. Around each cell
it scrapes /metrics, which makes rank 0 log its cumulative `round_clock_us` / `round_launches` / `profile` lines, and
differences the cumulative totals (per-round mean x rounds) so the cell's own rounds are isolated.
"""
import concurrent.futures as cf, glob, json, subprocess, sys, time, urllib.request
from tokenizers import Tokenizer

URL = "http://127.0.0.1:8000"
C = "dsv41-1017-exl3-tf"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
DEPTHS = [4096, 32768, 131072]
STREAMS = [1, 4]
MAX_TOKENS = 320
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."

# WORKLOAD_CORPUS (a JSON list of token ids): a frozen corpus. Without it the corpus is rebuilt from the repo's
# RESULTS.md files, which change as runs are recorded - prompts (and so draft acceptance and rounds a cell) then differ
# between suites; compare round times across suites only on a frozen corpus.
import os
if os.environ.get("WORKLOAD_CORPUS"):
    corpus_ids = json.load(open(os.environ["WORKLOAD_CORPUS"]))
else:
    files = sorted(glob.glob("runs/**/RESULTS.md", recursive=True)) + sorted(glob.glob("docs/*.md"))
    corpus_ids = []
    for f in files:
        corpus_ids += TOK.encode(open(f, errors="replace").read()).ids
        if len(corpus_ids) > max(DEPTHS) * 5:
            break


# The cells: DEPTHS/STREAMS/MAX_TOKENS override the defaults above (the capacity suite runs 8 and 16 streams).
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", ",".join(map(str, DEPTHS))).split(",") if x.strip()]
STREAMS = [int(x) for x in os.environ.get("STREAMS", ",".join(map(str, STREAMS))).split(",") if x.strip()]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", str(MAX_TOKENS)))


def prompt(depth, salt):
    off = (salt * 7919) % max(1, len(corpus_ids) - depth)
    return TOK.decode(corpus_ids[off:off + depth]) + ASK


def chat(p):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}],
            "max_tokens": MAX_TOKENS, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        o = json.load(r)
    return {"s": time.time() - t, "usage": o.get("usage"), "finish": o["choices"][0].get("finish_reason")}


def scrape():
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 1))
    urllib.request.urlopen(URL + "/metrics", timeout=30).read()
    time.sleep(1.0)
    log = subprocess.run(["docker", "logs", "--since", since, C], capture_output=True, text=True).stderr + \
          subprocess.run(["docker", "logs", "--since", since, C], capture_output=True, text=True).stdout
    got = {}
    for line in log.splitlines():
        i = line.find("{\"rank\": 0, ")
        if i < 0:
            continue
        try:
            o = json.loads(line[i:])
        except ValueError:
            continue
        for k in ("profile", "round_clock_us", "round_launches", "phases_ms"):
            if k in o:
                got[k] = o[k]
    return got


def totals(s):
    rc = s.get("round_clock_us") or {}
    n = rc.get("rounds", 0)
    out = {"rounds": n}
    for k, v in rc.items():
        if isinstance(v, list):
            out[k] = (v[0] * n, v[1] * n)   # us total, marks total
    p = s.get("profile") or {}
    pr = p.get("rounds", 0)
    for k, v in (s.get("phases_ms") or {}).items():   # per-round means over the timer's rounds (= profile rounds)
        out["ph_" + k] = (v * pr, 0)
    out["_prof_rounds"] = pr
    out["_forward_ms_total"] = p.get("forward_ms", 0) * p.get("rounds", 0)
    return out


def diff(a, b):
    n = b["rounds"] - a["rounds"]
    d = {"rounds": n}
    pr = b["_prof_rounds"] - a["_prof_rounds"]
    for k, v in b.items():
        if k.startswith("ph_") and pr > 0:
            d[k] = round((v[0] - a.get(k, (0, 0))[0]) / pr, 3)
        elif isinstance(v, tuple):
            u0, c0 = a.get(k, (0, 0))
            if n > 0:
                d[k] = [round((v[0] - u0) / n, 2), round((v[1] - c0) / n, 2)]
    d["forward_ms_per_round"] = round((b["_forward_ms_total"] - a["_forward_ms_total"]) / pr, 3) if pr > 0 else None
    d["profile_rounds"] = pr
    return d


label, out = sys.argv[1], sys.argv[2]
res = {"label": label, "depths": DEPTHS, "streams": STREAMS, "max_tokens": MAX_TOKENS, "cells": []}
chat(prompt(512, 999))  # warm
# PREFILL_CELL=1: cold prefill of fresh 32K and 96K prompts (salted with the label and the clock, so no cache or spill
# entry serves them), 1 output token each: prefill tok/s for config sweeps
if os.environ.get("PREFILL_CELL") == "1":
    res["prefill"] = []
    for depth in (32768, 98304):
        p = f"[cold {label} {time.time():.6f}]\n" + prompt(depth, 31 * depth + 5)
        body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": 1,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
            o = json.load(r)
        dt = time.time() - t
        u = o.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        res["prefill"].append({"depth": depth, "s": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"), "cached": cached,
                               "tok_s": round((u.get("prompt_tokens", 0) - cached) / dt, 1)})
        print(json.dumps({"prefill": res["prefill"][-1]}), flush=True)
for depth in DEPTHS:
    for n in STREAMS:
        ps = [prompt(depth, 100 * depth + i) for i in range(n)]
        # prefill each history first (one at a time) so the measured cell is decode-dominated
        for p in ps:
            body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": 1,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                                          {"Content-Type": "application/json"}), timeout=1800).read()
        a = totals(scrape())
        t = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            reqs = list(ex.map(chat, ps))
        wall = time.time() - t
        b = totals(scrape())
        cell = {"depth": depth, "streams": n, "wall_s": round(wall, 2), "requests": reqs, "per_round": diff(a, b)}
        res["cells"].append(cell)
        print(json.dumps({k: cell[k] for k in ("depth", "streams", "wall_s")}), json.dumps({k: v for k, v in cell["per_round"].items() if k in ("profile_rounds", "forward_ms_per_round", "ph_sh_fetch", "ph_sh_rows", "ph_moe", "ph_a_idx", "ph_attn")}), flush=True)
        json.dump(res, open(out, "w"), indent=1)
print("done", out)
