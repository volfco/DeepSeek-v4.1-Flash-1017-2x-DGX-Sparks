#!/usr/bin/env python3
"""Decode at fixed context depth + warm-turn appends, against the pair as it serves (client requests only).

Runs on the head from runs/depth/. Reuses tools/bench_div.py (Corpus, guarded, multi) and the overnight corpus.
  python3 -I runs/depth/depth.py ladder|warm|all
Every request's server-side numbers come from its `done` line in the rank-0 container log (by chatcmpl id).
Memory: mem.sh samples MemAvailable on both machines each second (mem.head / mem.worker); the driver gates every
cell predictively against a 3.5 GiB floor (watchdog 2.5 GiB + ~1 GiB margin) and never cuts a request in flight.
"""
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = os.path.expanduser("~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks")
sys.path.insert(0, f"{ROOT}/tools")
import bench_div as bd  # noqa: E402

OUT = f"{ROOT}/runs/depth"
URL = "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4.1-Flash-1017"
CONTAINER = "dsv41-1017-exl3-tf"
SALT = "[depth-1117] "
SEED = "depth-1117-hay"
FLOOR_GIB = 3.5
DEC_TOKENS = 1024
LADDER = [100_000, 125_000, 140_000, 200_000, 250_000, 270_000, 300_000]
SHORT = 2_000
srv = bd.Server(URL, MODEL)
corpus = bd.Corpus(f"{ROOT}/runs/overnight/corpus")
HAY = corpus.text("div", 380_000, SEED, fixed=True)
STATE = {"cpt": 4.0}
DOC_PATH = f"{OUT}/depth.json"
DOC = json.load(open(DOC_PATH)) if os.path.exists(DOC_PATH) else {"cells": [], "mem_model": {}}


def log(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    print(line, flush=True)
    with open(f"{OUT}/depth.log", "a") as fh:
        fh.write(line + "\n")


def save():
    with open(DOC_PATH + ".tmp", "w") as fh:
        json.dump(DOC, fh, indent=1)
    os.replace(DOC_PATH + ".tmp", DOC_PATH)


def hay(tokens):
    """A prefix of HAY of about `tokens` tokens, cut at a paragraph boundary (every rung extends the previous)."""
    c = int(tokens * STATE["cpt"])
    cut = HAY.rfind("\n\n", 0, c)
    return HAY[:cut if cut > 0 else c]


def learn(text, r):
    pt = r.get("prompt_tokens")
    if pt and pt > 1000:
        STATE["cpt"] = len(text) / (pt - 40)


def mem_low(t0, t1):
    out = {}
    for who in ("head", "worker"):
        lo = None
        try:
            for ln in open(f"{OUT}/mem.{who}"):
                p = ln.split()
                if len(p) >= 2 and p[0].isdigit() and t0 - 1 <= int(p[0]) <= t1 + 1:
                    v = int(p[1]) / 1048576
                    lo = v if lo is None else min(lo, v)
        except OSError:
            pass
        out[who] = round(lo, 2) if lo is not None else None
    return out


def mem_now():
    return mem_low(time.time() - 3, time.time())


DONE_RE = re.compile(r"done (chatcmpl-\w+) (.*)")


def done_lines(since):
    txt = subprocess.run(["docker", "logs", "--since", str(int(since) - 5), CONTAINER],
                         capture_output=True, text=True).stdout
    txt += subprocess.run(["docker", "logs", "--since", str(int(since) - 5), CONTAINER],
                          capture_output=True, text=True).stderr
    d = {}
    for m in DONE_RE.finditer(txt):
        kv = dict(x.split("=", 1) for x in m.group(2).split() if "=" in x)
        d[m.group(1)] = kv
    graphs = re.findall(r".*CUDA graphs:.*", txt)
    return d, graphs


def enrich(att, since):
    time.sleep(1.5)
    dl, graphs = done_lines(since)
    for r in att.get("results", []):
        kv = dl.get(r.get("id"))
        if not kv:
            continue
        r["done"] = kv
        try:
            rounds = int(kv["rounds"])
            a, b = (int(x) for x in kv["accepted"].split("/"))
            dec = float(kv["decode_s"])
            r["rounds"], r["drafted"], r["accepted_n"] = rounds, b, a
            r["rows_per_round"] = 1 + b / rounds if rounds else None
            r["ms_per_round"] = 1000 * dec / rounds if rounds else None
            # client-side cross-check (decode_s has one decimal on this image)
            if r.get("t_last") and r.get("t_first") and rounds > 1:
                r["ms_per_round_client"] = 1000 * (r["t_last"] - r["t_first"]) / (rounds - 1)
            r["tok_per_round"] = int(kv["tokens"]) / rounds if rounds else None
        except (KeyError, ValueError):
            pass
    if graphs:
        att["graph_lines"] = graphs[-4:]
    return att


def clocks():
    q = "nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader"
    h = subprocess.run(q, shell=True, capture_output=True, text=True).stdout.strip()
    w = subprocess.run(["ssh", "-o", "BatchMode=yes", "10.50.0.11", q], capture_output=True, text=True).stdout.strip()
    return {"head": h, "worker": w}


def cell(name, depth, prompts, max_tokens, thinking, kind):
    t0 = time.time()
    n = len(prompts)
    log(f"cell {name}: depth~{depth} streams={n} max_tokens={max_tokens} thinking={thinking} mem_now={mem_now()}")
    atts = bd.guarded(srv, lambda: bd.multi(srv, prompts, max_tokens, thinking), n, retries=3, idle_max=300)
    t1 = time.time()
    for a in atts:
        enrich(a, a["t"])
    rec = {"name": name, "kind": kind, "depth_target": depth, "streams": n, "t0": t0, "t1": t1,
           "mem_low_gib": mem_low(t0, t1), "clocks_after": clocks(), "attempts": atts}
    DOC["cells"].append(rec)
    save()
    last = atts[-1]
    rs = last.get("results", [])
    summ = []
    for r in rs:
        dn = r.get("done", {})
        summ.append(f"pt={r.get('prompt_tokens')} c={r.get('cached_tokens')} ct={r.get('completion_tokens')} "
                    f"msr={r.get('ms_per_round') and round(r['ms_per_round'], 1)} rows={r.get('rows_per_round') and round(r['rows_per_round'], 2)} "
                    f"pf={dn.get('prefill_s')} q={dn.get('queue_s')} rs={dn.get('restore_s')} err={r.get('error')}")
    log(f"  {name} clean={last.get('clean')} attempts={len(atts)} mem_low={rec['mem_low_gib']} | " + " ; ".join(summ))
    return rec


def low_of(rec):
    v = [x for x in rec["mem_low_gib"].values() if x is not None]
    return min(v) if v else None


def task(label):
    return (f"\n\nTask {label}: write a long, detailed commentary on the text above, section by section, quoting it. "
            "Do not stop early.")


def ok(rec):
    last = rec["attempts"][-1]
    return bool(last.get("results")) and not any("error" in r for r in last["results"])


def ladder():
    # discarded warm-up, then the short reference
    cell("warmup", 0, ["Write a short story about a lighthouse keeper."], 256, True, "warmup")
    rungs = [SHORT] + LADDER
    prev_low, prev_depth, extra4 = None, None, None
    slope = 1.0 / 100_000  # GiB per token of history (recorded: 6.5 -> 4.2 GiB over 261K on one conversation)
    first_low = None
    reached = []
    for d in rungs:
        h = hay(d)
        hay_fixed[d] = h
        if prev_low is not None:
            proj = prev_low - slope * (d - prev_depth) - 0.3
            if proj < FLOOR_GIB:
                log(f"STOP climbing before {d}: projected low {proj:.2f} GiB < {FLOOR_GIB} (prev low {prev_low}, slope {slope*1e5:.2f} GiB/100K)")
                DOC["stopped_before"] = d
                break
        pr = cell(f"prime-{d}", d, [SALT + h + "\n\nReply with the single word OK."], 1, False, "prime")
        if not ok(pr):
            log(f"prime failed at {d}; stop climbing")
            DOC["stopped_before"] = d
            break
        r0 = pr["attempts"][-1]["results"][0]
        learn(SALT + h, r0)
        pt = r0.get("prompt_tokens") or 0
        for edge in (131072, 262144):
            if pt < edge <= pt + DEC_TOKENS + 64:
                log(f"WARN rung {d}: prompt {pt} + decode crosses {edge}")
        if prev_depth and (r0.get("cached_tokens") or 0) == 0:
            log(f"WARN prime at {d} got cached=0 (cold prefill)")
        low1 = []
        for rep in range(2):
            rc = cell(f"d{d}-s1-r{rep}", d, [SALT + h + task(f"d{d}-s1-r{rep}")], DEC_TOKENS, True, "dec1")
            if not ok(rc):
                log(f"1-stream failed at {d}; stop")
                DOC["stopped_before"] = d
                save()
                return reached
            low1.append(low_of(rc))
        lo1 = min(x for x in low1 + [low_of(pr)] if x is not None)
        # project the 4-stream low
        if extra4 is None:
            proj4 = lo1 - 0.8
        else:
            proj4 = lo1 - max(extra4, 0.2) * max(1.0, d / max(prev_depth or d, 1)) * 1.5
        if proj4 < FLOOR_GIB:
            log(f"SKIP 4-stream at {d}: projected low {proj4:.2f} GiB < {FLOOR_GIB}")
            DOC.setdefault("skipped4", []).append(d)
            lo = lo1
        else:
            r4 = cell(f"d{d}-s4", d, [SALT + h + task(f"d{d}-s4-{s}") for s in range(4)], DEC_TOKENS, True, "dec4")
            lo4 = low_of(r4)
            if lo4 is not None:
                extra4 = max(0.0, lo1 - lo4)
            lo = min(lo1, lo4 if lo4 is not None else lo1)
            if not ok(r4):
                log(f"4-stream failed at {d}; stop")
                DOC["stopped_before"] = d
                save()
                return reached
        reached.append(d)
        if first_low is None:
            first_low = (d, lo)
        elif d > first_low[0]:
            slope = max(slope * 0.5, (first_low[1] - lo) / (d - first_low[0]), 0.3 / 100_000)
        DOC["mem_model"] = {"slope_gib_per_100k": slope * 1e5, "extra4": extra4, "last_low": lo}
        prev_low, prev_depth = lo, d
        save()
    # bookends: repeat 100K and the short cells after the deepest point
    for d in (100_000, SHORT):
        h = hay_fixed[d]
        for rep in range(2):
            cell(f"end-d{d}-s1-r{rep}", d, [SALT + h + task(f"end-d{d}-s1-r{rep}")], DEC_TOKENS, True, "end1")
        cell(f"end-d{d}-s4", d, [SALT + h + task(f"end-d{d}-s4-{s}") for s in range(4)], DEC_TOKENS, True, "end4")
    return reached


hay_fixed = {}


def warm():
    for H in (8_000, 100_000, 200_000):
        h = hay_fixed.get(H) or hay(H)
        hay_fixed[H] = h
        cell(f"hist-{H}", H, [SALT + h + "\n\nReply with the single word OK."], 1, False, "hist")
        for k in (100, 500, 2000):
            for rep in range(2):
                app = corpus.text("div", k, f"warm-{H}-{k}-{rep}", fixed=True)
                cell(f"warm-{H}-a{k}-r{rep}", H, [SALT + h + "\n\n" + app +
                                                  "\n\nIn one sentence, what is the last paragraph about?"],
                     16, True, "warm")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"start {what} pid={os.getpid()} hay_chars={len(HAY)}")
    try:
        if what in ("ladder", "all"):
            # fix the 100K and short prefixes up-front so the bookends reuse the exact same text
            DOC["started"] = time.time()
            DOC["reached"] = ladder()
        if what in ("warm", "all"):
            warm()
    except Exception as exc:  # noqa: BLE001
        log(f"driver error: {type(exc).__name__}: {exc}")
        raise
    finally:
        DOC["finished"] = time.time()
        save()
        log("end")
