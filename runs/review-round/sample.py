"""Paired sampled warm turns. Run on the head from the repository root.

python3 -I runs/review-round/sample.py ARM SALT > ARM.jsonl
Use a new one-digit SALT for each arm, to keep token counts comparable.
For --warm-chat use the same previously unused SALT on both sides of a restart
to replay identical messages. A prime with restored KV is recorded separately.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "tools"))
import bench_div as bench

parser = argparse.ArgumentParser()
parser.add_argument("arm")
parser.add_argument("salt", choices=list("0123456789"))
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--finish-grace", action="store_true", help="Measure short bursts and mixed arrivals at fixed reply limits")
parser.add_argument("--grace-depth", action="store_true", help="Use two matched warm-prefill sizes at 100K in the finish-grace trial")
parser.add_argument("--adaptive-mix", action="store_true", help="Include an 80-token lead that can plausibly finish within one second")
parser.add_argument("--chat-depth", type=int, choices=(250000, 500000), default=250000)
mode.add_argument("--decode-window", action="store_true", help="Sample physical rounds and generated tokens strictly inside common decode windows")
mode.add_argument("--replay-depth", action="store_true",
                  help="Replay a previous depth fixture, allowing a restored grid hit on its priming request")
mode.add_argument("--threshold", action="store_true",
                  help="Sweep the finish-window boundary with near-identical output budgets")
mode.add_argument("--admission", action="store_true", help="Cross prefix sharing, sampler and reply budget at 4K")
mode.add_argument("--occupancy", action="store_true",
                  help="Repeat the same 1,024-token 300K burst before and after seeding independent kept states")
mode.add_argument("--deep-admission", action="store_true", help="Bookend 256/1024-token bursts on a shared 300K prefix")
mode.add_argument("--warm-chat", action="store_true", help="Resend a fixed 250K transcript with 100/500/2000-token appends")
args = parser.parse_args() if __name__ == "__main__" else None
if args and args.grace_depth and not args.finish_grace:
    parser.error("--grace-depth requires --finish-grace")
worker = "10.50.0.11"
name = "dsv41-1017-exl3-tf"
srv = bench.Server("http://127.0.0.1:8000", "DeepSeek-V4.1-Flash-1017")
tasks = [
    "Design a bounded job queue for a Python server. Include code, shutdown behaviour, and a test.",
    "Write Python that streams a sorted merge of timestamped log files. Explain edge cases and give tests.",
    "Implement an idempotent retry helper with a deadline in Python. Include examples and test cases.",
    "Implement an LRU cache with a byte budget in Python. Include code and tests for eviction.",
]
flag_names = set(("TF_DS_ENGRAM_LEAD TF_DS_ENGRAM_SCALES TF_DS_ENGRAM_STRICT TENSORFOLD_TOKEN_CACHE_MB "
                  "TF_DS_DEVICE_ACCEPT TF_DS_COMPRESS_SIDE TF_DS_PDL TF_DS_FUSED_COMPRESS2 TF_DS_T13_EXPERT_PRUNE "
                  "TF_DS_T13_EXPERT_CAP TF_DS_T15_PLACE_BOUND TF_DS_T26_KEYED_DRAFT TF_DS_T27_DEVICE_DRAW "
                  "TF_DS_GRAPH_CAP TF_DS_GRAPHS TF_DS_KV_SHARD TF_DS_POOL TF_DS_REPLAY_FLOOR TF_DS_PREFILL_SLICE TF_DS_FINISH_GRACE_MS "
                  "TF_DS_FINISH_ADAPTIVE TF_DS_L2_PREFETCH TF_DS_SCORE").split())


def emit(kind, **kwargs):
    print(json.dumps({"kind": kind, "arm": args.arm, "at": time.time(), **kwargs}), flush=True)


def read_cmd(argv):
    return subprocess.check_output(argv, text=True, timeout=12)


def identity(prefix):
    d = json.loads(read_cmd(prefix + ["docker", "inspect", name]))[0]
    env = dict(item.split("=", 1) for item in d["Config"]["Env"])
    return {"image": d["Config"]["Image"], "image_id": d["Image"], "container_id": d["Id"],
            "start": d["State"]["StartedAt"], "env": {k: env.get(k) for k in sorted(flag_names)}}


def memory():
    return [int(read_cmd(prefix + ["grep", "MemAvailable", "/proc/meminfo"]).split()[1]) / 1048576
            for prefix in ([], ["ssh", "-o", "BatchMode=yes", worker])]


def health():
    with bench.OPENER.open(srv.url + "/health", timeout=10) as response:
        d = json.load(response)
    return {"status": d["status"], "pair": d["pair"], "per_part": d["kv_shard"]["per_part"],
            "refused_no_target": d["kv_shard"]["refused_no_target"], "live": d["live"],
            "kv_counts": {k: d["kv_shard"][k] for k in ("twins_made","staled","moves","carries","refused_pinned","no_part")}}


def request(content, seed, tokens, gate, temperature, *, lead_first=None, offset=None):
    messages = [{"role": "user", "content": content}] if isinstance(content, str) else content
    body = {"model": srv.model, "messages": messages, "max_tokens": tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": True}, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(srv.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    gate.wait(timeout=10)
    if offset is not None:
        if not lead_first.wait(timeout=90):
            raise RuntimeError("Lead request did not begin streaming")
        time.sleep(offset)
    t0 = time.monotonic()
    first = last = None
    gaps = []
    usage = rid = finish = None
    with bench.OPENER.open(req, timeout=900 if getattr(args, "warm_chat", False) and args.chat_depth == 500000 else 300) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            value = raw[5:].strip()
            if value == b"[DONE]":
                break
            ev = json.loads(value)
            rid = ev.get("id") or rid
            usage = ev.get("usage") or usage
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.monotonic()
                    if args.finish_grace and last is not None:
                        gaps.append(now - last)
                    last = now
                    first = first or last
                    if lead_first is not None and offset is None:
                        lead_first.set()
                finish = c.get("finish_reason") or finish
    if first is None or not rid or not usage:
        raise RuntimeError("Incomplete streamed reply")
    stream = {"events": len(gaps) + 1, "max_gap_s": max(gaps, default=0),
              "p95_gap_s": statistics.quantiles(gaps, n=100, method="inclusive")[94] if len(gaps) > 1 else None} if args.finish_grace else {}
    return {"id": rid, "seed": seed, **stream, "input_sha256": hashlib.sha256(req.data).hexdigest(), "prompt_tokens": usage["prompt_tokens"],
            "cached": usage["prompt_tokens_details"]["cached_tokens"], "tokens": usage["completion_tokens"],
            "finish": finish, "ttft": first - t0, "total": time.monotonic() - t0,
            "start": t0, "first": first, "last": last}


def cell(depth, kind, messages, seeds, *, budget=256, temperature=1.0, sharing=None, rep=None,
         append=None, expected_prompt=None, expected_cached=None, lead_budget=None, mixed=False,
         allow_early_stop=False):
    free = memory()
    if min(free) < 4.5:
        raise RuntimeError(f"Insufficient memory for next cell, {free} GiB")
    if not bench.idle_wait(srv, 90)[0]:
        raise RuntimeError("Pair busy")
    m0 = srv.metrics()
    h0 = health()
    if h0["status"] != "ok" or h0["pair"]["broken"] or m0.get("tensorfold:requests_running") != 0 or m0.get("tensorfold:requests_waiting") != 0:
        raise RuntimeError("Pair unhealthy or busy")
    since = str(int(time.time()) - 1)
    n = len(messages)
    tokens = 16 if kind == "prime" else budget
    gate = threading.Barrier(n)
    budgets = [lead_budget or tokens] + [tokens] * (n-1)
    lead_first = threading.Event() if mixed else None
    trace = []
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        futures = [pool.submit(request, msg, seed, limit, gate, temperature, lead_first=lead_first,
                               offset=(0.20, 0.55, 0.90)[i-1] if mixed and i else None)
                   for i, (msg, seed, limit) in enumerate(zip(messages, seeds, budgets))]
        if getattr(args, "decode_window", False) and kind in ("solo", "admission"):
            metrics = ("generation_tokens_running", "requests_running", "requests_waiting",
                       "kv_shard_fetch_rounds", "kv_shard_owner_rounds")
            while not all(f.done() for f in futures):
                before = time.monotonic()
                values = srv.metrics()
                trace.append({"t": (before + time.monotonic()) / 2, **{k: values.get("tensorfold:" + k) for k in metrics}})
                concurrent.futures.wait(futures, timeout=0.2)
        elif kind == "admission":
            while not all(f.done() for f in futures):
                try:
                    h = health()
                    trace.append({"t": time.monotonic(), "per_part": h["per_part"], "live": h["live"],
                                  "refused_no_target": h["refused_no_target"], "kv_counts": h["kv_counts"]})
                    if depth >= 300000 and len(trace) % 5 == 1:
                        trace[-1]["mem_gib"] = memory()
                except (OSError, ValueError) as exc:
                    trace.append({"t": time.monotonic(), "error": str(exc)})
                concurrent.futures.wait(futures, timeout=1)
        out = [f.result() for f in futures]
    time.sleep(0.5)
    m1 = srv.metrics()
    h1 = health()
    log = subprocess.run(["docker", "logs", "--since", since, name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         check=True, timeout=15, text=True).stdout
    for r in out:
        match = re.search(r"done " + re.escape(r["id"]) + r" (.*)", log)
        if not match:
            raise RuntimeError("Missing done line for " + r["id"])
        r["done"] = dict(x.split("=", 1) for x in match[1].split() if "=" in x)
    full_length = all(r["finish"] == "length" and r["tokens"] == limit for r, limit in zip(out, budgets))
    finished = full_length or (kind in ("admission", "chat-prime", "warm-chat") and
                               all(r["finish"] in ("length", "stop") for r in out))
    expected_cache = all(r["cached"] == expected_cached for r in out) if expected_cached is not None else (
        kind == "chat-prime" or (kind == "prime" and args.replay_depth)
        or (kind == "admission" and args.occupancy)
        or all((r["cached"] == 0 if kind == "prime" else r["cached"] > 0) for r in out))
    clean = (m1[bench.DONE_KEY] - m0[bench.DONE_KEY] == n
             and m1.get("tensorfold:requests_running") == m1.get("tensorfold:requests_waiting") == 0
             and h1["status"] == "ok" and not h1["pair"]["broken"]
             and finished and expected_cache and all(expected_prompt in (None, r["prompt_tokens"]) for r in out))
    physical = {}
    if getattr(args, "decode_window", False):
        physical = {k: m1["tensorfold:"+k] - m0["tensorfold:"+k] for k in
                    ("kv_shard_fetch_rounds","kv_shard_owner_rounds","decode_rounds_total","mtp_drafted_total","mtp_accepted_total")}
    emit("cell", depth=depth, cell=kind, clean=clean, health_before=h0, health_after=h1, mem_before_gib=free,
         mem_after_gib=memory(), own=n, completed_delta=m1[bench.DONE_KEY] - m0[bench.DONE_KEY], requests=out,
         overlap=min(r["last"] for r in out) - max(r["first"] for r in out), trace=trace,
         budget=budget, temperature=temperature, sharing=sharing, rep=rep, full_length=full_length,
         append=append, expected_prompt=expected_prompt, expected_cached=expected_cached, **({"counter_delta":physical} if physical else {}))
    if not clean:
        raise RuntimeError("Unclean cell")
    if args.finish_grace and not (full_length or allow_early_stop):
        raise RuntimeError("Unequal generation work in finish-grace comparison")
    if any(min(p["mem_gib"]) < 3.5 for p in trace if "mem_gib" in p):
        raise RuntimeError("Memory floor crossed; stopping after this cell")
    if depth >= 300000 and kind == "admission" and not args.occupancy and any(r["cached"] != 299008 for r in out):
        raise RuntimeError("Warm cut differs from 299008; stopping after this cell")
    return out


def depth_screen(corpus):
    for depth in (4000, 100000):
        history = corpus.text("div", depth, "review-1117-history", fixed=True)
        context = f"[nov17-review-{args.salt}-{depth}]\n{history}\n\n"
        emit("history", depth=depth, unsalted_sha256=hashlib.sha256(history.encode()).hexdigest())
        cell(depth, "prime", [context + "Summarize the preceding context in detail."], [4199])
        for i, task in enumerate(tasks[:3]):
            cell(depth, "solo", [context + task], [4200 + i])
        cell(depth, "burst4", [context + task for task in tasks], list(range(4200, 4204)))


def admission_screen(corpus):
    # Every arm uses this same text. Salt differences before token 2048 force four independent KV states.
    # Rotating the shared namespace on rep 2 lets two of the four distinct prefixes serve as shared controls.
    history = corpus.text("div", 4000, "review-1117-history", fixed=True)
    contexts = [f"[nov17-admission-{args.salt}-{i}]\n{history}\n\n" for i in range(4)]
    for ctx in contexts:
        cell(4000, "prime", [ctx + "Summarize the preceding context in detail."], [4199])
    # After >=1s of prefill the scheduler waits up to 5s to let a stream with <=256 tokens left finish.
    # Reversing the order balances drift. The two modes on each side of that threshold test admission.
    order = [(256, 1., "shared"), (512, 1., "distinct"), (512, 0., "shared"), (256, 0., "distinct"),
             (512, 1., "shared"), (256, 1., "distinct"), (256, 0., "shared"), (512, 0., "distinct")]
    for rep, cells in enumerate((order, order[::-1])):
        for budget, temperature, sharing in cells:
            prefixes = contexts if sharing == "distinct" else [contexts[rep]] * 4
            prompts = [ctx + task for ctx, task in zip(prefixes, tasks)]
            cell(4000, "admission", prompts, list(range(4200, 4204)), budget=budget, temperature=temperature,
                 sharing=sharing, rep=rep)


def threshold_screen(corpus):
    history = corpus.text("div", 4000, "review-1117-history", fixed=True)
    context = f"[nov17-admission-{args.salt}-0]\n{history}\n\n"
    cell(4000, "prime", [context + "Summarize the preceding context in detail."], [4199])
    prompts = [context + task for task in tasks]
    for rep, budgets in enumerate(((256, 257, 258), (258, 257, 256))):
        for budget in budgets:
            cell(4000, "admission", prompts, list(range(4200, 4204)), budget=budget,
                 sharing="shared", rep=rep)


def finish_grace_screen(corpus):
    history = corpus.text("div", 4000, "review-1117-history", fixed=True)
    ctx = f"[nov17-admission-{args.salt}-0]\n{history}\n\n"
    cell(4000, "chat-prime", [ctx + "Summarize the preceding context in detail."], [4199], budget=16)
    prompts = [ctx + task for task in tasks]
    seeds = list(range(4200, 4204))
    cell(4000, "solo", prompts[:1], seeds[:1], budget=256, expected_cached=2048)
    shapes = (("burst", False, 256), ("short-lead", True, 256), ("long-lead", True, 640))
    if args.adaptive_mix:
        shapes = (("near-finished", True, 80), *shapes)
    for rep, order in enumerate((shapes, shapes[::-1])):
        for shape, mixed, lead_budget in order:
            cell(4000, "admission", prompts, seeds, budget=256, lead_budget=lead_budget, mixed=mixed,
                 sharing=shape, rep=rep, expected_cached=2048)


def deep_grace_screen(corpus):
    grid = 98304
    source = corpus.text("div", 108000, "finish-grace-depth", fixed=True).replace("<", "(")
    contexts = {}
    for tail in (384, 1990):
        target = grid + tail
        cut = 4 * target
        for _ in range(9):
            ctx = f"[grace-100k-{args.salt}]\n{source[:cut]}\n\n"
            length = token_count([{"role": "user", "content": ctx + tasks[0]}])
            if abs(length - target) <= 4:
                break
            cut += round((target - length) * cut / length)
        else:
            raise RuntimeError(f"Could not calibrate the {tail}-token warm tail")
        contexts[tail] = [ctx + task for task in tasks]
        emit("history", depth=100000, tail=tail, measured=length,
             history_sha256=hashlib.sha256(ctx.encode()).hexdigest())
    seeds = list(range(4200, 4204))
    # The two contexts share their first 98,304 tokens. A different final
    # question allows both measured cells to resume from that same grid point.
    cell(100000, "chat-prime", [contexts[1990][0] + " Summarize."], [4199], budget=16)
    for tail in (384, 1990):
        cell(100000, "solo", contexts[tail][:1], seeds[:1], budget=256,
             sharing=f"tail-{tail}", expected_cached=grid)
    for rep, order in enumerate(((384, 1990), (1990, 384))):
        for tail in order:
            cell(100000, "admission", contexts[tail], seeds, budget=256, mixed=True,
                 sharing=f"tail-{tail}", rep=rep, expected_cached=grid)
        if args.adaptive_mix:
            cell(100000, "admission", contexts[1990], seeds, budget=256, lead_budget=80, mixed=True,
                 sharing="near-finished", rep=rep, expected_cached=grid)


def decode_windows(corpus):
    # Prefixes differ in length but the four questions and both sets of seeds repeat at both depths.
    source = corpus.text("div", 360000, "decode-window-1118", fixed=True)
    for depth, grid in ((4000, 2048), (300000, 299008)):
        target = grid + 384
        cut = 4 * target
        for _ in range(9):
            ctx = f"[decode-window-{args.salt}]\n{source[:cut]}\n\n"
            length = token_count([{"role":"user", "content":ctx + tasks[0]}])
            if abs(length-target) <= 4:
                break
            cut += round((target-length)*cut/length)
        else:
            raise RuntimeError(f"Cannot size depth fixture at {depth}")
        prompts = [ctx + task for task in tasks]
        cell(depth, "chat-prime", [ctx + "Summarize the notes."], [4199], budget=16)
        for rep in range(2):
            seeds = [4200 + rep*10+i for i in range(4)]
            order = ("solo", "admission") if rep == 0 else ("admission", "solo")
            for kind in order:
                if kind == "solo":
                    for i in range(4):
                        cell(depth, kind, prompts[i:i+1], seeds[i:i+1], budget=512, sharing=f"task-{i}", rep=rep, expected_cached=grid)
                else:
                    cell(depth, kind, prompts, seeds, budget=512, sharing="shared", rep=rep, expected_cached=grid)


def deep_admission(corpus):
    # The earlier ladder consumed up to 1.7 GiB by 300K; preserve its 3.5 GiB floor.
    if min(memory()) < 3.5 + 1.7:
        raise RuntimeError("Available memory below 300K projection")
    source = corpus.text("div", 360000, "depth-1117-hay", fixed=True)
    target = 300576
    cut = 4 * target
    for attempt in range(4):
        ctx = f"[nov17-deep-admission-{args.salt}]\n" + source[:cut] + "\n\n"
        messages = [{"role": "user", "content": ctx + tasks[0]}]
        body = {"model": srv.model, "messages": messages, "chat_template_kwargs": {"enable_thinking": True}}
        req = urllib.request.Request(srv.url + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with bench.OPENER.open(req, timeout=30) as response:
            measured = json.load(response)["count"]
        if abs(measured-target) <= 32:
            break
        cut = round(cut * target / measured)
    else:
        raise RuntimeError(f"Could not match the earlier 300K ladder depth ({measured})")
    emit("history", depth=300000, unsalted_sha256=hashlib.sha256(source[:cut].encode()).hexdigest(), measured=measured)
    prompts = [ctx + task for task in tasks]
    prime_task = "Summarize the preceding context in detail."
    if args.occupancy:
        # Use the earlier deep fixture byte-for-byte. Its prime may restore a disk state.
        cell(300000, "chat-prime", [ctx + prime_task], [4199], budget=16)
        seeds = list(range(4200, 4204))
        def kept_rows():
            return sum(p["kept_rows"] for p in health()["per_part"])
        start_rows = kept_rows()
        emit("occupancy", stage="low", kept_rows=start_rows, health=health())
        for rep in range(2):
            cell(300000, "admission", prompts, seeds, budget=1024, sharing="shared", rep=rep)
        # 1.04M kept rows preceded the earlier long queues. Use the measured live counters
        # to select the second condition; identical independent warmups can restore from disk.
        for i in range(4):
            if kept_rows() >= 1000000:
                break
            independent = f"[review-occupancy-{args.salt}-{i}]\n" + source[:cut] + "\n\n" + prime_task
            cell(300000, "chat-prime", [independent], [4250+i], budget=16)
        before = kept_rows()
        emit("occupancy", stage="high", kept_rows=before, health=health())
        if before < 1000000 or before - start_rows < 300000:
            raise RuntimeError(f"Insufficient occupancy contrast: {start_rows} vs {before}")
        for rep in range(2, 4):
            cell(300000, "admission", prompts, seeds, budget=1024, sharing="shared", rep=rep)
        return
    cell(300000, "prime", [ctx + prime_task], [4199])
    for rep, budgets in enumerate(((256,1024), (1024,256))):
        for budget in budgets:
            cell(300000, "admission", prompts, list(range(4200,4204)), budget=budget, sharing="shared", rep=rep)


def token_count(messages):
    body = {"model": srv.model, "messages": messages, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(srv.url + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with bench.OPENER.open(req, timeout=60) as response:
        return json.load(response)["count"]


def warm_chat(corpus):
    if min(memory()) < 3.5 + 1.7:
        raise RuntimeError("Available memory below deep-turn projection")
    grid = args.chat_depth // 2048 * 2048
    # A keep needs 128 rows after its grid cut. Even the 2000-token append
    # ends <128 beyond the next cut, including the 16-token completion.
    target = grid + 145
    assistant = "The evidence is indexed. Send the next change and I will check it against the prior review."
    shadow_digit = str((int(args.salt) + 1) % 10)

    def transcript(history, last):
        return [{"role": "user", "content": history}, {"role": "assistant", "content": assistant},
                {"role": "user", "content": last}]

    def history(digit, source):
        return f"[review-chat-{digit}]\nUse these engineering notes in the review.\n" + source

    # Size a different plain tokenizer piece. /tokenize on the scored long message
    # would prewarm 0119 before the first generation. Exclude literal template tags.
    source = corpus.text("div", (260000 if args.chat_depth == 250000 else 550000), "review-chat-history", fixed=True).replace("<", "(")
    cut = 4 * target
    base = "Review trace."
    for _ in range(9):
        measured = token_count(transcript(history(shadow_digit, source[:cut]), base))
        if abs(measured - target) <= 3:
            break
        cut += round((target - measured) * cut / measured)
    else:
        raise RuntimeError(f"Could not place history at {target} tokens, measured {measured}")

    # Size each changed user append using a short shadow transcript and a different
    # final-message prefix. Its 0119 keys differ from every scored user message.
    shadow = "Sizing trace."
    shadow_base = token_count(transcript("Sizing stub.", shadow))

    def make_append(size, rep):
        lead = "\n" + tasks[rep] + "\nEvidence for this case:\n"
        notes = corpus.text("div", size + 200, f"review-chat-{size}-{rep}", fixed=True).replace("<", "(")
        char_cut = 4 * size
        for _ in range(9):
            tail = lead + notes[:char_cut]
            n = token_count(transcript("Sizing stub.", shadow + tail)) - shadow_base
            if 0 <= size - n <= 10:
                tail += " item" * (size - n)
                if token_count(transcript("Sizing stub.", shadow + tail)) - shadow_base == size:
                    return base + tail
            char_cut = max(0, char_cut + 4 * (size - 4 - n))
        raise RuntimeError(f"Could not size {size} appended tokens")

    schedule = [(size, rep, make_append(size, rep)) for rep in range(3)
                for size in ((100, 500, 2000), (500, 2000, 100), (2000, 100, 500))[rep]]
    fixed = history(args.salt, source[:cut])
    emit("history", depth=args.chat_depth, history_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
         assistant_sha256=hashlib.sha256(assistant.encode()).hexdigest(), calibrated_prime_tokens=measured,
         expected_warm_grid=grid, schedule=[[s,r] for s,r,_ in schedule])
    prime = cell(args.chat_depth, "chat-prime", [transcript(fixed, base)], [5199], budget=16,
                 expected_prompt=measured)[0]
    for size, rep, last in schedule:
        cell(args.chat_depth, "warm-chat", [transcript(fixed, last)], [5200 + 10 * rep + (100,500,2000).index(size)],
             budget=16, rep=rep, append=size, expected_prompt=prime["prompt_tokens"] + size, expected_cached=grid)


if __name__ == "__main__":
    corpus = bench.Corpus("runs/combo4/corpus")
    ids = [identity(p) for p in ([], ["ssh", "-o", "BatchMode=yes", worker])]
    emit("start", salt=args.salt, ranks=ids, health=health(), memory_gib=memory(), tasks=tasks)
    if args.decode_window:
        decode_windows(corpus)
    elif args.grace_depth:
        deep_grace_screen(corpus)
    elif args.finish_grace:
        finish_grace_screen(corpus)
    elif args.threshold:
        threshold_screen(corpus)
    elif args.admission:
        admission_screen(corpus)
    elif args.deep_admission or args.occupancy:
        deep_admission(corpus)
    elif args.warm_chat:
        warm_chat(corpus)
    else:
        depth_screen(corpus)
    finish_ids = [identity(p) for p in ([], ["ssh", "-o", "BatchMode=yes", worker])]
    if finish_ids != ids:
        raise RuntimeError("Serving identity changed")
    emit("complete", ranks=finish_ids, health=health())
