"""Tokens delivered per decode round on fixed prompts, joined with the engine's done lines.

  python3 runs/suite-fused-20261121/suite_probe.py <label> <out.json> [--quick]

The server sends one content event per round, so each event's token count is that round's delivered tokens: the
pending row's token and the drafts that landed. Counts come from tokenizing the running text and differencing, so a
merge across an event boundary cannot miscount. Runs on the head (needs the `tokenizers` package and the model's
tokenizer.json), one request at a time.
"""
import collections, json, subprocess, sys, time, urllib.request
from tokenizers import Tokenizer

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
TETRIS = "implement the tetris game. output HTML code directly - no explanation."
PROSE = ["Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
         "Explain in depth how a compiler turns source code into machine code, step by step."]
CODE = ("Write a complete Python module implementing an LRU cache class with get, put, delete and a "
        "thread-safe variant, with type hints and unit tests. Output code only.")
JSONP = ("Output a JSON array of 40 objects describing fictional books, each with the fields id, title, author, "
         "year, genres (an array) and rating. Output JSON only.")


EXTRA = [
    ("go greedy", "Write a complete Go program: an HTTP server with JSON endpoints to create, list, update and delete notes kept in memory, with input validation and graceful shutdown. Output code only.", 1280),
    ("ts greedy", "Write a TypeScript React component for a sortable, filterable data table with pagination, typed props and a small usage example. Output code only.", 1280),
    ("sql greedy", "Write a PostgreSQL schema for an online bookstore (authors, books, customers, orders, order items, reviews) with constraints and indexes, then ten analytical queries with comments. Output SQL only.", 1280),
    ("yaml greedy", "Write Kubernetes manifests for a three-tier web application: namespace, config map, secret, deployments, services, an ingress and a horizontal pod autoscaler. Output YAML only.", 1280),
]


def run(label, prompt, sampled, seed, max_tokens, thinking=False):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "seed": seed, "stream": True,
            "chat_template_kwargs": {"enable_thinking": thinking}, "stream_options": {"include_usage": True}}
    body.update(dict(temperature=1.0, top_p=0.95, top_k=20) if sampled else dict(temperature=0))
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    pieces, stamps, rid, usage, problem = [], [], None, None, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                o = json.loads(p)
                if o.get("error"):
                    problem = json.dumps(o["error"])[:300]
                rid = o.get("id", rid)
                usage = o.get("usage") or usage
                d = (o.get("choices") or [{}])[0].get("delta", {}) if o.get("choices") else {}
                text = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if text:
                    pieces.append(text)
                    stamps.append(time.perf_counter())
    except Exception as e:  # an HTTP error or a dropped stream: the row says so
        problem = f"{type(e).__name__}: {e}"[:300]
    counts, text, prev = [], "", 0
    for piece in pieces:
        text += piece
        n = len(TOK.encode(text, add_special_tokens=False).ids)
        counts.append(n - prev)
        prev = n
    time.sleep(1)
    log = subprocess.run(["docker", "logs", "--since", "10m", "dsv41-1017-exl3-tf"], capture_output=True, text=True)
    line = next((l for l in (log.stdout + log.stderr).splitlines() if " done " + str(rid) in l), "")
    f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
    hist = collections.Counter(counts)
    row = {"label": label, "sampled": sampled, "seed": seed, "events": len(pieces), "tokenized_total": prev,
           "completion_tokens": (usage or {}).get("completion_tokens"), "hist": dict(sorted(hist.items())),
           "rounds": f.get("rounds"), "accepted": f.get("accepted"), "tokens_per_round": f.get("tokens_per_round"),
           "tok_per_s": f.get("tok_per_s"), "decode_s": f.get("decode_s"), "sha": f.get("sha"), "per_round": counts,
           **gaps(stamps, prev)}
    if problem or not f.get("rounds"):
        row["failed"] = problem or f"no done line ({(usage or {}).get('completion_tokens')} of {max_tokens} tokens streamed)"
    if f.get("rounds") and f.get("decode_s"):
        rounds, tokens = int(f["rounds"]), int(f.get("tokens", 0) or 0)
        acc, drafted = (int(x) for x in f["accepted"].split("/"))
        row.update(ms_per_round=round(1000 * float(f["decode_s"]) / rounds, 2), drafted_per_round=round(drafted / rounds, 3),
                   exact_tokens_per_round=round(tokens / rounds, 3), exact_tok_s=round(tokens / float(f["decode_s"]), 2))
    return row


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


def gaps(stamps, tokens):
    """Client-side timing of the content events (one a round): inter-chunk gaps (a chunk carries a round's tokens, so
    this is inter-round latency as the client sees it, not inter-token), first chunk excluded; and the first-to-last
    chunk duration, which resolves finer than the engine's decode_s."""
    g = [1000 * (b - a) for a, b in zip(stamps, stamps[1:])]
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    return {"chunk_gap_ms_p50": pct(g, 0.50), "chunk_gap_ms_p95": pct(g, 0.95), "client_span_s": round(span, 4) if span else None,
            "client_tok_s": round(tokens / span, 2) if span else None}


def stream_burst(label, width, seed0, max_tokens):
    """`width` sampled Tetris requests at once, streamed: pooled inter-chunk gap p50/p95 across the streams (the
    multi-stream latency the non-streamed burst() cannot see) and aggregate tokens a second by wall."""
    import threading
    res = [None] * width

    def one(i):
        body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": TETRIS}], "max_tokens": max_tokens,
                "seed": seed0 + i, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        stamps, usage, err = [], None, None
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    o = json.loads(line[5:].strip())
                    usage = o.get("usage") or usage
                    d = (o.get("choices") or [{}])[0].get("delta", {}) if o.get("choices") else {}
                    if (d.get("reasoning_content") or "") + (d.get("content") or ""):
                        stamps.append(time.perf_counter())
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
        res[i] = (stamps, (usage or {}).get("completion_tokens") or 0, err)

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(i,)) for i in range(width)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0
    g = [1000 * (b - a) for st, _, _ in res for a, b in zip(st, st[1:])]
    tokens = sum(n for _, n, _ in res)
    bad = [e for _, _, e in res if e]
    return {**({"failed": json.dumps(bad)[:300]} if bad else {}), "label": label, "width": width, "seed0": seed0, "tokens": tokens,
            "wall_s": round(wall, 3), "agg_tok_s_wall": round(tokens / wall, 2), "chunk_gap_ms_p50": pct(g, 0.50), "chunk_gap_ms_p95": pct(g, 0.95)}


def burst(label, width, seed0, max_tokens):
    """`width` sampled Tetris requests at once (different seeds): aggregate tokens a second by wall and by rounds."""
    import threading
    res = [None] * width

    def one(i):
        body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": TETRIS}], "max_tokens": max_tokens,
                "seed": seed0 + i, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            res[i] = json.load(urllib.request.urlopen(req, timeout=900))
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"[:300]}

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(i,)) for i in range(width)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0
    time.sleep(1)
    log = subprocess.run(["docker", "logs", "--since", "10m", "dsv41-1017-exl3-tf"], capture_output=True, text=True)
    lines = (log.stdout + log.stderr).splitlines()
    rows = []
    for r in res:
        line = next((l for l in lines if " done " + str(r.get("id")) in l), "")
        rows.append(dict(kv.split("=", 1) for kv in line.split() if "=" in kv))
    tokens = sum(int(f.get("tokens", 0)) for f in rows)
    rounds = sum(int(f.get("rounds", 0)) for f in rows)
    drafted = sum(int(f["accepted"].split("/")[1]) for f in rows if f.get("accepted"))
    accepted = sum(int(f["accepted"].split("/")[0]) for f in rows if f.get("accepted"))
    bad = [r.get("error") for r in res if r.get("error")] + [f"no done line for request {i}" for i, f in enumerate(rows) if not f.get("rounds")]
    return {**({"failed": json.dumps(bad)[:300]} if bad else {}),
            "label": label, "width": width, "seed0": seed0, "tokens": tokens, "wall_s": round(wall, 3), "agg_tok_s_wall": round(tokens / wall, 2),
            "tokens_per_stream_round": round(tokens / rounds, 3) if rounds else None, "drafted_per_stream_round": round(drafted / rounds, 3) if rounds else None,
            "accept_pct": round(100 * accepted / drafted, 1) if drafted else None, "decode_s": [f.get("decode_s") for f in rows], "sha": [f.get("sha") for f in rows]}


def main():
    label, out = sys.argv[1], sys.argv[2]
    quick = "--quick" in sys.argv
    narrow = "--narrow" in sys.argv   # the shortened suite: five solo rows, two bursts, the four extra code rows
    stream_bursts = "--stream-bursts" in sys.argv   # also streamed x1/x4 bursts: pooled inter-chunk gap p50/p95 (new rows)
    sampled_all = "--sampled" in sys.argv   # every class at the clients' sampling (T=1.0, top_p 0.95, top_k 20), two seeds each
    jobs = [("tetris greedy", TETRIS, False, 1, 2048)]
    if sampled_all:
        jobs = [(f"{nm} sampled s{sd}", pr, True, sd, n) for sd in (700, 701)
                for nm, pr, n in (("tetris", TETRIS, 2048), ("code", CODE, 1536), ("json", JSONP, 1536), ("prose", PROSE[0], 768))]
        jobs += [(f"{nm.replace(' greedy', '')} sampled s{sd}", pr, True, sd, n) for sd in (700, 701) for nm, pr, n in EXTRA]
    elif narrow:
        jobs += [("tetris sampled", TETRIS, True, 700, 2048), ("code greedy", CODE, False, 1, 1536),
                 ("json greedy", JSONP, False, 1, 1536), ("prose greedy", PROSE[0], False, 1, 768)]
    elif not quick:
        jobs += [("tetris sampled", TETRIS, True, 700 + k, 2048) for k in range(3)]
        jobs += [("code greedy", CODE, False, 1, 1536), ("json greedy", JSONP, False, 1, 1536)]
        jobs += [("prose greedy", p, False, 1, 768) for p in PROSE]
        jobs += [("prose sampled", PROSE[0], True, 700, 768)]
        jobs += [("tetris greedy", TETRIS, False, 1, 2048)]  # a second pass: the engine's depth rule has settled
    rows = []

    def stop(r):
        json.dump(rows, open(out, "w"))
        print(label, "FAILED:", r["label"], r["failed"], flush=True)
        sys.exit(2)

    for name, prompt, sampled, seed, n in jobs:
        r = run(name, prompt, sampled, seed, n)
        rows.append(r)
        if r.get("failed"):
            stop(r)
        print(label, name, "seed", seed, {k: r.get(k) for k in ("rounds", "accepted", "exact_tokens_per_round", "drafted_per_round", "ms_per_round", "exact_tok_s", "hist", "sha")}, flush=True)
    if not quick:
        for width, seed0 in (((2, 800), (4, 810)) if narrow else ((2, 800), (4, 810), (8, 820), (2, 800), (4, 810))):
            b = burst(f"tetris sampled x{width}", width, seed0, 768)
            rows.append(b)
            if b.get("failed"):
                stop(b)
            print(label, b["label"], {k: b[k] for k in ("agg_tok_s_wall", "tokens_per_stream_round", "drafted_per_stream_round", "accept_pct", "wall_s")}, flush=True)
        for width, seed0 in (((1, 830), (4, 840)) if stream_bursts else ()):
            b = stream_burst(f"tetris sampled x{width} stream", width, seed0, 768)
            rows.append(b)
            if b.get("failed"):
                stop(b)
            print(label, b["label"], {k: b[k] for k in ("agg_tok_s_wall", "chunk_gap_ms_p50", "chunk_gap_ms_p95", "wall_s")}, flush=True)
    if not quick:
        # more code-like and structured prompts than the first 15 rows had (one a class), and one with thinking on
        for name, prompt, n in ([] if sampled_all else EXTRA):
            r = run(name, prompt, False, 1, n)
            rows.append(r)
            print(label, name, {k: r.get(k) for k in ("rounds", "accepted", "exact_tokens_per_round", "drafted_per_round", "ms_per_round", "exact_tok_s", "sha")}, flush=True)
            if r.get("failed"):
                stop(r)
        if not narrow:
            r = run("code thinking", CODE, False, 1, 1536, thinking=True)
            rows.append(r)
            print(label, "code thinking", {k: r.get(k) for k in ("rounds", "accepted", "exact_tokens_per_round", "ms_per_round", "exact_tok_s", "sha")}, flush=True)
            if r.get("failed"):
                stop(r)
    json.dump(rows, open(out, "w"))


main()
