"""A short screen of one serving arm: single-stream cells with tokens per round, then mixed bursts at 4 and 8 streams.

  python3 runs/screen-20261118/screen.py <label> <out.json> [sampled Tetris seeds, default 1]

Runs on the head against 127.0.0.1:8000. Single-stream rates are the engine's own (the done line); burst rates are
tokens over wall time. A failed request stops the run with a non-zero code, with what was measured written.
"""
import collections, json, subprocess, sys, threading, time, urllib.request
from tokenizers import Tokenizer

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
TETRIS = "implement the tetris game. output HTML code directly - no explanation."
MIX = [
    "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
    "Write a complete Python module implementing an LRU cache class with get, put, delete and a thread-safe variant, with type hints and unit tests. Output code only.",
    "Explain in depth how a compiler turns source code into machine code, step by step.",
    "Output a JSON array of 40 objects describing fictional books, each with the fields id, title, author, year, genres (an array) and rating. Output JSON only.",
    "Describe the rise and fall of the Roman Republic, from the expulsion of the kings to Augustus, in detail.",
    "Write a Python script that reads a CSV of bank transactions, groups them by month and category, and prints a formatted report. Output code only.",
    "Explain how TCP congestion control works, covering slow start, congestion avoidance, fast retransmit and modern variants.",
    TETRIS,
]


def done_line(rid):
    log = subprocess.run(["docker", "logs", "--since", "15m", "dsv41-1017-exl3-tf"], capture_output=True, text=True)
    line = next((l for l in (log.stdout + log.stderr).splitlines() if " done " + str(rid) in l), "")
    return dict(kv.split("=", 1) for kv in line.split() if "=" in kv)


def body(prompt, sampled, seed, max_tokens, stream):
    b = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(dict(temperature=1.0, top_p=0.95, top_k=20) if sampled else dict(temperature=0))
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    return json.dumps(b).encode()


def solo(label, prompt, sampled, seed, max_tokens):
    pieces, rid, problem = [], None, None
    try:
        req = urllib.request.Request(URL, body(prompt, sampled, seed, max_tokens, True), {"Content-Type": "application/json"})
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
                d = (o.get("choices") or [{}])[0].get("delta", {}) if o.get("choices") else {}
                if d.get("content"):
                    pieces.append(d["content"])
    except Exception as e:
        problem = f"{type(e).__name__}: {e}"[:300]
    counts, text, prev = [], "", 0
    for piece in pieces:
        text += piece
        n = len(TOK.encode(text, add_special_tokens=False).ids)
        counts.append(n - prev)
        prev = n
    time.sleep(1)
    f = done_line(rid)
    row = {"cell": label, "seed": seed, "rounds": f.get("rounds"), "accepted": f.get("accepted"), "tokens": f.get("tokens"),
           "tok_per_s": f.get("tok_per_s"), "decode_s": f.get("decode_s"), "sha": f.get("sha"),
           "hist": dict(sorted(collections.Counter(counts).items()))}
    if f.get("rounds"):
        rounds, tokens = int(f["rounds"]), int(f["tokens"])
        acc, drafted = (int(x) for x in f["accepted"].split("/"))
        row.update(tokens_per_round=round(tokens / rounds, 3), drafted_per_round=round(drafted / rounds, 3),
                   accept_pct=round(100 * acc / drafted, 1) if drafted else None, ms_per_round=round(1000 * float(f["decode_s"]) / rounds, 2))
    if problem or not f.get("rounds"):
        row["failed"] = problem or "no done line"
    return row


def burst(label, prompts, seed0, max_tokens):
    width = len(prompts)
    res = [None] * width

    def one(i):
        try:
            req = urllib.request.Request(URL, body(prompts[i], True, seed0 + i, max_tokens, False), {"Content-Type": "application/json"})
            res[i] = json.load(urllib.request.urlopen(req, timeout=900))
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"[:300]}

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(i,)) for i in range(width)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    time.sleep(1)
    rows = [done_line(r.get("id")) if r.get("id") else {} for r in res]
    tokens = sum(int(f.get("tokens", 0)) for f in rows)
    rounds = sum(int(f.get("rounds", 0)) for f in rows)
    drafted = sum(int(f["accepted"].split("/")[1]) for f in rows if f.get("accepted"))
    accepted = sum(int(f["accepted"].split("/")[0]) for f in rows if f.get("accepted"))
    bad = [r.get("error") for r in res if r.get("error")] + [f"no done line for request {i}" for i, f in enumerate(rows) if not f.get("rounds")]
    out = {"cell": label, "width": width, "seed0": seed0, "tokens": tokens, "wall_s": round(wall, 3), "agg_tok_s_wall": round(tokens / wall, 2),
           "tokens_per_stream_round": round(tokens / rounds, 3) if rounds else None, "drafted_per_stream_round": round(drafted / rounds, 3) if rounds else None,
           "accept_pct": round(100 * accepted / drafted, 1) if drafted else None,
           "rounds": [int(f.get("rounds", 0)) for f in rows], "decode_s": [float(f.get("decode_s", 0)) for f in rows],
           "stream_tokens": [int(f.get("tokens", 0)) for f in rows],
           # a stream's seconds a round while it decodes: the round time, whatever the drafts delivered
           "ms_per_round": round(1000 * sum(float(f.get("decode_s", 0)) for f in rows) / rounds, 2) if rounds else None,
           "comparable": tokens == width * max_tokens, "sha": [f.get("sha") for f in rows]}
    if bad:
        out["failed"] = json.dumps(bad)[:300]
    return out


def main():
    label, out = sys.argv[1], sys.argv[2]
    rows = []

    def add(r):
        rows.append(r)
        json.dump(rows, open(out, "w"))
        print(label, {k: v for k, v in r.items() if k not in ("sha", "hist") or k == "hist" and "tetris" in r["cell"]}, flush=True)
        if r.get("failed"):
            print(label, "FAILED:", r["cell"], r["failed"], flush=True)
            sys.exit(2)

    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    add(solo("tetris greedy", TETRIS, False, 1, 2048))
    for k in range(seeds):
        add(solo("tetris sampled", TETRIS, True, 700 + k, 2048))
    add(solo("code greedy", MIX[1], False, 1, 1536))
    add(solo("prose greedy", MIX[0], False, 1, 768))
    for rep in range(2):
        add(burst("mixed x4", MIX[:4], 900, 512))
        add(burst("mixed x8", MIX, 910, 512))
    add(burst("tetris x8", [TETRIS] * 8, 820, 768))


main()
