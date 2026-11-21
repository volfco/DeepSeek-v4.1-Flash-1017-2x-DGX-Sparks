"""The ladder's probe: the narrow suite probe's nine solo prompts, eleven more code-like prompts, two sampled code
prompts, sampled Tetris at two and four streams, and two different code prompts at once. One request at a time
except the bursts. Each row carries the engine's rounds, verified drafts and decode rate (its done line).

  python3 runs/draft-ladder-20261119/ladder_probe.py <label> <out.json>      (head, recipe root)
"""
import json, sys, threading, time, types, urllib.request, subprocess

# the suite probe's request functions and prompts, without running it: that file ends in a bare main() call
_src = open("runs/suite-20261119/suite_probe.py").read().rstrip()
assert _src.endswith("\nmain()")
sp = types.ModuleType("suite_probe")
exec(compile(_src[:-len("main()")], "runs/suite-20261119/suite_probe.py", "exec"), sp.__dict__)

MORE = [
    ("rust greedy", "Write a complete Rust command-line program that counts word frequencies in a text file and prints the top N words, with argument parsing, error handling and unit tests. Output code only."),
    ("c greedy", "Write a C implementation of a generic ring buffer (header and source) with push, pop, peek, is_full and is_empty, plus a test program. Output code only."),
    ("java greedy", "Write a Java class library for a bank account service: Account, Transaction, InsufficientFundsException and an AccountService with deposit, withdraw, transfer and a statement method, plus JUnit tests. Output code only."),
    ("bash greedy", "Write a Bash script that backs up a directory to a dated tar.gz archive, rotates backups older than N days, logs to a file and handles errors and usage. Output code only."),
    ("html greedy", "Write a single HTML file with embedded CSS and JavaScript for a responsive pricing page with three plans, a monthly/yearly toggle and a FAQ accordion. Output code only."),
    ("cpp greedy", "Write a C++17 Matrix class template with addition, multiplication, transpose, an identity factory and stream output, plus a small test main. Output code only."),
    ("pandas greedy", "Write a Python script using pandas that loads a CSV of sales records, cleans missing values, computes monthly revenue per region and writes a summary CSV, with functions and docstrings. Output code only."),
    ("jest greedy", "Write a JavaScript module with functions to validate email addresses, phone numbers and postal codes using regular expressions, and a Jest test file covering them. Output code only."),
    ("docker greedy", "Write a Dockerfile and a docker-compose.yml for a Python web application with a PostgreSQL database and a Redis cache, with health checks, volumes and environment variables. Output the files only."),
    ("readme greedy", "Write a detailed README in Markdown for an open-source command-line tool that syncs files between directories: features, installation, usage examples, a configuration table, FAQ and contributing."),
    ("csharp greedy", "Write a C# console application that manages a to-do list stored in a JSON file, with add, list, complete and remove commands. Output code only."),
]


def pair(label, prompts, max_tokens):
    """Different greedy prompts at once: each stream's rounds, verified drafts and rate, and the aggregate by wall."""
    res = [None] * len(prompts)

    def one(i):
        body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": prompts[i]}], "max_tokens": max_tokens,
                "seed": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(sp.URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            res[i] = json.load(urllib.request.urlopen(req, timeout=900))
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"[:300]}

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
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
    bad = [r.get("error") for r in res if r.get("error")] + [f"no done line for request {i}" for i, f in enumerate(rows) if not f.get("rounds")]
    tokens = sum(int(f.get("tokens", 0)) for f in rows)
    rounds = sum(int(f.get("rounds", 0)) for f in rows)
    drafted = sum(int(f["accepted"].split("/")[1]) for f in rows if f.get("accepted"))
    accepted = sum(int(f["accepted"].split("/")[0]) for f in rows if f.get("accepted"))
    return {**({"failed": json.dumps(bad)[:300]} if bad else {}), "label": label, "width": len(prompts), "seed0": 1, "tokens": tokens,
            "wall_s": round(wall, 3), "agg_tok_s_wall": round(tokens / wall, 2), "rounds_total": rounds, "drafted_total": drafted,
            "tokens_per_stream_round": round(tokens / rounds, 3) if rounds else None, "drafted_per_stream_round": round(drafted / rounds, 3) if rounds else None,
            "accept_pct": round(100 * accepted / drafted, 1) if drafted else None, "decode_s": [f.get("decode_s") for f in rows],
            "tok_per_s": [f.get("tok_per_s") for f in rows], "sha": [f.get("sha") for f in rows]}


def main():
    label, out = sys.argv[1], sys.argv[2]
    jobs = [("tetris greedy", sp.TETRIS, False, 1, 2048), ("tetris sampled", sp.TETRIS, True, 700, 2048),
            ("code greedy", sp.CODE, False, 1, 1536), ("json greedy", sp.JSONP, False, 1, 1536), ("prose greedy", sp.PROSE[0], False, 1, 768)]
    jobs += [(name, prompt, False, 1, n) for name, prompt, n in sp.EXTRA]
    jobs += [(name, prompt, False, 1, 1280) for name, prompt in MORE]
    jobs += [("code sampled", sp.CODE, True, 701, 1280), ("go sampled", sp.EXTRA[0][1], True, 702, 1280)]
    rows = []

    def stop(r):
        json.dump(rows, open(out, "w"))
        print(label, "FAILED:", r["label"], r["failed"], flush=True)
        sys.exit(2)

    for name, prompt, sampled, seed, n in jobs:
        r = sp.run(name, prompt, sampled, seed, n)
        rows.append(r)
        if r.get("failed"):
            stop(r)
        print(label, name, "seed", seed, {k: r.get(k) for k in ("rounds", "accepted", "exact_tokens_per_round", "tok_per_s", "sha")}, flush=True)
    for width, seed0 in ((2, 800), (4, 810)):
        b = sp.burst(f"tetris sampled x{width}", width, seed0, 768)
        rows.append(b)
        if b.get("failed"):
            stop(b)
        print(label, b["label"], {k: b[k] for k in ("agg_tok_s_wall", "tokens_per_stream_round", "drafted_per_stream_round", "accept_pct", "wall_s")}, flush=True)
    b = pair("code and go x2", [sp.CODE, sp.EXTRA[0][1]], 1024)
    rows.append(b)
    if b.get("failed"):
        stop(b)
    print(label, b["label"], {k: b[k] for k in ("agg_tok_s_wall", "tokens_per_stream_round", "drafted_per_stream_round", "accept_pct", "wall_s")}, flush=True)
    json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    main()
