"""Depth-quality probe: identical greedy prompts at 4K..530K with 8 planted facts; recall + sum scored, token_sha kept.
   python3 -I runs/depth-quality-20261120/probe.py <label>      (head, recipe root; writes runs/depth-quality-20261120/<label>.json)"""
import json, os, random, sys, time, urllib.request
sys.path.insert(0, "tools")
import bench_div as bd

OUT = "runs/depth-quality-20261120"
URL, MODEL = "http://127.0.0.1:8000", "DeepSeek-V4.1-Flash-1017"
DEPTHS = [4000, 64000, 256000, 530000]
FRACS = [0.02, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.98]
NAMES = ["Alder", "Brisk", "Cobalt", "Dunmore", "Ember", "Fallow", "Garnet", "Hollin"]
corpus = bd.Corpus("runs/suite-20261119/corpus")


def build(depth):
    rng = random.Random(9000 + depth)
    names = NAMES[:]; rng.shuffle(names)
    codes = [rng.randint(10000, 99999) for _ in NAMES]
    paras = corpus.text("div", depth, f"depthq-{depth}", fixed=True).split("\n\n")
    marks = sorted((int(f * len(paras)), i) for i, f in enumerate(FRACS))
    for pos, i in reversed(marks):
        paras.insert(pos, f"Logbook entry: the vault code for station {names[i]} is {codes[i]}.")
    q = ("Above, scattered through the text, are logbook entries giving a vault code for each of several stations. "
         "List every station and its vault code in the order the entries appear, one 'Station: code' per line, "
         "then a final line 'Sum: N' with the sum of all the codes.")
    return f"[depthq-{depth}]\n" + "\n\n".join(paras) + "\n\n" + q, names, codes


def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0,
            "seed": 1234, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    j["_wall_s"] = time.monotonic() - t0
    return j


def score(text, names, codes):
    lines = [l.strip().strip("*-• ").replace("**", "") for l in (text or "").splitlines() if l.strip()]
    got = []
    for l in lines:
        for n in names:
            if l.lower().startswith(n.lower()):
                digits = "".join(c for c in l[len(n):] if c.isdigit())
                got.append((n, int(digits) if digits else None))
    correct = sum(1 for n, c in got if n in names and codes[names.index(n)] == c)
    order_ok = [n for n, _ in got] == names
    sum_line = next((l for l in lines if l.lower().startswith("sum")), "")
    s = "".join(ch for ch in sum_line if ch.isdigit())
    return {"codes_correct": correct, "order_ok": order_ok, "sum_ok": s == str(sum(codes)), "sum_got": s,
            "sum_true": sum(codes)}


label = sys.argv[1]
rows = []
plan = [(d, "cold") for d in DEPTHS] + [(DEPTHS[-1], "resume")]
for depth, mode in plan:
    prompt, names, codes = build(depth)
    j = ask(prompt)
    msg = j["choices"][0]["message"]
    tf = j.get("tensorfold", {})
    row = {"depth": depth, "mode": mode, "prompt_tokens": j["usage"]["prompt_tokens"],
           "cached": j["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
           "out_tokens": j["usage"]["completion_tokens"], "finish": j["choices"][0]["finish_reason"],
           "token_sha": tf.get("token_sha"), "ttft_s": tf.get("time_to_first_token"), "wall_s": j["_wall_s"],
           "accept": j.get("speculative", {}).get("acceptance_rate"), **score(msg.get("content"), names, codes),
           "content": msg.get("content"), "reasoning_chars": len(msg.get("reasoning_content") or "")}
    rows.append(row)
    print(f"{label} {depth:>7} {mode:6} prompt={row['prompt_tokens']} cached={row['cached']} out={row['out_tokens']} "
          f"sha={row['token_sha']} codes={row['codes_correct']}/8 order={row['order_ok']} sum={row['sum_ok']} "
          f"finish={row['finish']} wall={row['wall_s']:.0f}s", flush=True)
    json.dump(rows, open(f"{OUT}/{label}.json", "w"), indent=1)
