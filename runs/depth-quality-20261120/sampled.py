"""Sampled depth probe: same prompts as probe.py, at the clients' sampling (T=1.0, top_p 0.95, top_k 20) with explicit seeds.
   python3 -I runs/depth-quality-20261120/sampled.py <label>   (head, recipe root, against production; writes <label>.json)"""
import json, sys, time, urllib.request
src = open("runs/depth-quality-20261120/probe.py").read().split("label = sys.argv[1]")[0]   # reuse build()/score() only
ns = {"__name__": "probe_build"}; sys.path.insert(0, "tools"); exec(src, ns)
build, score, URL, MODEL, OUT = ns["build"], ns["score"], ns["URL"], ns["MODEL"], ns["OUT"]
DEPTHS = [64000, 256000, 530000]
SEEDS = [11, 22, 33, 44, 55]
MAX = 8192

def ask(prompt, seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX, "temperature": 1.0,
            "top_p": 0.95, "top_k": 20, "seed": seed, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=7200) as r:
        j = json.load(r)
    j["_wall_s"] = time.monotonic() - t0
    return j

def repeat_frac(text, k=64):
    """Share of k-char chunks (stride k) seen earlier in the text: ~0 for normal reasoning, high when it loops."""
    chunks = [text[i:i + k] for i in range(0, max(len(text) - k, 0), k)]
    seen, dup = set(), 0
    for c in chunks:
        dup += c in seen; seen.add(c)
    return round(dup / len(chunks), 3) if chunks else 0.0

label = sys.argv[1]; rows = []
for depth in DEPTHS:
    prompt, names, codes = build(depth)
    for seed in SEEDS:
        j = ask(prompt, seed)
        msg = j["choices"][0]["message"]; tf = j.get("tensorfold", {})
        rc = msg.get("reasoning_content") or ""
        row = {"depth": depth, "seed": seed, "prompt_tokens": j["usage"]["prompt_tokens"],
               "cached": j["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
               "out_tokens": j["usage"]["completion_tokens"], "finish": j["choices"][0]["finish_reason"],
               "token_sha": tf.get("token_sha"), "wall_s": j["_wall_s"], "reasoning_chars": len(rc),
               "reasoning_repeat": repeat_frac(rc), "accept": j.get("speculative", {}).get("acceptance_rate"),
               **score(msg.get("content"), names, codes), "content": msg.get("content")}
        rows.append(row)
        print(f"{label} {depth:>7} seed={seed} cached={row['cached']} out={row['out_tokens']} finish={row['finish']} "
              f"codes={row['codes_correct']}/8 order={row['order_ok']} sum={row['sum_ok']} rep={row['reasoning_repeat']} "
              f"sha={row['token_sha']} wall={row['wall_s']:.0f}s", flush=True)
        json.dump(rows, open(f"{OUT}/{label}.json", "w"), indent=1)
