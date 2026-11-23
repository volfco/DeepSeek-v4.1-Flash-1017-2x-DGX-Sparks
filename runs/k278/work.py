"""k278 gate workload: (1) cold 32K x2 / 96K prefill (salted) for 0278 TAPS_PACKED; (2) long-context greedy decode at
~300K (indexer nbk 32,768) and ~600K (nbk 65,536) for 0279 IDX_PICK_WIDE, which only engages past 16,384 blocks.
Long prompts are NOT salted: they are identical in every arm (the spill tier may serve later arms' prefill), so the
decode text sha is comparable across arms (exactness at depth) and decode tok/s is measured after TTFT.
  python3 work.py <label> <out.json>"""
import hashlib, json, os, sys, time, urllib.request
from tokenizers import Tokenizer
URL = "http://127.0.0.1:8000"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
ids = json.load(open("runs/overnight-20261122/corpus.json"))
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."
def prompt(depth, salt):
    off = (salt * 7919) % max(1, len(ids) - depth); return TOK.decode(ids[off:off + depth]) + ASK
def long_prompt(depth):
    parts, n, k = [], 0, 0
    while n < depth:
        take = min(40000, depth - n); off = (k * 15485) % (len(ids) - take)
        parts.append(f"\n\n## Part {k + 1}\n\n" + TOK.decode(ids[off:off + take])); n += take; k += 1
    return "".join(parts) + ASK
def post(body, stream=False):
    return urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
def gen(p, max_tokens):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); ttft = None; text = []; usage = {}
    with post(body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if ttft is None: ttft = time.time() - t0
                    text.append(c)
    tot = time.time() - t0; out = usage.get("completion_tokens", 0)
    return {"ttft": round(ttft or -1, 3), "s": round(tot, 3), "prompt_tokens": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "tokens": out,
            "decode_tok_s": round((out - 1) / (tot - ttft), 2) if ttft and out > 1 else None,
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:12]}
label, outp = sys.argv[1], sys.argv[2]
res = {"label": label, "prefill": [], "long": []}
gen(prompt(512, 999), 8)
for depth in (32768, 32768, 98304):
    r = gen(f"[cold {label} {time.time():.6f}]\n" + prompt(depth, 31 * depth + 5), 1); r["depth"] = depth
    r["tok_s"] = round((r["prompt_tokens"] - r["cached"]) / r["s"], 1); res["prefill"].append(r); print(json.dumps(r), flush=True)
for depth in (300000, 600000):
    p = long_prompt(depth)
    for rep in range(2):
        r = gen(p, 256); r["depth"] = depth; r["rep"] = rep; res["long"].append(r); print(json.dumps(r), flush=True)
json.dump(res, open(outp, "w"), indent=1)
