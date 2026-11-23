"""NCCL screen workload (short, ~2 min an arm): cold prefill (32K x2, 96K x1, salted so nothing is cached) and greedy
decode at 1 and 4 streams on a 4K prompt (fixed prompts, so tokens and rounds repeat across arms).
NCCL carries only the prompt pass's all-gathers and send/recv (no reductions), so these knobs cannot move bits.
  python3 short.py <label> <out.json>"""
import concurrent.futures as cf, json, os, sys, time, urllib.request
sys.path.insert(0, "runs/kvreuse-size-20261122")
from tokenizers import Tokenizer
URL = "http://127.0.0.1:8000"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
ids = json.load(open(os.environ.get("WORKLOAD_CORPUS", "runs/overnight-20261122/corpus.json")))
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."
def prompt(depth, salt):
    off = (salt * 7919) % max(1, len(ids) - depth); return TOK.decode(ids[off:off + depth]) + ASK
def chat(p, max_tokens):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        o = json.load(r)
    return time.time() - t, o.get("usage") or {}
label, outp = sys.argv[1], sys.argv[2]
res = {"label": label, "prefill": [], "decode": []}
chat(prompt(512, 999), 8)
for depth in (32768, 32768, 98304):
    dt, u = chat(f"[cold {label} {time.time():.6f}]\n" + prompt(depth, 31 * depth + 5), 1)
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    res["prefill"].append({"depth": depth, "s": round(dt, 3), "tok_s": round((u["prompt_tokens"] - c) / dt, 1), "cached": c})
    print(json.dumps(res["prefill"][-1]), flush=True)
for n in (1, 4):
    ps = [prompt(4096, 4096 * 100 + i) for i in range(n)]
    for p in ps: chat(p, 1)   # prefill (cached afterwards: the decode cell is decode-dominated)
    t = time.time()
    with cf.ThreadPoolExecutor(n) as ex: rs = list(ex.map(lambda p: chat(p, 256), ps))
    wall = time.time() - t; tok = sum(u["completion_tokens"] for _, u in rs)
    res["decode"].append({"streams": n, "tokens": tok, "wall_s": round(wall, 3), "tok_s": round(tok / wall, 2)})
    print(json.dumps(res["decode"][-1]), flush=True)
json.dump(res, open(outp, "w"), indent=1)
