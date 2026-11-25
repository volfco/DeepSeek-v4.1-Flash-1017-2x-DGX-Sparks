"""Mixed prefill+decode cell: one cold 128K+ prefill beside 7 decoding streams.

The cell no suite has. Live showed a 166-192K cold pass stretching other streams' rounds from 92 to 155-312 ms and
prefill dropping to 1,670 tok/s from 2,040-2,100. This measures that interference directly and per arm, so the
TF_DS_PREFILL_INTERLEAVE_MS / TF_DS_PREFILL_SLICE knobs can be swept against production's 250 ms / 5 layers.

  python3 runs/mixed-20261125/mixed.py <label> <out.json> [--control]

One server restart per arm; the arm's knobs come from the environment start.sh was given. The decoders' prompts are
warmed with a 1-token request first so they hit the prompt cache and decode immediately, putting the cold prefill in
the decode period rather than in a competing prefill. --control runs the same 7 decoders with no prefill.

Metrics come from the engine's own `done` line per request: prompt, cached, tokens, rounds, prefill_s, ttft_s,
decode_s, tok_per_s. Round time = decode_s*1000/rounds. The interference is the decoders' decode tok/s in the mixed
cell against the control's.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import concurrent.futures as cf

URL = "http://127.0.0.1:8000"
C = "dsv41-1017-exl3-tf"
TOK_PATH = "/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json"

DECODERS = 7
DECODER_DEPTH = 8192
DECODER_TOKENS = 1024
PREFILL_DEPTH = 131072
PREFILL_TOKENS = 8
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOK_PATH)


def corpus(tok):
    """The frozen corpus the other suites use, so prompts match across arms."""
    p = os.environ.get("WORKLOAD_CORPUS", "runs/overnight-20261122/corpus.json")
    if os.path.exists(p):
        return json.load(open(p))
    import glob
    ids = []
    for f in sorted(glob.glob("runs/**/RESULTS.md", recursive=True)) + sorted(glob.glob("docs/*.md")):
        ids += tok.encode(open(f, errors="replace").read()).ids
        if len(ids) > PREFILL_DEPTH * 5:
            break
    return ids


def prompt(tok, ids, depth, salt):
    off = (salt * 7919) % max(1, len(ids) - depth)
    return tok.decode(ids[off:off + depth]) + ASK


def chat(p, max_tokens):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}],
            "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        o = json.load(r)
    return {"s": round(time.time() - t, 3), "usage": o.get("usage"),
            "finish": o["choices"][0].get("finish_reason")}


def done_lines(since):
    """The engine's per-request `done` lines after `since` (UTC 'YYYY-MM-DDTHH:MM:SS')."""
    log = subprocess.run(["docker", "logs", "--since", since, C], capture_output=True, text=True)
    out = []
    for line in (log.stdout + log.stderr).splitlines():
        i = line.find("done chatcmpl")
        if i < 0:
            continue
        # [tensorfold] done chatcmpl-... prompt=... cached=... ... key=value pairs
        kv = {}
        for tok in line[i:].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        try:
            kv["prompt"] = int(kv.get("prompt", 0))
            kv["cached"] = int(kv.get("cached", 0))
            kv["tokens"] = int(kv.get("tokens", 0))
            kv["rounds"] = int(kv.get("rounds", 0))
            kv["prefill_s"] = float(kv.get("prefill_s", 0))
            kv["ttft_s"] = float(kv.get("ttft_s", 0))
            kv["decode_s"] = float(kv.get("decode_s", 0))
            kv["tok_per_s"] = float(kv.get("tok_per_s", 0))
        except ValueError:
            continue
        out.append(kv)
    return out


def main():
    label, out = sys.argv[1], sys.argv[2]
    control = "--control" in sys.argv or os.environ.get("MIXED_CONTROL") == "1"
    tok = load_tokenizer()
    ids = corpus(tok)
    stamp = time.time()

    dec_prompts = [prompt(tok, ids, DECODER_DEPTH, 1000 + i + int(stamp) % 1000) for i in range(DECODERS)]
    pre_prompt = prompt(tok, ids, PREFILL_DEPTH, 9000 + int(stamp) % 1000)

    # warm the decoders' prompts so the cell's requests hit the cache and decode at once
    with cf.ThreadPoolExecutor(DECODERS) as ex:
        list(ex.map(lambda p: chat(p, 1), dec_prompts))
    time.sleep(2)

    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 2))
    t0 = time.time()
    with cf.ThreadPoolExecutor(DECODERS + 1) as ex:
        decs = [ex.submit(chat, p, DECODER_TOKENS) for p in dec_prompts]
        time.sleep(1.5)  # let the decoders reach decode before the cold pass lands
        pre = None if control else ex.submit(chat, pre_prompt, PREFILL_TOKENS)
        dec_res = [f.result() for f in decs]
        pre_res = pre.result() if pre else None
    wall = round(time.time() - t0, 2)
    time.sleep(2)

    lines = done_lines(since)
    # the cell's requests: long generations are decoders, a 128K prompt is the prefill
    dec = [d for d in lines if d["tokens"] >= 100]
    prel = [d for d in lines if d["prompt"] >= 120000]
    for d in dec:
        d["ms_per_round"] = round(d["decode_s"] * 1000 / d["rounds"], 1) if d["rounds"] else None
    for d in prel:
        d["prefill_tok_s"] = round((d["prompt"] - d["cached"]) / d["prefill_s"], 0) if d["prefill_s"] else None

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    res = {
        "label": label, "control": control, "wall_s": wall,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TF_DS_PREFILL", "PREFILL_"))},
        "decoders": {"n": len(dec), "requested": DECODERS,
                     "decode_tok_s_mean": mean([d["tok_per_s"] for d in dec]),
                     "ms_per_round_mean": mean([d["ms_per_round"] for d in dec]),
                     "ttft_s_mean": mean([d["ttft_s"] for d in dec]),
                     "rounds_sum": sum(d["rounds"] for d in dec),
                     "tokens_sum": sum(d["tokens"] for d in dec)},
        "prefill": ({"n": len(prel), "prefill_tok_s": mean([d["prefill_tok_s"] for d in prel]),
                     "prefill_s": mean([d["prefill_s"] for d in prel]),
                     "ttft_s": mean([d["ttft_s"] for d in prel])} if prel else None),
        "rows": dec + prel,
        "client": {"decoder_s": [r["s"] for r in dec_res],
                   "prefill_s": pre_res["s"] if pre_res else None},
    }
    json.dump(res, open(out, "w"), indent=1)
    d = res["decoders"]
    print(json.dumps({"label": label, "control": control, "wall": wall,
                      "dec_tok_s": d["decode_tok_s_mean"], "ms_per_round": d["ms_per_round_mean"],
                      "prefill_tok_s": res["prefill"]["prefill_tok_s"] if res["prefill"] else None,
                      "n_dec": d["n"], "n_pre": res["prefill"]["n"] if res["prefill"] else 0}))


if __name__ == "__main__":
    main()
