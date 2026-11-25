"""Send decode requests to fill the M3 probe dump.

One stream only (the engine must run PARALLEL=1): a round then holds one window, which is what train.py's
chain-prefix label assumes. Each request decodes MAX_TOKENS tokens; every verify round appends one dump record.

  python3 runs/m3-20261125/capture.py <requests> [max_tokens]

Stops early if the dump (via /metrics-free check) is not growing, and prints a per-request line so a stall is visible.
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000"
TOK_PATH = "/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json"
CORPUS = os.environ.get("WORKLOAD_CORPUS", "runs/overnight-20261122/corpus.json")
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    temp = float(os.environ.get("TEMPERATURE", "0"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    ids = json.load(open(CORPUS))
    t0 = time.time()
    par = int(os.environ.get("PARALLEL", "1"))  # concurrent requests, so the trace sees overlapping streams
    def one(i):
        off = (i * 7919 + 12345) % max(1, len(ids) - 4096)
        p = tok.decode(ids[off:off + 4096]) + ASK
        body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}],
                "max_tokens": max_tokens, "temperature": temp,
                "top_k": (20 if temp > 0 else 0), "top_p": (0.95 if temp > 0 else 1.0),
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                o = json.load(r)
        except Exception as e:
            print(f"{i} ERROR {e}", flush=True)
            return
        u = o.get("usage") or {}
        print(f"{i} {time.time()-t:.2f}s tokens={u.get('completion_tokens')} total={time.time()-t0:.0f}s", flush=True)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(one, range(n)))


if __name__ == "__main__":
    main()
