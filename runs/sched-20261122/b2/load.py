"""Round B2 workload for arms.sh (WORKLOAD_SCRIPT): B2_STREAMS concurrent greedy requests (distinct short prompts,
384 tokens each), twice. Only the trace matters; this just keeps the decode busy at a known concurrency.
  python3 load.py <label> <out.json>"""
import json, os, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
P = ["Write a Python function that parses a CSV file and returns a list of dicts, with tests.",
     "Explain how a hash map handles collisions, with examples in C.",
     "Write a bash script that rotates log files older than seven days.",
     "Describe the TCP three-way handshake and what happens on packet loss."]
n = int(sys.argv[1].rsplit("x", 1)[1]); out = []   # streams from the arm name (k0x4 -> 4)
def one(i, k):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": P[i % 4] + f" (variant {k})"}],
            "max_tokens": 384, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    out.append({"i": i, "k": k, "s": round(time.perf_counter() - t, 3), "tokens": r["usage"]["completion_tokens"]})
for k in range(2):
    th = [threading.Thread(target=one, args=(i, k)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
json.dump({"label": sys.argv[1], "streams": n, "requests": out}, open(sys.argv[2], "w"))
print(sys.argv[1], "streams", n, "tok", sum(r["tokens"] for r in out), "max_s", max(r["s"] for r in out))
