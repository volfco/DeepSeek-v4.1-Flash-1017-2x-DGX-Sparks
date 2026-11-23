"""Prompt-cache replay workload (0276 CACHE_REPLACE_STALE / 0271 RESUME_ASSEMBLE gate).
  python3 replay.py <label> <out.json> [mode]
mode "repeat" (default): K distinct ~WORDS-word documents, each asked R times, C concurrent; the same prompt returns
  after other streams' placements may have taken its kept rows (the stale-duplicate case 0276 targets).
mode "turns": K conversations of T turns each, C concurrent; each turn resends the whole history plus the greedy
  answer and a new question (the exact-resume prepend case 0271 targets).
Records per-request TTFT/total/usage and the /metrics prompt-cache counter deltas."""
import json, os, random, re, sys, threading, time, urllib.request
BASE = os.environ.get("URL", "http://127.0.0.1:8000")
K = int(os.environ.get("K", 6)); R = int(os.environ.get("R", 3)); C = int(os.environ.get("C", 4))
T = int(os.environ.get("T", 4)); WORDS = int(os.environ.get("WORDS", 7000)); MAXT = int(os.environ.get("MAXT", 48))
label, outp = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MODE", "repeat")
VOC = ("system buffer kernel thread queue packet route table index vector matrix tensor shard rank stream round "
       "token cache entry state window prefix layer expert gate norm scale block row column graph launch fence "
       "signal ring proxy flag credit budget spill tier point restore save evict refresh stale live dead").split()
def doc(i):
    rnd = random.Random(1000 + i)
    lines = [f"Document {i}: field log {rnd.randrange(10**9)}."]
    w = 0
    while w < WORDS:
        n = rnd.randrange(8, 20); lines.append(" ".join(rnd.choice(VOC) for _ in range(n)) + "."); w += n
    return "\n".join(lines)
Q = ["Summarize the document in two sentences.", "Which word appears most often? Answer briefly.",
     "List three distinct words from the first line.", "Count the sentences roughly and say how many.",
     "Name one word that never appears.", "Give the document a short title."]
def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^tensorfold:(prompt_cache_[a-z_]+|prefix_cache_[a-z_]+|prompt_tokens_cached_total) ([0-9.e+]+)$", t, re.M)}
def chat(msgs):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": msgs, "max_tokens": MAXT, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); ttft = None; text = []; usage = None
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if ttft is None: ttft = time.perf_counter() - t0
                    text.append(c)
    return {"ttft": round(ttft or -1, 4), "s": round(time.perf_counter() - t0, 4), "usage": usage, "text": "".join(text)}
out = []; lock = threading.Lock()
def run_jobs(jobs):
    it = iter(jobs); il = threading.Lock()
    def worker():
        while True:
            with il:
                j = next(it, None)
            if j is None: return
            r = j["fn"](); r.update(j["tag"])
            with lock: out.append(r)
    th = [threading.Thread(target=worker) for _ in range(C)]; [t.start() for t in th]; [t.join() for t in th]
m0 = metrics(); t0 = time.perf_counter()
if mode == "repeat":
    for rep in range(R):
        run_jobs([{"fn": (lambda i=i, rep=rep: chat([{"role": "user", "content": doc(i) + "\n\n" + Q[rep % len(Q)]}])),
                   "tag": {"doc": i, "rep": rep}} for i in range(K)])
else:
    hist = {i: [] for i in range(K)}
    def step(i, turn):
        msgs = hist[i] + [{"role": "user", "content": (doc(i) + "\n\n" if turn == 0 else "") + Q[turn % len(Q)]}]
        r = chat(msgs); hist[i] = msgs + [{"role": "assistant", "content": r["text"]}]; return r
    for turn in range(T):
        run_jobs([{"fn": (lambda i=i, turn=turn: step(i, turn)), "tag": {"conv": i, "turn": turn}} for i in range(K)])
m1 = metrics()
delta = {k: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
json.dump({"label": label, "mode": mode, "K": K, "R": R, "T": T, "C": C, "wall_s": round(time.perf_counter() - t0, 2),
           "metrics_delta": delta, "requests": out}, open(outp, "w"), indent=1)
first = [r["ttft"] for r in out if r.get("rep", r.get("turn")) == 0]; later = [r["ttft"] for r in out if r.get("rep", r.get("turn")) != 0]
print(label, mode, "wall", round(time.perf_counter() - t0, 1), "ttft first mean", round(sum(first)/len(first), 3),
      "later mean", round(sum(later)/max(1, len(later)), 3))
print(" ", {k: int(v) for k, v in delta.items() if any(s in k for s in ("hits", "misses", "kept", "evicted", "computed", "cached", "restore"))})
