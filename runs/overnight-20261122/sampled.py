"""Sampled-decode identity check (0261 TF_DS_TOPK_MERGE needs sampled rows; exact.py is greedy).

  python3 runs/overnight-20261122/sampled.py <label> <out.json> [--compare ref.json]

Three waves of four concurrent seeded requests at T=1.0 / top_p 0.95 with different top_k per request inside a wave
(so the round's kmax varies: 1, 5, 20, 56 / 3, 8, 40, 56 / 20, 20, 20, 20), 160 tokens each, thinking off, plus the
clients' setting (top_k 20) solo. Records each reply's tensorfold.token_sha and completion tokens. --compare: exit 1 if
any (wave, request) differs from the reference.
"""
import concurrent.futures as cf, json, sys, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = ["Write a short story about a lighthouse keeper and a storm.",
           "Explain how a hash map handles collisions, with examples.",
           "List twelve creative names for a coffee shop and why each works.",
           "Describe the water cycle to a ten year old."]
WAVES = [[1, 5, 20, 56], [3, 8, 40, 56], [20, 20, 20, 20]]


def run(prompt, top_k, seed):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": prompt}], "max_tokens": 160,
            "temperature": 1.0, "top_p": 0.95, "top_k": top_k, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        o = json.load(r)
    tf = o.get("tensorfold") or {}
    return {"top_k": top_k, "seed": seed, "sha": tf.get("token_sha"), "tokens": (o.get("usage") or {}).get("completion_tokens"),
            "finish": o["choices"][0].get("finish_reason")}


label, out = sys.argv[1], sys.argv[2]
res = {"label": label, "waves": []}
for w, ks in enumerate(WAVES):
    with cf.ThreadPoolExecutor(4) as ex:
        res["waves"].append(list(ex.map(lambda a: run(PROMPTS[a[0]], a[1], 1000 * w + a[0] + 7), enumerate(ks))))
res["solo"] = [run(PROMPTS[i], 20, 500 + i) for i in range(4)]
json.dump(res, open(out, "w"), indent=1)
print(label, "waves", [[(x["top_k"], (x["sha"] or "")[:8], x["tokens"]) for x in w] for w in res["waves"]])
if "--compare" in sys.argv:
    ref = json.load(open(sys.argv[sys.argv.index("--compare") + 1]))
    bad = [(i, j) for i, w in enumerate(res["waves"]) for j, x in enumerate(w) if (x["sha"], x["tokens"]) != (ref["waves"][i][j]["sha"], ref["waves"][i][j]["tokens"])]
    bad += [("solo", j) for j, x in enumerate(res["solo"]) if (x["sha"], x["tokens"]) != (ref["solo"][j]["sha"], ref["solo"][j]["tokens"])]
    print("sampled vs ref:", "IDENTICAL" if not bad else f"DIFFER {bad}")
    sys.exit(1 if bad else 0)
