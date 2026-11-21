#!/usr/bin/env python3
"""Short single-pass probes for the baseten-ultrafast switches (run on the head, from the repo root).

  python3 -I uf_probe.py sampled --label L --out F.json    # sampled + greedy decode tok/s, 1 and 4 streams
  python3 -I uf_probe.py longctx --label L --out F.json    # long prompt, then a resumed turn: ids for done-line lookup
  python3 -I uf_probe.py tetris --label L --out F.json     # Baseten's tetris prompt, 1 stream: greedy once, sampled per seed

Every reply's id is recorded so the done line (render_s / encode_s / ttft_s / tok_per_s / accept) can be joined
from `docker logs` afterwards.
"""
import argparse, json, os, sys, threading, time
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import checklib as C

S = C.Server("http://127.0.0.1:8000", timeout=900)
PROMPTS = ["Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
           "Explain in depth how a compiler turns source code into machine code, step by step.",
           "Describe a week-long hiking trip through the mountains, day by day, with vivid detail.",
           "Write a thorough essay on the history of the printing press and its effects on Europe."]


def chat(messages, max_tokens, sampled, seed):
    body = {"model": S.model, "messages": messages, "max_tokens": max_tokens, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    if sampled:
        body.update(temperature=1.0, top_p=0.95, top_k=20)
    else:
        body.update(temperature=0)
    r = S.post_json("/v1/chat/completions", body)
    out = {"id": r.get("id"), "seconds": r.get("_seconds"), "error": r.get("error"),
           "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
           "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
           "cached_tokens": ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
    try:
        out["text"] = r["choices"][0]["message"]["content"]
    except Exception:
        out["text"] = None
    return out


def sampled(a):
    rows = []
    for mode in ("greedy", "sampled"):
        for streams in (1, 4):
            for rep in range(a.reps):
                res = [None] * streams
                def go(i):
                    res[i] = chat([{"role": "user", "content": PROMPTS[i % 4]}], a.tokens, mode == "sampled",
                                  1000 + rep * 10 + i)
                t0 = time.monotonic()
                th = [threading.Thread(target=go, args=(i,)) for i in range(streams)]
                [t.start() for t in th]; [t.join() for t in th]
                wall = time.monotonic() - t0
                toks = sum((x.get("completion_tokens") or 0) for x in res)
                rows.append({"mode": mode, "streams": streams, "rep": rep, "wall": wall, "agg_tok_s": toks / wall,
                             "replies": [{k: v for k, v in x.items() if k != "text"} for x in res]})
                print(f"{a.label} {mode} streams={streams} rep={rep} agg_tok_s={toks / wall:.1f}", flush=True)
    return rows


TETRIS = "implement the tetris game. output HTML code directly - no explanation."


def tetris(a):
    """Baseten's like-for-like row: thinking off, 1 stream, greedy once then temperature 1 for each seed."""
    rows = []
    for mode, seed in [("greedy", 1)] + [("sampled", a.seed * 100 + k) for k in range(a.reps)]:
        r = chat([{"role": "user", "content": TETRIS}], a.tokens, mode == "sampled", seed)
        toks = r.get("completion_tokens") or 0
        row = {"mode": mode, "seed": seed, "tok_s_wall": toks / r["seconds"] if r.get("seconds") else None,
               **{k: v for k, v in r.items() if k != "text"}}
        rows.append(row)
        print(f"{a.label} tetris {mode} seed={seed} tokens={toks} wall_tok_s={row['tok_s_wall']}", flush=True)
    return rows


def longctx(a):
    rows = []
    for size in [int(x) for x in a.sizes.split(",")]:
        hist = C.words(C.words_for(size), seed=size + a.seed)
        m1 = [{"role": "user", "content": "Read this and reply OK.\n\n" + hist}]
        r1 = chat(m1, 8, False, 1)
        m2 = m1 + [{"role": "assistant", "content": r1.get("text") or "OK"},
                   {"role": "user", "content": "Now summarise the first sentence in five words."}]
        r2 = chat(m2, 8, False, 1)
        for turn, r in (("cold", r1), ("resume", r2)):
            r = {k: v for k, v in r.items() if k != "text"}
            rows.append({"size": size, "turn": turn, **r})
            print(f"{a.label} size={size} {turn} {r}", flush=True)
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("sampled", "longctx", "tetris"))
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--sizes", default="100000,250000")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rows = {"sampled": sampled, "longctx": longctx, "tetris": tetris}[a.mode](a)
json.dump({"label": a.label, "mode": a.mode, "rows": rows}, open(a.out, "w"), indent=1)
