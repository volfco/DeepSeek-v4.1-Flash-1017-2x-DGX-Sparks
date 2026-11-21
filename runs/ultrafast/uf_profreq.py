#!/usr/bin/env python3
"""Request files for tf-dsv41-lanes --requests (rank 0's round profile), built on the live server.
usage: uf_profreq.py OUT_DIR    -> OUT_DIR/r1.json (run with --parallel 1) and OUT_DIR/r4.json (--parallel 4)
Each request's expect_sha is the served reply's token_sha for the same ids and sampling (greedy ones must match)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import checklib as C

S = C.Server("http://127.0.0.1:8000", timeout=900)
PROSE = ["Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
         "Explain in depth how a compiler turns source code into machine code, step by step.",
         "Describe a week-long hiking trip through the mountains, day by day, with vivid detail.",
         "Write a thorough essay on the history of the printing press and its effects on Europe."]
TETRIS = "implement the tetris game. output HTML code directly - no explanation."


def ids_of(text):
    t = S.post_json("/tokenize", {"model": S.model, "messages": [{"role": "user", "content": text}],
                                  "chat_template_kwargs": {"enable_thinking": False}})
    return t["tokens"]


def req(name, text, max_tokens, seed=None):
    ids = ids_of(text)
    body = {"model": S.model, "prompt": ids, "max_tokens": max_tokens, "return_token_ids": True}
    if seed is None:
        body["temperature"] = 0
    else:
        body.update(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, seed=seed)
    r = S.post_json("/v1/completions", body)
    tf = r.get("tensorfold") or {}
    q = {"name": name, "prompt": ids, "max_tokens": max_tokens, "expect_sha": tf.get("token_sha")}
    if seed is not None:
        q.update(seed=seed, temperature=1.0, top_k=20, top_p=0.95)
    print(name, len(ids), (r.get("usage") or {}).get("completion_tokens"), q["expect_sha"], r.get("error") or "", flush=True)
    return q


out = sys.argv[1]
r1 = [req(f"prose{i}", p, 256) for i, p in enumerate(PROSE)]
r1 += [req("tetris-greedy", TETRIS, 1024), req("tetris-s1", TETRIS, 1024, 1), req("prose0-s2", PROSE[0], 256, 2)]
json.dump({"requests": r1}, open(os.path.join(out, "r1.json"), "w"))
json.dump({"requests": [q for q in r1 if q["name"].startswith("prose") and "seed" not in q]},
          open(os.path.join(out, "r4.json"), "w"))
