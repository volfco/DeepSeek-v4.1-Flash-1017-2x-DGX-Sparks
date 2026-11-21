#!/usr/bin/env python3
"""The W4 prefill-profile request file (item 3, `TF_DS_PREFILL_PROFILE=1`): prompts of ~256 / 512 / 1,024 / 2,048 /
8,192 / 32,768 tokens of tools/bench.py's prose, tokenized on the live server (chat template, thinking off), each with
max_tokens 1 (a prompt pass, no decode to speak of). No expect_sha: all_equal is meaningless for this file.
usage (repo root, server up): w4_prefreq.py OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import checklib as C
import bench as B

S = C.Server("http://127.0.0.1:8000", timeout=300)
reqs = []
for size in (256, 512, 1024, 2048, 8192, 32768):
    text = B.prose(int(size / B.TOK_PER_SIZE), f"w4-prefill-{size}") + "\n\nReply with the single word: ok"
    t = S.post_json("/tokenize", {"model": S.model, "messages": [{"role": "user", "content": text}],
                                  "chat_template_kwargs": {"enable_thinking": False}})
    ids = t["tokens"]
    reqs.append({"name": f"pf{size}", "prompt": ids, "max_tokens": 1})
    print(f"pf{size}", len(ids), flush=True)
json.dump({"requests": reqs}, open(sys.argv[1], "w"))
