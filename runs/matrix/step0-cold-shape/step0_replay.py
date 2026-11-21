#!/usr/bin/env python3
"""Step 0 (0121): one or more single-turn requests, printed for the split-identity experiment.

Modes:
  --turn-6          the exact turn-6 prompt of runs/ultrafast/full/resume.json
                    (meta + the cold arm's recorded replies 1-5 as history)
  --tokens N        a synthetic prose prompt of ~N tokens (checklib words, seed fixed)
Every arm prints prompt_tokens, cached_tokens, ttft/prefill, token_sha (first 12).
"""
import argparse, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/tools")
import checklib as C  # noqa: E402
import resume_exact as RE  # noqa: E402  (turn_messages/doc/words reuse)

p = argparse.ArgumentParser()
p.add_argument("--turn-6", action="store_true")
p.add_argument("--tokens", type=int, default=0)
p.add_argument("--needle", action="store_true")
p.add_argument("--max-tokens", type=int, default=24)
p.add_argument("--seed", type=int, default=987654)
p.add_argument("--url", default=os.environ.get("API_URL", "http://100.95.49.21:8000"))
p.add_argument("--model", default=os.environ.get("MODEL", C.DEFAULT_MODEL))
a = p.parse_args()

HDR = {"User-Agent": "step0", "Content-Type": "application/json"}
st = None
if a.turn_6:
    st = json.load(open("/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/ultrafast/full/resume.json"))
    m = st["meta"]
    replies = [st["arms"]["cold"]["turns"][str(i + 1)].get("text") or "" for i in range(5)]
    msgs = RE.turn_messages(m["salt"], m["deltas"], m["seed"], replies)
    label = "turn6"
else:
    assert a.tokens > 0
    q = "\n\nWhat is the first word of the text above? Answer with exactly that word and nothing else." if a.needle else "\n\nReply with one short sentence about the text above."
    msgs = [{"role": "user", "content": C.words(C.words_for(a.tokens), a.seed) + q}]
    label = f"needle{a.tokens}s{a.seed}" if a.needle else f"t{a.tokens}s{a.seed}"

srv = C.Server(a.url, a.model, 1800, 1234)
r = srv.stream(msgs, a.max_tokens, keep_text=False)
print(f"{label}: prompt {r.get('prompt_tokens')} cached {r.get('cached_tokens')} "
      f"prefill {r.get('prefill_seconds')}s ttft {r.get('ttft')}s "
      f"sha {(r.get('token_sha') or '')[:12]} ({r.get('token_sha_source')}) err={r.get('error')}")
