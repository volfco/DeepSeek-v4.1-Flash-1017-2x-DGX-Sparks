"""Three cold natural-text prefills at 32K on the pair as it runs (fresh passages, so nothing is cached).
The November 19 narrow run measured 1,994 tok/s without the lead reader and 2,093 with it. Head, recipe root."""
import json, os, sys, time
sys.path.insert(0, "tools")
import bench_div as bd
srv = bd.Server("http://127.0.0.1:8000", "DeepSeek-V4.1-Flash-1017")
corpus = bd.Corpus("runs/suite-20261119/corpus")
salt = f"{int(time.time()):x}"
rows = []
for rep in range(4):
    text = corpus.text("div", 32000, f"{salt}-chk-{rep}")
    r = srv.chat([{"role": "user", "content": f"[{salt}:chk-{rep}] " + text + "\n\nReply with the single word: ok"}], 1)
    rows.append({k: r.get(k) for k in ("prompt_tokens", "cached_tokens", "ttft_s", "prefill_tok_s")})
    print(f"32K rep {rep}: prompt {r.get('prompt_tokens')} cached {r.get('cached_tokens')} prefill {r.get('prefill_tok_s'):.0f} tok/s" + ("  (first of its size after the restart)" if rep == 0 else ""))
json.dump(rows, open("runs/deploy-lead-20261119/prefill_check.json", "w"), indent=1)
