#!/usr/bin/env bash
# Candidate 3: the pool-block union a 16/128-row block of prompt rows shares at the candidate source (layer 20), on cold
# natural-text prompts at 24K/48K/96K/160K tokens. Image: production's (84806bb + 0259) + the TF_DS_POOL_UNION_STAT
# diagnostic (branch pool-union-stat). One restart with the diagnostic, prompts, the log lines, then production restored.
#   IMG=tensorfold-dsv41:bench-fused-next-<sha> setsid nohup bash runs/pool-union-20261122/run.sh > runs/pool-union-20261122/run.log 2>&1 < /dev/null &
set -uo pipefail
out=runs/pool-union-20261122; mkdir -p "$out"
: "${IMG:?set IMG}"
C=dsv41-1017-exl3-tf
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
echo "$(date -u +%FT%TZ) diag restart"
env IMAGE="$IMG" PREPARE=0 TF_DS_POOL_UNION_STAT=1 ./start.sh restart > "$out/diag.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$out/diag.restart.log" || { echo "restart failed"; tail -5 "$out/diag.restart.log"; }
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
since=$(date -u +%FT%TZ)
python3 - <<'PY'
import glob, json, time, urllib.request
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
ids = []
for f in sorted(glob.glob("runs/**/RESULTS.md", recursive=True)) + sorted(glob.glob("docs/*.md")) + sorted(glob.glob("*.md")):
    ids += TOK.encode(open(f, errors="replace").read()).ids
print("corpus tokens", len(ids), flush=True)
for depth, off in ((24576, 3), (49152, 50000), (98304, 120000), (163840, 7)):
    off = off % max(1, len(ids) - depth)
    p = f"[salt {depth}-{off}]\n" + TOK.decode(ids[off:off + depth]) + "\n\nIn one sentence, what is the document above about?"
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        o = json.load(r)
    print(json.dumps({"depth": depth, "s": round(time.time() - t, 1), "usage": o.get("usage")}), flush=True)
PY
docker logs --since "$since" $C 2>&1 | grep 'TF_DS_POOL_UNION_STAT' > "$out/union.log"
wc -l < "$out/union.log"
echo "$(date -u +%FT%TZ) restore production"
./start.sh restart > "$out/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$out/restore.restart.log" || echo "restore restart failed"
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$out/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
tail -2 "$out/restored.validate.log"
echo "$(date -u +%FT%TZ) finished"
