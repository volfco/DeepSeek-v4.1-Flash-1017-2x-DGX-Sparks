#!/usr/bin/env bash
# Items 1+2 (rank balance in decode, the keyed prefill indexer's share): both ranks' eager phase timers on production's
# config. Image: the diag-follow-phases build (84806bb + 0259 + the follower printing decode phases_ms at exit).
# Per arm: restart with TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PREFILL_PROFILE=1, the arm's workload,
# rank 0's phases via /metrics, then `docker stop` rank 0 so rank 1 prints its aggregated lines (start.sh's note),
# rank 1's log saved, ./stop.sh. Arms: d32x4 = four cold 32K histories + 4-stream greedy decode (320 tokens);
#                                     p128 = one cold 128K prompt, 32 tokens.
#   IMG=tensorfold-dsv41:bench-fused-next-<sha> setsid nohup bash runs/overnight-20261122/phases.sh > runs/overnight-20261122/phases/run.log 2>&1 < /dev/null &
set -uo pipefail
out=runs/overnight-20261122/phases; mkdir -p "$out"
: "${IMG:?set IMG}"
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
for arm in d32x4 p128; do
  echo "$(date -u +%FT%TZ) arm $arm"
  env IMAGE="$IMG" PREPARE=0 TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PREFILL_PROFILE=1 ./start.sh restart > "$out/$arm.restart.log" 2>&1 \
    || grep -q 'the smoke test request failed' "$out/$arm.restart.log" || { echo "restart failed"; tail -5 "$out/$arm.restart.log"; continue; }
  for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
  since=$(date -u +%FT%TZ)
  ARM=$arm python3 - <<'PY'
import concurrent.futures as cf, glob, json, os, time, urllib.request
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
ids = []
for f in sorted(glob.glob("runs/**/RESULTS.md", recursive=True)) + sorted(glob.glob("docs/*.md")) + sorted(glob.glob("*.md")):
    ids += TOK.encode(open(f, errors="replace").read()).ids
def chat(p, mt, ask):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p + ask}], "max_tokens": mt,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        o = json.load(r)
    return round(time.time() - t, 2), o.get("usage")
arm = os.environ["ARM"]
if arm == "d32x4":
    ps = [f"[phases {i}]\n" + TOK.decode(ids[30000 * i + 11: 30000 * i + 11 + 32768]) for i in range(4)]
    for p in ps: print("prefill", chat(p, 1, "\n\nSummarize the document above in detail, section by section, in plain prose."), flush=True)
    t = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        r = list(ex.map(lambda p: chat(p, 320, "\n\nSummarize the document above in detail, section by section, in plain prose."), ps))
    print("decode", round(time.time() - t, 2), r, flush=True)
else:
    print("prefill", chat("[phases 128]\n" + TOK.decode(ids[5: 5 + 131072]), 32, "\n\nIn one sentence, what is the document above about?"), flush=True)
urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=30).read()
PY
  sleep 2
  docker logs --since "$since" $C 2>&1 | grep -E '"rank": 0, "(profile|phases_ms|profile_note)"' > "$out/$arm.rank0.txt"
  docker stop -t 60 $C > /dev/null 2>&1
  sleep 5
  ssh -o BatchMode=yes "$worker" "timeout 120 docker wait $C >/dev/null 2>&1; docker logs $C 2>&1 | grep -E '\"rank\": 1'" > "$out/$arm.rank1.txt"
  echo "  rank0 lines $(wc -l < "$out/$arm.rank0.txt"), rank1 lines $(wc -l < "$out/$arm.rank1.txt")"
  ./stop.sh > "$out/$arm.stop.log" 2>&1 || true
done
echo "$(date -u +%FT%TZ) restore production (.env alone)"
./start.sh restart > "$out/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$out/restore.restart.log" || echo "restore restart failed"
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$out/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
tail -2 "$out/restored.validate.log"
echo "$(date -u +%FT%TZ) finished"
