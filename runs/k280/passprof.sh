#!/usr/bin/env bash
# 0280: where a drafter pass's time goes. One attribution arm on the new image with the pass's phase timer on
# (TF_DS_PASS_PROFILE=1: the pass runs eagerly under timing events), modelled on runs/overnight-20261122/phases.sh.
# Cells, one server start each so no cell's passes mix with another's:
#   s1 = one sampled stream (three seeded T=1.0 replies in turn), s4 = four sampled streams (two waves of four),
#   g1 = one greedy stream, the cell the trace review's numbers are for (runs/trace-review-20261123/RESULTS.md s.5).
# Per cell: restart with TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PASS_PROFILE=1, 4K prompts from the
# frozen corpus, rank 0's lines scraped before and after the cell (the cell's own means are the difference), then
# `docker stop` rank 0 so rank 1 prints its aggregated lines, rank 1's log saved, ./stop.sh. At the end production is
# restored from .env alone and validated. Attribution only: every figure includes the profile's syncs and the eager
# launches; never quote one as a speed.
#   mkdir -p runs/k280/passprof && setsid nohup bash runs/k280/passprof.sh > runs/k280/passprof/run.log 2>&1 < /dev/null &
# About 20 minutes. CELLS="s1 s4" drops the greedy cell.
set -uo pipefail
out=runs/k280/passprof; mkdir -p "$out"
IMG=${IMG:-tensorfold-dsv41:bench-fused-next-1421187}
CELLS=${CELLS:-s1 s4 g1}
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
docker image inspect "$IMG" > /dev/null || { echo "$IMG missing on the head"; exit 1; }
ssh -n -o BatchMode=yes "$worker" "docker image inspect '$IMG' >/dev/null" || { echo "$IMG missing on the worker"; exit 1; }
for arm in $CELLS; do
  echo "$(date -u +%FT%TZ) cell $arm"
  env IMAGE="$IMG" PREPARE=0 TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PASS_PROFILE=1 ./start.sh restart > "$out/$arm.restart.log" 2>&1 \
    || grep -q 'the smoke test request failed' "$out/$arm.restart.log" || { echo "restart failed"; tail -5 "$out/$arm.restart.log"; continue; }
  ok=0; for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && { ok=1; break; }; sleep 5; done
  (( ok )) || { echo "  no /health"; continue; }
  for r in 0 1; do
    if [[ $r == 0 ]]; then e=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C); else e=$(ssh -n -o BatchMode=yes "$worker" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C"); fi
    echo "rank$r $(grep -E '^(TF_DS_|TENSORFOLD_|NCCL_)' <<< "$e" | grep -vE '_(KEY|TOKEN)=' | sort | tr '\n' ' ')"
  done > "$out/$arm.armed.txt"
  since=$(date -u +%FT%TZ)
  ARM=$arm OUT=$out python3 - <<'PY'
import concurrent.futures as cf, json, os, subprocess, time, urllib.request
from tokenizers import Tokenizer
URL, C = "http://127.0.0.1:8000", "dsv41-1017-exl3-tf"
TOK = Tokenizer.from_file("/nfs1/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/model/tokenizer.json")
ids = json.load(open("runs/overnight-20261122/corpus.json"))   # the frozen corpus (runs/k280/work.py's)
ASK = "\n\nSummarize the document above in detail, section by section, in plain prose."
arm, out = os.environ["ARM"], os.environ["OUT"]
KEYS = ("profile", "markov_cache", "pass_counts", "pass_phases_ms", "phases_ms")

def prompt(i):
    off = (7919 * (280 + i)) % (len(ids) - 4096)
    return f"[passprof {i}]\n" + TOK.decode(ids[off:off + 4096]) + ASK

def chat(p, max_tokens, seed=None):
    body = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        body.update(temperature=1.0, top_p=0.95, top_k=20, seed=seed)
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        o = json.load(r)
    return {"s": round(time.time() - t, 3), "seed": seed, "usage": o.get("usage"), "sha": (o.get("tensorfold") or {}).get("token_sha")}

def scrape():
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 1))
    urllib.request.urlopen(URL + "/metrics", timeout=30).read()
    time.sleep(1.0)
    p = subprocess.run(["docker", "logs", "--since", since, C], capture_output=True, text=True)
    got = {}
    for line in (p.stderr + p.stdout).splitlines():
        i = line.find('{"rank": 0, ')
        if i < 0:
            continue
        try:
            o = json.loads(line[i:])
        except ValueError:
            continue
        for k in KEYS:
            if k in o:
                got[k] = o[k]
                if k == "pass_phases_ms":
                    got["phase_passes"] = o.get("passes", 0)
    return got

def cell(a, b):
    """The cell's own figures from two cumulative scrapes: counters subtracted, means un-averaged and re-averaged."""
    d = {}
    ca, cb = a.get("pass_counts") or {}, b.get("pass_counts") or {}
    if cb:
        d["pass_counts"] = {k: ({s: cb[k][s] - ca.get(k, {}).get(s, 0) for s in cb[k]} if isinstance(cb[k], dict) else cb[k] - ca.get(k, 0)) for k in cb}
        c = d["pass_counts"]
        look = c["hits"] + c["misses"]
        d["miss_rate"] = round(c["misses"] / look, 4) if look else None
        d["rows_a_pass"] = round(c["rows"] / c["passes"], 3) if c["passes"] else None
        d["miss_steps_a_pass"] = round(c["miss_steps"] / c["passes"], 3) if c["passes"] else None
    na, nb = a.get("phase_passes", 0), b.get("phase_passes", 0)
    if nb > na:
        pa, pb = a.get("pass_phases_ms") or {}, b["pass_phases_ms"]
        ph = {k: round((pb[k] * nb - pa.get(k, 0) * na) / (nb - na), 4) for k in pb}
        d["pass_phases_ms"], d["phase_passes"] = ph, nb - na
        d["phases_sum_ms"] = round(sum(ph.values()), 4)
        grp = {"head": ("head",), "moe": ("gate", "experts", "gather_m"), "attention": ("attn_in", "attn_q", "attn_kv", "attn_core", "attn_out", "gather_a"),
               "mix": ("mix",), "markov": ("mk_bias", "mk_score", "mk_gather", "mk_pick"), "gathers": ("gather_a", "gather_m", "mk_gather"),
               "edges": ("upload", "embed", "post", "conf", "read")}
        d["groups_ms"] = {g: round(sum(ph[k] for k in ks), 4) for g, ks in grp.items()}   # (gathers overlaps moe/attention/markov)
        ms = (d.get("pass_counts") or {}).get("miss_steps")
        if ms:
            d["mk_bias_ms_a_miss_step"] = round(ph["mk_bias"] * (nb - na) / ms, 4)   # upper bound: hit steps' _bias is in it too
    fa, fb = a.get("profile") or {}, b.get("profile") or {}
    if fb.get("passes", 0) > fa.get("passes", 0):
        n = fb["passes"] - fa.get("passes", 0)
        d["profile"] = {"passes": n, "pass_ms": round((fb["pass_ms"] * fb["passes"] - fa.get("pass_ms", 0) * fa.get("passes", 0)) / n, 3),
                        "streams_a_pass": round((fb["streams_a_pass"] * fb["passes"] - fa.get("streams_a_pass", 0) * fa.get("passes", 0)) / n, 3)}
    return d

n = 4 if arm == "s4" else 1
waves = 2 if arm == "s4" else 3
ps = [prompt(i) for i in range(n * waves)]
for p in ps:
    chat(p, 1)                                     # each history's prefill first: the cell is decode
a = scrape()
reqs, t = [], time.time()
for w in range(waves):
    batch = ps[w * n:(w + 1) * n]
    seeds = [None] * n if arm == "g1" else [9280 + w * n + i for i in range(n)]
    with cf.ThreadPoolExecutor(n) as ex:
        reqs += list(ex.map(lambda x: chat(x[0], 320, x[1]), zip(batch, seeds)))
wall = round(time.time() - t, 2)
b = scrape()
res = {"cell": arm, "streams": n, "sampled": arm != "g1", "wall_s": wall, "requests": reqs, "before": a, "after": b, "own": cell(a, b)}
json.dump(res, open(f"{out}/{arm}.json", "w"), indent=1)
print(json.dumps({"cell": arm, "wall_s": wall, **{k: res["own"].get(k) for k in ("phase_passes", "phases_sum_ms", "groups_ms", "miss_rate", "rows_a_pass", "miss_steps_a_pass", "mk_bias_ms_a_miss_step", "profile")}}), flush=True)
PY
  sleep 2
  docker logs --since "$since" $C 2>&1 | grep -E '"rank": 0, "(profile|phases_ms|markov_cache|pass_counts|pass_phases_ms|profile_note)"' > "$out/$arm.rank0.txt"
  echo "  rank0 banner $(docker logs $C 2>&1 | grep -c 'TF_DS_PASS_PROFILE: the drafter pass runs eagerly') errors $(docker logs $C 2>&1 | grep -ciE 'panic|error\.|CudaFailed|segfault')"
  echo "  rank1 banner $(ssh -n -o BatchMode=yes "$worker" "docker logs $C 2>&1 | grep -c 'TF_DS_PASS_PROFILE: the drafter pass runs eagerly'") errors $(ssh -n -o BatchMode=yes "$worker" "docker logs $C 2>&1 | grep -ciE 'panic|error\\.|CudaFailed|segfault'")"
  docker stop -t 60 $C > /dev/null 2>&1
  sleep 5
  ssh -n -o BatchMode=yes "$worker" "timeout 120 docker wait $C >/dev/null 2>&1; docker logs $C 2>&1 | grep -E '\"rank\": 1'" > "$out/$arm.rank1.txt"
  echo "  rank0 lines $(wc -l < "$out/$arm.rank0.txt"), rank1 lines $(wc -l < "$out/$arm.rank1.txt") (pass lines: $(grep -cE 'pass_counts|pass_phases_ms' "$out/$arm.rank1.txt"))"
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
