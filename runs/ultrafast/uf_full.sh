#!/usr/bin/env bash
# Full run for 0119 (token cache) and 0121 (exact resume), plus the Tetris row and keyed-draft on/off. Runs on the
# head from the repo root. Three passes, arms alternated within each pass, one restart per arm, same image throughout.
#   x-base  production (.env as is)                      bench block, validate, longctx, tetris, sampled
#   x-tok   + TENSORFOLD_TOKEN_CACHE_MB=32                bench block, validate, longctx
#   x-k0    TF_DS_T26_KEYED_DRAFT=0                       validate, tetris, sampled
#   x-rbase SPILL_GIB=0 KV_SHARD=0 TF_DS_PREFILL_SLICE=0  bench block, validate, resume_exact warm, prompt_reuse, longctx
#   x-ron   x-rbase + TF_DS_EXACT_RESUME=1                same as x-rbase
# Once first: x-cold (x-rbase with --prompt-cache-gib 0) records resume_exact's cold rows.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PORT=8000
U=http://127.0.0.1:8000; M=DeepSeek-V4.1-Flash-1017; C=dsv41-1017-exl3-tf
O=runs/ultrafast/full; P=runs/ultrafast/uf_probe.py
mkdir -p $O
PASSES=${PASSES:-3}
RES0="SPILL_GIB=0 KV_SHARD=0 TF_DS_PREFILL_SLICE=0"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $O/status.log; }

restart() {   # name, then NAME=value words, then optionally -- serve args
  local n=$1; shift; local -a envs=() extra=()
  while (( $# )); do [[ $1 == -- ]] && { shift; extra=("$@"); break; }; envs+=("$1"); shift; done
  log "$n restart [${envs[*]}] ${extra[*]}"
  T0=$(date -u +%FT%TZ)
  ( (( ${#envs[@]} )) && export "${envs[@]}"; ./start.sh restart "${extra[@]}" ) > $O/$n.restart.log 2>&1
  local rc=$?
  (( rc == 0 )) || { log "$n RESTART-FAILED rc=$rc"; return 1; }
  log "$n up image=$(docker inspect -f '{{.Config.Image}}' $C 2>/dev/null)"
}
gate() { timeout 3600 python3 -I tools/validate.py $U --model $M --save $O/$1.validate.json --compare runs/matrix/base.validate.json > $O/$1.validate.log 2>&1
  log "$1 validate rc=$? $(tail -1 $O/$1.validate.log | cut -c1-120)"; }
bench() { timeout 3600 python3 -I tools/bench.py --url $U --model $M --suite prefill,decode,multi --streams 1,4 --reps 3 \
    --prefill-sizes 2000,8000,32000 --metrics --label $1 --out $O/$1.bench.json > $O/$1.bench.log 2>&1
  local rc=$?
  log "$1 bench rc=$rc $(python3 - $O/$1.bench.json <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    keys = (("prefill.2000.tok_s", "p2k"), ("prefill.8000.tok_s", "p8k"), ("prefill.32000.tok_s", "p32k"),
            ("decode.tok_s", "decode"), ("multi.4.aggregate_tok_s", "multi4"))
    print(" ".join(f"{l}={s[k]:.1f}" if s.get(k) is not None else f"{l}=n/a" for k, l in keys))
except Exception as e:
    print("no-summary", e)
EOF
)"; }
longctx() { timeout 3600 python3 -I $P longctx --sizes 100000 --seed $2 --label $1 --out $O/$1.longctx.json > $O/$1.longctx.log 2>&1; log "$1 longctx rc=$?"; }
tetris() { timeout 3600 python3 -I $P tetris --tokens 2048 --reps 3 --seed $2 --label $1 --out $O/$1.tetris.json > $O/$1.tetris.log 2>&1
  log "$1 tetris rc=$? $(tr '\n' ';' < $O/$1.tetris.log | cut -c1-400)"; }
sampled() { timeout 3600 python3 -I $P sampled --reps 2 --tokens 256 --label $1 --out $O/$1.sampled.json > $O/$1.sampled.log 2>&1
  log "$1 sampled rc=$? $(grep -o 'sampled streams=[0-9] rep=[0-9] agg_tok_s=[0-9.]*' $O/$1.sampled.log | tr '\n' ';')"; }
resume() { timeout 3600 python3 -I tools/resume_exact.py warm --state $O/resume.json --label $1 > $O/$1.resume.log 2>&1; log "$1 resume rc=$?"
  python3 -I tools/prompt_reuse.py 12000 > $O/$1.reuse.log 2>&1; log "$1 reuse rc=$? $(tail -2 $O/$1.reuse.log | tr '\n' ' ' | cut -c1-200)"; }
logs() { docker logs -t --since "$T0" $C > $O/$1.engine.log 2>&1; grep 'done ' $O/$1.engine.log > $O/$1.done.log; }

log "full run start passes=$PASSES"
if [[ ! -s $O/resume.json ]]; then
  if restart x-cold $RES0 -- --prompt-cache-gib 0; then
    python3 -I tools/resume_exact.py cold --state $O/resume.json --label cold > $O/x-cold.resume.log 2>&1; log "x-cold resume cold rc=$?"
    logs x-cold
  fi
fi
for p in $(seq 1 $PASSES); do
  n=x-base-$p;  if restart $n; then gate $n; bench $n; longctx $n $((100 + p)); tetris $n $p; sampled $n; logs $n; fi
  n=x-tok-$p;   if restart $n TENSORFOLD_TOKEN_CACHE_MB=32; then gate $n; bench $n; longctx $n $((200 + p)); logs $n; fi
  n=x-k0-$p;    if restart $n TF_DS_T26_KEYED_DRAFT=0; then gate $n; tetris $n $p; sampled $n; logs $n; fi
  n=x-rbase-$p; if restart $n $RES0; then gate $n; bench $n; resume $n; longctx $n $((300 + p)); logs $n; fi
  n=x-ron-$p;   if restart $n $RES0 TF_DS_EXACT_RESUME=1; then gate $n; bench $n; resume $n; longctx $n $((400 + p)); logs $n; fi
  log "pass $p done"
done
timeout 3600 python3 -I tools/resume_exact.py report --state $O/resume.json --off x-rbase-1 --on x-ron-1 --md $O/resume-report.md > $O/resume-report.log 2>&1
log "resume report rc=$?"
restart x-final && gate x-final
log "full run done"
