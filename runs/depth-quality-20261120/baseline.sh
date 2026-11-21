#!/usr/bin/env bash
# Repo-baseline arm: production .env minus every tuning key (ABLIT and setup kept), same image, private spill; probe;
# then restore production + validate. .env is swapped only for the start and always put back (trap).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/depth-quality-20261120
IMG=tensorfold-dsv41:exl3-tp2-ea522d353626
DROP='^(PARALLEL|KV_SHARD|TENSORFOLD_TOKEN_CACHE_MB|TF_DS_T13_EXPERT_PRUNE|TF_DS_T26_KEYED_DRAFT|TF_DS_T27_DEVICE_DRAW|TF_DS_PREFILL_SLICE|TF_DS_DRAFT_BLOCK|TF_DS_ENGRAM_LEAD|TF_DS_RDMA_CPUS)='
export ARM_ALLOW_PAST_TRAFFIC=1
idle() { [[ "$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null)" == "0 0" ]]; }
wait_idle() { local q=0; echo "$(date -u +%T) waiting for 120 s idle"; while (( q < 12 )); do if idle; then q=$((q+1)); else q=0; fi; sleep 10; done; echo "$(date -u +%T) idle"; }
bak=$HOME/dsv41-env-backups/.env.bak-20261120-pre-baseline
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { echo ".env differs from its backup: stopping"; exit 2; }
restore_env() { install -m 600 "$bak" .env; cmp -s .env "$bak" && echo "$(date -u +%T) .env restored"; }
trap restore_env EXIT
grep -vE "$DROP" "$bak" > .env.baseline; chmod 600 .env.baseline
echo "baseline .env: $(wc -l < .env.baseline) of $(wc -l < "$bak") lines; dropped: $(grep -E "$DROP" "$bak" | cut -d= -f1 | tr '\n' ' ')"
until wait_idle && { install -m 600 .env.baseline .env; runs/draft-block-20261118/arm.sh dq-baseline "$IMG"; rc=$?; restore_env; (( rc == 0 )); }; do sleep 30; done
rm -f .env.baseline
echo "lead banners (want 0): $(docker logs dsv41-1017-exl3-tf 2>&1 | grep -c 'engram-lead: on')"
echo "rank0 TF env:"; docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf | grep -E '^(TF_DS_|TENSORFOLD_)' | grep -vE 'REVISION|RANK_CACHE|KIT|TOKEN_MAP|ENGINE_ID|CUDA_KERNELS' | sort | tr '\n' ' '; echo
python3 -I "$D/probe.py" baseline || echo "probe baseline failed"
until wait_idle && runs/draft-block-20261118/arm.sh dq-restore2 production; do sleep 30; done
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$D/restored2.json" \
  --compare runs/matrix/base.validate.json > "$D/restore2-validate.log" 2>&1; echo "restore validate rc=$?"
tail -2 "$D/restore2-validate.log"
echo "== $(date -u +%T) baseline done"
