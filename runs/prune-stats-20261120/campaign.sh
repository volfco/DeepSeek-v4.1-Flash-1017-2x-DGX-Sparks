#!/usr/bin/env bash
set -euo pipefail
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
B=runs/prune-stats-20261120
mkdir -p "$B"
exec > >(tee -a "$B/campaign.log") 2>&1
restore() {
 rc=$?; trap - EXIT
 bash runs/draft-block-20261118/arm.sh prune-stats-restore production > "$B/restore.start.log" 2>&1 || exit 90
 python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$B/restored.json" --compare runs/matrix/base.validate.json > "$B/restore-validate.log" 2>&1 || exit 91
 curl -fsS http://127.0.0.1:8000/health > "$B/restored-health.json"
 echo "RESTORED rc=$rc $(date -u +%FT%TZ)"; exit "$rc"
}
trap restore EXIT
for spec in control1:1 candidate:0 control2:1; do
 label=${spec%:*}; stats=${spec#*:}
 echo "START $label stats=$stats $(date -u +%FT%TZ)"
 bash runs/draft-block-20261118/arm.sh "prune-stats-$label" tensorfold-dsv41:bench-fused-rework-20261120 TF_DS_DRAFT_TRIAGE=conf:0.2 TF_DS_FUSED_ROUTE=1 TF_DS_T13_PRUNE_STATS=$stats > "$B/$label.start.log" 2>&1
 cp runs/draft-block-20261118/prune-stats-$label/armed.txt "$B/$label.ranks.txt"
 for host in local colum@10.50.0.11; do
   if [[ $host == local ]]; then docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf; else ssh -o BatchMode=yes "$host" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf"; fi | grep -E "^(TF_DS_DRAFT_TRIAGE|TF_DS_T13_EXPERT_PRUNE|TF_DS_T13_PRUNE_STATS|TF_DS_FUSED_ROUTE)=" | sort > "$B/$label.$host.env"
 done
 python3 "$B/probe.py" "$B/$label.timing.json" > "$B/$label.timing.log" 2>&1
 tail -6 "$B/$label.timing.log"
done
