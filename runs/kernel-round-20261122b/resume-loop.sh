#!/usr/bin/env bash
# Resume a pi agent session until its worktree has a commit past 27baa93 (or the report declares the premise failed).
n=$1; D=/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261122b
cd ~/Workspace/tfref-k$n || exit 1
for i in $(seq 8); do
  [[ $(git rev-parse --short HEAD) != 27baa93 ]] && { echo "$(date -u +%T) committed $(git log --oneline -1)"; exit 0; }
  grep -qiE 'premise (fails|does not hold)|not worth building' $D/report-$n.md 2>/dev/null && [[ -z $(git status --short) ]] && { echo "$(date -u +%T) premise failed; report written"; exit 0; }
  echo "$(date -u +%T) resume $i"
  pi -p --provider xiaomi-token-plan-sgp --model mimo-v2.6-pro --session-dir $D/sessions/$n -c "Your process was interrupted mid-task (not by you). Continue exactly where you left off per $D/BRIEF-common.md and $D/BRIEF-$n.md: finish, run the full local gate, make exactly one commit with the brief's subject, and write $D/report-$n.md with the real commit hash. Work in small steps and commit as soon as the gate passes. Do not stop until committed (or, if the premise fails, revert the worktree and write the evidence as the report)." < /dev/null
  echo "$(date -u +%T) pi exited rc=$?"
done
echo "gave up after 8 resumes"
