#!/usr/bin/env bash
# One pi agent for patch $1: first run, then resume (-c) up to 8 times until a commit past 27baa93 lands
# (or the report declares the premise failed). Agents get OOM-killed mid-task; resuming continues the session.
n=$1; D=/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261122d
cd ~/Workspace/tfref-k$n || exit 1
P="You are implementing kernel patch $n. Read $D/BRIEF-common.md first, then $D/BRIEF-$n.md, and do exactly what they say in the worktree $PWD: implement, run the full local gate (zig build test, the cuda syntax check, your CPU-emulation harness in $D/h$n/), make the commit(s) the brief specifies, with its subject(s), and write $D/report-$n.md with the real commit hash. Work in small steps; keep harness thread counts small (emulate blocks sequentially, not one OS thread per CUDA thread) so you do not run out of memory. If the premise fails, revert the worktree and write the evidence as the report instead."
echo "$(date -u +%T) start"; pi -p --provider xiaomi-token-plan-sgp --model mimo-v2.6-pro --session-dir $D/sessions/$n "$P" < /dev/null; echo "$(date -u +%T) pi exited rc=$?"
for i in $(seq 8); do
  want=1; [[ $n == 0273 ]] && want=2   # 0273's brief is two commits (0273 then 0274)
  [[ $(git rev-list --count 27baa93..HEAD) -ge $want ]] && { echo "$(date -u +%T) committed $(git log --oneline -1)"; exit 0; }
  grep -qiE 'premise (fails|failed|does not hold)|not worth building' $D/report-$n.md 2>/dev/null && [[ -z $(git status --short) ]] && { echo "$(date -u +%T) premise failed; report written"; exit 0; }
  echo "$(date -u +%T) resume $i"
  pi -p --provider xiaomi-token-plan-sgp --model mimo-v2.6-pro --session-dir $D/sessions/$n -c "Your process was interrupted or stopped before committing. Continue exactly where you left off per $D/BRIEF-common.md and $D/BRIEF-$n.md: finish, run the full local gate, make the commit(s) the brief specifies, with its subject(s), and write $D/report-$n.md with the real commit hash. Do not stop until committed (or, if the premise fails, revert the worktree and write the evidence as the report)." < /dev/null
  echo "$(date -u +%T) pi exited rc=$?"
done
echo "gave up after 8 resumes"
