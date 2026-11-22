#!/usr/bin/env bash
# Round B2: per-kernel GPU timings on both ranks under production's config (image 10c9866 wrapped in nsys launch).
# Captures: x1 (one sampled tetris request, 512 tokens) and x4 (four at once). Reports copied out of each container,
# then production restored from .env and validated.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
D=runs/sched-20261122/b2; C=dsv41-1017-exl3-tf; W=colum@10.50.0.11
insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes $W "$2"; fi; }
req() { curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"DeepSeek-V4.1-Flash-1017\",\"messages\":[{\"role\":\"user\",\"content\":\"implement the tetris game. output HTML code directly - no explanation.\"}],\"max_tokens\":512,\"seed\":$1,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > /dev/null; }
echo "$(date -u +%T) restart under nsys"
env IMAGE=tensorfold-dsv41:nsys-10c9866 PREPARE=0 ./start.sh restart > $D/restart.log 2>&1 || grep -q 'the smoke test request failed' $D/restart.log || { echo "restart failed"; tail -5 $D/restart.log; }
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
for r in 0 1; do echo "rank$r image $(insp $r "docker inspect -f '{{.Config.Image}}' $C") sessions: $(insp $r "docker exec $C nsys sessions list 2>&1 | tail -n +2 | tr '\n' ' '")"; done
echo "$(date -u +%T) warmup"; for s in 1 2 3; do req $((960+s)); done; (for s in 1 2 3 4; do req $((970+s)) & done; wait)
for case in x1 x4; do
  for r in 0 1; do insp $r "docker exec $C nsys start --session=tf --sample=none --cpuctxsw=none --output=/tmp/b2-$case-r$r --force-overwrite=true" 2>&1 | tail -1; done
  t0=$(date +%s.%N)
  if [[ $case == x1 ]]; then req 981; else (for s in 1 2 3 4; do req $((990+s)) & done; wait); fi
  t1=$(date +%s.%N)
  for r in 0 1; do insp $r "docker exec $C nsys stop --session=tf" 2>&1 | tail -1; done
  echo "$(date -u +%T) $case captured ($(echo "$t1 - $t0" | bc) s)"
done
sleep 20
for case in x1 x4; do
  docker cp $C:/tmp/b2-$case-r0.nsys-rep $D/ 2>&1 | tail -1
  ssh -n -o BatchMode=yes $W "docker cp $C:/tmp/b2-$case-r1.nsys-rep /tmp/" && scp -q $W:/tmp/b2-$case-r1.nsys-rep $D/
done
ls -la $D/*.nsys-rep
echo "$(date -u +%T) restoring production"
PREPARE=0 ./start.sh restart > $D/restore.log 2>&1 || grep -q 'the smoke test request failed' $D/restore.log || echo "restore restart failed"
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
echo "rank0 image $(docker inspect -f '{{.Config.Image}}' $C)"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/restored.validate.json \
  --compare runs/matrix/base.validate.json > $D/validate.log 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
echo "$(date -u +%T) finished"
