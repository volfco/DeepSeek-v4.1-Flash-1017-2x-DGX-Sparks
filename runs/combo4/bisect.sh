#!/usr/bin/env bash
# usage: bisect.sh <label> <patch globs from runs/combo4/pulled...>; builds, restarts, 3 chats, restores the combo3 set
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
export PATH=$HOME/.local/bin:$PATH
L=$1; shift
exec 9>/tmp/combo4-bisect.lock; flock -n 9 || { echo "already running" >&2; exit 1; }
R=runs/combo4/bisect.log
log(){ echo "$(date -u +%FT%TZ) $L: $*" >> $R; }
for g in "$@"; do cp runs/combo4/pulled/$g zig/patches/; done
log "set: $(cd zig/patches && ls 018* 019* 2>/dev/null | cut -c1-5 | tr '\n' ' ')"
./scripts/prepare.sh > runs/combo4/bisect-$L.build.log 2>&1 || { log "BUILD FAILED"; }
timeout 600 ./start.sh restart > runs/combo4/bisect-$L.restart.log 2>&1; log "restart rc=$?"
ok=0
for i in 1 2 3; do
  out=$(curl -s -m 150 localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}')
  echo "$out" | grep -q '"content": "Hello' && ok=$((ok+1)); echo "chat $i: ${out:0:200}" >> runs/combo4/bisect-$L.chats
done
log "RESULT chats_ok=$ok/3 $(docker logs --tail 200 dsv41-1017-exl3-tf 2>&1 | grep -cE 'ILLEGAL|pair broke') crash-lines"
docker logs dsv41-1017-exl3-tf > runs/combo4/bisect-$L.rank0.log 2>&1; ssh 10.50.0.11 docker logs dsv41-1017-exl3-tf > runs/combo4/bisect-$L.rank1.log 2>&1
rm -f zig/patches/018* zig/patches/019*
timeout 600 ./start.sh restart > runs/combo4/bisect-restore.log 2>&1
curl -s -m 60 localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null
out=$(curl -s -m 60 localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}')
echo "$out" | grep -q '"content"' && log "restored combo3 image ok" || log "RESTORE CHAT FAILED"
log "DONE"
