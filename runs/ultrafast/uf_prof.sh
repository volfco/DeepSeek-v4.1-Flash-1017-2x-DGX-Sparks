#!/usr/bin/env bash
# Rank 0's round profile: tf-dsv41-lanes as rank 0 (batch, --requests) and as rank 1, with the served pair's settings
# except the spill tier (off, so the batch run never writes the served store). Attribution only (the profile syncs
# every round; no HTTP or text path). Runs on the head from the repo root; ends with the production pair back up.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PORT=8000
O=runs/ultrafast/prof; R=runs/ultrafast; C=dsv41-1017-exl3-tf
W="$(grep -m1 '^WORKER=' .env | cut -d= -f2-)"
mkdir -p $O
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $O/status.log; }

log "profile start"
( export SPILL_GIB=0; ./start.sh restart ) > $O/restart.log 2>&1 || { log "restart failed"; exit 1; }
timeout 1800 python3 -I $R/uf_profreq.py $O > $O/req.log 2>&1 || { log "requests failed"; exit 1; }
log "requests: $(tr '\n' ';' < $O/req.log)"
docker inspect $C > $O/rank0.inspect.json
ssh -o BatchMode=yes "$W" "docker inspect $C" > $O/rank1.inspect.json
./stop.sh > $O/stop.log 2>&1; log "pair stopped rc=$?"

# label  requests  parallel  graphs  profile
for spec in "g-r1 r1 1 1 1" "g-r4 r4 4 1 1" "e-r1 r1 1 0 2" "e-r4 r4 4 0 2"; do
  read -r lab file par graphs prof <<<"$spec"
  line1=$(python3 - $O/rank1.inspect.json $graphs $R/uf_relaunch.py <<'PY'
import json, subprocess, sys
c = json.load(open(sys.argv[1]))[0]["Config"]["Cmd"]
i = c.index("--graphs"); c[i + 1] = sys.argv[2]
print(subprocess.check_output(["python3", "-I", sys.argv[3], sys.argv[1], "tf-prof1", "--"] + c + ["--profile", "1"], text=True).strip())
PY
)
  line0=$(python3 - $O/rank0.inspect.json $O/rank1.inspect.json $file $par $graphs $prof $R/uf_relaunch.py "$PWD/$O" <<'PY'
import json, subprocess, sys
i0, i1, f, par, graphs, prof, tool, odir = sys.argv[1:]
env = dict(e.split("=", 1) for e in json.load(open(i0))[0]["Config"]["Env"])
c = json.load(open(i1))[0]["Config"]["Cmd"]
c = [c[0], c[1], env["TF_DS_RANK_CACHE"], "0", c[4], c[5], c[6], c[7]] + c[8:]
c[c.index("--rdma") + 1] = env["TF_RDMA_DEVICES"]
c[c.index("--graphs") + 1] = graphs
c += ["--requests", f"/reqs/{f}.json", "--parallel", par, "--profile", prof]
print(subprocess.check_output(["python3", "-I", tool, i0, "tf-prof0", "--mount", f"{odir}:/reqs:ro", "--"] + c, text=True).strip())
PY
)
  echo "$line1" > $O/$lab.rank1.cmd; echo "$line0" > $O/$lab.rank0.cmd
  log "$lab launch"
  ssh -o BatchMode=yes "$W" "docker rm -f tf-prof1 >/dev/null 2>&1; $line1" > $O/$lab.rank1.id 2>&1 || { log "$lab rank1 launch failed: $(tail -2 $O/$lab.rank1.id)"; continue; }
  docker rm -f tf-prof0 >/dev/null 2>&1
  eval "$line0" > $O/$lab.rank0.id 2>&1 || { log "$lab rank0 launch failed: $(tail -2 $O/$lab.rank0.id)"; ssh "$W" "docker rm -f tf-prof1"; continue; }
  rc=$(timeout 2400 docker wait tf-prof0); log "$lab rank0 exit=${rc:-timeout}"
  docker logs tf-prof0 > $O/$lab.rank0.log 2>&1
  ssh -o BatchMode=yes "$W" "timeout 180 docker wait tf-prof1; docker logs tf-prof1" > $O/$lab.rank1.log 2>&1
  docker rm -f tf-prof0 >/dev/null 2>&1; ssh -o BatchMode=yes "$W" "docker rm -f tf-prof1" >/dev/null 2>&1
  log "$lab rank0: $(grep -h '"profile"\|"all_equal"' $O/$lab.rank0.log | tr '\n' ' ' | cut -c1-900)"
  log "$lab rank1: $(grep -h 'follow_profile' $O/$lab.rank1.log | tail -1 | cut -c1-400)"
done
./start.sh restart > $O/final.restart.log 2>&1; log "production restart rc=$?"
timeout 1800 python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $O/final.validate.json \
  --compare runs/matrix/base.validate.json > $O/final.validate.log 2>&1; log "final validate rc=$?"
log "profile done"
