#!/usr/bin/env bash
# SCHEDULING.md row 3 gate, read-only: proxy thread scheduling on both ranks across one 4-stream sampled burst (and a
# 1-stream request), against production as it serves. No restart. Decides whether a CPU-reservation arm is warranted.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
D=runs/sched-20261122/proxysched; W=colum@10.50.0.11
scp -q $D/snap.sh $W:/tmp/proxysnap.sh
burst() {  # width seed0
  for i in $(seq $1); do
    curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"DeepSeek-V4.1-Flash-1017\",\"messages\":[{\"role\":\"user\",\"content\":\"implement the tetris game. output HTML code directly - no explanation.\"}],\"max_tokens\":768,\"seed\":$(($2+i)),\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > /dev/null &
  done; wait
}
for case in "x1 1 900" "x4 4 910"; do
  set -- $case
  sh $D/snap.sh > $D/$1.r0.before; ssh -o BatchMode=yes $W sh /tmp/proxysnap.sh > $D/$1.r1.before
  t0=$(date +%s.%N); burst $2 $3; t1=$(date +%s.%N)
  sh $D/snap.sh > $D/$1.r0.after; ssh -o BatchMode=yes $W sh /tmp/proxysnap.sh > $D/$1.r1.after
  echo "$1 wall $(echo "$t1 - $t0" | bc) s"
done
python3 - "$D" <<'PY'
import re, sys, os
D = sys.argv[1]
def proxy(f):
    for l in open(f):
        if l.startswith("proxy"):
            kv = dict(re.findall(r"(\w+)=(\S+)", l)); m = re.search(r"Cpus_allowed_list: (\S+)", l)
            vol = int(re.search(r"voluntary_ctxt_switches: (\d+)", l).group(1)); inv = int(re.search(r"nonvoluntary_ctxt_switches: (\d+)", l).group(1))
            run, wait, slices = (int(x) for x in re.search(r"schedstat=(\d+) (\d+) (\d+)", l).groups())
            return dict(cpus=m.group(1) if m else "?", vol=vol, inv=inv, mig=int(kv.get("migrations") or 0), run=run, wait=wait, slices=slices)
for case in ("x1", "x4"):
    for r in ("r0", "r1"):
        a, b = proxy(f"{D}/{case}.{r}.before"), proxy(f"{D}/{case}.{r}.after")
        if not a or not b: print(case, r, "no proxy thread"); continue
        def has15(al):
            for part in al.split(","):
                lo, _, hi = part.partition("-")
                if lo.isdigit() and int(lo) <= 15 <= int(hi or lo): return True
            return False
        others = [l.split()[2] for l in open(f"{D}/{case}.{r}.after") if l.startswith("other") and has15(re.search(r"allowed=(\S+)", l).group(1))]
        print(f"{case} {r}: cpus={b['cpus']} vol+{b['vol']-a['vol']} invol+{b['inv']-a['inv']} migr+{b['mig']-a['mig']} "
              f"run+{(b['run']-a['run'])/1e6:.1f}ms runq_wait+{(b['wait']-a['wait'])/1e6:.2f}ms slices+{b['slices']-a['slices']}; "
              f"other threads allowed on 15: {len(others)}")
PY
