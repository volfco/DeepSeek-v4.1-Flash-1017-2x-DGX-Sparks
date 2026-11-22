#!/usr/bin/env bash
# suite-fused-next2-20261121 preflight (on the WORKSTATION, from the recipe root): build and ship the
# bench-fused-next-84806bb image to both Sparks (no container restart), sync this suite folder to the head, then
# start launch.sh detached there. NEVER run `just sync-head` for this: it would push the recipe's zig/patches over
# the head's and change the production build hash.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=colum@100.95.49.21
O=runs/suite-fused-next2-20261121

sha=$(git -C ~/Workspace/tfref-fused-next rev-parse --short 84806bb)
[ "$sha" = "84806bb" ] || { echo "unexpected short sha $sha"; exit 1; }

avail_kb=$(ssh -o BatchMode=yes "$H" "awk '/MemAvailable/{print \$2}' /proc/meminfo")
if [ "${avail_kb:-0}" -lt 4718592 ]; then   # 4.5 GiB in kB
  echo "head MemAvailable ${avail_kb} kB < 4.5 GiB: the image build (zig build -j4 + nvcc) needs headroom; retry when quieter"
  exit 1
fi

echo "== build & ship bench-fused-next-84806bb"
bash runs/fused-next-20261121/bench/build.sh 84806bb

echo "== sync suite folder to the head (folder only; never just sync-head)"
rsync -a --exclude='blocks/' "$O/" "$H:$O/"

echo "== start launch.sh on the head (waits for 30 idle minutes, snapshots env, captures 0215 timings, starts suite)"
ssh -o BatchMode=yes "$H" "cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks && { nohup bash $O/launch.sh >> $O/launch.log 2>&1 < /dev/null & } && echo launched"
