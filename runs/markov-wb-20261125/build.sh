#!/usr/bin/env bash
# Build tensorfold-dsv41:bench-probe-dump on both ranks: the M3 hidden-state dump (probe-dump = 2bb052f +
# hidden_probe.zig) cross-built on the workstation with production's own fatbins and layered over production's image.
# Nothing is started, stopped or restarted. Based on runs/k280/build/build.sh, with the jump host the tailnet needs.
#   bash runs/m3-20261125/build.sh
set -euo pipefail
H=colum@10.50.0.10                     # head (LAN; 100.95.49.21 has been offline since 2026-11-24)
W=colum@10.50.0.11                     # worker, reached FROM the head (same LAN)
SH() { ssh -J ai-node -o BatchMode=yes -o ConnectTimeout=20 "$@"; }
SC() { scp -J ai-node -o BatchMode=yes -o ConnectTimeout=20 "$@"; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=~/Workspace/tfref-kstack4
REV=markov-trace
BASE=tensorfold-dsv41:bench-fused-next-2bb052f
TAG=tensorfold-dsv41:bench-markov-trace
ZIG=/home/colum/.local/zig-0.17.0/zig
WORK=$(mktemp -d -p "${TMPDIR:-/tmp}" m3-build.XXXXXX)
STAGE=~/.cache/mwb-stage

echo "== 1/5 the base image's executable and its fatbins"
SH $H "set -e; mkdir -p $STAGE/base; cid=\$(docker create $BASE); docker cp \$cid:/opt/tensorfold/native/bin/tensorfold-native $STAGE/base/; docker rm \$cid >/dev/null"
mkdir -p "$WORK/base"
SC -q "$H:$STAGE/base/tensorfold-native" "$WORK/base/"
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/base/tensorfold-native" "$WORK/fatbins"
(cd "$WORK/fatbins" && sha256sum -c "$REPO/runs/k280/build/fatbins.sha256")

echo "== 2/5 cross-build $REV"
mkdir "$WORK/export" && git -C $SRC archive $REV | tar -xf - -C "$WORK/export"
python3 -I "$REPO/runs/overnight-opt/cross-build.py" "$WORK/export" "$WORK/fatbins" "$WORK/out" --zig $ZIG

echo "== 3/5 the new executables embed the base's fatbins and the new switches"
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/out/native/bin/tensorfold-native" "$WORK/fatbins-new"
for f in "$WORK"/fatbins/*.fatbin; do cmp "$f" "$WORK/fatbins-new/$(basename "$f")"; done
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/out/bin/tf-dsv41-lanes" "$WORK/fatbins-lanes"
for f in "$WORK"/fatbins-lanes/*.fatbin; do cmp "$f" "$WORK/fatbins/$(basename "$f")"; done
for e in native/bin/tensorfold-native bin/tf-dsv41-lanes; do
  # (a subshell without pipefail: `strings | grep -q` makes strings die of SIGPIPE once grep has found it)
  for v in TF_DS_MARKOV_TRACE; do
    ( set +o pipefail; strings -a "$WORK/out/$e" | grep -q "$v" ) || { echo "$v missing in $e"; exit 1; }
  done
done

echo "== 4/5 the image on each rank (executables layered over $BASE)"
mkdir -p $STAGE/build
cp "$WORK/out/native/bin/tensorfold-native" "$WORK/out/bin/tf-dsv41-lanes" "$REPO/runs/k280/build/Dockerfile" $STAGE/build/
(cd $STAGE/build && sha256sum tensorfold-native tf-dsv41-lanes | tee SHA256SUMS)
SH $H "mkdir -p $STAGE/ctx"
SC -q $STAGE/build/{tensorfold-native,tf-dsv41-lanes,Dockerfile,SHA256SUMS} "$H:$STAGE/ctx/"
SH $H "set -e; cd $STAGE/ctx; sha256sum -c SHA256SUMS; docker build --network none -q -t $TAG .; tar -cf - Dockerfile tensorfold-native tf-dsv41-lanes | ssh -o BatchMode=yes $W 'docker build --network none -q -t $TAG -'"

echo "== 5/5 the executables inside the new image on each rank"
chk="set -e; d=\$(mktemp -d); cid=\$(docker create $TAG); docker cp \$cid:/opt/tensorfold/native/bin/tensorfold-native \$d/; docker cp \$cid:/opt/tensorfold/bin/tf-dsv41-lanes \$d/; docker rm \$cid >/dev/null; (cd \$d && sha256sum tensorfold-native tf-dsv41-lanes); rm -rf \$d"
SH $H "$chk" | diff - $STAGE/build/SHA256SUMS
SH $H "ssh -o BatchMode=yes $W \"$chk\"" | diff - $STAGE/build/SHA256SUMS
echo "built $TAG on both ranks"
rm -rf "$WORK"
