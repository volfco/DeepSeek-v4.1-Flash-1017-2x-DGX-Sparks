#!/usr/bin/env bash
# How tensorfold-dsv41:bench-fused-next-1421187 was built (2026-11-23), kept as a record and for a rebuild.
# Run from the workstation; it touches the pair only to read the base image's executable, copy the build context and
# `docker build` the new tag on each rank. Nothing is started, stopped or restarted.
#   bash runs/k280/build/build.sh            # everything
# NOT RUN END TO END: on 2026-11-23 its steps were run by hand, one at a time (build.log). Rerunning it rebuilds the
# executables and RETAGS the image on both ranks; if the new sha256s differ from SHA256SUMS the suites would then run
# different bytes from the ones recorded, so compare before trusting the tag.
# Tip: kernel-stack-0280 = 1421187 (2bb052f + 0280..0283) in ~/Workspace/tfref-kstack5. No .cu file changed, so the
# CUDA fatbins are the base image's own, recovered from its tensorfold-native.
set -euo pipefail
H=colum@100.95.49.21          # head
W=colum@10.50.0.11            # worker, reached from the head
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=~/Workspace/tfref-kstack5
REV=1421187
BASE=tensorfold-dsv41:bench-fused-next-2bb052f
TAG=tensorfold-dsv41:bench-fused-next-$REV
ZIG=/home/colum/.local/zig-0.17.0/zig
WORK=$(mktemp -d -p "${TMPDIR:-/tmp}" k280-build.XXXXXX)
STAGE=~/.cache/k280-stage     # the same path on the workstation and on the head, outside every repo checkout

# 1. the base image's executable, read out of a throwaway (never started) container on the head
ssh -o BatchMode=yes $H "set -e; mkdir -p $STAGE/base; cid=\$(docker create $BASE); docker cp \$cid:/opt/tensorfold/native/bin/tensorfold-native $STAGE/base/; docker rm \$cid >/dev/null"
mkdir -p "$WORK/base" && scp -q -o BatchMode=yes $H:.cache/k280-stage/base/tensorfold-native "$WORK/base/"
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/base/tensorfold-native" "$WORK/fatbins"          # 20 fatbins
(cd "$WORK/fatbins" && sha256sum -c "$REPO/runs/k280/build/fatbins.sha256")

# 2. cross-build from an export (cross-build.py rewrites zig/build/cuda.zig: never point it at the worktree)
mkdir "$WORK/export" && git -C $SRC archive $REV | tar -xf - -C "$WORK/export"
python3 -I "$REPO/runs/overnight-opt/cross-build.py" "$WORK/export" "$WORK/fatbins" "$WORK/out" --zig $ZIG

# 3. checks before anything leaves the workstation: the embedded fatbins are the base's, the new switches are in
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/out/native/bin/tensorfold-native" "$WORK/fatbins-new"
for f in "$WORK"/fatbins/*.fatbin; do cmp "$f" "$WORK/fatbins-new/$(basename "$f")"; done
python3 -I "$REPO/runs/overnight-opt/fatbins.py" "$WORK/out/bin/tf-dsv41-lanes" "$WORK/fatbins-lanes"   # rank 1's embeds 6 of them
for f in "$WORK"/fatbins-lanes/*.fatbin; do cmp "$f" "$WORK/fatbins/$(basename "$f")"; done
for e in native/bin/tensorfold-native bin/tf-dsv41-lanes; do
  for v in TF_DS_PASS_PROFILE TF_DS_MARKOV_CACHE TF_DS_DRAFT_PREFETCH TF_DS_L2_DELAY_NS TF_DS_L2_RATE_GBPS TF_DS_L2_BUDGET_MIB; do
    strings -a "$WORK/out/$e" | grep -q "$v" || { echo "$v missing in $e"; exit 1; }
  done
done
mkdir -p $STAGE/build && cp "$WORK/out/native/bin/tensorfold-native" "$WORK/out/bin/tf-dsv41-lanes" "$REPO/runs/k280/build/Dockerfile" $STAGE/build/
(cd $STAGE/build && sha256sum tensorfold-native tf-dsv41-lanes | tee SHA256SUMS)
# (a rebuild is not promised to be byte-identical: compare with runs/k280/build/SHA256SUMS and say so if it differs)

# 4. the image on each rank: the two executables layered over production's image (labels inherited)
ssh -o BatchMode=yes $H "mkdir -p $STAGE/ctx"
scp -q -o BatchMode=yes $STAGE/build/{tensorfold-native,tf-dsv41-lanes,Dockerfile,SHA256SUMS} $H:.cache/k280-stage/ctx/
ssh -o BatchMode=yes $H "set -e; cd $STAGE/ctx; sha256sum -c SHA256SUMS; docker build --network none -q -t $TAG .; tar -cf - Dockerfile tensorfold-native tf-dsv41-lanes | ssh -o BatchMode=yes $W 'docker build --network none -q -t $TAG -'"

# 5. the executables inside the new image on each rank, read out of a throwaway container (no container is run)
chk="set -e; d=\$(mktemp -d); cid=\$(docker create $TAG); docker cp \$cid:/opt/tensorfold/native/bin/tensorfold-native \$d/; docker cp \$cid:/opt/tensorfold/bin/tf-dsv41-lanes \$d/; docker rm \$cid >/dev/null; (cd \$d && sha256sum tensorfold-native tf-dsv41-lanes); rm -rf \$d"
ssh -o BatchMode=yes $H "bash -c '$chk'" | diff - $STAGE/build/SHA256SUMS
ssh -o BatchMode=yes $H "ssh -o BatchMode=yes $W '$chk'" | diff - $STAGE/build/SHA256SUMS
echo "built $TAG on both ranks"
rm -rf "$WORK"
