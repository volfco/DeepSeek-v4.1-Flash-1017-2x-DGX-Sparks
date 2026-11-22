#!/usr/bin/env bash
# Standalone RDMA ring bench at production's hidden-gather payload (SCHEDULING.md step 1). Runs on the head with the
# engine STOPPED on both boards; restores production and validates afterwards.
#   width 5120 floats (a hidden row: rows x 5120 x 4 bytes a rank; 7 rows = 140 KiB), rows 1,7,14,21,28 (1-stream
#   modal ~7, four streams ~20-28), production NICs/GID 3/PDL on, TF_RDMA_IDLE_S=2 TF_RDMA_NAP_US=500, cpuset 15-19,
#   proxy thread alone pinned to CPU 15. A throwaway rows=1 first. 3 repeats; plus one width-2560 pass (the bench's
#   default, for comparison with older numbers). Engine-side plain gather -> compare the us_gather column.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
D=runs/sched-20261122/rdmabench; W=colum@10.50.0.11; WD=DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/$D
IMG=tensorfold-dsv41:bench-fused-next-27baa93
ssh -o BatchMode=yes $W "mkdir -p $WD"; scp -q $D/rdma_gather.cu $D/rank.sh $W:$WD/
NV='/usr/local/cuda/bin/nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121 -o /b/rdma.fatbin /b/rdma_gather.cu'
docker run --rm -v $PWD/$D:/b --entrypoint sh $IMG -c "$NV" || { echo "nvcc failed (head)"; exit 1; }
ssh -o BatchMode=yes $W "docker run --rm -v \$HOME/$WD:/b --entrypoint sh $IMG -c '$NV'" || { echo "nvcc failed (worker)"; exit 1; }
echo "$(date -u +%T) fatbins built; stopping the engine"
./stop.sh > $D/stop.log 2>&1
ARGS="--rm --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband --cpuset-cpus 15-19 -e TF_RDMA_IDLE_S=2 -e TF_RDMA_NAP_US=500 --entrypoint sh"
pass() {  # tag rows width reps
  docker run $ARGS -v $PWD/$D:/b $IMG /b/rank.sh 0 rocep1s0f0,roceP2p1s0f0 $2 $3 $4 &   # rank 0 listens first
  sleep 4
  timeout 600 ssh -o BatchMode=yes $W "docker run $ARGS -v \$HOME/$WD:/b $IMG /b/rank.sh 1 rocep1s0f1,roceP2p1s0f1 $2 $3 $4"
  wait
  scp -q $W:$WD/out.1.json $D/$1.r1.json; scp -q $W:$WD/err.1.log $D/$1.r1.err; scp -q $W:$WD/pin.1.txt $D/$1.r1.pin; scp -q "$W:$WD/proxy1.1.txt" $D/$1.r1.proxy 2>/dev/null
  cp $D/out.0.json $D/$1.r0.json; cp $D/err.0.log $D/$1.r0.err; cp $D/pin.0.txt $D/$1.r0.pin; cp $D/proxy1.0.txt $D/$1.r0.proxy 2>/dev/null
  echo "$(date -u +%T) $1: $(cat $D/$1.r0.json | wc -l) / $(cat $D/$1.r1.json | wc -l) lines; $(tail -n1 $D/$1.r0.err) $(tail -n1 $D/$1.r1.err)"
}
for k in 1 2 3; do pass w5120-$k 1,1,7,14,21,28 5120 1000; done
pass w2560-1 1,1,6,16 2560 1000
echo "$(date -u +%T) restoring production"
PREPARE=0 ./start.sh restart > $D/restore.log 2>&1 || grep -q 'the smoke test request failed' $D/restore.log || echo "restore restart failed"
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/restored.validate.json \
  --compare runs/matrix/base.validate.json > $D/validate.log 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
echo "$(date -u +%T) finished"
