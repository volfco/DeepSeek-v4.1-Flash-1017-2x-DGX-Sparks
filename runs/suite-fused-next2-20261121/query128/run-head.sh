#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
head=${QUERY_BENCH_HEAD:-colum@100.95.49.21}
bench=runs/suite-fused-next2-20261121/query128/bench.py
reverse=${1:-0}
[[ "$reverse" == 0 || "$reverse" == 1 ]]
remote=$(cat <<'REMOTE'
set -euo pipefail
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
tar -xf - -C "$tmp"
chmod 755 "$tmp"
tag=tensorfold-dsv41:bench-fused-next-84806bb
image=$(docker image inspect --format '{{.Id}}' "$tag")
order=()
[[ "$REVERSE" == 0 ]] || order+=(--reverse-batches)
timeout 35s flock -w 2 ~/tf-bench/gpu.lock docker run --rm --gpus all --network host \
    --mount "type=bind,src=$tmp,dst=/querybench,readonly" \
    --mount type=bind,src=/nfs1/tensorfold-dsv41,dst=/artifacts,readonly \
    --entrypoint /usr/bin/python3 "$image" \
    -B /querybench/runs/suite-fused-next2-20261121/query128/bench.py --image-id "$image" \
    --aot /artifacts/kit-abe07c7d/kit/aot \
    --config /artifacts/45ee5d0f-ablit-0bdc2036/model/config.json "${order[@]}"
REMOTE
)
tar -cf - "$bench" tools/fused_bench/bench.py |
    timeout 45s ssh -o BatchMode=yes -o ConnectTimeout=6 "$head" "REVERSE=$reverse; $remote"
