#!/usr/bin/env bash
# Reference duration is dominated by the 3 x 4 s storage reads. Refuse measurements contaminated by new HTTP completions.
set -euo pipefail
dest=runs/overnight-opt/hardware01
mkdir -p "$dest"
requests() { curl -fsS http://127.0.0.1:8000/metrics | awk '/^tensorfold:request_latency_seconds_count[{ ]/{s+=$NF} END{print s+0}'; }
requests > "$dest/before.txt"
bash runs/overnight-opt/fabric.sh "$dest/fabric" || echo "fabric reference failed" >&2
dir=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/engram"}}{{.Source}}{{end}}{{end}}' dsv41-1017-exl3-tf)
python3 runs/overnight-opt/storage-read.py "$dir" > "$dest/head-drive.jsonl" 2> "$dest/head-drive.err" & h=$!
ssh -n -o BatchMode=yes 10.50.0.11 'python3 /tmp/gb10-storage-read.py "$(docker inspect dsv41-1017-exl3-tf | python3 -c '"'"'import json,sys; print(next(m["Source"] for m in json.load(sys.stdin)[0]["Mounts"] if m["Destination"]=="/engram"))'"'"')"' > "$dest/worker-drive.jsonl" 2> "$dest/worker-drive.err" & w=$!
wait "$h"; wait "$w"
requests > "$dest/after.txt"
cmp "$dest/before.txt" "$dest/after.txt"
