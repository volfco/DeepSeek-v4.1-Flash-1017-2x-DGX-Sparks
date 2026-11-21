#!/usr/bin/env bash
# Run on one idle rank. Compare PM-sampling counters with a timed scan larger than L2 and the SLC.
set -euo pipefail
bin=$1
dest=$2
metric=${3:-lts__d_sectors_fill_device.sum}
mkdir -p "$dest"
sudo -n "$bin/pm_sampling" -i 1000000 -b 16777216 -s 10000 -m "$metric" > "$dest/counters.csv" 2> "$dest/counters.err" &
counter=$!
for ((i=0;i<100;i++)); do
    if grep -q '^anchor,' "$dest/counters.csv"; then break; fi
    if ! kill -0 "$counter" 2>/dev/null; then wait "$counter"; exit 1; fi
    sleep .1
done
grep -q '^anchor,' "$dest/counters.csv"
"$bin/calibrate" > "$dest/scan.csv"
wait "$counter"
