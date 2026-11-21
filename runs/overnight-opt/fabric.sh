#!/usr/bin/env bash
# Bidirectional bulk and small-message reference. TCP is the control connection; the timed data use RC RoCE.
set -euo pipefail
out=$1
mkdir -p "$out"
peer=10.50.0.11
run_rail() {
    local nic=$1 port=$2 tool=$3
    local opt
    if [[ $tool = ib_write_bw ]]; then opt=(-b --report_gbits -s 1048576 -D 4); else opt=(-s 64 -n 5000); fi
    ssh -n -o BatchMode=yes "$peer" timeout 30 "$tool" -F -d "${nic}f1" -x 3 -p "$port" "${opt[@]}" > "$out/$nic-$tool.server" 2>&1 &
    local server=$!
    sleep 1
    timeout 25 "$tool" -F -d "${nic}f0" -x 3 -p "$port" "${opt[@]}" "$peer" > "$out/$nic-$tool.client" 2>&1
    wait "$server"
}
run_rail roceP2p1s0 28721 ib_write_bw & a=$!
run_rail rocep1s0 28722 ib_write_bw & b=$!
wait "$a"; wait "$b"
run_rail rocep1s0 28723 ib_write_lat
