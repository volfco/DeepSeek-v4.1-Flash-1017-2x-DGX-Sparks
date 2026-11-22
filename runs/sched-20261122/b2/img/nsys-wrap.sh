#!/bin/sh
# Round B2: run the engine under an idle nsys session ("tf"); a capture window is opened later with
#   nsys start --session=tf --output=/spill/nsys/<name> ... ; nsys stop --session=tf
exec nsys launch --session-new=tf --trace=cuda,nvtx,osrt --cuda-graph-trace=node --wait=all "$@"
