#!/usr/bin/env bash
OUT=runs/nccl-screen-20261123 EXACT=0 SAMPLED=0 PROBE=0 WORKLOAD_SCRIPT=runs/nccl-screen-20261123/short.py BANNERS='NCCL' \
  ARMS_FILE=runs/nccl-screen-20261123/arms.txt bash runs/overnight-20261122/arms.sh
