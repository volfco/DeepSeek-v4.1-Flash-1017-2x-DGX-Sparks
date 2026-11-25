#!/usr/bin/env python3
# h0272's extraction step (p0194/splice.py's trick): pull TF_DS_RING_TRACE's device-side record logic out of the
# tree into trace_kernel_current.inc, so the CPU harness in trace_emu.cpp drives the shipped code and not a copy.
# The record logic lives in zig/kernels/cuda/dsv41/ring_trace_rec.cuh (included by rdma_gather.cu and
# dsv41_shard.cu); the kernels' call sites are one-liners on its functions, checked by build.sh's greps.
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("WORKTREE", "/home/colum/Workspace/tfref-k0272"), "zig/kernels/cuda/dsv41/ring_trace_rec.cuh")
src = open(root).read()

# the extracted unit: the whole header, minus its own include guard lines (the harness includes it once)
inc = "\n".join(l for l in src.splitlines(keepends=True) if l.strip() != "#pragma once")

# the functions the kernels call must all be present and exactly once each (drift stops the harness here)
for name, want in [("tf_tr_on", "TF_TR_HD bool tf_tr_on("), ("tf_tr_sampled", "TF_TR_HD bool tf_tr_sampled("),
                   ("tf_tr_slot", "TF_TR_HD int tf_tr_slot("), ("tf_tr_claim", "TF_TR_HD void tf_tr_claim("),
                   ("tf_tr_publish", "TF_TR_HD void tf_tr_publish("), ("tf_tr_entry", "TF_TR_HD void tf_tr_entry("),
                   ("tf_tr_flag", "TF_TR_HD void tf_tr_flag("), ("tf_tr_done", "TF_TR_HD void tf_tr_done(")]:
    count = src.count(want)
    assert count == 1, f"{name}: expected exactly one definition, found {count}"
# the layout indices must match ring_trace.GpuRec's field order (seq, round, op, layer, rows, bytes, sf, prev,
# stage, doorbell, collect, flag, copy): the Zig module's test pins the byte offsets
for line, want in [("TF_TR_SEQ = 0", 0), ("TF_TR_ROUND = 1", 1), ("TF_TR_OP = 2", 2), ("TF_TR_LAYER = 3", 3),
                   ("TF_TR_ROWS = 4", 4), ("TF_TR_BYTES = 5", 5), ("TF_TR_SF = 6", 6), ("TF_TR_PREV = 7", 7),
                   ("TF_TR_STAGE = 8", 8), ("TF_TR_DOORBELL = 9", 9), ("TF_TR_COLLECT = 10", 10),
                   ("TF_TR_FLAG = 11", 11), ("TF_TR_COPY = 12", 12), ("TF_TR_REC = 16", 16)]:
    assert line in src, f"missing {line}"

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_kernel_current.inc"), "w").write(inc)
print(f"spliced {len(inc)} bytes from {root}")
