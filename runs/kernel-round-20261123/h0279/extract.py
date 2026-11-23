#!/usr/bin/env python3
"""h0279: extract tf_ds_idx_pick_wide_kernel's exact source from the worktree's dsv41_ops.cu into
pick_wide_kernel.inc for the CPU harness: the kernel and the device helpers it calls (tf_ds_score_key,
tf_ds_pool_cut, tf_ds_tile_prefix, tf_ds_idx_select), verbatim. Extraction by each function's comment marker
through its closing brace at column 0 (the p0194 splice trick, kept simple).
"""
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()

MARKERS = [
    "// topk_indices' key of score x at column j",  # tf_ds_score_key
    "// An inclusive cutoff on the scores",  # tf_ds_pool_cut
    "// The prefix count of `flag`",  # tf_ds_tile_prefix
    "// The pool's selection and its two outputs",  # tf_ds_idx_select
    "// TF_DS_IDX_PICK_WIDE:",  # tf_ds_idx_pick_wide_kernel
]

blocks = []
for m in MARKERS:
    start = src.index(m)  # raises if a marker drifted: fix the marker, do not patch around it
    end = src.index("\n}\n", start) + len("\n}\n")
    blocks.append(src[start:end])

out = "// pick_wide_kernel.inc: extracted from zig/kernels/cuda/dsv41_ops.cu by extract.py - do not edit\n"
out += "\n".join(blocks)
open(out_path, "w").write(out)
print("extracted %d blocks into %s (%d bytes)" % (len(blocks), out_path, len(out)))
