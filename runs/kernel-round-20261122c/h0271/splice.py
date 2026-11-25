#!/usr/bin/env python3
"""splice.py [fresh_off_by_one|pos_plus_one|bad_slot]: extract tf_ds_resume_assemble_kernel's exact source from the
worktree's dsv41_ops.cu into fused_kernel_current.inc for the CPU harness. With a mutation name it writes a
seeded-bug variant instead (build.sh runs each and the matrix must catch it); a bare splice.py restores the kernel's
own text. The kernel has no __shared__ arrays and no barriers, so the extraction is the text unchanged.
"""
import os
import sys

SRC = os.environ.get("WT", "/home/colum/Workspace/tfref-k0271") + "/zig/kernels/cuda/dsv41_ops.cu"
src = open(SRC).read()
start = src.index("// TF_DS_RESUME_ASSEMBLE (resume_assemble.zig)")
end_marker = "// A greedy row's token as sampling.argmax takes it on the host"
assert src.count("// TF_DS_RESUME_ASSEMBLE (resume_assemble.zig)") == 1
assert src.count(end_marker) == 1
end = src.index(end_marker)
k = src[start:end]
assert "extern \"C\"" in k and k.count("extern \"C\"") == 1 and "__shared__" not in k

mut = sys.argv[1] if len(sys.argv) > 1 else ""
if mut == "fresh_off_by_one":
    # the fresh side reads one row past the cut
    want = "(unsigned long long)(first + i - pre)"
    assert k.count(want) == 1
    k = k.replace(want, "(unsigned long long)(first + i - pre + 1)")
elif mut == "pos_plus_one":
    # the positions run one ahead
    want = "pos[i] = (long long)p;"
    assert k.count(want) == 1
    k = k.replace(want, "pos[i] = (long long)(p + 1);")
elif mut == "bad_slot":
    # the ring's modular slot uses the wrong modulus
    want = "p % window"
    assert k.count(want) == 1
    k = k.replace(want, "p % (window - 1)")
elif mut != "":
    sys.exit("splice.py: unknown mutation %r (fresh_off_by_one|pos_plus_one|bad_slot)" % mut)

open("fused_kernel_current.inc", "w").write(k)
print("wrote fused_kernel_current.inc", len(k), "bytes" + (" [mutant %s]" % mut if mut else ""))
