#!/usr/bin/env python3
"""splice.py [better_ties|no_round|miss_mask]: extract tf_ds_markov_better and tf_ds_markov_fused_kernel's exact
source from the worktree's dsv41_ops.cu into fused_kernel_current.inc for the CPU harness. With a mutation name it
writes a seeded-bug variant instead (build.sh runs each and the matrix must catch it); a bare splice.py restores the
kernel's own text. The kernel's two __shared__ arrays move to the harness's SimBlock (`shm.`), as the CPU block
emulation needs; everything else is the .cu's text unchanged.
"""
import re
import sys

SRC = "/home/colum/Workspace/tfref-k0267/zig/kernels/cuda/dsv41_ops.cu"
src = open(SRC).read()
start = src.index("__device__ __forceinline__ int tf_ds_markov_better")
end = src.index("// TF_DS_SPEC_SAMPLE: out[i] = src[idx[i]]")
k = src[start:end]

# the shared reduce's arrays live in the harness's SimBlock (they are block-shared, not per-thread locals)
decl = "    __shared__ float sv[256];\n    __shared__ int si[256];\n"
assert k.count(decl) == 1, "the shared declaration moved: update splice.py"
k = k.replace(decl, "")
k = re.sub(r"\bsv\[", "shm.sv[", k)
k = re.sub(r"\bsi\[", "shm.si[", k)

# the harness's per-thread block handle, injected after the kernel's first statement
needle = "    pdl_enter(pdl);\n"
assert k.count(needle) == 1
k = k.replace(needle, needle + "    SimBlock& shm = *blk;\n", 1)
assert "__shared__" not in k and "SimBlock" in k

mut = sys.argv[1] if len(sys.argv) > 1 else ""
if mut == "better_ties":
    # a tie goes to the HIGHER index (the rule's tie half)
    want = "(v1 == v2) & (i1 < i2)"
    assert k.count(want) == 1
    k = k.replace(want, "(v1 == v2) & (i1 > i2)")
elif mut == "no_round":
    # the miss row's bias skips its round to fp16 (the staging round's rounding)
    want = ": __half2float(__float2half_rn(acc[r]));"
    assert k.count(want) == 1
    k = k.replace(want, ": acc[r];")
elif mut == "miss_mask":
    # a token with slot 0 is taken as a miss
    want = "const float b = slotv[r] >= 0"
    assert k.count(want) == 1
    k = k.replace(want, "const float b = slotv[r] > 0")
elif mut != "":
    sys.exit("splice.py: unknown mutation %r (better_ties|no_round|miss_mask)" % mut)

open("fused_kernel_current.inc", "w").write(k)
print("wrote fused_kernel_current.inc", len(k), "bytes" + (" [mutant %s]" % mut if mut else ""))
