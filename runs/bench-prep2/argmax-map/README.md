# 0186 (`TF_DS_GREEDY_ARGMAX`): the pair + reduce, checked host-side

`sim.cpp` compiles the two new kernels' own bodies (`tf_ds_argmax_pair_kernel`, `tf_ds_argmax_reduce_kernel` and the
refactored `tf_ds_block_argmax_v`) against stubs (`blockDim = 1`, shuffles as identity) and compares each case with a
brute force: the whole row's argmax as `tf_ds_block_argmax` computes it - NaN skipped, ties to the lower column, and
pick 0 when the row's first value is NaN.

    clang++ -std=c++17 -O2 -o sim sim.cpp && ./sim

`sim.out` is that run: 58 cases across world=2 and world=4, all exact. The cases: random rows, every value equal (so the
tie goes to the lower rank), a few distinct values with ties inside and across ranks, NaN in rank 0's first column (the
pick-0 rule), NaNs in later ranks (ignored), an all-NaN row (both sides return the same 0x7fffffff sentinel as the
non-mapped path), an all-NaN row with one finite value in rank 1, negative zero against zero, and 40 random rows mixing
those edges.

It caught one real bug before the patch was exported: the reduce returned the **rank-local** column, not the whole
row's (`want 85, got 21` - the rank offset missing). The kernel now offsets by `rank * n`, with `n` a rank's columns
(`Round.amap_n`, set where the head writes the pairs).

Not covered: the CUDA itself (warps, `__shfl_down_sync`, the block reduction), which is why the patch is
UNVERIFIED-BUILD until the image build and owes `exact.py` on the pair.
