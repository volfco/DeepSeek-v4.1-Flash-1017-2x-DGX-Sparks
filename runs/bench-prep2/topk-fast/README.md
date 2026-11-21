# 0182 (`TF_DS_TOPK_FAST`): the algorithm, checked host-side

`tf_ds_topk_indices_fast_kernel` (the kernel of `zig/patches/0182`) cannot run here: nvcc only runs at image build and
the pair runs the arms. What *can* be checked is the algorithm, so `sim.cpp` compiles the kernel's own body against
stubs (`blockDim = 1`, sequential atomics, no-op `__syncthreads`) and compares its output with a brute force (sort
every key, take the k largest, columns ascending, `-1` at or past `vis`) on 51 rows.

    clang++ -std=c++17 -O2 -o sim sim.cpp && ./sim

`sim.out` is that run: 51 cases, all exact. The cases: the indexer's shape (n=131072, k=2048, Gaussian-ish), n below
and at the 4096 sample, every score identical (the compaction overflows: the eight-digit fallback), heavy ties on
seven values, ascending and descending rows (a systematic sample misreads a trend), `vis` hiding most of the row,
negatives and zeros, k=n and 40 random shapes (n 200-100k, k 1-2048, three score distributions).

Not covered here: the CUDA itself (threads, `atomicAdd` order, the bitonic sort's parallelism), which is why the
patch is UNVERIFIED-BUILD until the image build and owes `a_idx` at layers 24-36 on the pair.
