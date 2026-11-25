// emu.h: the CUDA surface the swept kernels use, on the CPU. One cooperative fiber (ucontext) per CUDA thread, blocks
// sequential, one OS thread. __syncthreads and the warp collectives are exact at their yield points (a two-phase slot
// exchange per warp). The order in which a block's fibers are resumed is a parameter (ascending, descending, or a
// seeded shuffle redrawn every sweep), so anything that depends on the schedule - the slot an atomicAdd hands out -
// is exercised under several lane orders. A barrier releases when every LIVE thread of its group has arrived, so a
// kernel whose threads return early behaves as on the device.
// NOT a model of codegen or of the device's libm: it checks the kernel's C source in IEEE fp32/fp64 on x86-64
// (build with -ffp-contract=off). Include once per translation unit, before the extracted kernel sources.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <ucontext.h>

#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static   // one block alive at a time: a function static is the block's shared memory
#define __restrict__

struct dim3 { unsigned x = 1, y = 1, z = 1; };
struct uint4 { unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;

static inline void pdl_enter(int) {}
static inline float __int_as_float(int v) { float f; memcpy(&f, &v, 4); return f; }
static inline float __uint_as_float(unsigned v) { float f; memcpy(&f, &v, 4); return f; }
static inline unsigned __float_as_uint(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
static inline int __float_as_int(float v) { int u; memcpy(&u, &v, 4); return u; }
static inline double __longlong_as_double(long long v) { double d; memcpy(&d, &v, 8); return d; }
static inline int __popc(unsigned v) { return __builtin_popcount(v); }
static inline int __clz(int v) { return v == 0 ? 32 : __builtin_clz((unsigned)v); }
static inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
static inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
// single OS thread: an atomic is its plain form; the order of calls is the scheduler's lane order
static inline unsigned atomicAdd(unsigned* p, unsigned v) { const unsigned o = *p; *p += v; return o; }
static inline int atomicAdd(int* p, int v) { const int o = *p; *p += v; return o; }
static inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { const unsigned long long o = *p; *p += v; return o; }
static inline unsigned atomicOr(unsigned* p, unsigned v) { const unsigned o = *p; *p |= v; return o; }
static inline unsigned atomicExch(unsigned* p, unsigned v) { const unsigned o = *p; *p = v; return o; }

namespace emu {
enum Order { ASC = 0, DESC = 1, SHUFFLE = 2 };
static const int kMax = 1024;
static const size_t kStack = 128 * 1024;
struct Fiber { ucontext_t ctx; bool done; };
static ucontext_t sched;
static Fiber fib[kMax];
static std::vector<char> stacks;
static int cur = -1;
static void (*body)(const void*);
static const void* body_arg;
struct Bar { int count = 0; unsigned gen = 0; };
static Bar block_bar, warp_bar[32];
static int live_block, live_warp[32];
static unsigned long long slot[32][32];
static unsigned mslot[32][32];
static unsigned long progress;   // bumped at each release and each fiber end (deadlock detection)
static unsigned long barriers, shuffles, reductions;   // counted across a launch (harnesses report the work the source does)

static void release(Bar& b) { b.count = 0; ++b.gen; ++progress; }
static void arrive(Bar& b, const int& live) {
    const unsigned my = b.gen;
    if (++b.count >= live) { release(b); return; }
    while (b.gen == my) swapcontext(&fib[cur].ctx, &sched);
}
static void entry() {
    body(body_arg);
    const int t = cur, w = t >> 5;
    fib[t].done = true;
    ++progress;
    // a thread that returns leaves its groups: a barrier the others already wait at may now be complete
    if (--live_block > 0 && block_bar.count >= live_block) release(block_bar);
    if (--live_warp[w] > 0 && warp_bar[w].count >= live_warp[w]) release(warp_bar[w]);
}
// One block: `threads` fibers run `fn(arg)`; blockIdx is the caller's. Returns false on a deadlock.
static bool run_block(void (*fn)(const void*), const void* arg, int threads, Order order, std::mt19937& rng) {
    if ((int)(stacks.size() / kStack) < threads) stacks.assign((size_t)threads * kStack, 0);
    for (int i = 0; i < threads; ++i) {
        getcontext(&fib[i].ctx);
        fib[i].ctx.uc_stack.ss_sp = &stacks[(size_t)i * kStack];
        fib[i].ctx.uc_stack.ss_size = kStack;
        fib[i].ctx.uc_link = &sched;
        makecontext(&fib[i].ctx, entry, 0);
        fib[i].done = false;
    }
    body = fn; body_arg = arg;
    block_bar = Bar{};
    live_block = threads;
    for (int w = 0; w < 32; ++w) { warp_bar[w] = Bar{}; live_warp[w] = std::max(0, std::min(32, threads - 32 * w)); }
    blockDim.x = (unsigned)threads;
    std::vector<int> perm(threads);
    for (int i = 0; i < threads; ++i) perm[i] = order == DESC ? threads - 1 - i : i;
    for (;;) {
        if (order == SHUFFLE) std::shuffle(perm.begin(), perm.end(), rng);
        const unsigned long before = progress;
        bool any = false;
        for (int j = 0; j < threads; ++j) {
            const int i = perm[j];
            if (fib[i].done) continue;
            any = true;
            cur = i;
            threadIdx.x = (unsigned)i;
            swapcontext(&sched, &fib[i].ctx);
        }
        if (!any) return true;
        if (progress == before) { fprintf(stderr, "emu: deadlock (a barrier some live thread never reaches)\n"); return false; }
    }
}
}  // namespace emu

static inline void __syncthreads() { ++emu::barriers; emu::arrive(emu::block_bar, emu::live_block); }
static inline void __syncwarp() { const int w = threadIdx.x >> 5; emu::arrive(emu::warp_bar[w], emu::live_warp[w]); }

// A warp collective: publish, wait for the warp, read, wait again (so no lane republishes before every lane has read).
template <typename T, typename F> static inline T emu_shuffle(T v, F src_of) {
    static_assert(sizeof(T) <= 8, "shuffle slot");
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    ++emu::shuffles;
    unsigned long long raw = 0;
    memcpy(&raw, &v, sizeof(T));
    emu::slot[w][l] = raw;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    T out;
    memcpy(&out, &emu::slot[w][src_of(l)], sizeof(T));
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    return out;
}
template <typename T> static inline T __shfl_xor_sync(unsigned, T v, int delta, int width = 32) {
    return emu_shuffle(v, [=](int l) { const int s = l ^ delta; return s / width == l / width ? s : l; });
}
template <typename T> static inline T __shfl_down_sync(unsigned, T v, int delta, int width = 32) {
    return emu_shuffle(v, [=](int l) { return (l % width) + delta < width ? l + delta : l; });
}
template <typename T> static inline T __shfl_up_sync(unsigned, T v, int delta, int width = 32) {
    return emu_shuffle(v, [=](int l) { return (l % width) >= delta ? l - delta : l; });
}
static inline unsigned __ballot_sync(unsigned mask, int pred) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    emu::slot[w][l] = pred ? 1u : 0u;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    unsigned bits = 0;
    for (int i = 0; i < 32; ++i) if (((mask >> i) & 1u) && emu::slot[w][i]) bits |= 1u << i;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    return bits;
}
// The lanes named by each caller's own mask (the kernels pass a group mask that contains the caller).
static inline unsigned __reduce_max_sync(unsigned mask, unsigned v) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    ++emu::reductions;
    emu::slot[w][l] = v;
    emu::mslot[w][l] = mask;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) if ((mask >> i) & 1u) m = std::max(m, (unsigned)emu::slot[w][i]);
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    return m;
}
static inline unsigned __reduce_add_sync(unsigned mask, unsigned v) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    emu::slot[w][l] = v;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) if ((mask >> i) & 1u) m += (unsigned)emu::slot[w][i];
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    return m;
}

// -- bf16 and the explicit-rounding intrinsics (dsv41_torch.cu) -----------------------------------------------------
// __float2bfloat16_rn is cuda_bf16.hpp's host form (round to nearest even on the low 16 bits, a NaN as 0x7fff); the
// device's cvt.rn.bf16.f32 may carry a NaN's payload differently. Both revisions of a kernel run through the same
// form here, so an old-vs-new comparison does not depend on it.
struct __nv_bfloat16 { unsigned short x; };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
static inline float __bfloat162float(__nv_bfloat16 h) { return __uint_as_float((unsigned)h.x << 16); }
static inline __nv_bfloat16 __float2bfloat16_rn(float f) {
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __nv_bfloat16{0x7fffu};
    unsigned short r = (unsigned short)(u >> 16);
    const unsigned rem = u << 16;
    if (rem > 0x80000000u || (rem == 0x80000000u && (r & 1u))) ++r;
    return __nv_bfloat16{r};
}
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fmaf(a, b, c); }
