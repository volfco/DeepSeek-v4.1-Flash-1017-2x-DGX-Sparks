// h0279: CPU harness that executes tf_ds_idx_pick_wide_kernel's exact source (pick_wide_kernel.inc, extracted from
// the worktree's dsv41_ops.cu by extract.py) with emulated blocks/warps/threads, against an INDEPENDENT reference
// written from BRIEF-0279's spec: the k largest keys under the total order (a score's bits, -0 as +0, ordered as
// signed integers, above the inverted column: ties to the lower column), their columns ascending, a column at or
// past `every` or whose maximum is not above -inf as -1. The reference sorts (a radix select is the kernel's, not
// the reference's); it is not copied from the kernel.
//
// Emulation model: one cooperative fiber (ucontext) per CUDA thread, blocks sequential (one block alive at a time),
// one OS thread total. __syncthreads and the warp collectives (__ballot_sync, __shfl_up_sync) are exact at their
// yield points (a two-phase slot exchange per warp). NOT a model of codegen: it checks this C source, in fp32 IEEE
// on x86-64. The block maxima are the INPUT (bmax, as block_max - unchanged by 0279 - writes them).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>
#include <ucontext.h>

// -- CUDA surface the kernel's source uses -------------------------------------------------------------------------

#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
// shared across the fibers of a block; blocks run sequentially and every shared object is written before it is read
#define __shared__ static

struct Dim3 {
    unsigned x, y, z;
};
static Dim3 threadIdx, blockIdx, blockDim, gridDim; // set by the scheduler per fiber and per block

static inline void pdl_enter(int) {}
static inline float __int_as_float(int v) { float f; memcpy(&f, &v, 4); return f; }
static inline unsigned __float_as_uint(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
static inline int __popc(unsigned v) { return __builtin_popcount(v); }
static inline unsigned atomicAdd(unsigned* p, unsigned v) { const unsigned old = *p; *p += v; return old; }

// -- fiber scheduler, barriers and warp collectives ----------------------------------------------------------------

static const int kMaxThreads = 1024;
static const size_t kStackBytes = 64 * 1024;

struct Fiber {
    ucontext_t ctx;
    bool done;
};
static ucontext_t g_sched;
static Fiber g_fib[kMaxThreads];
static std::vector<char> g_stacks;
static Fiber* g_cur = nullptr;
static void (*g_block_fn)(const void*);
static const void* g_block_arg;

struct Barrier {
    int n = 0; // arrivals per generation
    int count = 0;
    unsigned gen = 0; // bumped at each release
};
static Barrier g_block_bar; // __syncthreads
static Barrier g_warp_bar[32];
static unsigned g_bal[32][32]; // ballot slots
static unsigned long long g_sh[32][32]; // shuffle slots (values up to 8 bytes)

// spin-yield barrier: exact under the bulk-synchronous kernel (every thread reaches each barrier once per pass)
static void barrier(Barrier& b) {
    const unsigned my = b.gen;
    if (++b.count == b.n) {
        b.count = 0;
        ++b.gen;
        return;
    }
    while (b.gen == my) swapcontext(&g_cur->ctx, &g_sched);
}

static void __syncthreads() { barrier(g_block_bar); }

// all 32 lanes are converged at every call (the kernel's masks are 0xffffffff)
static inline unsigned __ballot_sync(unsigned /*mask*/, int pred) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    g_bal[w][l] = pred ? 1u : 0u;
    barrier(g_warp_bar[w]);
    unsigned bits = 0;
    for (int i = 0; i < 32; ++i)
        if (g_bal[w][i]) bits |= 1u << i;
    barrier(g_warp_bar[w]);
    return bits;
}

template <typename T>
static inline T __shfl_up_sync(unsigned /*mask*/, T v, int delta) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    static_assert(sizeof(T) <= 8, "shuffle slot size");
    memcpy(&g_sh[w][l], &v, sizeof(T));
    barrier(g_warp_bar[w]);
    T out;
    const int src = l >= delta ? l - delta : l; // out-of-range source: the caller's own value (never read)
    memcpy(&out, &g_sh[w][src], sizeof(T));
    barrier(g_warp_bar[w]);
    return out;
}

static void fiber_entry() {
    g_block_fn(g_block_arg);
    g_cur->done = true;
}

static void run_block(void (*fn)(const void*), const void* arg, int threads, unsigned block_x) {
    static int built = 0;
    if (built != threads) {
        g_stacks.assign((size_t)threads * kStackBytes, 0);
        built = threads;
    }
    // a completed fiber's context cannot be resumed: rebuild every context per block
    for (int i = 0; i < threads; ++i) {
        getcontext(&g_fib[i].ctx);
        g_fib[i].ctx.uc_stack.ss_sp = &g_stacks[(size_t)i * kStackBytes];
        g_fib[i].ctx.uc_stack.ss_size = kStackBytes;
        g_fib[i].ctx.uc_link = &g_sched;
        makecontext(&g_fib[i].ctx, fiber_entry, 0);
    }
    g_block_fn = fn;
    g_block_arg = arg;
    g_block_bar.n = threads;
    for (int w = 0; w < 32; ++w) g_warp_bar[w].n = 32;
    blockIdx.x = block_x;
    blockDim.x = (unsigned)threads;
    gridDim.x = 1;
    for (int i = 0; i < threads; ++i) g_fib[i].done = false;
    for (;;) {
        bool any = false;
        for (int i = 0; i < threads; ++i) {
            if (g_fib[i].done) continue;
            any = true;
            g_cur = &g_fib[i];
            threadIdx.x = (unsigned)i;
            swapcontext(&g_sched, &g_fib[i].ctx);
        }
        if (!any) break;
    }
}

// -- the kernel's exact source --------------------------------------------------------------------------------------

#include "pick_wide_kernel.inc"

struct Args {
    const float* bmax;
    long long bs;
    int nbk, k;
    const long long* every;
    int* cblk;
    int pdl;
};
static void kernel_block(const void* p) {
    const Args& a = *static_cast<const Args*>(p);
    tf_ds_idx_pick_wide_kernel(a.bmax, a.bs, a.nbk, a.k, a.every, a.cblk, a.pdl);
}

// -- the independent reference (from BRIEF-0279's spec; the key contract is tf_ds_topk_indices' documented one) ----

static uint64_t ref_key(float x, int j) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    if (bits == 0x80000000u) bits = 0; // -0 as +0
    int32_t sb;
    memcpy(&sb, &bits, 4);
    const int32_t ordered = sb < 0 ? (sb ^ 0x7FFFFFFF) : sb; // the bits ordered as signed integers
    const uint64_t key = ((uint64_t)(uint32_t)ordered << 32) | (uint64_t)(0xFFFFFFFFu - (unsigned)j);
    return key ^ 0x8000000000000000ull; // above the inverted column, ties to the lower column
}

// the k largest keys by a sort, columns ascending, every/inf/-NaN rule on the slots
static void reference(const std::vector<float>& bmax, int rows, int nbk, int k,
                      const std::vector<long long>& every, std::vector<int>& out) {
    std::vector<std::pair<uint64_t, int>> keys(nbk);
    std::vector<int> cols(k);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int r = 0; r < rows; ++r) {
        const float* row = &bmax[(size_t)r * nbk];
        for (int j = 0; j < nbk; ++j) keys[j] = {ref_key(row[j], j), j};
        std::partial_sort(keys.begin(), keys.begin() + k, keys.end(),
                          [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first > b.first; });
        for (int i = 0; i < k; ++i) cols[i] = keys[i].second;
        std::sort(cols.begin(), cols.end());
        for (int i = 0; i < k; ++i) {
            const int c = cols[i];
            out[(size_t)r * k + i] = (c < every[r] && row[c] > ninf) ? c : -1;
        }
    }
}

// -- cases ---------------------------------------------------------------------------------------------------------

enum Pattern { PAT_RANDOM = 0, PAT_EQUAL, PAT_ZEROS, PAT_NEGINF, PAT_MIXED, PAT_NAN, PAT_COUNT };
static const char* pat_name[PAT_COUNT] = {"random-finite", "equal-maxima", "signed-zeros", "all-neg-inf", "mixed-islands", "nan-maxima"};

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static float bits_f(uint32_t b) {
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static void fill_row(std::vector<float>& row, Pattern p, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    const float ninf = -std::numeric_limits<float>::infinity();
    switch (p) {
    case PAT_RANDOM:
        for (auto& x : row) {
            const uint32_t b = lcg(s);
            // a finite float: any mantissa, exponent 1..254 (normals; the specials have their own patterns)
            x = bits_f((b & 0x807fffffu) | (((lcg(s) % 254u) + 1u) << 23));
        }
        break;
    case PAT_EQUAL:
        for (auto& x : row) x = 1.0f; // every block tied; the tie rule alone orders the slots
        break;
    case PAT_ZEROS:
        for (auto& x : row) x = (lcg(s) & 1) ? 0.0f : bits_f(0x80000000u); // +0/-0 tie to the column order
        break;
    case PAT_NEGINF:
        for (auto& x : row) x = ninf;
        break;
    case PAT_MIXED: {
        for (auto& x : row) x = ninf;
        // finite islands, +inf anchors and lone finite maxima in a sea of -inf
        for (size_t j = 0; j + 3 < row.size(); j += 97) {
            row[j] = bits_f((lcg(s) & 0x807fffffu) | (((lcg(s) % 254u) + 1u) << 23));
            row[j + 1] = bits_f((lcg(s) & 0x807fffffu) | (((lcg(s) % 254u) + 1u) << 23));
            row[j + 2] = bits_f((lcg(s) & 0x807fffffu) | (((lcg(s) % 254u) + 1u) << 23));
        }
        row[0] = std::numeric_limits<float>::infinity();
        row[row.size() / 2] = -0.0f;
        row[row.size() - 1] = bits_f(0x7fc00000u); // a NaN maximum: ranks above +inf, lists as -1
        break;
    }
    case PAT_NAN:
        for (size_t j = 0; j < row.size(); ++j) {
            const uint32_t r = lcg(s) % 8u;
            if (r == 0) row[j] = bits_f(0x7fc00000u); // quiet NaN
            else if (r == 1) row[j] = bits_f(0xffc00000u); // negative quiet NaN
            else if (r == 2) row[j] = bits_f(0x7f800001u); // a signalling NaN payload
            else if (r == 3) row[j] = std::numeric_limits<float>::infinity();
            else if (r == 4) row[j] = ninf;
            else row[j] = bits_f((lcg(s) & 0x807fffffu) | (((lcg(s) % 254u) + 1u) << 23));
        }
        break;
    default:
        std::abort();
    }
}

static int g_fail = 0;

static void run_case(const char* name, int nbk, int rows, Pattern pat, int every_mode) {
    const int k = 2048; // candidate_blocks
    std::vector<float> bmax((size_t)rows * nbk);
    std::vector<long long> every(rows);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(nbk);
        fill_row(row, pat, 0x9E3779B9u * (uint32_t)(r + 1) + (uint32_t)nbk);
        memcpy(&bmax[(size_t)r * nbk], row.data(), (size_t)nbk * 4);
        every[r] = every_mode == 0 ? nbk : every_mode == 1 ? (nbk / 2 + 1) : 0;
    }
    std::vector<int> want((size_t)rows * k), got((size_t)rows * k, -777);
    reference(bmax, rows, nbk, k, every, want);
    Args a{bmax.data(), nbk, nbk, k, every.data(), got.data(), 0};
    for (int r = 0; r < rows; ++r) run_block(kernel_block, &a, 1024, (unsigned)r);
    const bool match = memcmp(got.data(), want.data(), (size_t)rows * k * 4) == 0;
    if (!match) {
        ++g_fail;
        for (size_t i = 0; i < want.size(); ++i) {
            if (got[i] != want[i]) {
                std::printf("case %s nbk=%d rows=%d pattern=%s every=mode%d: MISMATCH at slot %zu (kernel %d, reference %d)\n",
                            name, nbk, rows, pat_name[pat], every_mode, i, got[i], want[i]);
                break;
            }
        }
    } else {
        std::printf("case %s nbk=%d rows=%d pattern=%s every=mode%d: match (%d slots)\n",
                    name, nbk, rows, pat_name[pat], every_mode, rows * k);
    }
}

int main() {
    std::printf("h0279: CPU emulation of tf_ds_idx_pick_wide_kernel (extracted source) vs an independent sort reference\n");
    std::printf("k = 2048 (candidate_blocks); every modes: 0 = nbk (no masking), 1 = nbk/2 + 1, 2 = 0 (all slots -1)\n");
    // the brief's matrix: nbk 16,385 / 32,768 / 65,536 x rows 1/7 x adversarial tie patterns (equal maxima, signed
    // zeros, -inf) plus random, NaN maxima and mixed islands
    for (int nbk : {16385, 32768, 65536}) {
        for (int rows : {1, 7}) {
            for (int p = 0; p < PAT_COUNT; ++p) {
                run_case("wide", nbk, rows, (Pattern)p, (p + rows) % 3);
            }
        }
    }
    // one mixed case: each row of 7 a different pattern, every per row in all three modes
    {
        const int nbk = 32768, rows = 7, k = 2048;
        std::vector<float> bmax((size_t)rows * nbk);
        std::vector<long long> every(rows);
        for (int r = 0; r < rows; ++r) {
            std::vector<float> row(nbk);
            fill_row(row, (Pattern)(r % PAT_COUNT), 0xC0FFEEu + (uint32_t)r);
            memcpy(&bmax[(size_t)r * nbk], row.data(), (size_t)nbk * 4);
            every[r] = r % 3 == 0 ? nbk : r % 3 == 1 ? (nbk / 2 + 1) : 0;
        }
        std::vector<int> want((size_t)rows * k), got((size_t)rows * k, -777);
        reference(bmax, rows, nbk, k, every, want);
        Args a{bmax.data(), nbk, nbk, k, every.data(), got.data(), 0};
        for (int r = 0; r < rows; ++r) run_block(kernel_block, &a, 1024, (unsigned)r);
        const bool match = memcmp(got.data(), want.data(), (size_t)rows * k * 4) == 0;
        if (!match) {
            ++g_fail;
            std::printf("case mixed-rows nbk=%d rows=%d: MISMATCH\n", nbk, rows);
        } else {
            std::printf("case mixed-rows nbk=%d rows=%d: match (%d slots)\n", nbk, rows, rows * k);
        }
    }
    // negative control at nbk 65,537: the host gate refuses this shape (gate.zig asserts it; the unfused fallback
    // stays). Running the kernel anyway shows the bound is load-bearing: column 65,536's inverted id leaves 0xFFFE
    // in the radix prefix's constant bytes, so the selection can never pick it - while the reference does.
    {
        const int nbk = 65537, rows = 1, k = 2048;
        std::vector<float> bmax(nbk, 1.0f);
        std::vector<long long> every(1, nbk);
        bmax[nbk - 1] = std::numeric_limits<float>::infinity(); // the largest maximum sits at column 65,536
        std::vector<int> want(k), got(k, -777);
        reference(bmax, rows, nbk, k, every, want);
        Args a{bmax.data(), nbk, nbk, k, every.data(), got.data(), 0};
        run_block(kernel_block, &a, 1024, 0);
        const bool match = memcmp(got.data(), want.data(), (size_t)k * 4) == 0;
        if (match) {
            std::printf("negative control nbk=65537 rows=1: kernel matched the reference - the 65,536 bound is "
                        "conservative here (the host gate still refuses per todo.md)\n");
        } else {
            for (int i = 0; i < k; ++i) {
                if (got[i] != want[i]) {
                    std::printf("negative control nbk=65537 rows=1: kernel-vs-reference DIVERGES as expected at slot %d "
                                "(kernel %d, reference %d; the reference takes column 65,536, the kernel never can)\n",
                                i, got[i], want[i]);
                    break;
                }
            }
        }
    }
    std::printf("h0279: selection cases: %s (%d mismatched)\n", g_fail ? "FAIL" : "all match the reference", g_fail);
    return g_fail ? 1 : 0;
}
