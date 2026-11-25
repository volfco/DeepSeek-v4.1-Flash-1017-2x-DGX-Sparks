// CPU harness for TF_DS_MARKOV_FUSED's tf_ds_markov_fused_kernel (h0267). It executes the kernel's exact source
// (fused_kernel_current.inc, extracted by splice.py from zig/kernels/cuda/dsv41_ops.cu) with emulated blocks (256 OS
// threads a block, __syncthreads a barrier), against an INDEPENDENT reference written from BRIEF-0267 and the kit's
// markov.py spec: bias -> fp16 round -> + logit -> tile argmax with torch.argmax's tie rule (NaN above everything,
// ties -- and NaN against NaN -- to the lower index), then _finish's fold for the greedy draft id.
//
// NOT a model of CUDA codegen: it checks this C source's logic and fp32 IEEE arithmetic (x86-64, SSE, compiled with
// -ffp-contract=off so `acc += w * e` is the kernel's --fmad=false MUL then ADD). The fp16 round is cross-checked
// between the stub's bit routine (the kernel's __float2half_rn) and the reference's _Float16 conversion (libgcc).
// GPU-only rounding points are listed in report-0267.md.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <random>

// ---- CUDA stubs (p0194/fused_run.cpp's pattern) ---------------------------------------------------------------------
#define __global__
#define __device__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)
struct dim3 { unsigned x, y, z; };
thread_local dim3 threadIdx, blockIdx, gridDim, blockDim;

struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    int count = 0, generation = 0;
    void arrive(int n) {
        asm volatile("" ::: "memory");
        std::unique_lock lk(m);
        int g = generation;
        if (++count == n) { count = 0; ++generation; lk.unlock(); cv.notify_all(); asm volatile("" ::: "memory"); return; }
        cv.wait(lk, [&] { return generation != g; });
        asm volatile("" ::: "memory");
    }
};

struct SimBlock {
    float sv[256]; // the kernel's shared reduce partials (its __shared__ arrays, moved here by splice.py)
    int si[256];
    Barrier bar;
    void clear_shm() {
        std::memset(sv, 0, sizeof sv);
        std::memset(si, 0, sizeof si);
    }
};
thread_local SimBlock* blk = nullptr;
static SimBlock g_block;
static inline void __syncthreads() { blk->bar.arrive(256); }
static inline void pdl_enter(int) {}

struct __half { uint16_t __x; };
struct __nv_bfloat16 { uint16_t __x; };

// the widenings, bit for bit: bf16 -> fp32 is a shift (same exponent width); fp16 -> fp32 rebases the exponent and
// normalises subnormals (exact). Verified against _Float16 for all 65,536 half patterns in main().
static inline float h16_widen(uint16_t bits) {
    unsigned s = (unsigned)(bits & 0x8000u) << 16;
    unsigned e = (bits >> 10) & 0x1fu;
    unsigned m = bits & 0x3ffu;
    unsigned u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else { // subnormal: (m / 1024) * 2^-14, normalised into fp32
            int shift = 0;
            unsigned mm = m;
            while ((mm & 0x400u) == 0) { mm <<= 1; ++shift; }
            mm &= 0x3ffu;
            u = s | ((113u - (unsigned)shift) << 23) | (mm << 13);
        }
    } else if (e == 0x1fu) {
        u = s | (0xffu << 23) | (m << 13); // inf or NaN, the payload expanded
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline float __half2float(__half h) { return h16_widen(h.__x); }
static inline float __bfloat162float(__nv_bfloat16 b) {
    unsigned u = (unsigned)b.__x << 16; // bf16 -> fp32: the same exponent width, exact
    float f;
    memcpy(&f, &u, 4);
    return f;
}
// one fp32 op each: the kernel builds --fmad=false MUL then ADD from these (the harness compiles -ffp-contract=off)
static inline float __fmul_rn(float x, float y) { return x * y; }
static inline float __fadd_rn(float x, float y) { return x + y; }

// __float2half_rn: fp32 -> fp16, round to nearest even (IEEE-754), a NaN to the canonical quiet 0x7e00 with its sign
// kept (cvt.rn.f16.f32 canonicalises). Bit manipulation, kept apart from the reference's _Float16 conversion.
static inline __half __float2half_rn(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xffu) - 127 + 15;
    unsigned m = u & 0x7fffffu;
    __half h;
    if (((u >> 23) & 0xffu) == 0xffu && m != 0) { h.__x = (uint16_t)(s | 0x7e00u); return h; } // NaN -> canonical
    if (e <= 0) {
        if (e < -10) { h.__x = (uint16_t)s; return h; }
        m |= 0x800000u;
        int shift = 1 - e + 13;
        unsigned rem = m & ((1u << shift) - 1);
        unsigned half = m >> shift;
        unsigned halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) ++half;
        h.__x = (uint16_t)(s | half);
        return h;
    }
    if (e >= 31) { h.__x = (uint16_t)(s | 0x7c00u); return h; }
    unsigned lsb = (m >> 13) & 1;
    m += 0xfffu + lsb;
    if (m & 0x800000u) { m = 0; ++e; if (e >= 31) { h.__x = (uint16_t)(s | 0x7c00u); return h; } }
    h.__x = (uint16_t)(s | ((unsigned)e << 10) | (m >> 13));
    return h;
}

// ---- the kernel's exact source ---------------------------------------------------------------------------------------
#include "fused_kernel_current.inc"

// ---- the independent reference (from BRIEF-0267 and markov.py's spec, not from the kernel) --------------------------
// The tie rule, as the spec words it: NaN above everything, then the larger value, ties (and NaN against NaN) to the
// lower index. A predicate over "a beats b", written in the opposite direction from the kernel's fold.
static bool ref_beats(float va, int ia, float vb, int ib) {
    const bool na = std::isnan(va), nb = std::isnan(vb);
    if (na != nb) return na; // NaN above everything
    if (na) return ia < ib; // NaN against NaN: the lower index
    if (va != vb) return va > vb; // the larger value
    return ia < ib; // a tie: the lower index
}

// fp32 -> fp16 bits, round to nearest even: the reference asks _Float16 (libgcc's conversion), with the kernel's
// rounding's NaN rule (a NaN canonicalises to 0x7e00, sign kept).
static uint16_t ref_round16(float x) {
    if (std::isnan(x)) return (uint16_t)(0x7e00u | (std::signbit(x) ? 0x8000u : 0));
    _Float16 h = (_Float16)x;
    uint16_t bits;
    memcpy(&bits, &h, 2);
    return bits;
}
static float ref_widen16(uint16_t bits) { return h16_widen(bits); }
static float ref_widen_bf16(uint16_t bits) {
    unsigned u = (unsigned)bits << 16; // bf16 -> fp32: the same exponent width, exact (not h16_widen: different layout)
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// The step pipeline per row: bias = the token's cached row (slot >= 0) or fp16(head[v] . fp16(emb[tok])) per column
// (the fp32 dot in rank order), widened, plus the logit; each tile of 1024 columns emits its best (value, id) under
// ref_beats into tiles / ids ([rows, n_part], p = seg * s_tiles + tile); wins[r] folds row r's tiles (_finish).
struct RefOut {
    std::vector<float> tiles;
    std::vector<int> ids;
    std::vector<long long> wins;
};

static RefOut ref_step(int rows, int n_cols, int segs, int seg0, int rank, long long l_seg, long long l_row,
                       const float* LG, const int64_t* TOK, long long t_stride, const int* SLOT,
                       const uint16_t* CACHE, long long c_row, const uint16_t* EMB, const uint16_t* HEAD) {
    const int s_tiles = (n_cols + 1023) / 1024;
    const int n_part = s_tiles * segs;
    RefOut o;
    o.tiles.assign((size_t)rows * n_part, 0.0f);
    o.ids.assign((size_t)rows * n_part, 0);
    o.wins.assign(rows, -1);
    for (int r = 0; r < rows; ++r) {
        int64_t tok = TOK[(long long)r * t_stride];
        int slot = SLOT[tok];
        for (int s = 0; s < segs; ++s) {
            for (int t = 0; t < s_tiles; ++t) {
                float bv = -INFINITY;
                int bi = 1 << 30; // _score's masked columns fold (-inf, 1 << 30)
                int c0 = t * 1024;
                for (int c = c0; c < c0 + 1024 && c < n_cols; ++c) {
                    float b;
                    if (slot >= 0) {
                        b = ref_widen16(CACHE[(long long)slot * c_row + (long long)s * n_cols + c]);
                    } else {
                        // fp16(head[v] . fp16(emb[tok])): the fp32 dot in rank order, rounded to fp16, widened back
                        float acc = 0.0f;
                        for (int k = 0; k < rank; ++k) {
                            float wk = ref_widen16(HEAD[(size_t)((long long)(seg0 + s) * n_cols + c) * (unsigned)rank + k]);
                            float ek = ref_widen16(ref_round16(ref_widen_bf16(EMB[(size_t)tok * (unsigned)rank + k])));
                            acc += wk * ek;
                        }
                        b = ref_widen16(ref_round16(acc));
                    }
                    float sc = LG[(long long)s * l_seg + (long long)r * l_row + c] + b;
                    int v = (seg0 + s) * n_cols + c;
                    if (ref_beats(sc, v, bv, bi)) { bv = sc; bi = v; }
                }
                int p = s * s_tiles + t;
                o.tiles[(size_t)r * n_part + p] = bv;
                o.ids[(size_t)r * n_part + p] = bi;
            }
        }
        float bv = -INFINITY;
        int bi = 1 << 30;
        for (int p = 0; p < n_part; ++p)
            if (ref_beats(o.tiles[(size_t)r * n_part + p], o.ids[(size_t)r * n_part + p], bv, bi)) {
                bv = o.tiles[(size_t)r * n_part + p];
                bi = o.ids[(size_t)r * n_part + p];
            }
        o.wins[r] = bi;
    }
    return o;
}

// ---- the emulated launch -------------------------------------------------------------------------------------------
// Blocks are independent (no grid-wide synchronisation), so they run one after another, each with 256 OS threads.
static void launch(int s_tiles, int segs, int rows, int n_cols, int seg0, int rank, long long l_seg, long long l_row,
                   const float* LG, const int64_t* TOK, long long t_stride, const int* SLOT, const uint16_t* CACHE,
                   long long c_row, const uint16_t* EMB, const uint16_t* HEAD, float* PV, int* PI, int n_part) {
    for (int s = 0; s < segs; ++s) {
        for (int t = 0; t < s_tiles; ++t) {
            g_block.clear_shm();
            std::vector<std::thread> ts;
            for (int i = 0; i < 256; ++i) {
                ts.emplace_back([=]() {
                    blk = &g_block;
                    threadIdx = { (unsigned)i, 0, 0 };
                    blockIdx = { (unsigned)t, (unsigned)s, 0 };
                    gridDim = { (unsigned)s_tiles, (unsigned)segs, 1 };
                    blockDim = { 256, 0, 0 };
                    tf_ds_markov_fused_kernel(LG, (uint64_t)l_seg, (uint64_t)l_row, TOK, (uint64_t)t_stride, SLOT,
                                              (const __half*)CACHE, (uint64_t)c_row, (const __nv_bfloat16*)EMB,
                                              (const __half*)HEAD, rank, PV, PI, rows, n_cols, seg0, n_part, 0);
                });
            }
            for (auto& th : ts) th.join();
        }
    }
}

// ---- checking ------------------------------------------------------------------------------------------------------
static int g_fail = 0;
static long long g_cases = 0, g_rows = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct Case {
    int rows, n_cols, segs, seg0, rank;
    long long l_seg, l_row, t_stride, c_row;
    std::vector<float> lg;
    std::vector<int64_t> tok;
    std::vector<int> slot;
    std::vector<uint16_t> cache, emb, head;
    long long tok_space;
};

// Run one case through the emulated kernel and the reference; compare every tile winner's bits and id and the drafts.
// Returns the kernel-side draft ids (the planted cases assert their expected winners on them).
static std::vector<long long> run_case(const char* name, Case& c) {
    ++g_cases;
    g_rows += c.rows;
    const int s_tiles = (c.n_cols + 1023) / 1024;
    const int n_part = s_tiles * c.segs;
    std::vector<long long> wins(c.rows, -1);
    std::vector<float> pv((size_t)c.rows * n_part, 12345.0f);
    std::vector<int> pi((size_t)c.rows * n_part, -999);
    launch(s_tiles, c.segs, c.rows, c.n_cols, c.seg0, c.rank, c.l_seg, c.l_row, c.lg.data(), c.tok.data(), c.t_stride,
           c.slot.data(), c.cache.data(), c.c_row, c.emb.data(), c.head.data(), pv.data(), pi.data(), n_part);
    RefOut ref = ref_step(c.rows, c.n_cols, c.segs, c.seg0, c.rank, c.l_seg, c.l_row, c.lg.data(), c.tok.data(), c.t_stride,
                          c.slot.data(), c.cache.data(), c.c_row, c.emb.data(), c.head.data());
    for (int r = 0; r < c.rows; ++r) {
        for (int p = 0; p < n_part; ++p) {
            float kv = pv[(size_t)r * n_part + p], rv = ref.tiles[(size_t)r * n_part + p];
            uint32_t kb, rb;
            memcpy(&kb, &kv, 4);
            memcpy(&rb, &rv, 4);
            // NaN winners compare by kind: C leaves the sign/payload a NaN add returns unspecified (the two sides are
            // separate compilations), while the ids below stay exact. On the GPU both arms issue the same add of the
            // same operands, so the bytes match there (report-0267's rounding points).
            const bool same = (kb == rb) || (std::isnan(kv) && std::isnan(rv));
            CHECK(same, "%s row %d p %d: value bits %08x vs %08x", name, r, p, kb, rb);
            CHECK(pi[(size_t)r * n_part + p] == ref.ids[(size_t)r * n_part + p], "%s row %d p %d: id %d vs %d", name, r, p,
                  pi[(size_t)r * n_part + p], ref.ids[(size_t)r * n_part + p]);
        }
        // the greedy draft id: _finish's fold over the kernel's own partials under the kernel's own tie rule
        float bv = -INFINITY;
        int bi = 1 << 30;
        for (int p = 0; p < n_part; ++p)
            if (tf_ds_markov_better(pv[(size_t)r * n_part + p], pi[(size_t)r * n_part + p], bv, bi)) {
                bv = pv[(size_t)r * n_part + p];
                bi = pi[(size_t)r * n_part + p];
            }
        CHECK(bi == ref.wins[r], "%s row %d: draft id %lld vs %lld", name, r, (long long)bi, ref.wins[r]);
        wins[r] = bi;
    }
    return wins;
}

// ---- case construction ---------------------------------------------------------------------------------------------
static uint16_t f16_bits(float f) { return __float2half_rn(f).__x; }
static uint16_t b16_bits(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static Case make_case(int rows, int n_cols, int segs, int seg0, int rank, long long t_stride, long long block) {
    Case c;
    c.rows = rows;
    c.n_cols = n_cols;
    c.segs = segs;
    c.seg0 = seg0;
    c.rank = rank;
    c.t_stride = t_stride;
    c.c_row = (long long)segs * n_cols;
    c.l_row = block * n_cols;
    c.l_seg = (long long)rows * block * n_cols;
    c.tok_space = 64;
    return c;
}

// fills head/emb/cache/lg with tame or wild patterns; hit rows get their random cached row, miss rows nothing
static void fill(Case& c, std::mt19937_64& rng, bool wild, double hit_rate, double special_rate) {
    std::uniform_real_distribution<double> u(-4.0, 4.0);
    std::uniform_int_distribution<int> hitd(0, 999);
    c.tok.assign((size_t)c.rows * c.t_stride, 0);
    c.slot.assign((size_t)c.tok_space, -1);
    for (int r = 0; r < c.rows; ++r) {
        c.tok[(size_t)r * c.t_stride] = (int64_t)(rng() % (uint64_t)c.tok_space);
        if (hitd(rng) < hit_rate * 1000) c.slot[c.tok[(size_t)r * c.t_stride]] = (int)(rng() % 4); // a cached row of 4
    }
    c.cache.assign((size_t)4 * c.c_row, 0);
    for (auto& x : c.cache) x = wild ? (uint16_t)rng() : f16_bits((float)u(rng));
    c.emb.assign((size_t)c.tok_space * c.rank, 0);
    for (auto& x : c.emb) x = wild ? (uint16_t)rng() : b16_bits((float)u(rng));
    c.head.assign((size_t)((long long)(c.seg0 + c.segs) * c.n_cols) * c.rank, 0);
    for (auto& x : c.head) x = wild ? (uint16_t)rng() : f16_bits((float)u(rng));
    c.lg.assign((size_t)(c.segs * c.l_seg), 0.0f);
    for (auto& x : c.lg) x = (float)u(rng);
    // special logits at random columns: NaN, +-inf, -0
    const long long cols = (long long)c.segs * c.n_cols;
    std::uniform_int_distribution<long long> cd(0, cols - 1);
    std::uniform_int_distribution<int> sd(0, 4);
    const int n_special = (int)(special_rate * cols) + 3;
    for (int i = 0; i < n_special; ++i) {
        float v = (float[]){ NAN, INFINITY, -INFINITY, -0.0f, 0.0f }[sd(rng)];
        for (int r = 0; r < c.rows; ++r) {
            long long cc = cd(rng);
            long long s = cc / c.n_cols, col = cc % c.n_cols;
            c.lg[(long long)s * c.l_seg + (long long)r * c.l_row + col] = v;
        }
    }
}

int main() {
    std::mt19937_64 rng(20261122);

    // 1. the fp16 roundings agree (the stub's bit routine vs the reference's _Float16): edges, then random bits;
    //    and every half's widening against _Float16's
    {
        long long widen_bad = 0, widen_nan = 0;
        for (unsigned bits = 0; bits < 0x10000u; ++bits) {
            _Float16 h;
            memcpy(&h, &bits, 2);
            float a = h16_widen((uint16_t)bits), b = (float)h;
            uint32_t ua, ub;
            memcpy(&ua, &a, 4);
            memcpy(&ub, &b, 4);
            if (std::isnan(a) || std::isnan(b)) { ++widen_nan; continue; }
            if (ua != ub && ++widen_bad < 5) CHECK(false, "widen16 %04x: %08x vs %08x", bits, ua, ub);
        }
        CHECK(widen_bad == 0, "widen16: %lld mismatches", widen_bad);
        printf("fp16 widen cross-check: 65536 patterns (%lld NaNs compared by kind)\n", widen_nan);
    }
    const float edges[] = { 0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, 65520.0f, 65536.0f, 1e8f, -1e8f,
                            0x1p-24f, -0x1p-24f, 0x1p-25f, 0x1.8p-25f, 0x1p-14f, 0x1p-11f, 1.0f + 0x1p-11f,
                            1.0f + 0x1.8p-11f, 0x1.fffffep15f, 0x1.002p16f, INFINITY, -INFINITY, NAN, -NAN,
                            0x1.8p-10f, 2.9802322e-8f };
    for (float f : edges) {
        uint16_t a = __float2half_rn(f).__x, b = ref_round16(f);
        CHECK(a == b, "round16 edge %g: %04x vs %04x", (double)f, a, b);
    }
    long long rnd_ok = 0, rnd_nan = 0;
    for (int i = 0; i < 200000; ++i) {
        float f;
        uint32_t bits = (uint32_t)rng();
        memcpy(&f, &bits, 4);
        uint16_t a = __float2half_rn(f).__x, b = ref_round16(f);
        if (std::isnan(f)) { CHECK(a == b, "round16 NaN %08x: %04x vs %04x", bits, a, b); ++rnd_nan; }
        else { CHECK(a == b, "round16 random %g (%08x): %04x vs %04x", (double)f, bits, a, b); ++rnd_ok; }
    }
    printf("fp16 round cross-check: %lld finite bit patterns + %lld NaNs\n", rnd_ok, rnd_nan);

    // 2. misses for 1..4 streams (every row a miss), then 7 and 16 rows
    for (int rows = 1; rows <= 4; ++rows) {
        Case c = make_case(rows, 2176, 1, 1, 8, 1, 6);
        fill(c, rng, false, 0.0, 0.001);
        run_case("miss-rows", c);
    }
    for (int rows : { 7, 16 }) {
        Case c = make_case(rows, 1100, 1, 0, 4, 1, 5);
        fill(c, rng, false, 0.0, 0.001);
        run_case("miss-rows-bound", c);
    }

    // 3. mixed hits and misses (the cached rows are random: the kernel must READ them, not recompute)
    for (int rows = 1; rows <= 4; ++rows) {
        Case c = make_case(rows, 3000, 1, 2, 33, 1, 6);
        fill(c, rng, false, 0.5, 0.002);
        run_case("mixed-hit-miss", c);
        Case h = make_case(rows, 1500, 1, 0, 2, 1, 5);
        fill(h, rng, false, 1.0, 0.002);
        run_case("all-hit", h);
    }

    // 4. ties: equal scores to the lower id, in a tile, across tiles, and against the masked tail
    {
        Case c = make_case(2, 2176, 1, 1, 2, 1, 6); // ids 2176..: three tiles, the last 128 columns
        fill(c, rng, false, 0.0, 0.0);
        for (auto& x : c.head) x = 0; // bias 0 everywhere, then exact ties in the logits
        for (int r = 0; r < 2; ++r)
            for (long long i = 0; i < (long long)c.segs * c.n_cols; ++i) c.lg[(long long)r * c.l_row + i] = -7.0f;
        c.lg[0 * c.l_row + 5] = 3.0f; // row 0 tile 0: two columns at 3.0
        c.lg[0 * c.l_row + 900] = 3.0f;
        c.lg[0 * c.l_row + 1030] = 5.0f; // tile 1
        c.lg[0 * c.l_row + 2100] = 5.0f; // a tie across tiles 1 and 2 -> the fold takes the lower id
        c.lg[1 * c.l_row + 2175] = -7.0f; // row 1: every valid column ties at -7 -> the lowest id
        auto w = run_case("ties", c);
        // the planted ties' rule: the lower id (row 0's tie across tiles 1 and 2; row 1's all-tie row)
        CHECK(w[0] == 2176 + 1030, "ties row 0: winner %lld want %d", w[0], 2176 + 1030);
        CHECK(w[1] == 2176, "ties row 1: winner %lld want 2176", w[1]);
    }

    // 5. NaN and inf logits: NaN above everything, NaN vs NaN to the lower index, +inf ties, an all -inf row
    {
        Case c = make_case(3, 1025, 1, 0, 2, 1, 6); // tile 1 has one valid column and 1023 masked ones
        fill(c, rng, false, 0.0, 0.0);
        for (auto& x : c.head) x = 0;
        for (int r = 0; r < 3; ++r)
            for (long long i = 0; i <= 1025; ++i) c.lg[(long long)r * c.l_row + i] = (float)i;
        c.lg[0 * c.l_row + 7] = NAN; // row 0: two NaNs (columns 7 and 12) -> id 7
        c.lg[0 * c.l_row + 12] = NAN;
        c.lg[1 * c.l_row + 40] = INFINITY;
        c.lg[1 * c.l_row + 90] = INFINITY; // row 1: +inf tie -> id 40
        for (long long i = 0; i <= 1025; ++i) c.lg[(long long)2 * c.l_row + i] = -INFINITY; // row 2: all -inf
        auto w = run_case("nan-inf", c);
        CHECK(w[0] == 7, "nan row 0: winner %lld want 7 (NaN vs NaN: the lower index)", w[0]);
        CHECK(w[1] == 40, "inf row 1: winner %lld want 40 (+inf tie: the lower index)", w[1]);
        CHECK(w[2] == 0, "row 2: winner %lld want 0 (an all -inf row: the lowest valid id, not the sentinel)", w[2]);
    }

    // 6. partial tiles and exact tile widths
    for (int n_cols : { 1, 1023, 1024, 1025, 2176, 3000 }) {
        Case c = make_case(3, n_cols, 1, 3, 16, 1, 6);
        fill(c, rng, false, 0.34, 0.01);
        run_case("tile-widths", c);
    }

    // 7. two segments (the non-split geometry): the segment's cache columns and global ids
    {
        Case c = make_case(4, 1500, 2, 5, 9, 1, 6);
        fill(c, rng, false, 0.5, 0.002);
        run_case("segs-2", c);
        Case w = make_case(2, 1200, 2, 0, 3, 1, 5);
        fill(w, rng, true, 0.4, 0.02);
        run_case("segs-2-wild", w);
    }

    // 8. a token stride > 1 (the caller's out row stride)
    {
        Case c = make_case(3, 1300, 1, 1, 5, 3, 6);
        fill(c, rng, false, 0.4, 0.002);
        run_case("strides", c);
    }

    // 9. wild bit patterns (NaN / inf / subnormal weights, embeddings and cached rows)
    for (int i = 0; i < 24; ++i) {
        Case c = make_case(1 + (int)(rng() % 4), 1 + (int)(rng() % 2200), 1 + (int)(rng() % 2), (int)(rng() % 3),
                           1 + (int)(rng() % 40), 1, 5 + (long long)(rng() % 2));
        fill(c, rng, true, 0.3, 0.02);
        run_case("wild", c);
    }

    // 10. tame fuzz
    for (int i = 0; i < 120; ++i) {
        Case c = make_case(1 + (int)(rng() % 8), 1 + (int)(rng() % 2500), 1 + (int)(rng() % 2), (int)(rng() % 3),
                           1 + (int)(rng() % 33), (long long)(1 + rng() % 2), 5 + (long long)(rng() % 2));
        fill(c, rng, false, (double)(rng() % 100) / 100.0, 0.005);
        run_case("fuzz", c);
    }

    // 11. the served shape: 129,280 vocabulary on TP2, 64,640 columns a rank, rank 256, four streams
    {
        Case c = make_case(4, 64640, 1, 1, 256, 1, 6);
        fill(c, rng, false, 0.5, 0.0005);
        run_case("served-shape", c);
    }

    printf("%lld cases, %lld rows compared\n", g_cases, g_rows);
    if (g_fail) {
        printf("%d FAILURES\n", g_fail);
        return 1;
    }
    printf("all match\n");
    return 0;
}
