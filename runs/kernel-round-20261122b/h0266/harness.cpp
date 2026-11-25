// h0266: CPU harness executing the exact source of 0197's decode compressor (tf_ds_compress2x_norm{128,512}_kernel)
// and the TF_DS_COMPRESS2_PAIR_ONCE variant (tf_ds_compress2x_norm{128,512}_pair_kernel) on emulated blocks/warps/
// threads, against each other and against an independent per-row reference written from BRIEF-0266's contract.
// NOT a model of CUDA codegen: it checks this C source, bit for bit, in fp32 IEEE (x86-64), where both kernels run the
// same extracted text through the same primitives (expf, the kit sqrt/reciprocal stubs), so old-vs-new byte equality
// holds exactly and reference-vs-kernel byte equality validates the pairing/broadcast logic.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <condition_variable>
#include <cfloat>
#include <string>
#include <atomic>

#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict__
#define __ffs(x) __builtin_ffs(x)
struct int2 { int x, y; };
struct dim3 { unsigned x, y, z; };
static inline int2 make_int2(int x, int y) { return {x, y}; }

// one mutex for the atomics the kernels use
static std::recursive_mutex mu;

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

// one emulated block's state: the kernels' __shared__ arrays (renamed by the extractor to blk->*) and the sync slots
struct SimBlock {
    float row[512];
    float warp_sum[4];
    Barrier bar;              // blockDim.x lanes
    Barrier war[16];          // 32 lanes a warp
    unsigned sh_u[16][32];    // shuffle slots as bits
    unsigned bpred[16][32];   // ballot predicates
    int nthreads = 0;
    void clear(int n) {
        std::memset(row, 0, sizeof row);
        std::memset(warp_sum, 0, sizeof warp_sum);
        std::memset(sh_u, 0, sizeof sh_u);
        std::memset(bpred, 0, sizeof bpred);
        nthreads = n;
    }
};
thread_local SimBlock* blk = nullptr;
thread_local dim3 threadIdx, blockIdx, gridDim, blockDim;

static inline void __syncthreads() { blk->bar.arrive(blk->nthreads); }

template <typename T> static inline T __shfl_xor_sync(unsigned, T v, int lane) {
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    memcpy(&blk->sh_u[w][l], &v, sizeof(T));
    asm volatile("" ::: "memory");
    blk->war[w].arrive(32);
    asm volatile("" ::: "memory");
    T out;
    memcpy(&out, &blk->sh_u[w][l ^ lane], sizeof(T));
    asm volatile("" ::: "memory");
    blk->war[w].arrive(32);
    asm volatile("" ::: "memory");
    return out;
}

static inline unsigned __ballot_sync(unsigned, bool pred) {
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    blk->bpred[w][l] = pred ? 1u : 0u;
    asm volatile("" ::: "memory");
    blk->war[w].arrive(32);
    asm volatile("" ::: "memory");
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) m |= blk->bpred[w][i] << i;
    asm volatile("" ::: "memory");
    blk->war[w].arrive(32);
    asm volatile("" ::: "memory");
    return m;
}

static inline unsigned atomicExch(unsigned* p, unsigned v) {
    asm volatile("" ::: "memory");
    std::lock_guard g(mu);
    unsigned r = *p;
    *p = v;
    asm volatile("" ::: "memory");
    return r;
}

static inline float __fadd_rn(float x, float y) { return x + y; }
static inline float __fsub_rn(float x, float y) { return x - y; }
static inline float __fmul_rn(float x, float y) { return x * y; }
static inline float __fdiv_rn(float x, float y) { return x / y; }
static inline float __fmaf_rn(float x, float y, float z) { return std::fmaf(x, y, z); }

struct __nv_bfloat16 { uint16_t __x; };
static inline float __bfloat162float(__nv_bfloat16 v) {
    unsigned u = ((unsigned)v.__x) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline __nv_bfloat16 __float2bfloat16_rn(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned lsb = (u >> 16) & 1;
    u += 0x7fff + lsb;
    __nv_bfloat16 r;
    r.__x = (uint16_t)(u >> 16);
    return r;
}

// the kit's approximate PTX (SASS MUFU.SQRT then MUFU.RCP) as harness primitives: both kernels' extracted text and the
// reference below call these, so comparisons are bitwise on this CPU
static inline float kit_sqrt_approx(float x) { return std::sqrt(x); }
static inline float kit_div_approx(float a, float b) { return a / b; }

static inline void pdl_launch(int) {}
static inline void pdl_wait(int) {}
static inline void pdl_enter(int) {}

#include "kernels_host.inc"

// -- independent reference (written from BRIEF-0266's contract and the documented arithmetic, not from the kernel) ----

// _compress's ratio-2 pair (compress2_norm.pairSum's documented rule): the softmax over the pair with the max
// subtracted, the sum accumulated from 0, the two products and the sum as the reduce adds it (four adds of 0).
static float ref_pair(float s0, float s1, float kv0, float kv1) {
    float m = -3.402823466e+38f;
    if (s0 > m) m = s0;
    if (s1 > m) m = s1;
    float sum = 0.f;
    sum += expf(s0 - m);
    sum += expf(s1 - m);
    const float p0 = expf(s0 - m) / sum;
    const float p1 = expf(s1 - m) / sum;
    const float y0 = kv0 * p0;
    const float y1 = kv1 * p1;
    float t = (0.f + y0) + (0.f + y1);
    t = t + 0.f;
    t = t + 0.f;
    return t;
}

// the pinned kit's _rmsnorm reduction (tf_ds_kit_*'s documented order): D == 512 first reduces four adjacent channels
// per logical lane in the squares4 fma order, D == 128 squares its one channel; then the xor-sum tree of each warp,
// the (w0 + w2) + (w1 + w3) combine, mean + eps fma, the approximate root and reciprocal, and the bf16 store.
static void ref_kit_norm(const float* x, int D, float eps, const __nv_bfloat16* w, __nv_bfloat16* dst) {
    const int lanes = D == 512 ? 128 : 128; // 128 logical lanes at both served widths (512: four channels a lane)
    float lane[128];
    for (int i = 0; i < lanes; ++i) {
        if (D == 512) {
            const int b = i * 4;
            float sum = x[b + 1] * x[b + 1];
            sum = std::fmaf(x[b + 0], x[b + 0], sum);
            sum = std::fmaf(x[b + 2], x[b + 2], sum);
            sum = std::fmaf(x[b + 3], x[b + 3], sum);
            lane[i] = sum;
        } else {
            lane[i] = x[i] * x[i];
        }
    }
    // the warp xor tree (one warp = 32 lanes; both widths are four warps of logical lanes)
    float warp_sum[4];
    for (int w = 0; w < 4; ++w) {
        float s[32];
        for (int l = 0; l < 32; ++l) s[l] = lane[w * 32 + l];
        for (int off = 16; off > 0; off >>= 1) {
            float t[32];
            for (int l = 0; l < 32; ++l) t[l] = s[l] + s[l ^ off];
            memcpy(s, t, sizeof s);
        }
        warp_sum[w] = s[0];
    }
    const float total = (warp_sum[0] + warp_sum[2]) + (warp_sum[1] + warp_sum[3]);
    const float mean_eps = std::fmaf(total, 1.0f / D, eps);
    const float inv = kit_div_approx(1.0f, kit_sqrt_approx(mean_eps));
    for (int t = 0; t < D; ++t) {
        const float y = x[t] * inv;
        dst[t] = __float2bfloat16_rn(__bfloat162float(w[t]) * y);
    }
}

// -- case construction (setGlue's documented gi/rslot rule: rslot = slot*RAW + pos % RAW; gi = slot*RAW +
// (2*(pos/2) + [0,1]) % RAW) ----------------------------------------------------------------------------------------------------------------

static const long long RAW = 64;

struct RowSpec {
    long long slot, pos;
};

struct Case {
    std::string name;
    int D;
    std::vector<RowSpec> rows;
    long long slots;       // the ring's streams (slot ids must be < slots)
    int bad_partner = -1;  // this row's gi[0] forced out of range (the invalid flag's zero-substitution path)
};

struct Buffers {
    std::vector<float> ckv, cgate, ring_kv, ring_score;
    std::vector<__nv_bfloat16> w, out;
    std::vector<long long> rslot, gi;
    unsigned invalid;
    long long src_rows, ldo;
    int G, D;
};

static uint64_t rng_state;
static uint64_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void build(const Case& cs, Buffers& b, uint64_t seed) {
    rng_state = seed;
    b.G = (int)cs.rows.size();
    b.D = cs.D;
    b.src_rows = cs.slots * RAW;
    b.ldo = cs.D + 8; // stride padding the kernels must not touch
    b.ckv.assign((size_t)b.G * cs.D, 0);
    b.cgate.assign((size_t)b.G * cs.D, 0);
    b.ring_kv.assign((size_t)b.src_rows * cs.D, 0);
    b.ring_score.assign((size_t)b.src_rows * cs.D, 0);
    b.w.resize(cs.D);
    b.out.assign((size_t)b.G * b.ldo, __nv_bfloat16{0xABCD});
    b.rslot.resize(b.G);
    b.gi.resize((size_t)b.G * 2);
    b.invalid = 0;
    for (int g = 0; g < b.G; ++g) {
        for (int t = 0; t < cs.D; ++t) {
            b.ckv[(size_t)g * cs.D + t] = (float)((rnd() % 8001) - 4000) / 2000.f;
            b.cgate[(size_t)g * cs.D + t] = (float)((rnd() % 12001) - 6000) / 2000.f;
        }
    }
    // the ring keeps an earlier round's bytes (a distinct range: ring reads are then unmistakable)
    for (size_t i = 0; i < b.ring_kv.size(); ++i) {
        b.ring_kv[i] = -5.f + (float)(rnd() % 1001) / 500.f;
        b.ring_score[i] = -4.f + (float)(rnd() % 1001) / 500.f;
    }
    for (int t = 0; t < cs.D; ++t) b.w[t] = __float2bfloat16_rn(0.5f + (float)(rnd() % 1001) / 1000.f);
    for (int g = 0; g < b.G; ++g) {
        const long long s = cs.rows[g].slot, p = cs.rows[g].pos;
        b.rslot[g] = s * RAW + p % RAW;
        const long long groups = p / 2;
        b.gi[2 * g] = s * RAW + (groups * 2) % RAW;
        b.gi[2 * g + 1] = s * RAW + (groups * 2 + 1) % RAW;
    }
    if (cs.bad_partner >= 0) b.gi[2 * cs.bad_partner] = b.src_rows + 5; // out of range: zero row + invalid flag
}

// -- the independent per-row reference over a case's buffers ---------------------------------------------------------------------------------

static void ref_case(Buffers& b, float eps) {
    for (int g = 0; g < b.G; ++g) {
        const long long own = b.rslot[g];
        const long long q0 = b.gi[2 * g], q1 = b.gi[2 * g + 1];
        // every input row writes its own raw-ring slot
        for (int t = 0; t < b.D; ++t) {
            b.ring_kv[own * b.D + t] = b.ckv[(size_t)g * b.D + t];
            b.ring_score[own * b.D + t] = b.cgate[(size_t)g * b.D + t];
        }
        const bool ok0 = q0 >= 0 && q0 < b.src_rows, ok1 = q1 >= 0 && q1 < b.src_rows;
        if (!ok0 || !ok1) b.invalid = 1;
        // resolve each partner: the first current row naming the slot, else the ring, else (out of range) a zero row
        std::vector<float> kv0(b.D), kv1(b.D), sc0(b.D), sc1(b.D);
        for (int side = 0; side < 2; ++side) {
            const long long q = side == 0 ? q0 : q1;
            const bool ok = side == 0 ? ok0 : ok1;
            std::vector<float>& kvo = side == 0 ? kv0 : kv1;
            std::vector<float>& sco = side == 0 ? sc0 : sc1;
            if (!ok) continue; // zeros
            int row = -1;
            for (int j = 0; j < b.G; ++j)
                if (b.rslot[j] == q) { row = j; break; }
            for (int t = 0; t < b.D; ++t) {
                if (row >= 0) { // the current row's bytes (the kernel's own-row arm reads the same element)
                    kvo[t] = b.ckv[(size_t)row * b.D + t];
                    sco[t] = b.cgate[(size_t)row * b.D + t];
                } else {
                    kvo[t] = b.ring_kv[q * b.D + t];
                    sco[t] = b.ring_score[q * b.D + t];
                }
            }
        }
        std::vector<float> x(b.D);
        for (int t = 0; t < b.D; ++t) {
            const float value = ref_pair(sc0[t], sc1[t], kv0[t], kv1[t]);
            x[t] = __bfloat162float(__float2bfloat16_rn(value));
        }
        ref_kit_norm(x.data(), b.D, eps, b.w.data(), &b.out[(size_t)g * b.ldo]);
    }
}

// -- kernel runners (one block at a time: no kernel here lets one block's writes feed another block's reads) ----------------

template <class F>
static void run_blocks(int G, int D, F&& fn) {
    static SimBlock blocks[64];
    for (int g = 0; g < G; ++g) {
        SimBlock& sb = blocks[g];
        sb.clear(D);
        std::vector<std::thread> ts;
        ts.reserve(D);
        for (int t = 0; t < D; ++t)
            ts.emplace_back([&, t] {
                blk = &sb;
                blockIdx = {(unsigned)g, 0, 0};
                threadIdx = {(unsigned)t, 0, 0};
                blockDim = {(unsigned)D, 0, 0};
                gridDim = {(unsigned)G, 0, 0};
                fn();
            });
        for (auto& th : ts) th.join();
    }
}

static void run_old(Buffers& b, float eps, int pdl) {
    run_blocks(b.G, b.D, [&] {
        if (b.D == 128)
            tf_ds_compress2x_norm128_kernel(b.ckv.data(), b.D, b.cgate.data(), b.D, b.rslot.data(), b.ring_kv.data(),
                                            b.ring_score.data(), b.gi.data(), b.src_rows, b.w.data(), b.out.data(),
                                            b.ldo, b.G, b.D, eps, &b.invalid, pdl);
        else
            tf_ds_compress2x_norm512_kernel(b.ckv.data(), b.D, b.cgate.data(), b.D, b.rslot.data(), b.ring_kv.data(),
                                            b.ring_score.data(), b.gi.data(), b.src_rows, b.w.data(), b.out.data(),
                                            b.ldo, b.G, b.D, eps, &b.invalid, pdl);
    });
}

static void run_new(Buffers& b, float eps, int pdl) {
    run_blocks(b.G, b.D, [&] {
        if (b.D == 128)
            tf_ds_compress2x_norm128_pair_kernel(b.ckv.data(), b.D, b.cgate.data(), b.D, b.rslot.data(), b.ring_kv.data(),
                                                 b.ring_score.data(), b.gi.data(), b.src_rows, b.w.data(), b.out.data(),
                                                 b.ldo, b.G, b.D, eps, &b.invalid, pdl);
        else
            tf_ds_compress2x_norm512_pair_kernel(b.ckv.data(), b.D, b.cgate.data(), b.D, b.rslot.data(), b.ring_kv.data(),
                                                 b.ring_score.data(), b.gi.data(), b.src_rows, b.w.data(), b.out.data(),
                                                 b.ldo, b.G, b.D, eps, &b.invalid, pdl);
    });
}

// -- comparison and reporting ------------------------------------------------------------------------------------------

static int failures = 0;

static void compare(const std::string& name, const char* what, const void* a, const void* bb, size_t n) {
    if (memcmp(a, bb, n) != 0) {
        const unsigned char* x = (const unsigned char*)a;
        const unsigned char* y = (const unsigned char*)bb;
        for (size_t i = 0; i < n; ++i)
            if (x[i] != y[i]) {
                printf("FAIL %s: %s differs at byte %zu (%02x vs %02x)\n", name.c_str(), what, i, x[i], y[i]);
                break;
            }
        ++failures;
    }
}

static int cases_run = 0;

static void run_case(const Case& cs, uint64_t seed) {
    Buffers bn, bo, br;
    build(cs, bn, seed);
    bo = bn;
    br = bn;
    const float eps = 1e-20f;
    run_new(bn, eps, 0);
    run_old(bo, eps, 0);
    ref_case(br, eps);
    compare(cs.name, "new-vs-off raw ring kv", bn.ring_kv.data(), bo.ring_kv.data(), bn.ring_kv.size() * 4);
    compare(cs.name, "new-vs-off raw ring score", bn.ring_score.data(), bo.ring_score.data(), bn.ring_score.size() * 4);
    compare(cs.name, "new-vs-off latent (incl. stride padding)", bn.out.data(), bo.out.data(), bn.out.size() * 2);
    compare(cs.name, "new-vs-off invalid flag", &bn.invalid, &bo.invalid, 4);
    compare(cs.name, "new-vs-ref raw ring kv", bn.ring_kv.data(), br.ring_kv.data(), bn.ring_kv.size() * 4);
    compare(cs.name, "new-vs-ref raw ring score", bn.ring_score.data(), br.ring_score.data(), bn.ring_score.size() * 4);
    compare(cs.name, "new-vs-ref latent (incl. stride padding)", bn.out.data(), br.out.data(), bn.out.size() * 2);
    compare(cs.name, "new-vs-ref invalid flag", &bn.invalid, &br.invalid, 4);
    // every input row's own raw-ring slot holds exactly that row's bytes
    for (int g = 0; g < bn.G; ++g) {
        const long long own = bn.rslot[g];
        compare(cs.name, "own raw slot kv", &bn.ring_kv[own * bn.D], &bn.ckv[(size_t)g * bn.D], bn.D * 4);
        compare(cs.name, "own raw slot score", &bn.ring_score[own * bn.D], &bn.cgate[(size_t)g * bn.D], bn.D * 4);
    }
    ++cases_run;
}

// the distinct pair keys (pair computations) per window, for the report's savings table
static int distinct_keys(const std::vector<RowSpec>& rows) {
    std::vector<std::pair<long long, long long>> keys;
    for (auto& r : rows) {
        const long long groups = r.pos / 2;
        std::pair<long long, long long> k = {r.slot * RAW + (groups * 2) % RAW, r.slot * RAW + (groups * 2 + 1) % RAW};
        bool seen = false;
        for (auto& e : keys)
            if (e == k) { seen = true; break; }
        if (!seen) keys.push_back(k);
    }
    return (int)keys.size();
}

int main() {
    // the served widths
    for (int D : {128, 512}) {
        char buf[128];
        // A. one request's contiguous window, n = 1..7 rows, both starting parities
        for (int n = 1; n <= 7; ++n)
            for (long long start : {0LL, 1LL}) {
                Case cs;
                cs.D = D;
                cs.slots = 4;
                for (int i = 0; i < n; ++i) cs.rows.push_back({0, start + i});
                snprintf(buf, sizeof buf, "D=%d single n=%d start=%lld", D, n, start);
                cs.name = buf;
                run_case(cs, 0x5eed + n + start);
            }
        // B. ring wrap and a partner from an earlier round (odd starts leave the first pair's even row to the ring)
        for (long long start : {62LL, 63LL}) {
            for (int n : {2, 6, 7}) {
                Case cs;
                cs.D = D;
                cs.slots = 4;
                for (int i = 0; i < n; ++i) cs.rows.push_back({2, start + i});
                snprintf(buf, sizeof buf, "D=%d wrap n=%d start=%lld", D, n, start);
                cs.name = buf;
                run_case(cs, 0x600d + n + start);
            }
        }
        // C. mixed requests: 2-4 streams a launch, distinct slots, odd/even and wrapping starts
        {
            Case cs;
            cs.D = D;
            cs.slots = 4;
            for (int i = 0; i < 5; ++i) cs.rows.push_back({0, 0 + i});   // even start
            for (int i = 0; i < 5; ++i) cs.rows.push_back({1, 1 + i});   // odd start
            cs.name = "D=" + std::to_string(D) + " mixed 2x5";
            run_case(cs, 0x21);
        }
        {
            Case cs;
            cs.D = D;
            cs.slots = 4;
            for (int i = 0; i < 6; ++i) cs.rows.push_back({0, 62 + i}); // wrapping
            for (int i = 0; i < 6; ++i) cs.rows.push_back({1, 63 + i}); // wrapping, odd
            for (int i = 0; i < 6; ++i) cs.rows.push_back({2, 3 + i});
            cs.name = "D=" + std::to_string(D) + " mixed 3x6";
            run_case(cs, 0x22);
        }
        {
            Case cs;
            cs.D = D;
            cs.slots = 4;
            for (int s = 0; s < 4; ++s)
                for (int i = 0; i < 7; ++i) cs.rows.push_back({(long long)s, (long long)(s % 2) + i});
            cs.name = "D=" + std::to_string(D) + " mixed 4x7";
            run_case(cs, 0x23);
        }
        // C2. past the first ballot chunk: 33-48 rows (max_rows) with duplicate keys at rows 32+, single and mixed
        for (long long start : {0LL, 1LL, 62LL}) {
            for (int n : {33, 48}) {
                Case cs;
                cs.D = D;
                cs.slots = 4;
                for (int i = 0; i < n; ++i) cs.rows.push_back({0, start + i});
                snprintf(buf, sizeof buf, "D=%d long n=%d start=%lld", D, n, start);
                cs.name = buf;
                run_case(cs, 0x1000 + n + start);
            }
        }
        {
            Case cs;
            cs.D = D;
            cs.slots = 4;
            for (int s = 0; s < 4; ++s)
                for (int i = 0; i < 9; ++i) cs.rows.push_back({(long long)s, (long long)(s % 2) + i});
            cs.name = "D=" + std::to_string(D) + " mixed 4x9";
            run_case(cs, 0x1005);
        }
        // D. an out-of-range partner (the invalid flag's zero substitution) at both parities
        for (long long start : {0LL, 1LL}) {
            Case cs;
            cs.D = D;
            cs.slots = 4;
            for (int i = 0; i < 4; ++i) cs.rows.push_back({0, start + i});
            cs.bad_partner = (int)start; // a member row of the first key (and the first row at start=0)
            snprintf(buf, sizeof buf, "D=%d bad-partner start=%lld", D, start);
            cs.name = buf;
            run_case(cs, 0xbad + start);
        }
        // E. randomized rounds: 1-4 streams, 1-7 rows a stream, random start positions (wrap variety)
        for (int k = 0; k < 60; ++k) {
            rng_state = 0xC0FFEE + k * 17 + D;
            Case cs;
            cs.D = D;
            cs.slots = 4;
            int S = 1 + (int)(rnd() % 4);
            for (int s = 0; s < S; ++s) {
                long long slot = (s * 3 + k) % cs.slots; // distinct slots a stream: rows of different requests never share
                long long start = (long long)(rnd() % 128);
                int n = 1 + (int)(rnd() % 7);
                for (int i = 0; i < n; ++i) cs.rows.push_back({slot, start + i});
            }
            if (rnd() % 10 == 0) cs.bad_partner = (int)(rnd() % cs.rows.size());
            cs.name = "D=" + std::to_string(D) + " rand " + std::to_string(k);
            run_case(cs, 0xC0FFEE + k * 17 + D);
        }
    }

    // the savings table (pair computations the pair-once kernel skips)
    printf("\npair computations per window (rows -> distinct pairs, saved):\n");
    for (int n = 5; n <= 7; ++n)
        for (long long start : {0LL, 1LL}) {
            std::vector<RowSpec> rows;
            for (int i = 0; i < n; ++i) rows.push_back({0, start + i});
            printf("  n=%d start=%lld: %d -> %d (saved %d)\n", n, start, n, distinct_keys(rows), n - distinct_keys(rows));
        }
    {
        long long lo = 100, hi = 0, tot_lo = 100, tot_hi = 0, s_lo = 100, s_hi = 0;
        // every combination of four windows of 5..7 rows at odd/even starts
        for (int c = 0; c < 3 * 3 * 3 * 3 * 16; ++c) {
            std::vector<RowSpec> rows;
            int tot = 0;
            int q = c;
            for (int s = 0; s < 4; ++s) {
                int n = 5 + q % 3;
                q /= 3;
                long long start = (q & 1) ? 1 : 0;
                q >>= 1;
                for (int i = 0; i < n; ++i) rows.push_back({(long long)s, start + i});
                tot += n;
            }
            int d = distinct_keys(rows);
            tot_lo = tot < tot_lo ? tot : tot_lo;
            tot_hi = tot > tot_hi ? tot : tot_hi;
            lo = d < lo ? d : lo;
            hi = d > hi ? d : hi;
            long long saved = tot - d;
            s_lo = saved < s_lo ? saved : s_lo;
            s_hi = saved > s_hi ? saved : s_hi;
        }
        printf("  4x(5..7)-row round (all 1296 combinations): %lld..%lld rows -> %lld..%lld distinct (saved %lld..%lld)\n",
               tot_lo, tot_hi, lo, hi, s_lo, s_hi);
    }

    printf("\n%d cases, %d failures\n", cases_run, failures);
    if (failures == 0) printf("h0266: all match\n");
    return failures == 0 ? 0 : 1;
}
