// CPU harness: executes tf_ds_conf_pack_kernel's exact source (kernel.inc, extracted from dsv41_ops.cu) with a
// per-thread loop (the kernel has no barriers or shared memory) and compares conf_in bytes, the invalid flag and the
// resulting embs bytes with an independent C++ transcription of the served chain (draft.zig body: gatherRows +
// toF32 a stream, toF32 of the gathered rows, two copyRows). Checks the C source's indexing and rules, not codegen.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
struct __nv_bfloat16 { uint16_t v; };
static inline float __bfloat162float(__nv_bfloat16 h) { uint32_t u = uint32_t(h.v) << 16; float f; memcpy(&f, &u, 4); return f; }
static inline unsigned atomicExch(unsigned* a, unsigned b) { unsigned o = *a; *a = b; return o; }
static inline void pdl_enter(int) {}
#define TF_DS_KERNEL_UNDER_TEST
#include "kernel.inc"

struct Dims { int N, steps, n, d, rank; uint64_t vocab; };

static void run_kernel(const Dims& m, const std::vector<uint16_t>& xc, const std::vector<uint16_t>& embed,
                       const std::vector<int64_t>& ids, std::vector<uint16_t>& embs, std::vector<float>& conf, uint32_t& inv) {
    gridDim = {unsigned(m.N * m.steps), 1, 1};
    blockDim = {256, 1, 1};
    for (unsigned b = 0; b < gridDim.x; ++b)
        for (unsigned t = 0; t < blockDim.x; ++t) {
            blockIdx = {b, 0, 0};
            threadIdx = {t, 0, 0};
            tf_ds_conf_pack_kernel((const __nv_bfloat16*)xc.data(), (const __nv_bfloat16*)embed.data(), ids.data(),
                                   (__nv_bfloat16*)embs.data(), conf.data(), &inv, m.n, m.steps, m.d, m.rank,
                                   (unsigned long long)m.vocab, 0);
        }
}

// the reference: the chain's launches in order, written from the brief's semantics
static void run_chain(const Dims& m, const std::vector<uint16_t>& xc, const std::vector<uint16_t>& embed,
                      const std::vector<int64_t>& ids, std::vector<uint16_t>& embs, std::vector<float>& conf, uint32_t& inv) {
    const int ts = m.steps + 1, w = m.d + m.rank;
    std::vector<float> xcf((size_t)m.N * m.steps * m.d), ef((size_t)m.N * m.steps * m.rank);
    auto f32 = [](uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; };
    for (int i = 0; i < m.N; ++i) {
        // gatherRows(markov_embed, vocab, out + i*ts*8, embs + i*steps*rank*2, rank*2, steps, invalid): byte rows
        const uint8_t* src = (const uint8_t*)embed.data();
        uint8_t* dst = (uint8_t*)embs.data() + (size_t)i * m.steps * m.rank * 2;
        for (int r = 0; r < m.steps; ++r) {
            const int64_t p = ids[(size_t)i * ts + r];
            if (p < 0 || uint64_t(p) >= m.vocab) { inv = 1; continue; }
            memcpy(dst + (size_t)r * m.rank * 2, src + (size_t)p * m.rank * 2, (size_t)m.rank * 2);
        }
        // toF32(xc + i*n*d*2, xcf + i*steps*d*4, steps*d)
        for (size_t k = 0; k < (size_t)m.steps * m.d; ++k) xcf[(size_t)i * m.steps * m.d + k] = f32(xc[(size_t)i * m.n * m.d + k]);
    }
    for (size_t k = 0; k < ef.size(); ++k) ef[k] = f32(embs[k]);
    for (int r = 0; r < m.N * m.steps; ++r) {
        memcpy(&conf[(size_t)r * w], &xcf[(size_t)r * m.d], (size_t)m.d * 4);
        memcpy(&conf[(size_t)r * w + m.d], &ef[(size_t)r * m.rank], (size_t)m.rank * 4);
    }
}

static int fails = 0, cases = 0;
struct State { std::vector<uint16_t> embs; std::vector<float> conf; uint32_t inv; };

// one pass on both sides from the same prior state; returns the post state (identical on success)
static void pass(const char* tag, const Dims& m, const std::vector<uint16_t>& xc, const std::vector<uint16_t>& embed,
                 const std::vector<int64_t>& ids, State& ka, State& rb) {
    ka.inv = 0; rb.inv = 0;  // the cuMemsetD8Async before the launch(es)
    run_kernel(m, xc, embed, ids, ka.embs, ka.conf, ka.inv);
    run_chain(m, xc, embed, ids, rb.embs, rb.conf, rb.inv);
    const bool ok = ka.inv == rb.inv && ka.embs == rb.embs &&
                    memcmp(ka.conf.data(), rb.conf.data(), ka.conf.size() * 4) == 0;
    ++cases;
    if (!ok) { ++fails; printf("FAIL %s N=%d steps=%d n=%d d=%d rank=%d inv %u/%u\n", tag, m.N, m.steps, m.n, m.d, m.rank, ka.inv, rb.inv); }
}

static void one(const char* tag, const Dims& m, std::mt19937_64& rng, int invalid_mode) {
    const int ts = m.steps + 1;
    std::vector<uint16_t> xc((size_t)m.N * m.n * m.d), embed(m.vocab * m.rank);
    for (auto& x : xc) x = uint16_t(rng());
    for (auto& x : embed) x = uint16_t(rng());
    std::vector<int64_t> ids((size_t)m.N * ts);
    for (auto& x : ids) x = int64_t(rng() % m.vocab);
    for (int i = 0; i < m.N; ++i) ids[(size_t)i * ts + m.steps] = -5;  // never read
    const int64_t bad[] = {(int64_t)m.vocab, -1, INT64_MIN, INT64_MAX, (int64_t)m.vocab + 1};
    if (invalid_mode) ids[(rng() % m.N) * ts + rng() % m.steps] = bad[rng() % 5];
    if (invalid_mode == 2) for (int i = 0; i < m.N; ++i) ids[(size_t)i * ts] = bad[i % 5];
    State a, b;
    a.embs.resize((size_t)m.N * m.steps * m.rank);
    for (auto& x : a.embs) x = uint16_t(rng());
    b.embs = a.embs;
    a.conf.assign((size_t)m.N * m.steps * (m.d + m.rank), 3.0f);
    b.conf.assign(a.conf.size(), -7.0f);
    pass(tag, m, xc, embed, ids, a, b);
    // a second pass on the same buffers with every read id invalid where the first was valid at one row: stale parity
    std::vector<int64_t> ids2 = ids;
    ids2[(rng() % m.N) * ts + rng() % m.steps] = -1;
    pass(tag, m, xc, embed, ids2, a, b);
}

int main() {
    std::mt19937_64 rng(0x0260);
    // the matrix: streams x depth x block rows (n >= steps) x invalid mode, small synthetic shapes
    for (int N : {1, 2, 4})
        for (int steps : {1, 3, 5, 6})
            for (int extra : {0, 1, 2, 3})
                for (int mode : {0, 1, 2}) {
                    Dims m{N, steps, steps + extra, 24 + 40 * extra, 8 + 4 * (steps % 3), 37};
                    one("matrix", m, rng, mode);
                }
    // widths crossing the 256-thread stride (d not a multiple, d + rank > 256 and > 512)
    for (int d : {255, 256, 257, 300, 511}) one("stride", Dims{2, 5, 6, d, 256 - (d % 7), 64}, rng, 1);
    // the served shape: d = 5120, rank = 256, vocab 129280, block 6 rows for 5 steps, four streams
    one("real", Dims{4, 5, 6, 5120, 256, 129280}, rng, 0);
    one("real", Dims{4, 5, 6, 5120, 256, 129280}, rng, 1);
    one("real", Dims{1, 5, 6, 5120, 256, 129280}, rng, 2);
    printf("%d passes over %d cases, %d failures\n", cases, cases / 2, fails);
    if (fails) return 1;
    printf("h0260: all match\n");
}
