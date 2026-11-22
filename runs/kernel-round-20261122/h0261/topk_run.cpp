// CPU harness for TF_DS_TOPK_MERGE. Executes the exact source (kernels.inc, extracted from dsv41_ops.cu) of
// tf_ds_topk_select_kernel, tf_ds_topk_merge_kernel and tf_ds_draw_rows_topk_kernel, plus the untouched full-row
// tf_ds_draw_rows_kernel (and tf_ds_block_argmax) as the served path, one std::thread per CUDA thread, a std::barrier
// per block for __syncthreads and one per warp for the shuffles. Checks:
//   (a) each rank's list against an independent per-shard sort by the draw key (+ rank 0's row[0] NaN flag);
//   (b) the merged list against a sort of the whole concatenated row;
//   (c) the merged path's picks against the served full-row kernel's picks on the concatenated rows (every row), and
//       against a direct C++ transcription of sampling.zig's sampleRow / argmax (rows without NaN).
// Checks this C source's logic in IEEE fp64/fp32 with the host's libm on both sides; not CUDA codegen or device libm.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cfloat>
#include <memory>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static   // blocks run one at a time: function statics are the block's shared memory
struct dim3 { unsigned x, y, z; };
static thread_local dim3 threadIdx, blockIdx;
static dim3 blockDim, gridDim;
static std::mutex mu;
static std::barrier<>* g_bar = nullptr;
static std::barrier<>* g_warp[32];
static uint64_t g_slot[32][32];
static inline void __syncthreads() { g_bar->arrive_and_wait(); }
template <typename T> static inline T __shfl_down_sync(unsigned, T v, int off) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    uint64_t b = 0;
    memcpy(&b, &v, sizeof(T));
    g_slot[w][l] = b;
    g_warp[w]->arrive_and_wait();
    T out = v;
    if (l + off < 32) memcpy(&out, &g_slot[w][l + off], sizeof(T));
    g_warp[w]->arrive_and_wait();
    return out;
}
static inline unsigned atomicAdd(unsigned* p, unsigned v) { std::lock_guard g(mu); unsigned r = *p; *p += v; return r; }
static inline unsigned __float_as_uint(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
static inline float __uint_as_float(unsigned v) { float f; memcpy(&f, &v, 4); return f; }
static inline float __int_as_float(int v) { float f; memcpy(&f, &v, 4); return f; }
static inline double __longlong_as_double(long long v) { double d; memcpy(&d, &v, 8); return d; }
static inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
static inline void pdl_enter(int) {}
#include "kernels_host.inc"  // kernels.inc with `extern "C" ` stripped (build.sh)

// Run a kernel body over a grid: each block's threads as std::threads with a fresh block barrier.
static void launch(unsigned grid, unsigned block, const std::function<void()>& body) {
    gridDim = {grid, 1, 1};
    blockDim = {block, 1, 1};
    for (unsigned b = 0; b < grid; ++b) {
        std::barrier<> bar(block);
        std::vector<std::unique_ptr<std::barrier<>>> warps;
        for (unsigned w = 0; w < (block + 31) / 32; ++w) {
            warps.emplace_back(new std::barrier<>(std::min(32u, block - 32 * w)));
            g_warp[w] = warps.back().get();
        }
        g_bar = &bar;
        std::vector<std::thread> ts;
        ts.reserve(block);
        for (unsigned t = 0; t < block; ++t)
            ts.emplace_back([=, &body] { threadIdx = {t, 0, 0}; blockIdx = {b, 0, 0}; body(); });
        for (auto& t : ts) t.join();
    }
}

// ---- independent references ----------------------------------------------------------------------------------------
static uint64_t ref_key(float v, uint32_t id) {
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7fffffffu) == 0) b = 0;                      // -0 as +0
    uint32_t hi = std::isnan(v) ? 0u : (b >> 31 ? ~b : (b | 0x80000000u));
    return (uint64_t(hi) << 32) | (0xFFFFFFFFu - id);
}
struct P { uint32_t vb, id; };
static std::vector<P> ref_sorted(const float* row, int n, uint32_t col0, int k) {
    std::vector<P> all(n);
    for (int i = 0; i < n; ++i) all[i] = {__float_as_uint(row[i]), col0 + (uint32_t)i};
    std::sort(all.begin(), all.end(), [](P a, P b) { return ref_key(__uint_as_float(a.vb), a.id) > ref_key(__uint_as_float(b.vb), b.id); });
    all.resize(std::min(k, n));
    return all;
}
// sampling.zig, transcribed: argmax (first largest; rows without NaN), topCandidates (value desc, id asc), served()
// and sampleRow
struct Rule { uint64_t seed; double temp, top_p, min_p; uint64_t pos; uint32_t top_k, pad; };
static uint64_t mix(uint64_t x) { x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; return x ^ (x >> 31); }
static double uniform_(uint64_t seed, uint64_t pos, uint64_t id) {
    uint64_t x = mix(seed + 0x9E3779B97F4A7C15ull); x = mix(x ^ (pos * 0xD1B54A32D192ED03ull)); x = mix(x ^ id);
    return double(x >> 11) * 0x1p-53 + 0x1p-54;
}
static double pairwise(const double* a, size_t n) {
    if (n < 8) { double r = 0; for (size_t i = 0; i < n; ++i) r += a[i]; return r; }
    double r[8]; for (int j = 0; j < 8; ++j) r[j] = a[j];
    size_t i = 8; for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}
static int64_t sample_row(const std::vector<float>& row, const Rule& s) {
    if (s.temp <= 0) { size_t b = 0; for (size_t i = 0; i < row.size(); ++i) if (row[i] > row[b]) b = i; return (int64_t)b; }
    const size_t count = std::min(row.size(), (size_t)s.top_k + 8);
    std::vector<std::pair<float, int64_t>> c;
    for (size_t i = 0; i < row.size(); ++i) c.push_back({row[i], (int64_t)i});
    std::stable_sort(c.begin(), c.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    c.resize(count);
    const size_t k = std::max<size_t>(1, std::min<size_t>(s.top_k, c.size()));
    const double t = std::max(s.temp, 1e-6);
    double scaled[64], score[64]; bool keep[64];
    for (size_t j = 0; j < k; ++j) { scaled[j] = double(c[j].first) / t; keep[j] = true; }
    if (0.0 < s.top_p && s.top_p < 1.0) {
        double mx = scaled[0]; for (size_t j = 1; j < k; ++j) mx = std::max(mx, scaled[j]);
        double probs[64]; for (size_t j = 0; j < k; ++j) probs[j] = std::exp(scaled[j] - mx);
        const double sum = pairwise(probs, k);
        for (size_t j = 0; j < k; ++j) probs[j] /= sum;
        double acc = 0; size_t below = 0;
        for (size_t j = 0; j < k; ++j) { acc += probs[j]; if (acc < s.top_p) ++below; }
        for (size_t j = below + 1; j < k; ++j) keep[j] = false;
    }
    if (s.min_p > 0.0) { const double fl = scaled[0] + std::log(s.min_p); for (size_t j = 0; j < k; ++j) if (scaled[j] < fl) keep[j] = false; }
    for (size_t j = 0; j < k; ++j) score[j] = keep[j] ? scaled[j] - std::log(-std::log(uniform_(s.seed, s.pos, (uint64_t)c[j].second))) : -INFINITY;
    size_t best = 0; for (size_t j = 0; j < k; ++j) if (score[j] > score[best]) best = j;
    return c[best].second;
}

// ---- one round ------------------------------------------------------------------------------------------------------
static int fails = 0, cases = 0, picks = 0;
static void fail(const char* what, int c, int r) { ++fails; if (fails < 30) printf("FAIL case %d row %d: %s\n", c, r, what); }

static void round_case(int c, int hn, const std::vector<std::vector<float>>& rows, const std::vector<Rule>& rules, bool nanfree) {
    const int R = (int)rows.size();
    int kmax = 0;
    for (auto& r : rules) if (r.temp > 0) kmax = std::max<int>(kmax, r.top_k);
    if (kmax == 0) return;  // a greedy-only round never takes this path
    ++cases;
    // each rank's head columns: its half of every row ([R, hn], stride hn)
    std::vector<unsigned> pairs((size_t)2 * R * kmax * 2, 0xdeadbeefu);
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float> hl((size_t)R * hn);
        for (int r = 0; r < R; ++r) memcpy(&hl[(size_t)r * hn], &rows[r][(size_t)rank * hn], hn * 4);
        unsigned* out = pairs.data() + (size_t)rank * R * kmax * 2;  // the all-gather's rank-major layout
        launch(R, 1024, [&] { tf_ds_topk_select_kernel(hl.data(), hn, hn, (unsigned)(rank * hn), kmax, out, 0); });
        for (int r = 0; r < R; ++r) {  // (a)
            auto want = ref_sorted(&hl[(size_t)r * hn], hn, rank * hn, kmax);
            const unsigned* o = out + (size_t)r * kmax * 2;
            for (int j = 0; j < kmax; ++j) {
                const unsigned flag = (j == 0 && std::isnan(hl[(size_t)r * hn])) ? 0x80000000u : 0u;
                if (o[2 * j] != want[j].vb || o[2 * j + 1] != (want[j].id | flag)) { fail("select list", c, r); break; }
            }
        }
    }
    std::vector<unsigned> merged((size_t)R * kmax * 2, 0xdeadbeefu);
    launch((R + 63) / 64, 64, [&] { tf_ds_topk_merge_kernel(pairs.data(), R, kmax, merged.data(), 0); });
    for (int r = 0; r < R; ++r) {  // (b)
        auto want = ref_sorted(rows[r].data(), 2 * hn, 0, kmax);
        const unsigned* m = merged.data() + (size_t)r * kmax * 2;
        for (int j = 0; j < kmax; ++j) {
            const unsigned flag = (j == 0 && std::isnan(rows[r][0])) ? 0x80000000u : 0u;
            if (m[2 * j] != want[j].vb || m[2 * j + 1] != (want[j].id | flag)) { fail("merged list", c, r); break; }
        }
    }
    std::vector<unsigned> got(R, 0xdeadbeefu), full(R, 0xdeadbeefu);
    launch(R, 64, [&] { tf_ds_draw_rows_topk_kernel(merged.data(), kmax, (const TfDsDrawRow*)rules.data(), got.data(), 0); });
    // the served path: the full rows (the ranks' halves in rank order) through the untouched kernel
    std::vector<float> logits((size_t)R * 2 * hn);
    for (int r = 0; r < R; ++r) memcpy(&logits[(size_t)r * 2 * hn], rows[r].data(), (size_t)2 * hn * 4);
    launch(R, 1024, [&] { tf_ds_draw_rows_kernel(logits.data(), 2 * hn, 2 * hn, (const TfDsDrawRow*)rules.data(), full.data(), 0); });
    for (int r = 0; r < R; ++r) {  // (c)
        ++picks;
        if (got[r] != full[r]) fail("pick vs full-row tf_ds_draw_rows_kernel", c, r);
        if (nanfree && (int64_t)got[r] != sample_row(rows[r], rules[r])) fail("pick vs sampleRow transcription", c, r);
    }
}

int main() {
    static_assert(sizeof(Rule) == 48, "TfDsDrawRow layout");
    std::mt19937_64 rng(0x0261);
    auto unif = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    auto grid_row = [&](int n) {  // sampling.testRow's shape: a 1/8 grid (many ties), a few outliers
        std::vector<float> row(n);
        std::normal_distribution<float> nd(0, 1);
        for (auto& v : row) v = std::round(nd(rng) * 8.0f) / 8.0f;
        for (int i = 0; i < 24; ++i) row[rng() % n] = 4.0f + std::round(unif(0, 1) * 16.0f) / 8.0f;
        return row;
    };
    auto rule = [&](bool greedy, int top_k) {
        const double tps[] = {1.0, 0.95, 0.5, 0.0};
        Rule r{rng(), greedy ? 0.0 : unif(0.3, 1.5), tps[rng() % 4], (rng() % 3 == 0) ? 0.05 : 0.0, rng() % (1 << 20), (uint32_t)top_k, 0};
        return r;
    };
    int c = 0;
    // 1) fuzz rounds: k = 1..56, mixed greedy/sampled rows (top_k < kmax rows exercise the prefix rule)
    for (int k = 1; k <= 56; ++k, ++c) {
        const int hn = 96 + (int)(rng() % 400), R = 1 + (int)(rng() % 5);
        std::vector<std::vector<float>> rows; std::vector<Rule> rules;
        for (int r = 0; r < R; ++r) { rows.push_back(grid_row(2 * hn)); rules.push_back(rule(r == 1, r == 0 ? k : 1 + (int)(rng() % k))); }
        round_case(c, hn, rows, rules, true);
    }
    // 2) cross-rank ties at the cutoff: kmax-1 winners on rank 1, a tie block straddling the cutoff on both ranks
    for (int k : {1, 2, 7, 20, 33, 56}) for (int variant = 0; variant < 2; ++variant, ++c) {
        const int hn = 128;
        std::vector<float> row(2 * hn, -1.0f);
        for (int j = 0; j < k - 1; ++j) row[hn + j] = 10.0f + j;
        for (int j = 0; j < 8; ++j) { row[hn - 1 - j] = 3.0f; row[hn + 64 + j] = 3.0f; }
        if (variant) std::fill(row.begin(), row.end(), 0.5f);  // every value equal: ids 0..k-1, all rank 0's
        std::vector<std::vector<float>> rows{row, row}; std::vector<Rule> rules{rule(false, k), rule(true, 0)};
        round_case(c, hn, rows, rules, true);
    }
    // 3) specials: +-0, +-inf, NaN (incl. rank 0's row[0], rank 1's first column, all-NaN rows), max/min, subnormal
    const float sp[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, FLT_MAX, -FLT_MAX, 1e-45f};
    for (int i = 0; i < 40; ++i, ++c) {
        const int hn = 64 + (int)(rng() % 64), R = 2 + (int)(rng() % 3);
        std::vector<std::vector<float>> rows; std::vector<Rule> rules;
        bool nanfree = true;
        for (int r = 0; r < R; ++r) {
            std::vector<float> row(2 * hn);
            for (auto& v : row) v = (rng() & 1) ? sp[rng() % 10] : std::round(unif(-4, 4) * 4) / 4;
            if (i % 4 == 0) row[0] = NAN;
            if (i % 4 == 1) row[hn] = NAN;
            if (i % 5 == 2 && r == 0) std::fill(row.begin(), row.end(), NAN);
            if (i % 7 == 3) for (auto& v : row) v = (rng() & 1) ? 0.0f : -0.0f;
            for (float v : row) if (std::isnan(v)) nanfree = false;
            rows.push_back(row);
            rules.push_back(rule(r % 2 == 1, 1 + (int)(rng() % 56)));
        }
        rules[0].temp = 0.9;  // at least one sampled row
        round_case(c, hn, rows, rules, nanfree);
    }
    // 4) the served width: hn = 64,640 a rank (vocabulary 129,280), a greedy and a sampled row, k = 56 and 20
    for (int k : {56, 20}) {
        std::vector<std::vector<float>> rows{grid_row(129280), grid_row(129280)};
        std::vector<Rule> rules{rule(false, k), rule(true, 0)};
        round_case(c++, 64640, rows, rules, true);
    }
    printf("%d rounds, %d picks, %d failures\n", cases, picks, fails);
    if (fails) return 1;
    printf("h0261: all match\n");
}
