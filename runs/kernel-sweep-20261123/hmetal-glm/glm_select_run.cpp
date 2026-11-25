// hmetal-glm: GLM_INDEX_CUT (unverified/0001-glm-index-select-cut.patch). Runs glm_index_select's text from
// glm_glue.metal, rewritten by metal2c.py, on the CPU (emu/emu.h: a fiber a thread, 1,024 threads a threadgroup, a
// threadgroup a row):
//   old   = the file at 2bb052f
//   cut0  = the patched file with GlmSelectArgs.cut = 0 (the switch off)
//   cut1  = the patched file with cut = 1
// and compares the key lists' bytes (old == cut0, old == cut1) under three lane orders. NOT a Metal compile.
#include "../emu/emu.h"
typedef unsigned uint;
static inline unsigned simd_prefix_exclusive_sum(unsigned v) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    emu::slot[w][l] = v;
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    unsigned s = 0;
    for (int i = 0; i < l; ++i) s += (unsigned)emu::slot[w][i];
    emu::arrive(emu::warp_bar[w], emu::live_warp[w]);
    return s;
}

namespace oldk {
#include "old.inc"
}
namespace newk {
#include "new.inc"
}

struct Args { const float* scores; int* indices; uint p0, top, width, s_stride, i_stride; int which; };
static void body(const void* p) {
    const Args& a = *(const Args*)p;
    if (a.which == 0) {
        const oldk::GlmSelectArgs g{a.p0, a.top, a.width, a.s_stride, a.i_stride};
        oldk::glm_index_select(a.scores, a.indices, g, blockIdx.x, threadIdx.x);
    } else {
        const newk::GlmSelectArgs g{a.p0, a.top, a.width, a.s_stride, a.i_stride, a.which == 2 ? 1u : 0u};
        newk::glm_index_select(a.scores, a.indices, g, blockIdx.x, threadIdx.x);
    }
}

static std::mt19937 g_rng(0x61);
static long g_fail = 0, g_cases = 0, g_launches = 0, g_bytes = 0, g_index = 0, g_shard = 0, g_nshards = 1, g_blocks = 0, g_skipped = 0;
static double g_digits[3];

enum Cls { GAUSS, GRID, EQUAL, NINF, PINF, NANS, ZEROS, STRADDLE, EXACT, SPECIAL, CLS_N };
static const char* cls_name[CLS_N] = {"gauss", "grid-ties", "all-equal", "neg-inf", "pos-inf", "nans", "signed-zeros", "tie-straddles-top", "distinct", "specials"};
static void fill(float* s, uint blocks, Cls c, uint top) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity();
    const float sp[] = {0.f, -0.f, inf, -inf, __uint_as_float(0x7fc00000u), __uint_as_float(0xffc00001u), 1.f, -1.f, std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()};
    for (uint b = 0; b < blocks; ++b) {
        switch (c) {
            case GAUSS: s[b] = nd(g_rng); break;
            case GRID: s[b] = std::round(nd(g_rng) * 4.f) / 4.f; break;
            case EQUAL: s[b] = 0.25f; break;
            case NINF: s[b] = (g_rng() % 50 == 0) ? nd(g_rng) : -inf; break;
            case PINF: s[b] = (g_rng() % 3 == 0) ? inf : nd(g_rng); break;
            case NANS: s[b] = (g_rng() % 4 == 0) ? __uint_as_float(0x7fc00000u | (g_rng() & 0xffu) | ((g_rng() & 1u) << 31)) : nd(g_rng); break;
            case ZEROS: s[b] = (g_rng() & 1) ? -0.f : 0.f; break;
            case STRADDLE: s[b] = -2.f - (float)(g_rng() % 5); break;
            case EXACT: s[b] = (float)((b * 2654435761u) % 1000003u) * 1e-3f - 400.f; break;
            default: s[b] = (g_rng() & 1) ? sp[g_rng() % (sizeof sp / sizeof sp[0])] : nd(g_rng); break;
        }
    }
    if (c == STRADDLE && blocks > 0) {   // fewer clear winners than `top`, then more ties than the slots left
        const uint winners = top > 2 ? top - 1 - (uint)(g_rng() % 2) : 0;
        for (uint j = 0; j < winners && j < blocks; ++j) s[g_rng() % blocks] = 5.f + (float)j;
        for (uint j = 0; j < 5 + g_rng() % 30; ++j) s[g_rng() % blocks] = 1.f;
    }
}

static const emu::Order orders[3] = {emu::ASC, emu::DESC, emu::SHUFFLE};
static const char* order_name[3] = {"asc", "desc", "shuffle"};

// rows p0 .. p0 + rows - 1; row r holds (p0 + r + 1) / 4 blocks
static void one_case(Cls c, uint rows, uint p0, uint top) {
    const long idx = g_index++;
    if (idx % g_nshards != g_shard) return;
    g_rng.seed((unsigned)(0x610000 + idx));
    ++g_cases;
    const uint most = (p0 + rows) / 4;
    const uint s_stride = most + 1 + (uint)(idx % 3);
    const uint width = 4 * std::min(top, most) + 3 + (uint)(idx % 2) * 5;   // the key width: the picked keys, the tail, padding
    std::vector<float> scores((size_t)rows * s_stride, 9.f);
    for (uint r = 0; r < rows; ++r) fill(&scores[(size_t)r * s_stride], (p0 + r + 1) / 4, c, top);
    std::vector<int> out[3];
    for (int o = 0; o < 3; ++o) {
        for (int w = 0; w < 3; ++w) {
            out[w].assign((size_t)rows * width + 8, 0x5A5A5A5A);
            const Args a{scores.data(), out[w].data(), p0, top, width, s_stride, width, w};
            const unsigned long b0 = emu::barriers;
            gridDim.x = rows;
            bool ok = true;
            for (uint r = 0; r < rows && ok; ++r) { blockIdx.x = r; ok = emu::run_block(body, &a, 1024, orders[o], g_rng); }
            if (!ok) { ++g_fail; printf("FAIL deadlock: class %s rows %u p0 %u top %u order %s arm %d\n", cls_name[c], rows, p0, top, order_name[o], w); return; }
            ++g_launches;
            // barriers a threadgroup: 3 a digit pass, then 7 (two scans of 3, the device barrier); the slot waits are not counted
            uint sel = 0;
            for (uint r = 0; r < rows; ++r) sel += top < (p0 + r + 1) / 4;
            if (sel) g_digits[w] += ((double)(emu::barriers - b0) / 1024.0 - 7.0 * rows) / 3.0;
            if (w == 0) { g_blocks += sel; g_skipped += rows - sel; }   // (counted once an order, as the digits are)
        }
        const size_t nb = out[0].size() * 4;
        g_bytes += 2 * (long)nb;
        const char* what = nullptr;
        if (memcmp(out[0].data(), out[1].data(), nb) != 0) what = "cut 0 differs from old";
        else if (memcmp(out[0].data(), out[2].data(), nb) != 0) what = "cut 1 differs from old";
        if (what && ++g_fail <= 20) printf("FAIL %s: class %s rows %u p0 %u top %u order %s\n", what, cls_name[c], rows, p0, top, order_name[o]);
    }
}

int main(int argc, char** argv) {
    if (argc == 3) { g_shard = atol(argv[1]); g_nshards = atol(argv[2]); }
    const uint row_counts[] = {1, 7, 16, 128};
    // block counts around the select's own boundaries: none, one, top, top + 1, a simdgroup, the threadgroup (1,024), past it
    const uint blocks_at[] = {0, 1, 2, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2051, 5000};
    const uint tops[] = {1, 2, 16, 512};
    int t = 0;
    for (int c = 0; c < CLS_N; ++c)
        for (uint nb : blocks_at)
            for (uint top : tops) {
                ++t;
                const uint rows = row_counts[t % 3];
                one_case((Cls)c, rows, nb * 4 + (uint)(t % 4), top);   // the first row has nb blocks (p0 + 1 in 4 nb .. 4 nb + 3)
            }
    // the served shape: 512 blocks picked (index_topk 2,048 keys at 4 a block) from long rows, and 128 rows at once
    for (int c = 0; c < CLS_N; ++c) one_case((Cls)c, 7, 4 * 16384 + 1, 512);
    one_case(GAUSS, 128, 4 * 3000, 512);
    one_case(GRID, 128, 4 * 600 + 2, 512);
    one_case(STRADDLE, 128, 4 * 1024 - 60, 16);
    printf("STAT %ld %ld %ld %ld %.3f %.3f %.3f %ld %ld\n", g_cases, g_launches, g_bytes, g_fail, g_digits[0], g_digits[1], g_digits[2], g_blocks, g_skipped);
    return g_fail == 0 ? 0 : 1;
}
