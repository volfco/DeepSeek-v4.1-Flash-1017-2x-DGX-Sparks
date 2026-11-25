// h0290: TF_DS_DRAW_CUT. Runs the exact source of the sampled selectors from two trees on the CPU (emu/emu.h):
//   old  = tfref-kstack4 @ 2bb052f           tf_ds_topk_select_kernel, tf_ds_draw_rows_kernel
//   new  = the sweep worktree, switch off    the same two symbols (the <false> instantiations)
//   cut  = the sweep worktree, switch on     tf_ds_topk_select_cut_kernel, tf_ds_draw_rows_cut_kernel
// and compares output bytes: old == new (off is unchanged) and old == cut (the complete-bin stop moves no bit), every
// case under three lane orders (the atomicAdd that hands out a candidate's slot follows the schedule). The select's
// list is also checked against an independent full sort by the draw key. The digit passes each arm ran are counted
// from the barriers the source executed (3 a digit + 2 for the select; the work removed, not a time).
#include "../emu/emu.h"
#include <cinttypes>

namespace oldk {
#include "old.inc"
}
namespace newk {
#include "new.inc"
}

struct SelArgs { const float* logits; unsigned long long ld; int n; unsigned col0; int kmax; unsigned* out; int which; };
static void sel_body(const void* p) {
    const SelArgs& a = *(const SelArgs*)p;
    if (a.which == 0) oldk::tf_ds_topk_select_kernel(a.logits, a.ld, a.n, a.col0, a.kmax, a.out, 0);
    else if (a.which == 1) newk::tf_ds_topk_select_kernel(a.logits, a.ld, a.n, a.col0, a.kmax, a.out, 0);
    else newk::tf_ds_topk_select_cut_kernel(a.logits, a.ld, a.n, a.col0, a.kmax, a.out, 0);
}
struct DrawArgs { const float* logits; unsigned long long ld; int n; const void* rules; unsigned* out; int which; };
static void draw_body(const void* p) {
    const DrawArgs& a = *(const DrawArgs*)p;
    if (a.which == 0) oldk::tf_ds_draw_rows_kernel(a.logits, a.ld, a.n, (const oldk::TfDsDrawRow*)a.rules, a.out, 0);
    else if (a.which == 1) newk::tf_ds_draw_rows_kernel(a.logits, a.ld, a.n, (const newk::TfDsDrawRow*)a.rules, a.out, 0);
    else newk::tf_ds_draw_rows_cut_kernel(a.logits, a.ld, a.n, (const newk::TfDsDrawRow*)a.rules, a.out, 0);
}
static_assert(sizeof(oldk::TfDsDrawRow) == sizeof(newk::TfDsDrawRow) && sizeof(oldk::TfDsDrawRow) == 48, "rule layout");

static std::mt19937 g_rng(0x290);
static long g_fail = 0, g_cases = 0, g_launches = 0, g_cmp_bytes = 0, g_sel = 0, g_draw = 0;
// a case runs in the shard its index falls in, with its own seed, so the matrix is the same however it is split
static long g_index = 0, g_shard = 0, g_nshards = 1;
static bool mine() {
    const long idx = g_index++;
    if (idx % g_nshards != g_shard) return false;
    g_rng.seed((unsigned)(0x290000 + idx));
    return true;
}
enum { CLS_MAX = 32 };
static double g_dsmall[3], g_dcls[CLS_MAX][3];   // digit passes summed over the select's blocks, per arm
static long g_bsmall, g_bcls[CLS_MAX];

static bool launch(void (*fn)(const void*), const void* arg, int rows, emu::Order order, unsigned long* barriers) {
    gridDim.x = (unsigned)rows;
    const unsigned long b0 = emu::barriers;
    for (int r = 0; r < rows; ++r) {
        blockIdx.x = (unsigned)r;
        if (!emu::run_block(fn, arg, 1024, order, g_rng)) return false;
    }
    ++g_launches;
    if (barriers) *barriers = emu::barriers - b0;
    return true;
}

// -- independent reference: the draw key's order by a full sort (not the kernel's radix) ---------------------------
static uint64_t ref_key(float v, uint32_t id) {
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7fffffffu) == 0) b = 0;
    const uint32_t hi = std::isnan(v) ? 0u : ((b >> 31) ? ~b : (b | 0x80000000u));
    return (uint64_t(hi) << 32) | (0xFFFFFFFFu - id);
}

// -- rows ----------------------------------------------------------------------------------------------------------
enum Cls { GAUSS, SERVED, GRID, EQUAL, ALLNAN, NINF, PINF, SPECIAL, STRADDLE, EXACT, RAMP_UP, RAMP_DN, TWO, NAN0, SUBNORMAL, CLS_N };
static const char* cls_name[CLS_N] = {"gauss", "served-like", "grid-ties", "all-equal", "all-nan", "all-neg-inf", "all-pos-inf", "specials", "tie-straddles-k", "tie-completes-k", "ramp-up", "ramp-down", "two-valued", "row0-nan", "subnormals"};
static void fill_row(float* row, int n, Cls c, int k) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float sp[] = {0.f, -0.f, inf, -inf, nan, 1.f, -1.f, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(), __uint_as_float(0x7fc12345u), __uint_as_float(0xffc00001u)};
    switch (c) {
        case GAUSS: for (int i = 0; i < n; ++i) row[i] = nd(g_rng); break;
        case SERVED: for (int i = 0; i < n; ++i) row[i] = nd(g_rng) * 2.5f - 3.f; for (int j = 0; j < 12 && j < n; ++j) row[g_rng() % n] = 8.f + nd(g_rng) * 3.f; break;
        case GRID: for (int i = 0; i < n; ++i) row[i] = std::round(nd(g_rng) * 8.f) / 8.f; break;
        case EQUAL: for (int i = 0; i < n; ++i) row[i] = 0.5f; break;
        case ALLNAN: for (int i = 0; i < n; ++i) row[i] = (i & 1) ? nan : __uint_as_float(0xffc00000u | (unsigned)(i & 0xffff)); break;
        case NINF: for (int i = 0; i < n; ++i) row[i] = -inf; break;
        case PINF: for (int i = 0; i < n; ++i) row[i] = inf; break;
        case SPECIAL: for (int i = 0; i < n; ++i) row[i] = (g_rng() & 1) ? sp[g_rng() % (sizeof sp / sizeof sp[0])] : std::round(nd(g_rng) * 4.f) / 4.f; break;
        case STRADDLE: {   // k - 1 - t clear winners, then more ties at the cutoff value than slots left
            for (int i = 0; i < n; ++i) row[i] = -1.f - (float)(g_rng() % 7);
            const int winners = std::max(0, k - 1 - (int)(g_rng() % 3));
            for (int j = 0; j < winners; ++j) row[g_rng() % n] = 10.f + (float)j;
            for (int j = 0; j < 9 + (int)(g_rng() % 40); ++j) row[g_rng() % n] = 3.f;
            break;
        }
        case EXACT: {      // distinct values everywhere: the cutoff bin completes at a value digit
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), g_rng);
            for (int i = 0; i < n; ++i) row[i] = (float)perm[i] * 0.37f - 5.f;
            break;
        }
        case RAMP_UP: for (int i = 0; i < n; ++i) row[i] = (float)i * 1e-3f; break;
        case RAMP_DN: for (int i = 0; i < n; ++i) row[i] = -(float)i * 1e-3f; break;
        case TWO: for (int i = 0; i < n; ++i) row[i] = (g_rng() % 5 == 0) ? 0.f : -0.f; break;
        case NAN0: for (int i = 0; i < n; ++i) row[i] = nd(g_rng); row[0] = nan; break;
        case SUBNORMAL: for (int i = 0; i < n; ++i) row[i] = __uint_as_float((g_rng() & 0x807fffffu)); break;
        default: break;
    }
}

static const emu::Order orders[3] = {emu::ASC, emu::DESC, emu::SHUFFLE};
static const char* order_name[3] = {"asc", "desc", "shuffle"};

static void fail(const char* what, const char* cls, int rows, int n, int k, const char* ord) {
    if (++g_fail <= 20) printf("FAIL %s: class %s rows %d n %d k %d order %s\n", what, cls, rows, n, k, ord);
}

static void select_case(Cls c, int rows, int n, int kmax, unsigned col0, int pad) {
    if (!mine()) return;
    ++g_sel;
    const unsigned long long ld = (unsigned long long)n + pad;
    std::vector<float> logits((size_t)rows * ld, 123.f);
    for (int r = 0; r < rows; ++r) fill_row(&logits[(size_t)r * ld], n, c, kmax);
    const size_t words = (size_t)rows * kmax * 2;
    std::vector<unsigned> out[3];
    ++g_cases;
    for (int o = 0; o < 3; ++o) {
        for (int w = 0; w < 3; ++w) {
            out[w].assign(words + 8, 0xDEADBEEFu);   // a guard past the end: an out-of-bounds store shows
            SelArgs a{logits.data(), ld, n, col0, kmax, out[w].data(), w};
            unsigned long bars = 0;
            if (!launch(sel_body, &a, rows, orders[o], &bars)) { fail("deadlock", cls_name[c], rows, n, kmax, order_name[o]); return; }
            (n == 64640 ? g_dcls[c][w] : g_dsmall[w]) += ((double)bars / 1024.0 - 2.0 * rows) / 3.0;
        }
        (n == 64640 ? g_bcls[c] : g_bsmall) += rows;
        g_cmp_bytes += 2 * (long)(words + 8) * 4;
        if (memcmp(out[0].data(), out[1].data(), (words + 8) * 4) != 0) fail("select: off differs from old", cls_name[c], rows, n, kmax, order_name[o]);
        if (memcmp(out[0].data(), out[2].data(), (words + 8) * 4) != 0) fail("select: cut differs from old", cls_name[c], rows, n, kmax, order_name[o]);
        if (o == 0) {   // the list against a full sort (the first min(kmax, n) pairs; the rest stay untouched)
            for (int r = 0; r < rows; ++r) {
                const float* row = &logits[(size_t)r * ld];
                std::vector<uint32_t> ids(n);
                for (int i = 0; i < n; ++i) ids[i] = col0 + (uint32_t)i;
                const int k = std::min(kmax, n);
                std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](uint32_t x, uint32_t y) { return ref_key(row[x - col0], x) > ref_key(row[y - col0], y); });
                for (int j = 0; j < k; ++j) {
                    const unsigned want_v = __float_as_uint(row[ids[j] - col0]);
                    const unsigned want_id = ids[j] | (j == 0 && std::isnan(row[0]) ? 0x80000000u : 0u);
                    if (out[2][((size_t)r * kmax + j) * 2] != want_v || out[2][((size_t)r * kmax + j) * 2 + 1] != want_id) { fail("select: cut differs from the sort reference", cls_name[c], rows, n, kmax, "asc"); break; }
                }
            }
        }
    }
}

static void draw_case(Cls c, int rows, int n, int pad, int kfix) {
    if (!mine()) return;
    ++g_draw;
    const unsigned long long ld = (unsigned long long)n + pad;
    std::vector<float> logits((size_t)rows * ld, 123.f);
    std::vector<oldk::TfDsDrawRow> rules(rows);
    const double temps[] = {0.0, 0.3, 0.7, 1.0, 1.7, 1e-9};
    const double tps[] = {1.0, 0.95, 0.5, 0.0, 0.999};
    const unsigned ks[] = {1, 2, 7, 20, 40, 55, 56};
    int kmaxrow = 1;
    for (int r = 0; r < rows; ++r) {
        auto& u = rules[r];
        u.seed = ((uint64_t)g_rng() << 32) | g_rng();
        u.temp = temps[g_rng() % 6];
        if (rows == 1 && u.temp <= 0.0) u.temp = 0.7;   // a one-row launch always runs the select
        u.top_p = tps[g_rng() % 5];
        u.min_p = (g_rng() % 3 == 0) ? 0.05 : 0.0;
        u.pos = g_rng() % (1u << 20);
        u.top_k = kfix > 0 ? (unsigned)kfix : ks[g_rng() % 7];
        u.pad = 0;
        kmaxrow = std::max(kmaxrow, (int)u.top_k);
        fill_row(&logits[(size_t)r * ld], n, c, (int)u.top_k);
    }
    std::vector<unsigned> out[3];
    ++g_cases;
    for (int o = 0; o < 3; ++o) {
        for (int w = 0; w < 3; ++w) {
            out[w].assign(rows + 4, 0xDEADBEEFu);
            DrawArgs a{logits.data(), ld, n, rules.data(), out[w].data(), w};
            if (!launch(draw_body, &a, rows, orders[o], nullptr)) { fail("deadlock", cls_name[c], rows, n, kmaxrow, order_name[o]); return; }
        }
        g_cmp_bytes += 2 * (long)(rows + 4) * 4;
        if (memcmp(out[0].data(), out[1].data(), (rows + 4) * 4) != 0) fail("draw: off differs from old", cls_name[c], rows, n, kmaxrow, order_name[o]);
        if (memcmp(out[0].data(), out[2].data(), (rows + 4) * 4) != 0) fail("draw: cut differs from old", cls_name[c], rows, n, kmaxrow, order_name[o]);
    }
}

int main(int argc, char** argv) {
    if (argc == 3) { g_shard = atol(argv[1]); g_nshards = atol(argv[2]); }
    const int row_counts[] = {1, 7, 16, 128};
    const int small_n[] = {56, 57, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4097};
    const int ks[] = {1, 2, 7, 20, 55, 56};
    const unsigned col0s[] = {0u, 64640u, 65536u - 17u};
    // 1. the select over every class at the small widths (alignment boundaries around the block and warp sizes)
    for (int c = 0; c < CLS_N; ++c)
        for (int n : small_n)
            for (int k : ks)
                select_case((Cls)c, row_counts[(c + n + k) % 3], n, k, col0s[(c + k) % 3], (n + k) % 2 ? 3 : 0);
    // 2. the served width: a rank's 64,640 head columns at both ranks' global id bases, rows 1 / 7 / 16 / 128
    for (int c = 0; c < CLS_N; ++c)
        for (int rank = 0; rank < 2; ++rank)
            for (int k : {1, 20, 56})
                select_case((Cls)c, row_counts[(c + rank + k) % 3], 64640, k, rank ? 64640u : 0u, 0);
    select_case(GAUSS, 128, 64640, 20, 64640u, 0);
    select_case(SERVED, 128, 64640, 56, 0u, 0);
    select_case(STRADDLE, 128, 4097, 20, 64640u, 3);
    // 3. the full-row draw (the fallback when the merged exchange is off): the whole 129,280-column row and smaller
    for (int c = 0; c < CLS_N; ++c) {
        for (int n : {56, 64, 257, 1024, 4097}) draw_case((Cls)c, row_counts[(c + n) % 3], n, n % 2 ? 5 : 0, 0);
        draw_case((Cls)c, row_counts[c % 3], 129280, 0, 0);
        for (int k : {1, 20, 56}) draw_case((Cls)c, 1, 129280, 0, k);
    }
    draw_case(GAUSS, 128, 129280, 0, 0);
    draw_case(SERVED, 128, 4097, 3, 0);
    // one machine-readable line a shard (build.sh sums them)
    printf("STAT %ld %ld %ld %ld %ld %ld %.3f %.3f %.3f %ld", g_cases, g_sel, g_draw, g_launches, g_cmp_bytes, g_fail, g_dsmall[0], g_dsmall[1], g_dsmall[2], g_bsmall);
    for (int c = 0; c < CLS_N; ++c) printf(" %s %.3f %.3f %.3f %ld", cls_name[c], g_dcls[c][0], g_dcls[c][1], g_dcls[c][2], g_bcls[c]);
    printf("\n");
    return g_fail == 0 ? 0 : 1;
}
