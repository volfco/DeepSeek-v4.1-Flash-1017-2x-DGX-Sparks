// h0291: TF_DS_QD_PAIR_SHFL. Runs the exact source of the decode/prompt indexer query's pair geometry from two trees
// on the CPU (emu/emu.h):
//   old   = tfref-kstack4 @ 2bb052f           tf_ds_fused_rope_qd_pair_kernel (half-warp masked __reduce_max_sync)
//   off   = the sweep worktree, switch off    the same symbol (the <false> instantiation)
//   shfl  = the sweep worktree, switch on     tf_ds_fused_rope_qd_pair_shfl_kernel (four width-16 xor shuffles)
//   one   = tfref-kstack4's one-element form  tf_ds_fused_rope_qd_kernel (0241: "both write the same bytes")
// and compares output bytes: old == off, old == shfl, one == shfl. Every bf16 encoding goes through a scale group in
// the exhaustive class (signalling and quiet NaNs, both infinities, both zeros, subnormals). The bf16 rounding and the
// fused multiply-add are the host's forms on every arm (see emu.h): the comparison isolates the group maximum.
#include "../emu/emu.h"

namespace oldk {
#include "old.inc"
}
namespace newk {
#include "new.inc"
}

struct Args { const __nv_bfloat16* x; __nv_bfloat16* out; const float* cos; const float* sin; const long long* pos; int H, HD, RD, which; };
static void body(const void* p) {
    const Args& a = *(const Args*)p;
    switch (a.which) {
        case 0: oldk::tf_ds_fused_rope_qd_pair_kernel(a.x, a.out, a.cos, a.sin, a.pos, a.H, a.HD, a.RD, 0); break;
        case 1: newk::tf_ds_fused_rope_qd_pair_kernel(a.x, a.out, a.cos, a.sin, a.pos, a.H, a.HD, a.RD, 0); break;
        case 2: newk::tf_ds_fused_rope_qd_pair_shfl_kernel(a.x, a.out, a.cos, a.sin, a.pos, a.H, a.HD, a.RD, 0); break;
        default: oldk::tf_ds_fused_rope_qd_kernel(a.x, a.out, a.cos, a.sin, a.pos, a.H, a.HD, a.RD, 0); break;
    }
}

static std::mt19937 g_rng(0x291);
static long g_fail = 0, g_cases = 0, g_launches = 0, g_bytes = 0, g_index = 0, g_shard = 0, g_nshards = 1;
static unsigned long g_red[4], g_shf[4];
static std::vector<bool> g_seen(65536, false);

static bool launch(const Args& a, int rows, emu::Order order) {
    const bool pair = a.which != 3;
    const int gy = pair ? a.H / 8 : a.H, threads = pair ? 256 : 128;
    gridDim.x = (unsigned)rows; gridDim.y = (unsigned)gy;
    for (int r = 0; r < rows; ++r)
        for (int y = 0; y < gy; ++y) {
            blockIdx.x = (unsigned)r; blockIdx.y = (unsigned)y;
            if (!emu::run_block(body, &a, threads, order, g_rng)) return false;
        }
    ++g_launches;
    return true;
}

enum Cls { RANDBITS, GAUSS, EXHAUST, ZEROS, NANGROUP, INFS, EQUAL, TINY, HUGE_, CLS_N };
static const char* cls_name[CLS_N] = {"random-bits", "gauss", "all-65536-encodings", "signed-zeros", "nan-groups", "infinities", "all-equal", "subnormals", "huge"};
static void fill(std::vector<__nv_bfloat16>& x, Cls c, long caseno) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        unsigned short b = 0;
        switch (c) {
            case RANDBITS: b = (unsigned short)g_rng(); break;
            case GAUSS: b = __float2bfloat16_rn(nd(g_rng) * 3.f).x; break;
            case EXHAUST: b = (unsigned short)((i * 40503u + (unsigned)caseno * 7919u) & 0xffffu); break;   // 40503 is odd: a permutation of the encodings
            case ZEROS: b = (g_rng() & 1) ? 0x8000u : 0u; if (g_rng() % 61 == 0) b = __float2bfloat16_rn(nd(g_rng)).x; break;
            case NANGROUP: b = ((i / 32) % 3 == 0) ? (unsigned short)(0x7fc0u | (g_rng() & 0x3f) | ((g_rng() & 1) << 15)) : ((g_rng() % 9 == 0) ? 0x7f81u : __float2bfloat16_rn(nd(g_rng)).x); break;
            case INFS: b = (g_rng() % 13 == 0) ? ((g_rng() & 1) ? 0x7f80u : 0xff80u) : __float2bfloat16_rn(nd(g_rng) * 100.f).x; break;
            case EQUAL: b = 0x3fc0u; break;
            case TINY: b = (unsigned short)(g_rng() & 0x807fu); break;
            case HUGE_: b = (unsigned short)(0x7f00u | (g_rng() & 0x7f) | ((g_rng() & 1) << 15)); break;
            default: break;
        }
        x[i].x = b;
        g_seen[b] = true;
    }
}

static const emu::Order orders[3] = {emu::ASC, emu::DESC, emu::SHUFFLE};
static const char* order_name[3] = {"asc", "desc", "shuffle"};

static void one_case(Cls c, int rows, int H, int HD, int RD) {
    const long idx = g_index++;
    if (idx % g_nshards != g_shard) return;
    g_rng.seed((unsigned)(0x291000 + idx));
    ++g_cases;
    const size_t n = (size_t)rows * H * HD;
    std::vector<__nv_bfloat16> x(n);
    fill(x, c, idx);
    const int P = 37;
    std::vector<float> cs((size_t)P * (RD / 2) + 1), sn((size_t)P * (RD / 2) + 1);
    std::uniform_real_distribution<float> ang(-3.2f, 3.2f);
    for (size_t i = 0; i < cs.size(); ++i) { const float t = ang(g_rng); cs[i] = std::cos(t); sn[i] = std::sin(t); }
    if (idx % 5 == 0) { cs[0] = 1.f; sn[0] = 0.f; cs[1] = 0.f; sn[1] = -1.f; }
    std::vector<long long> pos(rows);
    for (auto& p : pos) p = g_rng() % P;
    std::vector<__nv_bfloat16> out[4];
    for (int o = 0; o < 3; ++o) {
        for (int w = 0; w < 4; ++w) {
            out[w].assign(n + 16, __nv_bfloat16{0xBEEFu});
            const Args a{x.data(), out[w].data(), cs.data(), sn.data(), pos.data(), H, HD, RD, w};
            const unsigned long s0 = emu::shuffles, r0 = emu::reductions;
            if (!launch(a, rows, orders[o])) { ++g_fail; printf("FAIL deadlock: class %s rows %d H %d HD %d RD %d order %s arm %d\n", cls_name[c], rows, H, HD, RD, order_name[o], w); return; }
            g_shf[w] += emu::shuffles - s0;
            g_red[w] += emu::reductions - r0;
        }
        const size_t nb = (n + 16) * 2;
        g_bytes += 3 * (long)nb;
        const char* what = nullptr;
        if (memcmp(out[0].data(), out[1].data(), nb) != 0) what = "pair: off differs from old";
        else if (memcmp(out[0].data(), out[2].data(), nb) != 0) what = "pair: shfl differs from old";
        else if (memcmp(out[3].data(), out[2].data(), nb) != 0) what = "pair shfl differs from the one-element form";
        if (what && ++g_fail <= 20) printf("FAIL %s: class %s rows %d H %d HD %d RD %d order %s\n", what, cls_name[c], rows, H, HD, RD, order_name[o]);
    }
}

int main(int argc, char** argv) {
    if (argc == 3) { g_shard = atol(argv[1]); g_nshards = atol(argv[2]); }
    struct Shape { int H, HD, RD; };
    // the served query is 32 heads x 128 with a 64-element rotary suffix; the others move every boundary the chunk has
    const Shape shapes[] = {{32, 128, 64}, {8, 64, 64}, {8, 128, 64}, {16, 128, 64}, {8, 128, 32}, {8, 128, 128}, {16, 64, 2}, {8, 192, 64}, {24, 128, 0}, {8, 256, 62}};
    const int row_counts[] = {1, 7, 16, 32, 48, 128};
    int t = 0;
    for (int c = 0; c < CLS_N; ++c)
        for (const Shape& s : shapes)
            for (int rows : row_counts) {
                ++t;
                if (rows == 128 && (t % 4) != 0) continue;            // the 128-row launches on a quarter of the shapes
                if (rows == 128 && s.H > 16) continue;
                one_case((Cls)c, rows, s.H, s.HD, s.RD);
            }
    // the served shape at 128 rows, and every bf16 encoding four times over in one launch
    one_case(GAUSS, 128, 32, 128, 64);
    one_case(EXHAUST, 64, 32, 128, 64);
    one_case(RANDBITS, 48, 32, 128, 64);
    unsigned long seen = 0;
    for (int b = 0; b < 65536; ++b) seen += g_seen[b];
    printf("STAT %ld %ld %ld %ld %lu %lu %lu %lu\n", g_cases, g_launches, g_bytes, g_fail, g_red[0], g_shf[0], g_red[2], g_shf[2]);
    if (g_shard == 0) printf("shard 0 alone fed %lu of 65,536 bf16 encodings through a scale group\n", seen);
    return g_fail == 0 ? 0 : 1;
}
