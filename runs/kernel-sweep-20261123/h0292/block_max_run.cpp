// h0292: TF_DS_BLOCK_MAX8. Runs the exact source of the candidate pool's block maxima from two trees on the CPU
// (emu/emu.h; the kernels have no barrier, so every thread of every block simply runs to its end):
//   old   = tfref-kstack4 @ 2bb052f           tf_ds_block_max_kernel (eight scalar loads a block)
//   off   = the sweep worktree's              tf_ds_block_max_kernel (the launch when the switch is off or the gate says no)
//   eight = the sweep worktree's              tf_ds_block_max8_kernel (two 16-byte loads a whole block)
// on the launcher's geometry (grid (ceil(nb / 256), rows), 256 threads; bsize 8, a 16-byte-aligned score base and a
// row stride of whole 16-byte words: block_max8.ok) and on a clipped grid (the kernel's stride loop), and compares the
// maxima's bytes: old == off, old == eight. Rows 1 / 7 / 16 / 128; widths at every offset around a block, a warp, a
// CTA and the served pool sizes; NaN (payloads, either sign), +-inf, signed zeros, subnormals, ties; vis at, inside
// and past the row, zero and negative.
#include "../emu/emu.h"

namespace oldk {
#include "old.inc"
}
namespace newk {
#include "new.inc"
}

struct Args { const float* score; long long ss; int width; const long long* vis; float* out; long long os; int nb; int which; };
static void body(const void* p) {
    const Args& a = *(const Args*)p;
    if (a.which == 0) oldk::tf_ds_block_max_kernel(a.score, a.ss, a.width, 8, a.vis, a.out, a.os, a.nb, 0);
    else if (a.which == 1) newk::tf_ds_block_max_kernel(a.score, a.ss, a.width, 8, a.vis, a.out, a.os, a.nb, 0);
    else newk::tf_ds_block_max8_kernel(a.score, a.ss, a.width, 8, a.vis, a.out, a.os, a.nb, 0);
}

static std::mt19937 g_rng(0x292);
static long g_fail = 0, g_cases = 0, g_launches = 0, g_bytes = 0, g_whole = 0, g_partial = 0;

enum Cls { GAUSS, NINF, ALLNAN, NANPOS, PINF, ZEROS, SUBN, TIES, SPECIAL, CLS_N };
static const char* cls_name[CLS_N] = {"gauss", "all-neg-inf", "all-nan", "one-nan-a-block", "pos-inf", "signed-zeros", "subnormals", "ties", "specials"};
static void fill(float* row, int width, Cls c) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity();
    const float sp[] = {0.f, -0.f, inf, -inf, __uint_as_float(0x7fc00000u), __uint_as_float(0xffc00001u), __uint_as_float(0x7f800001u), 1.f, -1.f, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
    for (int i = 0; i < width; ++i) {
        switch (c) {
            case GAUSS: row[i] = nd(g_rng); break;
            case NINF: row[i] = (g_rng() % 97 == 0) ? nd(g_rng) : -inf; break;
            case ALLNAN: row[i] = __uint_as_float((0x7f800001u + (g_rng() & 0x7fffffu) % 0x7ffffeu) | ((g_rng() & 1u) << 31)); break;
            case NANPOS: row[i] = ((i % 8) == (i / 8) % 8) ? __uint_as_float(0x7fc00000u | (g_rng() & 0xffffu)) : nd(g_rng); break;
            case PINF: row[i] = (g_rng() % 5 == 0) ? inf : nd(g_rng) * 1e30f; break;
            case ZEROS: row[i] = (g_rng() & 1) ? -0.f : 0.f; break;
            case SUBN: row[i] = __uint_as_float(g_rng() & 0x807fffffu); break;
            case TIES: row[i] = (float)((int)(g_rng() % 3) - 1); break;
            default: row[i] = (g_rng() & 1) ? sp[g_rng() % (sizeof sp / sizeof sp[0])] : nd(g_rng); break;
        }
    }
}

static void launch(const Args& a, int rows, unsigned gx) {
    gridDim.x = gx; gridDim.y = (unsigned)rows;
    for (unsigned y = 0; y < (unsigned)rows; ++y)
        for (unsigned x = 0; x < gx; ++x) {
            blockIdx.x = x; blockIdx.y = y;
            if (!emu::run_block(body, &a, 256, emu::ASC, g_rng)) { ++g_fail; return; }
        }
    ++g_launches;
}

static void one_case(Cls c, int rows, int width, int pad4, int os_pad, bool clip) {
    ++g_cases;
    const int nb = (width + 7) / 8;
    const long long ss = (long long)((width + 3) / 4 * 4) + 4 * pad4;   // a stride of whole 16-byte words
    const long long os = nb + os_pad;
    // a 16-byte-aligned base (the launcher's gate): over-allocate and round up
    std::vector<float> store((size_t)rows * ss + 8, 777.f);
    float* score = store.data();
    while ((reinterpret_cast<uintptr_t>(score) & 15u) != 0) ++score;
    std::vector<long long> vis(rows);
    const long long vs[] = {0, 1, 7, 8, 9, width - 1, width, width + 5, -1, -8, -9, width / 2, 1LL << 40, -(1LL << 40)};
    for (int r = 0; r < rows; ++r) {
        fill(score + (long long)r * ss, width, c);
        vis[r] = vs[(r + g_cases) % (sizeof vs / sizeof vs[0])];
    }
    const unsigned gx = clip ? std::max(1u, (unsigned)((nb + 255) / 256) / 3u) : (unsigned)((nb + 255) / 256);
    std::vector<float> out[3];
    for (int w = 0; w < 3; ++w) {
        out[w].assign((size_t)rows * os + 4, 555.f);
        const Args a{score, ss, width, vis.data(), out[w].data(), os, nb, w};
        launch(a, rows, gx);
    }
    const size_t nbytes = out[0].size() * 4;
    g_bytes += 2 * (long)nbytes;
    g_whole += (long)rows * (width / 8);
    g_partial += (long)rows * (width % 8 != 0);
    const char* what = nullptr;
    if (memcmp(out[0].data(), out[1].data(), nbytes) != 0) what = "off differs from old";
    else if (memcmp(out[0].data(), out[2].data(), nbytes) != 0) what = "eight differs from old";
    if (what && ++g_fail <= 20) printf("FAIL %s: class %s rows %d width %d ss %lld os %lld clip %d\n", what, cls_name[c], rows, width, ss, os, (int)clip);
}

int main() {
    const int row_counts[] = {1, 7, 16, 128};
    std::vector<int> widths;
    for (int w = 1; w <= 80; ++w) widths.push_back(w);
    for (int w : {248, 255, 256, 257, 264, 2040, 2047, 2048, 2049, 2056, 4096, 8191, 8192, 8193, 49152, 65536, 131069, 131072}) widths.push_back(w);
    int t = 0;
    for (int c = 0; c < CLS_N; ++c)
        for (int w : widths) {
            ++t;
            const int rows = w > 9000 ? row_counts[t % 2] : row_counts[t % 4];
            one_case((Cls)c, rows, w, t % 3 == 0 ? 1 : 0, t % 2 ? 3 : 0, t % 5 == 0);
        }
    // the served pool sizes at every row count: 8,192 blocks (the small-row form), 16,385 and 65,536 (the wide pick)
    for (int rows : {1, 7, 16, 128}) one_case(GAUSS, rows, 65536, 0, 0, false);
    for (int rows : {1, 7}) { one_case(SPECIAL, rows, 131080 - 5, 0, 0, false); one_case(GAUSS, rows, 524288, 0, 0, false); }
    printf("h0292 TF_DS_BLOCK_MAX8: %ld cases, %ld launches; %ld whole blocks through the two-load form, %ld rows with a partial last block\n", g_cases, g_launches, g_whole, g_partial);
    printf("compared %ld output bytes old-vs-off and old-vs-eight: %ld failures\n", g_bytes, g_fail);
    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
