// h0269: TF_DS_DRAFT_TAPS_DIRECT's drafter input addressing on the CPU against an independent reference.
// 0269 adds no .cu kernel (the arm skips the gather launch and mainKeys reads rd.taps rows in place at the same
// row stride dr.at had), so there is nothing to splice out of the .cu: the harness emulates the arm's row
// addressing instead - absorb_many's contiguity decision on the items' byte addresses, the removed copyRows launch
// (a byte-exact row copy, emulated as its copy16 kernel does: one simulated thread a 16-byte word, no OS threads)
// and mainKeys' input row reads (x[r * ldx + c], ldx = taps_w). The reference is written from the brief's spec:
// the drafter must consume the participating windows' taps rows concatenated in round order, whatever the arm.
// stubs/ is the round's fixed CUDA syntax stubs (the shared cuda-syntax-stubs tree predates the base and fails on
// it; see report-0269.md).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// one window of a verify round: its first row in rd.taps, its rows, whether its stream's taps are absorbed
struct Window {
    long long row; // rd.taps row
    int n;
    bool absorb;
};

static int failures = 0;
static long long cases = 0, engaged_cases = 0;

static void fail(const char* what, long long row, long long n) {
    printf("FAIL %s (row %lld n %lld)\n", what, row, n);
    failures++;
}

// tf_ds_copy_rows16_kernel's copy: byte-exact rows at the given strides (a word a thread, executed in order)
static void copy_rows(const uint16_t* src, long long src_ld, uint16_t* dst, long long dst_ld, long long elems, int rows) {
    for (int r = 0; r < rows; r++) {
        const uint8_t* s = (const uint8_t*)src + (long long)r * src_ld * 2;
        uint8_t* d = (uint8_t*)dst + (long long)r * dst_ld * 2;
        for (long long w = 0; w < elems * 2; w += 16) memcpy(d + w, s + w, 16); // one simulated thread a word
    }
}

// draft_taps_direct.inPlace / Drafter.absorb_many's decision: the participating windows' byte addresses (the items
// are the round's windows in round order) must be consecutive rows at row_bytes; then the block's first address.
static long long in_place(const std::vector<long long>& taps, const std::vector<int>& ns, long long row_bytes) {
    if (taps.empty() || taps.size() != ns.size()) return -1;
    for (size_t i = 0; i + 1 < taps.size(); i++)
        if (taps[i + 1] != taps[i] + (long long)ns[i] * row_bytes) return -1;
    return taps[0];
}

// the arm under test (Drafter.absorb_many with the switch on): what mainKeys' input rows are. rd.taps is `taps`,
// W u16 a row; the decision runs on byte addresses exactly as the zig code's, and a gapped round falls back to the
// gather (copy_rows per item into dr.at, then rows [0, N) of it).
static std::vector<uint16_t> arm_reads(const uint16_t* taps, int W, const std::vector<Window>& wins, bool* engaged) {
    const long long row_bytes = (long long)W * 2;
    const long long base = 0x100000; // the device byte address stand-in
    std::vector<long long> addr;
    std::vector<int> ns;
    int N = 0;
    for (const Window& w : wins)
        if (w.absorb) {
            addr.push_back(base + w.row * row_bytes);
            ns.push_back(w.n);
            N += w.n;
        }
    long long first = in_place(addr, ns, row_bytes);
    std::vector<uint16_t> out((size_t)N * W);
    *engaged = first >= 0;
    if (first >= 0) {
        // mainKeys reads N rows at the source row stride: x[r * ldx + c], ldx = taps_w (one loop a GEMM input row)
        long long first_row = (first - base) / row_bytes;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < W; c++) out[(size_t)r * W + c] = taps[(first_row + r) * W + c];
    } else {
        // the off path's packing: copyRows per item into dr.at, then mainKeys reads rows [0, N) of it
        std::vector<uint16_t> at((size_t)N * W);
        int o = 0;
        for (const Window& w : wins) {
            if (!w.absorb) continue;
            copy_rows(taps + w.row * W, W, at.data() + (size_t)o * W, W, W, w.n);
            o += w.n;
        }
        for (int r = 0; r < N; r++)
            for (int c = 0; c < W; c++) out[(size_t)r * W + c] = at[(size_t)r * W + c];
    }
    return out;
}

// the independent reference (the brief's spec): the participating windows' taps rows concatenated in round order
static std::vector<uint16_t> spec_reference(const uint16_t* taps, int W, const std::vector<Window>& wins) {
    std::vector<uint16_t> out;
    for (const Window& w : wins) {
        if (!w.absorb) continue;
        out.insert(out.end(), taps + w.row * W, taps + (w.row + w.n) * W);
    }
    return out;
}

// whether the participating windows' rows are consecutive in the round (the arm's expected decision), from the
// round's row layout alone
static bool expect_engaged(const std::vector<Window>& wins) {
    long long end = -1;
    bool any = false;
    for (const Window& w : wins) {
        if (!w.absorb) continue;
        if (any and w.row != end) return false;
        end = w.row + w.n;
        any = true;
    }
    return any;
}

static void check(const std::vector<uint16_t>& taps_buf, int W, const std::vector<Window>& wins, const char* tag) {
    bool engaged = false;
    std::vector<uint16_t> got = arm_reads(taps_buf.data(), W, wins, &engaged);
    std::vector<uint16_t> want = spec_reference(taps_buf.data(), W, wins);
    cases++;
    if (engaged) engaged_cases++;
    if (engaged != expect_engaged(wins)) {
        fail("decision", (long long)wins.size(), W);
        printf("  %s: engaged %d expected %d\n", tag, engaged, expect_engaged(wins));
    }
    if (got.size() != want.size() or memcmp(got.data(), want.data(), got.size() * 2) != 0) {
        fail("bytes", (long long)wins.size(), W);
        printf("  %s: %zu vs %zu u16\n", tag, got.size(), want.size());
    }
}

int main() {
    // the case matrix: 1-4 streams (participating windows), window lengths 1-7, and the round's gap layouts
    // (no gap; a non-participating window before the first / after the last / between every pair / between the
    // first two only; gapped lengths 1-3)
    for (int k = 1; k <= 4; k++) {
        int L[4] = {1, 1, 1, 1};
        for (int a = 1; a <= 7; a++) {
            L[0] = a;
            for (int b = 1; b <= (k >= 2 ? 7 : 1); b++) {
                L[1] = b;
                for (int c = 1; c <= (k >= 3 ? 7 : 1); c++) {
                    L[2] = c;
                    for (int d = 1; d <= (k >= 4 ? 7 : 1); d++) {
                        L[3] = d;
                        for (int gap = 0; gap <= 3; gap++) {
                            for (int g = 1; g <= (gap ? 3 : 1); g++) {
                                // lay the round's windows out as lanes.Built.build does: window order, rows back
                                // to back; the gap layouts interleave non-participating windows of g rows
                                std::vector<Window> wins;
                                long long row = 0;
                                for (int i = 0; i < k; i++) {
                                    bool between = i > 0 and (gap == 1 or (gap == 3 and i == 1));
                                    bool before = i == 0 and gap == 2;
                                    if (before or between) {
                                        wins.push_back({row, g, false});
                                        row += g;
                                    }
                                    wins.push_back({row, L[i], true});
                                    row += L[i];
                                    if (i == k - 1 and gap == 2) {
                                        wins.push_back({row, g, false});
                                        row += g;
                                    }
                                }
                                if (row > 64) continue; // the matrix buffer below
                                std::vector<uint16_t> taps((size_t)row * 8);
                                for (size_t r = 0; r < (size_t)row; r++)
                                    for (int c = 0; c < 8; c++) taps[r * 8 + c] = (uint16_t)(r * 2654435761u + (unsigned)c * 40503u);
                                check(taps, 8, wins, "matrix");
                            }
                        }
                    }
                }
            }
        }
    }

    // the served width: three 5,120-channel BF16 taps (the brief's shape), the arm's indexing over real rows;
    // the byte accounting the item quotes (7 rows x 30,720 B = 210 KiB) re-checked here
    {
        const int W = 3 * 5120;
        for (int k = 1; k <= 4; k++) {
            int L[4] = {7, 7, 7, 7};
            for (int mix = 0; mix < 2; mix++) {
                if (mix) {
                    L[0] = 5;
                    L[1] = 6;
                    L[2] = 7;
                    L[3] = 7;
                }
                for (int gap = 0; gap <= 1; gap++) {
                    std::vector<Window> wins;
                    long long row = 0;
                    for (int i = 0; i < k; i++) {
                        if (gap and i > 0) {
                            wins.push_back({row, 1, false});
                            row += 1;
                        }
                        wins.push_back({row, L[i], true});
                        row += L[i];
                    }
                    std::vector<uint16_t> taps((size_t)row * W);
                    for (size_t r = 0; r < (size_t)row; r++)
                        for (int c = 0; c < W; c++) taps[r * W + c] = (uint16_t)(r * 2654435761u + (unsigned)c * 40503u);
                    check(taps, W, wins, "served");
                    int N = 0;
                    for (const Window& w : wins)
                        if (w.absorb) N += w.n;
                    if (!gap and k == 1 and L[0] == 7)
                        printf("served shape: %d rows x %d taps x 5120 ch x 2 B = %d B copied per rank per round off, 0 on\n", L[0], W / 5120, N * W * 2);
                }
            }
        }
    }

    printf("cases: %lld, engaged (in place): %lld, failures: %d\n", cases, engaged_cases, failures);
    if (failures == 0) printf("h0269: all cases match the packed-copy reference byte for byte\n");
    return failures == 0 ? 0 : 1;
}
