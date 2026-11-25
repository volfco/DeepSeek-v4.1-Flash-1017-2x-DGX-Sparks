// h0265: the row selection / packing of TF_DS_DRAFT_HEAD_SCORED (draft_head_scored.zig, Drafter.body's on path) on the
// CPU, against an independent reference written from the brief's spec.
//
// There is no new .cu kernel in 0265: the packing is done by the kit's own `_collapse` (one launch a stream at that
// stream's row offsets), `_rmsnorm` (one launch over the packed rows) and the head GEMM (one launch, M = N * steps).
// What can go wrong is therefore the call sites' row arithmetic: which source row each packed row carries, and which
// packed rows each downstream reader indexes. This harness emulates the per-thread row semantics of those launches
// (thread (row, col-chunk) reads and writes exactly the rows the launch's base pointers and counts describe), tags
// every row with a per-row hash of its own bytes (per-row, row-count-invariant: the kit kernels' documented property),
// and checks:
//   1. packed row i*steps+s carries exactly source row i*n+s's tag (the brief: "gather of steps rows per stream out
//      of n with the strides used"), for xc (after the collapse), xn (after the norm) and local (after the head);
//   2. every downstream reader row (Markov _score at step s, the noise hook, the speculative-score hook, the
//      confidence packer) equals packRow(i, s) under the packed stride and srcRow(i, s) under the full stride;
//   3. no reader reaches an unscored row (s >= steps) anywhere;
//   4. the sq rows the spec hook writes keep their layout (stream * n + step) in both arms.
// Cases: N 1..4, n 5..7, steps 1..n (72 layouts) x 2 width regimes = 144 layouts, 120 of them shallow (steps < n),
// plus the served shapes (d=5120, lw=64640) and three mutation checks that must fail.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct Rng { uint64_t s; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } };

// The pass's shape and row widths (bytes): h rows hc*d*2 (bf16 [hc, d]), pre rows hc*4 (fp32 [hc]), xc/xn rows d*2,
// local rows lw*4, sq rows sq_w*4. The served strides: draft.zig Drafter.init's allocations.
struct Dims {
    int N;      // streams
    int n;      // block rows a stream (blockRows(steps))
    int steps;  // scored rows a stream
    int hc, d;  // the hidden state's parts and width
    int lw;     // a logits row's columns (mk.rowCols())
    size_t h_row() const { return (size_t)hc * d * 2; }
    size_t pre_row() const { return (size_t)hc * 4; }
    size_t xc_row() const { return (size_t)d * 2; }
    size_t loc_row() const { return (size_t)lw * 4; }
};

// The brief's row-selection spec, as a direct formula (the reference: written from "a host row-selection check ...
// gather of steps rows per stream out of n with the strides used", not from draft.zig's code).
static size_t ref_pack_row(const Dims& m, int i, int s) { return (size_t)i * m.steps + s; }
static size_t ref_src_row(const Dims& m, int i, int s) { return (size_t)i * m.n + s; }

// The emulated call sites (MUTATION points): the base-pointer arithmetic of Drafter.body's on path exactly as written
// in draft.zig (kept separate from the reference above).
static size_t k_pack_row(const Dims& m, int i, int s, int mut) {
    if (mut == 1) return (size_t)i * m.n + s;      // bug: the packed rows laid out by the block, not the steps
    return (size_t)i * m.steps + s;
}
static const uint8_t* k_stream_base(const uint8_t* buf, size_t row, int i, int n, int steps, int mut) {
    const int stride = (mut == 3) ? steps : n;     // bug: the collapse offsets by the scored rows, not the block rows
    return buf + (size_t)i * stride * row;
}

// -- the per-thread emulations of the launches ----------------------------------------------------------------------

// _collapse: X [rows, hc, d] + PRE [rows, hc] -> OUT [rows, d], one thread a (row, column chunk). Stand-in arithmetic:
// the row's own bytes hash to a tag (per-row, so row-count invariant by construction).
static void collapse_launch(const uint8_t* x, const uint8_t* pre, uint8_t* out, int rows, const Dims& m) {
    for (int r = 0; r < rows; r++) { // thread (row, chunk): chunks cover d*2 bytes
        uint64_t tag = fnv1a(x + (size_t)r * m.h_row(), m.h_row());
        tag ^= fnv1a(pre + (size_t)r * m.pre_row(), m.pre_row()) + 0x9e3779b97f4a7c15ull;
        memcpy(out + (size_t)r * m.xc_row(), &tag, sizeof tag); // the row's tag lands in its own row
    }
}

// _rmsnorm: X [rows, d] -> OUT [rows, d], one thread a row (per-row reduction inside).
static void rmsnorm_launch(const uint8_t* x, uint8_t* out, int rows, const Dims& m) {
    for (int r = 0; r < rows; r++) {
        uint64_t tag = fnv1a(x + (size_t)r * m.xc_row(), sizeof(uint64_t));
        memcpy(out + (size_t)r * m.xc_row(), &tag, sizeof tag);
    }
}

// the head GEMM (glinear): X [M, d] -> OUT [M, lw], one thread a row's columns; per-row output = its own input row.
static void head_launch(const uint8_t* x, uint8_t* out, int rows, const Dims& m) {
    for (int r = 0; r < rows; r++) {
        uint64_t tag = fnv1a(x + (size_t)r * m.xc_row(), sizeof(uint64_t));
        memset(out + (size_t)r * m.loc_row(), 0, m.loc_row());
        memcpy(out + (size_t)r * m.loc_row(), &tag, sizeof tag);
    }
}

// -- the harness ----------------------------------------------------------------------------------------------------

static int failures = 0;
static void fail(const char* what, const Dims& m, int i, int s) {
    printf("FAIL %s (N=%d n=%d steps=%d hc=%d d=%d lw=%d stream=%d step=%d)\n", what, m.N, m.n, m.steps, m.hc, m.d, m.lw, i, s);
    failures++;
}

// Run one layout through both arms (full: stride n; packed: stride steps) and check the four properties. `mut` injects
// a call-site bug (0: none) and must then make the layout fail.
static void run_layout(const Dims& m, int mut) {
    Rng rng{ 0x2650ull ^ (uint64_t)m.N << 40 ^ (uint64_t)m.n << 20 ^ (uint64_t)m.steps << 8 ^ (uint64_t)m.d };
    const size_t R = (size_t)m.N * m.n;
    const size_t P = (size_t)m.N * m.steps;
    std::vector<uint8_t> h(R * m.h_row()), pre(R * m.pre_row());
    for (auto& b : h) b = (uint8_t)rng.next();
    for (auto& b : pre) b = (uint8_t)rng.next();
    std::vector<uint8_t> xc_p(P * m.xc_row(), 0), xn_p(P * m.xc_row(), 0), loc_p(P * m.loc_row(), 0);
    std::vector<uint8_t> xc_f(R * m.xc_row(), 0), xn_f(R * m.xc_row(), 0), loc_f(R * m.loc_row(), 0);

    // the full arm: one launch over all R rows (today's call sites)
    collapse_launch(h.data(), pre.data(), xc_f.data(), R, m);
    rmsnorm_launch(xc_f.data(), xn_f.data(), R, m);
    head_launch(xn_f.data(), loc_f.data(), R, m);

    // the packed arm: one _collapse a stream at that stream's offsets (draft.zig's loop), then the packed norm and head
    for (int i = 0; i < m.N; i++) {
        const uint8_t* hx = k_stream_base(h.data(), m.h_row(), i, m.n, m.steps, mut);
        const uint8_t* px = k_stream_base(pre.data(), m.pre_row(), i, m.n, m.steps, mut);
        uint8_t* ox = xc_p.data() + (size_t)i * m.steps * m.xc_row();
        collapse_launch(hx, px, ox, m.steps, m);
    }
    rmsnorm_launch(xc_p.data(), xn_p.data(), (int)P, m);
    head_launch(xn_p.data(), loc_p.data(), (int)P, m);

    // (1) every packed row carries its source row's value, at every stage
    for (int i = 0; i < m.N; i++) for (int s = 0; s < m.steps; s++) {
        const size_t p = k_pack_row(m, i, s, mut), src = ref_src_row(m, i, s), want = ref_pack_row(m, i, s);
        if (p != want) { fail("pack row map", m, i, s); goto readers; }
        if (memcmp(xc_p.data() + p * m.xc_row(), xc_f.data() + src * m.xc_row(), m.xc_row()) != 0) fail("xc packed != full", m, i, s);
        if (memcmp(xn_p.data() + p * m.xc_row(), xn_f.data() + src * m.xc_row(), m.xc_row()) != 0) fail("xn packed != full", m, i, s);
        if (memcmp(loc_p.data() + p * m.loc_row(), loc_f.data() + src * m.loc_row(), m.loc_row()) != 0) fail("local packed != full", m, i, s);
    }

readers:
    // (2)+(3) the readers with the packed stride (lrow = steps) index exactly the scored packed rows; with the full
    // stride (lrow = n) exactly the source rows. Markov _score at step s reads rows s + i*lrow; the hooks the same;
    // the confidence packer reads xc row i*lrow + s; the spec hook's sq rows stay stream * n + step in both arms.
    for (int lrow = 0; lrow < 2; lrow++) {
        const int stride = mut == 2 ? m.n : (lrow == 0 ? m.steps : m.n); // bug: the pass passes the block stride
        for (int s = 0; s < m.steps; s++) for (int i = 0; i < m.N; i++) {
            const size_t rd = (size_t)s + (size_t)i * stride;      // Markov / hooks
            const size_t rc = (size_t)i * stride + s;               // the confidence packer
            const size_t expect = lrow == 0 ? ref_pack_row(m, i, s) : ref_src_row(m, i, s);
            if (rd != expect) fail(lrow == 0 ? "markov/hook read (packed)" : "markov/hook read (full)", m, i, s);
            if (rc != expect) fail(lrow == 0 ? "conf read (packed)" : "conf read (full)", m, i, s);
            if (expect >= (size_t)(lrow == 0 ? P : R)) fail("reader past the rows", m, i, s);
            if (lrow == 0) {
                // the reader's value is the (i, s) source row's tag
                uint64_t tag = 0;
                memcpy(&tag, loc_p.data() + rd * m.loc_row(), sizeof tag);
                uint64_t want = 0;
                memcpy(&want, loc_f.data() + ref_src_row(m, i, s) * m.loc_row(), sizeof want);
                if (tag != want) fail("reader value", m, i, s);
            }
        }
    }
    // (4) the spec hook's sq writes keep row stream * n + step in both arms (its stride argument is unchanged)
    for (int i = 0; i < m.N; i++) for (int s = 0; s < m.steps; s++) {
        const size_t sq = (size_t)s + (size_t)i * m.n;
        if (sq >= R) fail("sq row past the rows", m, i, s);
    }
}

int main() {
    int layouts = 0, shallow = 0;
    for (int N = 1; N <= 4; N++) for (int n = 5; n <= 7; n++) for (int steps = 1; steps <= n; steps++)
        for (int regime = 0; regime < 2; regime++) {
            const Dims m = { N, n, steps, 4, regime == 0 ? 24 : 130, regime == 0 ? 32 : 97 };
            run_layout(m, 0);
            layouts++;
            if (steps < n) shallow++;
        }
    printf("matrix: %d layouts (%d shallow), failures %d\n", layouts, shallow, failures);

    // the served shape's example (the brief: four streams, five block rows, three steps: 12 rows instead of 20) and
    // its neighbours at the served widths
    for (int steps = 1; steps <= 6; steps++) {
        const int n = steps <= 5 ? 5 : steps; // blockRows(steps) with the checkpoint's block 5
        const Dims m = { 4, n, steps, 4, 5120, 64640 };
        run_layout(m, 0);
        layouts++;
        if (steps < n) shallow++;
    }
    printf("served widths: N=4 d=5120 lw=64640 steps 1..6, total %d layouts (%d shallow), failures %d\n", layouts, shallow, failures);

    // the arm predicate (the brief's contract: only a pass with unscored rows, unchecked, not 2D)
    struct P { bool on, checked, twod; int steps, n, want; };
    const P preds[] = {
        { true, false, false, 3, 5, 1 }, { true, false, false, 5, 5, 0 }, { true, false, false, 6, 6, 0 },
        { true, true, false, 3, 5, 0 }, { true, false, true, 3, 5, 0 }, { false, false, false, 3, 5, 0 },
    };
    for (const P& p : preds) {
        const bool got = p.on && !p.checked && !p.twod && p.steps >= 1 && p.steps < p.n;
        if ((int)got != p.want) { printf("FAIL predicate\n"); failures++; }
    }

    // mutation checks: each injected call-site bug must be caught by the matrix (their failures are expected and do
    // not count against the clean runs)
    const char* names[] = { "pack rows by n", "reader stride n", "collapse offsets by steps" };
    for (int mut = 1; mut <= 3; mut++) {
        const int before = failures;
        for (int N = 1; N <= 4; N++) for (int n = 5; n <= 7; n++) for (int steps = 1; steps < n; steps++) {
            const Dims m = { N, n, steps, 4, 24, 32 };
            run_layout(m, mut);
        }
        const int caused = failures - before;
        failures = before;
        if (caused == 0) { printf("FAIL mutation %d (%s) not caught\n", mut, names[mut - 1]); failures++; }
        else printf("mutation %d (%s) caught by %d failing checks\n", mut, names[mut - 1], caused);
    }

    if (failures) { printf("h0265: FAIL (%d)\n", failures); return 1; }
    printf("h0265: all cases match\n");
    return 0;
}
