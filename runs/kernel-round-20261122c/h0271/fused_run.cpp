// CPU harness for TF_DS_RESUME_ASSEMBLE's tf_ds_resume_assemble_kernel (h0271). It executes the kernel's exact
// source (fused_kernel_current.inc, extracted by splice.py from zig/kernels/cuda/dsv41_ops.cu) with emulated blocks
// run sequentially and each block's thread bodies in turn (the kernel has no barriers and every thread writes
// disjoint words, so any thread order is its semantics), against an INDEPENDENT reference of today's copy sequence
// written from BRIEF-0271 and prompt.replayCut's off path: the fresh rows copy to out rows [pre, ...), the tail rows
// restore from the ring's one or two segments (tailRowsCopy's wrap split), each field separately, and the positions
// fill on the host (the upload's bytes). Not a model of CUDA codegen: this is the C source's data movement (pure
// copies + integer positions). Byte for byte per case: residual rows, mixing weights and positions.
//
// The matrix covers >=150 layouts at the served shape (config: hc 4, hidden 5120 -> row_h 40,960 bytes, row_p 16
// bytes = the four mixing weights, ring window 128 = sliding_window): wrapped and unwrapped restored prefixes,
// short appends (1..7 fresh rows), long appends (to the 2,048-row prompt chunk). The 4- and 8-slot rings are extra
// wrap-density stress on the same slot rule.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ---- CUDA stubs (p0194/fused_run.cpp's pattern) -------------------------------------------------------------------
#define __global__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, gridDim, threadIdx;
struct alignas(16) uint4 { unsigned x, y, z, w; };
static inline void pdl_enter(int) {}

// ---- the kernel's exact source -----------------------------------------------------------------------------------
#include "fused_kernel_current.inc"

// ---- the independent reference (today's copy sequence, from BRIEF-0271 / prompt.replayCut, not from the kernel) ---
static void ref_restore(uint8_t* dst, const uint8_t* ring, size_t window, size_t rb, size_t p0, size_t rows) {
    const size_t slot = p0 % window;
    const size_t n1 = std::min(rows, window - slot);
    memcpy(dst, ring + slot * rb, n1 * rb);
    if (rows > n1) memcpy(dst + n1 * rb, ring, (rows - n1) * rb);
}

static void ref_assemble(const uint8_t* tail_h, const uint8_t* tail_p, const uint8_t* fresh_h, const uint8_t* fresh_p,
                         uint8_t* out_h, uint8_t* out_p, long long* pos, size_t row_h, size_t row_p, size_t window,
                         int pre, int first, int kept, long long replay) {
    const int n = pre + kept;
    memcpy(out_h + (size_t)pre * row_h, fresh_h + (size_t)first * row_h, (size_t)kept * row_h);
    if (pre > 0) ref_restore(out_h, tail_h, window, row_h, (size_t)replay, (size_t)pre);
    memcpy(out_p + (size_t)pre * row_p, fresh_p + (size_t)first * row_p, (size_t)kept * row_p);
    if (pre > 0) ref_restore(out_p, tail_p, window, row_p, (size_t)replay, (size_t)pre);
    for (int i = 0; i < n; ++i) pos[i] = replay + i;
}

// ---- the kernel's launch emulation: blocks sequential, each block's 256 thread bodies in turn --------------------
static void run_kernel(const uint4* tail_h, const uint4* tail_p, const uint4* fresh_h, const uint4* fresh_p,
                       uint4* out_h, uint4* out_p, long long* pos, unsigned long long row_h16,
                       unsigned long long row_p16, unsigned long long window, int pre, int first, int kept,
                       long long replay) {
    const unsigned n = (unsigned)(pre + kept);
    gridDim = {n, 1, 1};
    blockDim = {256, 1, 1};
    for (unsigned b = 0; b < n; ++b) {
        blockIdx.x = b;
        for (unsigned t = 0; t < 256u; ++t) {
            threadIdx.x = t;
            tf_ds_resume_assemble_kernel(tail_h, tail_p, fresh_h, fresh_p, out_h, out_p, pos,
                                         row_h16, row_p16, window, pre, first, replay, 0);
        }
    }
}

// ---- buffers and the case matrix ---------------------------------------------------------------------------------
struct Buf {
    uint8_t* p = nullptr;
    size_t cap = 0;
    bool need(size_t n) { // true when it reallocated (the new bytes carry no pattern yet)
        if (n > cap) {
            free(p);
            p = (uint8_t*)aligned_alloc(64, (n + 63) & ~(size_t)63);
            if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(2); }
            cap = n;
            return true;
        }
        return false;
    }
    ~Buf() { free(p); }
};

static Buf tail_h_b, tail_p_b, fresh_h_b, fresh_p_b, out_h_k, out_p_k, out_h_r, out_p_r;
static size_t tail_filled = 0, fresh_filled = 0; // rows already patterned (patterns depend on the row index only)
static size_t fill_row_h = 0, fill_row_p = 0; // the row sizes those patterns were written at

static void fill_tail(size_t window, size_t row_h, size_t row_p) {
    for (size_t s = tail_filled; s < window; ++s) {
        for (size_t j = 0; j < row_h; ++j) tail_h_b.p[s * row_h + j] = (uint8_t)(s * 31 + j * 7 + 1);
        for (size_t j = 0; j < row_p; ++j) tail_p_b.p[s * row_p + j] = (uint8_t)(s * 17 + j * 13 + 2);
    }
    if (window > tail_filled) tail_filled = window;
}

static void fill_fresh(size_t rows, size_t row_h, size_t row_p) {
    for (size_t r = fresh_filled; r < rows; ++r) {
        for (size_t j = 0; j < row_h; ++j) fresh_h_b.p[r * row_h + j] = (uint8_t)(r * 53 + j * 11 + 3);
        for (size_t j = 0; j < row_p; ++j) fresh_p_b.p[r * row_p + j] = (uint8_t)(r * 41 + j * 19 + 4);
    }
    if (rows > fresh_filled) fresh_filled = rows;
}

static long long* pos_k = nullptr;
static long long* pos_r = nullptr;
static size_t pos_cap = 0;

static long cases = 0, fails = 0, served = 0;

static void run_case(size_t window, int pre, int kept, int first, size_t mod, size_t row_h, size_t row_p, bool served_ring) {
    const size_t n = (size_t)pre + kept;
    const size_t n0 = (size_t)first + kept;
    const long long replay = (long long)(7 * window + mod);
    // a reallocation or a new row size leaves the fill frontiers' patterns stale: reset them and refill below
    const bool tail_new = tail_h_b.need(window * row_h) | tail_p_b.need(window * row_p);
    const bool fresh_new = fresh_h_b.need((n0 + 1) * row_h) | fresh_p_b.need((n0 + 1) * row_p); // +1 guard row: a mutant reading past the cut stays defined to be caught
    if (tail_new) tail_filled = 0;
    if (fresh_new) fresh_filled = 0;
    if (row_h != fill_row_h or row_p != fill_row_p) {
        tail_filled = 0;
        fresh_filled = 0;
        fill_row_h = row_h;
        fill_row_p = row_p;
    }
    out_h_k.need(n * row_h);
    out_p_k.need(n * row_p);
    out_h_r.need(n * row_h);
    out_p_r.need(n * row_p);
    if (n > pos_cap) {
        free(pos_k);
        free(pos_r);
        pos_k = (long long*)aligned_alloc(64, (n * 8 + 63) & ~(size_t)63);
        pos_r = (long long*)aligned_alloc(64, (n * 8 + 63) & ~(size_t)63);
        pos_cap = n;
    }
    fill_tail(window, row_h, row_p);
    fill_fresh(n0 + 1, row_h, row_p);
    memset(out_h_k.p, 0xAA, n * row_h); // poison: an unwritten byte cannot pass as the reference's
    memset(out_p_k.p, 0xAA, n * row_p);
    memset(pos_k, 0xAA, n * 8);
    ref_assemble(tail_h_b.p, tail_p_b.p, fresh_h_b.p, fresh_p_b.p, out_h_r.p, out_p_r.p, pos_r,
                 row_h, row_p, window, pre, first, kept, replay);
    run_kernel((const uint4*)tail_h_b.p, (const uint4*)tail_p_b.p, (const uint4*)fresh_h_b.p, (const uint4*)fresh_p_b.p,
               (uint4*)out_h_k.p, (uint4*)out_p_k.p, pos_k, row_h / 16, row_p / 16, window, pre, first, kept, replay);
    cases++;
    if (served_ring) served++;
    auto check = [&](const char* what, const void* a, const void* b, size_t bytes) {
        if (memcmp(a, b, bytes) != 0) {
            const uint8_t* x = (const uint8_t*)a;
            const uint8_t* y = (const uint8_t*)b;
            size_t at = 0;
            while (at < bytes && x[at] == y[at]) at++;
            printf("case %ld FAIL %s at byte %zu (kernel 0x%02x, reference 0x%02x): window %zu pre %d kept %d first %d mod %zu replay %lld\n",
                   cases, what, at, x[at], y[at], window, pre, kept, first, mod, replay);
            fails++;
            return false;
        }
        return true;
    };
    check("out_h", out_h_k.p, out_h_r.p, n * row_h) and
        check("out_p", out_p_k.p, out_p_r.p, n * row_p) and
        check("pos", pos_k, pos_r, n * 8);
}

int main() {
    const size_t row_h = 4 * 5120 * 2; // the served residual row (hc 4, hidden 5120, bf16)
    const size_t row_p = 4 * 4; // the served mixing-weight row (hc 4, fp32): four weights
    // 1. the served ring (window 128): wrapped and unwrapped restored prefixes, short appends 1..7
    for (int pre : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 63, 64, 65, 96, 127, 128})
        for (int kept : {1, 2, 3, 4, 5, 6, 7, 8, 32})
            for (int first : {0, 1, 5})
                for (size_t mod : {(size_t)0, (size_t)1, (size_t)7, (size_t)(128 - pre), (size_t)127})
                    run_case(128, pre, kept, first, mod, row_h, row_p, true);
    // 2. long appends to the 2,048-row prompt chunk, served ring
    for (int pre : {1, 7, 128})
        for (int kept : {256, 2048})
            for (int first : {0, 5})
                for (size_t mod : {(size_t)0, (size_t)127})
                    run_case(128, pre, kept, first, mod, row_h, row_p, true);
    // 3. one-word rows (the 16-byte copy loop's lower bound), served ring
    for (int pre : {1, 2, 127, 128})
        for (int kept : {1, 7})
            for (int first : {0, 1})
                for (size_t mod : {(size_t)0, (size_t)127})
                    run_case(128, pre, kept, first, mod, 16, 16, true);
    // 4. extra: 4- and 8-slot rings, dense wrap-around of the same slot rule
    for (size_t window : {(size_t)4, (size_t)8})
        for (int pre = 1; pre <= (int)window; ++pre)
            for (int kept : {1, 3, 7})
                for (int first : {0, 2})
                    for (size_t mod : {(size_t)0, (size_t)1, window - 1})
                        run_case(window, pre, kept, first, mod, row_h, row_p, false);
    printf("h0271: %ld layouts (%ld at the served ring), %ld mismatches\n", cases, served, fails);
    return fails ? 1 : 0;
}
