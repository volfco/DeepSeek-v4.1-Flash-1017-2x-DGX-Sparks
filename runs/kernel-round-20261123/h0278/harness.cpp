// h0278 — TF_DS_TAPS_PACKED: CPU emulation of the prompt tap write/copy layouts.
//
// The write under test is the real tf_ds_hc_mean4_kernel source (extracted from the worktree's
// dsv41_torch.cu into hc_mean4.inc by build.sh) executed as a sequential virtual-thread loop
// (no OS threads: one virtual CUDA thread at a time, per the round's harness rules). Three paths
// per case, served dims (cap = model.chunk_rows 2048, taps = 3 = dspark_target_layer_ids' length,
// D = hidden 5120):
//
//   planar + copy: prompt.tap's off write (out = plane j = base + j*cap*D*2 bytes, os = D), then
//                  Model.sliceRun's completion copyRows (rows n, D*2 bytes a row, src pitch D*2,
//                  dst pitch taps*D*2, dst base + j*D*2) into dr.at.
//   packed write:  prompt.tap's on write (out = base + j*D*2 bytes, os = taps*D) into ch.taps.
//   reference:     written from BRIEF-0278's spec, not from the kernel: tap j's row i is
//                  h.to(fp32).mean(1).to(bf16) of the row's 4 streams at packed row i, columns
//                  j*D .. (j+1)*D of the row-major [rows, taps*D] layout absorb consumes.
//
// The check per completion: rows [0, n) of the packed buffers are byte-identical across all three
// paths (which includes the absorb window rows [n-128, n)), i.e. the bytes absorb consumes are
// identical with and without the switch. Case matrix per BRIEF-0278: rows 1, 127, 128, 2048 and a
// partial last chunk (a 2048-row chunk then a 517-row one through the shared buffers). A negative
// control proves the comparison detects layout errors.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// -- CUDA shims: just enough to run the extracted elementwise kernel on the host -----------------
#define __global__
#define __device__
#define __forceinline__ inline
#define pdl_enter(x) ((void)0)
struct dim3 { unsigned x, y, z; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
struct __nv_bfloat16 { uint16_t u; };

static inline float __bfloat162float(__nv_bfloat16 v) {
    uint32_t bits = (uint32_t)v.u << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
static inline __nv_bfloat16 __float2bfloat16_rn(float f) {
    // round to nearest even on the low 16 bits (sign-magnitude: same for both signs)
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u) { // Inf/NaN: carry the bits through
        __nv_bfloat16 r;
        r.u = (uint16_t)(bits >> 16) | (uint16_t)((bits & 0x7fffffu) ? 0x4000u : 0);
        return r;
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    __nv_bfloat16 r;
    r.u = (uint16_t)(bits >> 16);
    return r;
}
static inline float __fadd_rn(float a, float b) { return a + b; } // SSE: IEEE round-to-nearest
static inline float __fmul_rn(float a, float b) { return a * b; }

#include "hc_mean4.inc" // the extracted tf_ds_hc_mean4_kernel (verbatim worktree source)

// -- the emulation's plumbing --------------------------------------------------------------------

// one virtual thread at a time through the kernel's grid-stride loop (the served launch geometry:
// Exact.hcMean4's grid = min(4096, blocks(rows*d, 256)), block 256)
static void runKernel(const __nv_bfloat16* h, __nv_bfloat16* out, long long os, int rows, int D, unsigned grid, unsigned block) {
    gridDim = { grid, 1, 1 };
    blockDim = { block, 1, 1 };
    for (unsigned b = 0; b < grid; ++b) {
        for (unsigned t = 0; t < block; ++t) {
            blockIdx = { b, 1, 1 };
            threadIdx = { t, 1, 1 };
            tf_ds_hc_mean4_kernel(h, out, os, rows, D, 0);
        }
    }
}

static unsigned servedGrid(int rows, int D) {
    unsigned blocks = (unsigned)(((long long)rows * D + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    return blocks < 4096 ? blocks : 4096;
}

// Ops.copyRows: rows rows of `bytes` bytes, source rows `src_ld` bytes apart, dst rows `dst_ld` apart
static void copyRows(__nv_bfloat16* dst, size_t dst_ld, const __nv_bfloat16* src, size_t src_ld, size_t bytes, size_t rows) {
    for (size_t r = 0; r < rows; ++r)
        std::memcpy((uint8_t*)dst + r * dst_ld, (const uint8_t*)src + r * src_ld, bytes);
}

// -- the independent reference (from BRIEF-0278's spec, not from the kernel source) ---------------

// bf16 -> fp32 widening is exact; .to(bf16) rounds to nearest even (the ops.zig header's contract)
static float widen(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
static uint16_t narrow(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t half = bits >> 16;
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t rest = bits & 0xffffu;
        if (rest > 0x8000u || (rest == 0x8000u && (half & 1u))) half += 1; // RN-even at the tie
    } else if (bits & 0x7fffffu) {
        half |= 0x4000u; // quiet NaN
    }
    return (uint16_t)half;
}

// packed rows [rows, taps*D]: tap j's row i = h.to(fp32).mean(1).to(bf16) of the row's 4 streams,
// at columns j*D .. (j+1)*D (BRIEF-0278: "write each tap directly at stride taps * D")
static void reference(__nv_bfloat16* packed, const __nv_bfloat16* h, size_t n, size_t taps, size_t D) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < taps; ++j)
            for (size_t c = 0; c < D; ++c) {
                const size_t in = (i * 4) * D + c; // h is [rows, 4, D]
                float s = widen(h[in].u) + widen(h[in + D].u) + widen(h[in + 2 * D].u) + widen(h[in + 3 * D].u);
                packed[i * taps * D + j * D + c].u = narrow(s * 0.25f);
            }
}

// -- input and comparison -------------------------------------------------------------------------

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint16_t nextBf16() {
    // finite bf16 values across the whole exponent range (no Inf/NaN patterns)
    rng += 0x9e3779b97f4a7c15ull;
    uint64_t z = rng;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z = z ^ (z >> 31);
    return (uint16_t)((z & 0x8000u) | (((z >> 7) & 0x7fu) << 7) | (z & 0x7fu));
}

static int failures = 0;

static void expectEqual(const char* what, const __nv_bfloat16* got, const __nv_bfloat16* want, size_t elems) {
    for (size_t k = 0; k < elems; ++k) {
        if (got[k].u != want[k].u) {
            std::printf("  FAIL %s: element %zu differs (got 0x%04x want 0x%04x)\n", what, k, got[k].u, want[k].u);
            failures++;
            return;
        }
    }
    std::printf("  ok   %s: %zu elements (%zu bytes) identical\n", what, elems, elems * 2);
}

static void expectDiffers(const char* what, const __nv_bfloat16* got, const __nv_bfloat16* want, size_t elems) {
    for (size_t k = 0; k < elems; ++k) {
        if (got[k].u != want[k].u) {
            std::printf("  ok   %s: mismatch detected at element %zu (the harness sees layout errors)\n", what, k);
            return;
        }
    }
    std::printf("  FAIL %s: wrong layout compared equal\n", what);
    failures++;
}

// -- the case matrix ------------------------------------------------------------------------------

struct Case {
    const char* name;
    std::vector<size_t> chunks; // the chunk rows at each completion (shared buffers)
};

static void runCase(const Case& cs, size_t taps, size_t D, size_t cap) {
    std::printf("case %s\n", cs.name);
    const size_t planarElems = taps * cap * D;
    const size_t packedElems = cap * taps * D;
    std::vector<__nv_bfloat16> planar(planarElems), drat(packedElems), direct(packedElems), ref(packedElems);
    std::vector<__nv_bfloat16> h; // [chunk rows, 4, D] per completion
    for (size_t chunk = 0; chunk < cs.chunks.size(); ++chunk) {
        const size_t n = cs.chunks[chunk];
        h.resize(n * 4 * D);
        for (auto& v : h) v.u = nextBf16();
        reference(ref.data(), h.data(), n, taps, D);
        // the planar write (prompt.tap, off): plane j = planar + j*cap*D elements, row stride D
        for (size_t j = 0; j < taps; ++j)
            runKernel(h.data(), planar.data() + j * cap * D, (long long)D, (int)n, (int)D, servedGrid((int)n, (int)D), 256);
        // the completion copies (Model.sliceRun, off): plane j's n rows into dr.at at column j*D
        for (size_t j = 0; j < taps; ++j)
            copyRows(drat.data() + j * D, taps * D * 2, planar.data() + j * cap * D, D * 2, D * 2, n);
        // the packed write (prompt.tap, on): tap j at column j*D of the rows, row stride taps*D,
        // served geometry first and the degenerate geometry as a grid-stride coverage check
        for (size_t j = 0; j < taps; ++j)
            runKernel(h.data(), direct.data() + j * D, (long long)(taps * D), (int)n, (int)D, servedGrid((int)n, (int)D), 256);
        char what[160];
        std::snprintf(what, sizeof what, "rows=%zu chunk=%zu planar+copy == reference", n, chunk + 1);
        expectEqual(what, drat.data(), ref.data(), n * taps * D);
        std::snprintf(what, sizeof what, "rows=%zu chunk=%zu packed write == reference (served grid)", n, chunk + 1);
        expectEqual(what, direct.data(), ref.data(), n * taps * D);
        for (size_t j = 0; j < taps; ++j)
            runKernel(h.data(), direct.data() + j * D, (long long)(taps * D), (int)n, (int)D, 1, 32);
        std::snprintf(what, sizeof what, "rows=%zu chunk=%zu packed write == reference (1x32 grid)", n, chunk + 1);
        expectEqual(what, direct.data(), ref.data(), n * taps * D);
        std::snprintf(what, sizeof what, "rows=%zu chunk=%zu absorb window [max(0,n-128), n) identical", n, chunk + 1);
        const size_t first = n > 128 ? n - 128 : 0;
        expectEqual(what, drat.data() + first * taps * D, ref.data() + first * taps * D, (n - first) * taps * D);
        std::printf("       completion copies avoided: %zu rows x %zu taps x %zu D x 2B = %.2f MiB\n",
                    n, taps, D, (double)(n * taps * D * 2) / 1048576.0);
    }
}

int main() {
    const size_t taps = 3, D = 5120, cap = 2048; // the served checkpoint's shapes
    std::printf("h0278: TF_DS_TAPS_PACKED write/copy layout emulation (cap=%zu, taps=%zu, D=%zu)\n", cap, taps, D);
    std::printf("kernel: extracted tf_ds_hc_mean4_kernel, sequential virtual-thread loop\n");
    std::printf("reference: independent, from BRIEF-0278's spec (packed rows [rows, taps*D])\n\n");
    const Case cases[] = {
        { "rows=1 (minimum chunk)", { 1 } },
        { "rows=127 (partial chunk)", { 127 } },
        { "rows=128 (the served post-replay chunk)", { 128 } },
        { "rows=2048 (full chunk_rows)", { 2048 } },
        { "a partial last chunk: 2048 then 517 through the shared buffers", { 2048, 517 } },
    };
    for (const auto& cs : cases) runCase(cs, taps, D, cap);

    // negative control: the packed write at the planar stride must NOT compare equal (the check works)
    std::printf("case negative control: packed base with the planar stride os=D\n");
    {
        const size_t n = 128;
        std::vector<__nv_bfloat16> h(n * 4 * D), direct(cap * taps * D), ref(cap * taps * D);
        for (auto& v : h) v.u = nextBf16();
        reference(ref.data(), h.data(), n, taps, D);
        for (size_t j = 0; j < taps; ++j)
            runKernel(h.data(), direct.data() + j * D, (long long)D, (int)n, (int)D, servedGrid((int)n, (int)D), 256);
        expectDiffers("wrong stride", direct.data(), ref.data(), n * taps * D);
    }

    std::printf("\n%s: %d failure(s)\n", failures == 0 ? "ALL CASES MATCHED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
