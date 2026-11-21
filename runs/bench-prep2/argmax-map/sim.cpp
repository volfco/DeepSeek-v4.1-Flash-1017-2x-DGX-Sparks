
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#define __global__
#define __device__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(x)
struct Dim3 { int x; };
static Dim3 threadIdx = {0}, blockIdx = {0}, blockDim = {1,};
static void __syncthreads() {}
static void pdl_enter(int) {}
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static unsigned __float_as_uint(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static float __uint_as_float(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
template <class T> T __shfl_down_sync(unsigned, T v, int) { return v; }
__device__ unsigned tf_ds_block_argmax_v(const float* row, int n, float* out_v) {
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v == v && (v > bv || (v == bv && i < bi))) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return 0;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    *out_v = bv;
    return (unsigned)bi;
}

__device__ unsigned tf_ds_block_argmax(const float* row, int n) {
    float v;
    const unsigned bi = tf_ds_block_argmax_v(row, n, &v);
    return row[0] != row[0] ? 0u : bi;
}

void tf_ds_argmax_pair_kernel(const float* logits, unsigned long long ld,
                                                                           int n, unsigned* out, int pdl) {
    pdl_enter(pdl);
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    float bv = 0.0f;
    const unsigned bi = tf_ds_block_argmax_v(row, n, &bv);
    if (threadIdx.x == 0) {
        out[2 * blockIdx.x] = __float_as_uint(bv);
        out[2 * blockIdx.x + 1] = row[0] != row[0] ? 0xffffffffu : bi;
    }
}

// The ranks' pairs [rows][world][2] in rank order reduced to the round's picks: tf_ds_argmax_rows_kernel's answer for
// the whole vocabulary (the first largest, the lower rank winning a tie, rank 0's row[0] NaN rule as pick 0). A thread a
// row (R <= 16), the pairs as the ranks' exchange left them; `n` is a rank's columns, so rank k's local index i is the
// whole row's k * n + i (the logits' layout: the ranks' halves in rank order).
void tf_ds_argmax_reduce_kernel(const unsigned* pairs, int world, int n, int rows, unsigned* out, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned* p = pairs + (long long)r * world * 2;
    if (p[1] == 0xffffffffu) {   // rank 0's first value was NaN: the row's pick is 0, as the whole-row rule says
        out[r] = 0u;
        return;
    }
    float bv = __uint_as_float(p[0]);
    unsigned bi = p[1];
    for (int k = 1; k < world; k++) {
        const float v = __uint_as_float(p[2 * k]);
        if (v > bv) {   // strict: an earlier rank wins a tie (each rank's own tie already went to its lower column)
            bv = v;
            bi = (unsigned)k * (unsigned)n + p[2 * k + 1];
        }
    }
    out[r] = bi;
}


static unsigned ref_argmax(const float* row, int n) {
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf; int bi = 0x7fffffff;
    for (int i = 0; i < n; i++) { float v = row[i]; if (v == v && (v > bv || (v == bv && i < bi))) { bv = v; bi = i; } }
    return row[0] != row[0] ? 0u : (unsigned)bi;
}

static int failures = 0;
static void one(const char* name, std::vector<float> full, int world) {
    int n = (int)full.size() / world;
    std::vector<unsigned> pairs(world * 2, 0);
    for (int k = 0; k < world; k++) {
        threadIdx.x = 0; blockIdx.x = 0;
        tf_ds_argmax_pair_kernel(full.data() + k * n, n, n, pairs.data() + k * 2, 0);
    }
    std::vector<unsigned> out(1, 0xDEADBEEF);
    threadIdx.x = 0; blockIdx.x = 0;
    tf_ds_argmax_reduce_kernel(pairs.data(), world, n, 1, out.data(), 0);
    const unsigned want = ref_argmax(full.data(), (int)full.size());
    if (out[0] != want) { printf("FAIL %-34s world=%d n=%d got %u want %u\n", name, world, n, out[0], want); failures++; return; }
}

int main() {
    srand(7);
    const float nan = NAN;
    int cases = 0;
    for (int world : {2, 4}) {
        int n = 64;
        { std::vector<float> r(world * n); for (auto& x : r) x = (float)(rand() % 100) / 7.0f; one("random", r, world); }
        { std::vector<float> r(world * n, 1.0f); one("all equal (ties across ranks)", r, world); }
        { std::vector<float> r(world * n); for (int i = 0; i < world * n; i++) r[i] = (float)(i % 5); one("few values, ties in rank order", r, world); }
        { std::vector<float> r(world * n); for (auto& x : r) x = -2.0f; r[0] = nan; one("rank 0 row[0] NaN", r, world); }
        { std::vector<float> r(world * n); for (auto& x : r) x = 3.5f; r[n] = nan; r[world*n-1] = nan; one("later rank NaNs", r, world); }
        { std::vector<float> r(world * n, nan); one("all NaN", r, world); }
        { std::vector<float> r(world * n, nan); r[n + 3] = 9.0f; one("all NaN, one value in rank 1", r, world); }
        { std::vector<float> r(world * n); for (auto& x : r) x = -0.0f; r[0] = 0.0f; one("neg zero and zero", r, world); }
        { std::vector<float> r(world * n); for (auto& x : r) x = (float)(rand() % 1000) / 3.0f; r[n - 1] = 500.0f; r[world*n-1] = 500.0f; one("tie at equal value across ranks", r, world); }
        for (int t = 0; t < 20; t++) {
            std::vector<float> r(world * n);
            int mode = t % 4;
            for (auto& x : r) x = mode == 0 ? (float)(rand() % 20) : mode == 1 ? (float)(rand() % 100000) / 1000.0f - 50.0f : mode == 2 ? (rand() % 3 == 0 ? nan : 1.0f) : 2.0f;
            char nm[64]; snprintf(nm, sizeof nm, "random t=%d mode=%d", t, mode);
            one(nm, r, world);
            cases++;
        }
    }
    printf(failures ? "%d FAILURES of %d\n" : "all %d cases exact\n", failures ? failures : cases + 18);
    return failures != 0;
}
