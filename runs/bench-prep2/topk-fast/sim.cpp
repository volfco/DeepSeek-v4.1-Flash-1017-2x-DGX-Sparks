
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define __global__
#define __device__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(x)
struct Dim3 { int x, y, z; };
static Dim3 threadIdx = {0,0,0}, blockIdx = {0,0,0}, blockDim = {1,1,1};
static void __syncthreads() {}
static void pdl_enter(int) {}
static unsigned int atomicAdd(unsigned int* p, unsigned int v) { unsigned int o=*p; *p+=v; return o; }
static int atomicAdd(int* p, int v) { int o=*p; *p+=v; return o; }
static unsigned long long tf_ds_score_key(float x, int j) {
    unsigned int bits = (unsigned int)*(uint32_t*)&x;
    if (bits == 0x80000000u) bits = 0u;
    int sb = (int)bits;
    int ordered = sb < 0 ? (sb ^ 0x7FFFFFFF) : sb;
    unsigned long long key = ((unsigned long long)(long long)ordered << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned int)j);
    return key ^ 0x8000000000000000ull;
}
// TF_DS_TOPK_FAST (tf_ds_topk_indices_fast_kernel): the same selection in three reads of the row instead of ten. A
// systematic sample of the row (every `stride`-th column, ~4096 of them) is radix-selected to the sample's rank of k,
// which estimates the k-th largest key; one pass then compacts every key at or above that estimate into shared memory;
// the exact k-th largest is selected among them by the same eight-digit radix rule. An estimate the row does not
// honour (fewer than k keys above it, or more than TF_DS_TOPK_CAP) falls back to the eight-digit select over the whole
// row, so whichever path runs, the k selected keys and their order are the exact top-k (the keys are unique, so no
// estimate can change the set).
#define TF_DS_TOPK_CAP 4096
#define TF_DS_TOPK_SAMPLE 4096

// The rank-th largest of a row's keys computed on the fly, every `stride`-th column of it (`stride` 1: the whole row):
// the same eight-digit radix select as tf_ds_topk_indices_kernel's (that loop, `count` keys wide).
__device__ __forceinline__ unsigned long long tf_ds_radix_kth_row(const float* row, int n, int stride, int count, int rank,
                                                                  unsigned int* hist, unsigned long long* prefix_p,
                                                                  unsigned long long* mask_p, int* want_p) {
    if (threadIdx.x == 0) {
        *prefix_p = 0;
        *mask_p = 0;
        *want_p = rank;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = *prefix_p, mask = *mask_p;
        if (stride == 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const unsigned long long u = tf_ds_score_key(row[i], i);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
            }
        } else {
            for (int s = threadIdx.x; s < count; s += blockDim.x) {
                const int i = s * stride;
                const unsigned long long u = tf_ds_score_key(row[i], i);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int want = *want_p;
            int b = 255;
            for (; b > 0; --b) {
                if ((int)hist[b] >= want) break;
                want -= (int)hist[b];
            }
            *want_p = want;
            *prefix_p = prefix | ((unsigned long long)b << (8 * d));
            *mask_p = mask | (0xffull << (8 * d));
        }
        __syncthreads();
    }
    return *prefix_p;
}

// The rank-th largest of `count` keys already in shared memory (`v`), by the same eight-digit radix rule: a positive
// shared scan a digit, `count` <= TF_DS_TOPK_CAP.
__device__ __forceinline__ unsigned long long tf_ds_radix_kth_shared(const unsigned long long* v, int count, int rank,
                                                                     unsigned int* hist, unsigned long long* prefix_p,
                                                                     unsigned long long* mask_p, int* want_p) {
    if (threadIdx.x == 0) {
        *prefix_p = 0;
        *mask_p = 0;
        *want_p = rank;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = *prefix_p, mask = *mask_p;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const unsigned long long u = v[i];
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int want = *want_p;
            int b = 255;
            for (; b > 0; --b) {
                if ((int)hist[b] >= want) break;
                want -= (int)hist[b];
            }
            *want_p = want;
            *prefix_p = prefix | ((unsigned long long)b << (8 * d));
            *mask_p = mask | (0xffull << (8 * d));
        }
        __syncthreads();
    }
    return *prefix_p;
}

void tf_ds_topk_indices_fast_kernel(const float* score, long long ss, int n,
                                                                                  int k, const long long* vis,
                                                                                  long long* out, int pdl) {
    pdl_enter(pdl);
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s, count_s;
    __shared__ unsigned long long cand[TF_DS_TOPK_CAP];
    const float* row = score + (long long)blockIdx.x * ss;
    long long* const cols = (long long*)cand;   // the tail's columns: the candidates are dead by then
    unsigned long long kth;
    bool have = false;
    const int stride = n > TF_DS_TOPK_SAMPLE ? n / TF_DS_TOPK_SAMPLE : 1;
    if (stride > 1) {
        const int sn = (n + stride - 1) / stride;
        const int rank = (int)(((long long)k * sn + n - 1) / n);   // the sample's rank of the row's k (k >= 1)
        const unsigned long long t = tf_ds_radix_kth_row(row, n, stride, sn, rank, hist, &prefix_s, &mask_s, &want_s);
        if (threadIdx.x == 0) count_s = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned long long u = tf_ds_score_key(row[i], i);
            if (u >= t) {
                const int at = atomicAdd(&count_s, 1);
                if (at < TF_DS_TOPK_CAP) cand[at] = u;
            }
        }
        __syncthreads();
        const int c = count_s;
        if (c >= k and c <= TF_DS_TOPK_CAP) {   // (uniform: every thread reads the same count)
            kth = tf_ds_radix_kth_shared(cand, c, k, hist, &prefix_s, &mask_s, &want_s);
            have = true;
        }
    }
    if (!have) kth = tf_ds_radix_kth_row(row, n, 1, n, k, hist, &prefix_s, &mask_s, &want_s);
    // from here tf_ds_topk_indices_kernel's own tail: the keys at or above the k-th, their columns sorted ascending,
    // a column at or past vis[r] as -1
    if (threadIdx.x == 0) count_s = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (tf_ds_score_key(row[i], i) >= kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) cols[at] = i;
        }
    }
    int p = 1;
    while (p < k) p <<= 1;
    for (int i = k + threadIdx.x; i < p; i += blockDim.x) cols[i] = 0x7FFFFFFFFFFFFFFFll;
    for (int size = 2; size <= p; size <<= 1) {
        for (int stride2 = size >> 1; stride2 > 0; stride2 >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < p; i += blockDim.x) {   // (p up to 2048: two elements a thread)
                const int j = i ^ stride2;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    const long long a = cols[i], b = cols[j];
                    if ((a > b) == up) {
                        cols[i] = b;
                        cols[j] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    const long long v = vis[blockIdx.x];
    long long* o = out + (long long)blockIdx.x * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) o[i] = cols[i] < v ? cols[i] : -1;
}


// brute force: the k largest keys' columns, ascending, a column at or past vis as -1
static void brute(const float* row, int n, int k, long long v, std::vector<long long>& out) {
    std::vector<unsigned long long> keys(n);
    for (int i = 0; i < n; i++) keys[i] = tf_ds_score_key(row[i], i);
    std::sort(keys.begin(), keys.end(), std::greater<unsigned long long>());
    std::vector<long long> cols(k);
    for (int i = 0; i < k; i++) cols[i] = (long long)(0xFFFFFFFFu - (unsigned int)(keys[i] & 0xFFFFFFFFull));
    std::sort(cols.begin(), cols.end());
    out.resize(k);
    for (int i = 0; i < k; i++) out[i] = cols[i] < v ? cols[i] : -1;
}

static int failures = 0;
static void one(const char* name, std::vector<float> row, int n, int k, long long vis) {
    std::vector<long long> out(k, 0), want;
    blockIdx.x = 0;
    tf_ds_topk_indices_fast_kernel(row.data(), n, n, k, &vis, out.data(), 0);
    brute(row.data(), n, k, vis, want);
    for (int i = 0; i < k; i++) {
        if (out[i] != want[i]) {
            printf("FAIL %s n=%d k=%d at %d: got %lld want %lld\n", name, n, k, i, out[i], want[i]);
            failures++;
            return;
        }
    }
    printf("ok   %-28s n=%-7d k=%d\n", name, n, k);
}

int main() {
    srand(12345);
    // 1. large row, gaussian-ish scores (the indexer's shape), k = 2048
    {
        std::vector<float> r(131072);
        for (auto& x : r) x = (float)((rand() % 10000) / 1000.0 - 5.0);
        one("gauss n=131072 k=2048", r, 131072, 2048, 1LL << 40);
    }
    // 2. small row: the plain path
    { std::vector<float> r(300); for (auto& x : r) x = (float)(rand() % 100) / 7.0f; one("small n=300 k=16", r, 300, 16, 1LL << 40); }
    // 3. exactly the sample size
    { std::vector<float> r(4096); for (auto& x : r) x = (float)(rand() % 1000) / 3.0f; one("n=4096 k=64", r, 4096, 64, 1LL << 40); }
    // 4. degenerate: every score identical (the compaction sees n equal keys -> fallback)
    { std::vector<float> r(20000, 1.5f); one("all equal n=20000 k=32", r, 20000, 32, 1LL << 40); }
    // 5. heavy ties on a few values
    { std::vector<float> r(50000); for (size_t i = 0; i < r.size(); i++) r[i] = (float)(i % 7); one("few values n=50000 k=1024", r, 50000, 1024, 1LL << 40); }
    // 6. sorted ascending (stride sampling would misread a trend)
    { std::vector<float> r(60000); for (size_t i = 0; i < r.size(); i++) r[i] = (float)i / 1000.0f; one("ascending n=60000 k=128", r, 60000, 128, 1LL << 40); }
    // 7. sorted descending
    { std::vector<float> r(60000); for (size_t i = 0; i < r.size(); i++) r[i] = (float)(r.size() - i) / 1000.0f; one("descending n=60000 k=128", r, 60000, 128, 1LL << 40); }
    // 8. vis hiding most of the row
    { std::vector<float> r(20000); for (auto& x : r) x = (float)(rand() % 1000) / 11.0f; one("vis hides most", r, 20000, 512, 100); }
    // 9. negatives and zero
    { std::vector<float> r(30000); for (auto& x : r) x = (float)(rand() % 2000 - 1000) / 13.0f; one("negatives n=30000 k=2048", r, 30000, 2048, 1LL << 40); }
    // 10. k == n
    { std::vector<float> r(5000); for (auto& x : r) x = (float)(rand() % 100) / 3.0f; one("k==n n=5000", r, 5000, 5000 > 2048 ? 2048 : 5000, 1LL << 40); }
    // 11. random rows, many shapes
    for (int t = 0; t < 40; t++) {
        int n = 200 + rand() % 100000;
        int k = 1 + rand() % std::min(2048, n);
        std::vector<float> r(n);
        int mode = rand() % 3;
        for (auto& x : r) x = mode == 0 ? (float)(rand() % 5000) / 7.0f : mode == 1 ? (float)(rand() % 40) : (float)((rand() % 2000000) / 1000.0 - 1000.0);
        char nm[64]; snprintf(nm, sizeof nm, "random t=%d mode=%d", t, mode);
        one(nm, r, n, k, (long long)(rand() % (n + 10)));
    }
    printf(failures ? "%d FAILURES\n" : "all cases exact\n", failures);
    return failures != 0;
}
