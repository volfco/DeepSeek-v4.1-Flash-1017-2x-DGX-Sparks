#include <cstdint>
extern "C" __global__ void k_coal(unsigned long long* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = (unsigned long long)i;
}
extern "C" __global__ void k_fold(unsigned long long* dst, int nb, int row) {
    for (int first = 0; first < nb; first += blockDim.x) {
        const int b = first + threadIdx.x;
        if (b >= nb) continue;
        for (int j = 0; j < 8; ++j) dst[(long long)blockIdx.x * nb * 8 + b * 8 + j] = (unsigned long long)(b * 8 + j);
    }
}
extern "C" __global__ void k_fold128(unsigned long long* dst, int nb, int row) {
    for (int first = 0; first < nb; first += blockDim.x) {
        const int b = first + threadIdx.x;
        if (b >= nb) continue;
        ulonglong2* p = (ulonglong2*)(dst + (long long)blockIdx.x * nb * 8 + b * 8);
        for (int j = 0; j < 4; ++j) p[j] = make_ulonglong2((unsigned long long)(b * 8 + 2 * j), 7);
    }
}
extern "C" __global__ void k_stage(unsigned long long* dst, int nb, int row) {
    __shared__ unsigned long long st[512 * 8];
    for (int first = 0; first < nb; first += blockDim.x) {
        const int b = first + threadIdx.x;
        if (b < nb) for (int j = 0; j < 8; ++j) st[threadIdx.x * 8 + j] = (unsigned long long)(b * 8 + j);
        __syncthreads();
        unsigned long long* out = dst + (long long)blockIdx.x * nb * 8 + (long long)first * 8;
        for (int i = threadIdx.x; i < 4096 && first * 8 + i < nb * 8; i += blockDim.x) out[i] = st[i];
        __syncthreads();
    }
}
