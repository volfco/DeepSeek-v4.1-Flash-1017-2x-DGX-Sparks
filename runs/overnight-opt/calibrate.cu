#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t rc) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
        std::exit(1);
    }
}

__global__ void scan(const uint4 *source, uint4 *outputs, int count) {
    uint4 total{};
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x) {
        uint4 v=source[i];
        total.x+=v.x; total.y+=v.y; total.z+=v.z; total.w+=v.w;
    }
    outputs[blockIdx.x*blockDim.x+threadIdx.x]=total;
}

int main() {
    const size_t bytes=256ul<<20;
    uint4 *source, *result;
    check(cudaMalloc(&source,bytes));
    check(cudaMalloc(&result,256*256*sizeof(uint4)));
    check(cudaMemset(source,17,bytes));
    scan<<<256,256>>>(source,result,bytes/sizeof(uint4));
    check(cudaDeviceSynchronize());
    auto start=std::chrono::system_clock::now();
    const int repeats=1024;
    for (int i=0;i<repeats;i++) scan<<<256,256>>>(source,result,bytes/sizeof(uint4));
    check(cudaDeviceSynchronize());
    auto end=std::chrono::system_clock::now();
    std::printf("read,%lld,%lld,%llu\n",
                (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count(),
                (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count(),
                (unsigned long long)bytes*repeats);
    check(cudaFree(source)); check(cudaFree(result));
}
