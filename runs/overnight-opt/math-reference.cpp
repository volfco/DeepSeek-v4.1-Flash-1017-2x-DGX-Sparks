// FP16 inputs and FP32 accumulation. Column-major cuBLAS multiplies row-major A[M,K] by B[K,N].
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <array>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(int rc, const char* call) {
    if (rc != 0) {
        std::fprintf(stderr, "%s failed (%d)\n", call, rc);
        std::exit(1);
    }
}
#define CHECK(call) check((call), #call)

int main(int argc, char** argv) {
    cublasHandle_t handle;
    cudaEvent_t begin, end;
    CHECK(cublasCreate(&handle));
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    std::puts("m,k,n,repeat,ms_gemm,TFLOP_s");
    // Q low-rank projections, KV, output, and one rank-local expert gate.
    std::vector<std::array<int,3>> cases;
    if (argc==2) {
        std::ifstream in(argv[1]);
        int m,k,n;
        while (in >> m >> k >> n) cases.push_back({m,k,n});
        if (!in.eof() || cases.empty()) return 2;
    } else if (argc==1) {
        for (auto [k,n] : std::vector<std::pair<int,int>>{
                 {5120,1280},{1280,16384},{5120,512},{4096,1024},{4096,5120},{5120,1152}})
            for (int m : {1,4,32,512,2048}) cases.push_back({m,k,n});
    } else return 2;
    for (auto [m,k,n] : cases) {
        if (m<=0 || k<=0 || n<=0 || m>100000 || n>200000 || k>200000) return 2;
        void *a,*b,*c;
        size_t a_bytes = size_t(2)*m*k, b_bytes=size_t(2)*k*n, c_bytes=size_t(2)*m*n;
        if (a_bytes+b_bytes+c_bytes > 384ull*1024*1024) {
            std::fprintf(stderr,"Shape exceeds calibration buffer budget: %d %d %d\n",m,k,n);
            continue;
        }
        CHECK(cudaMalloc(&a,a_bytes));
        CHECK(cudaMalloc(&b,b_bytes));
        CHECK(cudaMalloc(&c,c_bytes));
        CHECK(cudaMemset(a,0x35,a_bytes));
        CHECK(cudaMemset(b,0x34,b_bytes));
        float alpha=1.f, beta=0.f;
        auto gemm = [&]() {
            CHECK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,
                  b,CUDA_R_16F,n,a,CUDA_R_16F,k,&beta,c,CUDA_R_16F,n,
                  CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        for (int i=0; i<4; ++i) gemm();
        const int reps = m >= 512 ? 10 : 48;
        for (int r=0; r<3; ++r) {
            CHECK(cudaEventRecord(begin));
            for (int i=0; i<reps; ++i) gemm();
            CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end));
            float ms;
            CHECK(cudaEventElapsedTime(&ms,begin,end));
            std::printf("%d,%d,%d,%d,%.6f,%.5f\n",m,k,n,r,ms/reps,
                        (2.0*m*k*n)*reps/ms/1e9);
        }
        CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    }
    CHECK(cudaEventDestroy(end)); CHECK(cudaEventDestroy(begin)); CHECK(cublasDestroy(handle));
}
