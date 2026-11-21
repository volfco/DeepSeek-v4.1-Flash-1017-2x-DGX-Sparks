// Compare both actual CUDA kernels to greedy selection over the full logical row.
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void check(CUresult rc) {
    if (rc != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(rc, &error);
        std::fprintf(stderr, "%s\n", error);
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    check(cuInit(0));
    CUdevice device;
    CUcontext context;
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));
    CUmodule module;
    CUfunction pair, reduce;
    check(cuModuleLoad(&module, argv[1]));
    check(cuModuleGetFunction(&pair, module, "tf_ds_argmax_pair_kernel"));
    check(cuModuleGetFunction(&reduce, module, "tf_ds_argmax_reduce_kernel"));
    std::puts("ranks,rows,matched,total");
    for (int world : {2, 4}) for (int rows : {1, 4, 13}) {
        int n = 64, pdl = 0;
        unsigned long long ld = n;
        CUdeviceptr logits, pairs, output;
        check(cuMemAlloc(&logits, rows*n*sizeof(float)));
        check(cuMemAlloc(&pairs, world*rows*2*sizeof(unsigned)));
        check(cuMemAlloc(&output, rows*sizeof(unsigned)));
        std::vector<float> full(world*rows*n);
        for (int rank=0; rank<world; ++rank) {
            float *local = full.data()+rank*rows*n;
            for (int r=0; r<rows; ++r) {
                std::fill_n(local+r*n,n,-1.f);
                local[r*n+(7+r)%n] = rank==r%world ? 5.f : 2.f;
                if (r%7==1 && rank==0) local[r*n]=std::numeric_limits<float>::quiet_NaN();
                if (r%7==3 && rank==r%world) local[r*n]=std::numeric_limits<float>::quiet_NaN();
                if (r%7==5) local[r*n+2]=5.f;
            }
            check(cuMemcpyHtoD(logits,local,rows*n*sizeof(float)));
            CUdeviceptr target=pairs+rank*rows*2*sizeof(unsigned);
            void *params[]={&logits,&ld,&n,&target,&pdl};
            check(cuLaunchKernel(pair,rows,1,1,1024,1,1,0,nullptr,params,nullptr));
            check(cuCtxSynchronize());
        }
        void *params[]={&pairs,&world,&n,&rows,&output,&pdl};
        check(cuLaunchKernel(reduce,1,1,1,32,1,1,0,nullptr,params,nullptr));
        std::vector<unsigned> observed(rows);
        check(cuMemcpyDtoH(observed.data(),output,rows*sizeof(unsigned)));
        int matched=0;
        for (int r=0; r<rows; ++r) {
            unsigned best=0;
            float value=full[r*n];
            for(int k=0;k<world*n;++k) {
                float candidate=full[((k/n)*rows+r)*n+k%n];
                if(candidate>value) {value=candidate;best=k;}
            }
            matched+=observed[r]==best;
        }
        std::printf("%d,%d,%d,%d\n",world,rows,matched,rows);
        check(cuMemFree(logits));check(cuMemFree(pairs));check(cuMemFree(output));
    }
    check(cuModuleUnload(module));
    check(cuDevicePrimaryCtxRelease(device));
}
