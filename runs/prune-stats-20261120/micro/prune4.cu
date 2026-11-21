// DeepSeek-V4.1 (zig/src/families/dsv41): the torch casts its served forward makes that torch_ops/ does not have.
// Built like torch_ops (no fast math, --fmad=false --ftz=false); raw device pointers, the caller's stream.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// TF_DS_PDL: wait for the kernel this one follows (its memory flushed), then let the next one launch (programmatic
// dependent launches, sm_90+). Every kernel in this file takes `pdl` as its last argument (0: same instruction stream as
// before); the l2 prefetch kernel is the one exception: it reads the NEXT kernels' weights into L2 and must not run
// mid-flight against its predecessor's working set, so it neither takes `pdl` nor is launched with the attribute.
__device__ __forceinline__ void pdl_enter(int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

struct TfDsPrune { float topp; int min_k, renorm, apply, topk, slots, rows, cap, cap_min; };
#define TF_DS_PRUNE_ROWS 64
#define TF_DS_PRUNE_TOPK 8
#define TF_DS_PRUNE_EXPERTS 512
extern "C" __global__ void __launch_bounds__(TF_DS_PRUNE_ROWS) tf_ds_expert_prune_kernel(int* pick, float* wts, TfDsPrune a, unsigned long long* stats, const unsigned char* dead, int pdl) {
    pdl_enter(pdl);
    __shared__ unsigned mb[16], ma[16], acc[3], ins[16];
    __shared__ int sp[TF_DS_PRUNE_ROWS * TF_DS_PRUNE_TOPK];
    __shared__ float sf[TF_DS_PRUNE_ROWS * TF_DS_PRUNE_TOPK], score[TF_DS_PRUNE_EXPERTS];
    const int r = threadIdx.x;
    if (stats) {
        if (r < 16) { mb[r] = 0; ma[r] = 0; }
        if (r < 3) acc[r] = 0;
    }
    if (a.cap > 0 && r < 16) ins[r] = 0;
    // Without stats or a cap, each thread only touches its own row.
    if (stats || a.cap > 0) __syncthreads();
    int* pk = pick + r * a.slots;
    float* w = wts + r * a.slots;
    int p[8], rank[8], live = 0;
    float f[8], ex[8], tot = 0.f;
    bool keep[8], ok = false;
    if (r < a.rows) {
        for (int i = 0; i < a.topk; ++i) {
            p[i] = pk[i];
            keep[i] = true;
            if (p[i] < 0) continue;
            ++live;
            tot = __fadd_rn(tot, w[i]);
            if (stats && p[i] < 512) atomicOr(&mb[p[i] >> 5], 1u << (p[i] & 31));
        }
        ok = live > 0 && tot > 0.f && !isinf(tot);
        if (ok) {
            for (int i = 0; i < a.topk; ++i) f[i] = p[i] >= 0 ? __fdiv_rn(w[i], tot) : 0.f;
            for (int i = 0; i < a.topk; ++i) {
                rank[i] = 0;
                if (p[i] < 0) continue;
                for (int j = 0; j < a.topk; ++j) rank[i] += p[j] >= 0 && (f[j] > f[i] || (f[j] == f[i] && j < i));
            }
            float run = 0.f;
            for (int k = 0; k < live; ++k)
                for (int i = 0; i < a.topk; ++i)
                    if (p[i] >= 0 && rank[i] == k) { ex[i] = run; run = __fadd_rn(run, f[i]); }
            const int floor_k = a.min_k > 1 ? a.min_k : 1;
            for (int i = 0; i < a.topk; ++i)
                if (p[i] >= 0) keep[i] = rank[i] < floor_k || ex[i] < a.topp;
        }
        if (a.cap > 0)
            for (int i = 0; i < a.topk; ++i) {
                const bool in = ok && p[i] >= 0 && p[i] < TF_DS_PRUNE_EXPERTS && keep[i];
                sp[r * TF_DS_PRUNE_TOPK + i] = in ? p[i] : -1;
                sf[r * TF_DS_PRUNE_TOPK + i] = in ? f[i] : 0.f;
            }
    }
    if (a.cap > 0) {
        // over the window's picks only: an expert's score is summed once, at its first pick, in row and slot order;
        // score[x] < 0 marks a pick that is not an expert's first
        __syncthreads();
        const int ne = a.rows * TF_DS_PRUNE_TOPK;
        for (int x = r; x < ne; x += TF_DS_PRUNE_ROWS) {
            const int e = (x % TF_DS_PRUNE_TOPK) < a.topk ? sp[x] : -1;
            float s = -1.f;
            if (e >= 0) {
                bool first = true;
                for (int y = 0; y < x && first; ++y) first = !((y % TF_DS_PRUNE_TOPK) < a.topk && sp[y] == e);
                if (first) {
                    s = 0.f;
                    for (int y = x; y < ne; ++y)
                        if ((y % TF_DS_PRUNE_TOPK) < a.topk && sp[y] == e) s = __fadd_rn(s, sf[y]);
                }
            }
            score[x] = s;
        }
        __syncthreads();
        for (int x = r; x < ne; x += TF_DS_PRUNE_ROWS) {
            if (score[x] < 0.f) continue;
            const int e = sp[x];
            int k = 0;
            for (int y = 0; y < ne; ++y)
                k += score[y] >= 0.f && (score[y] > score[x] || (score[y] == score[x] && sp[y] < e));
            if (k < a.cap) atomicOr(&ins[e >> 5], 1u << (e & 31));
        }
        __syncthreads();
    }
    if (r < a.rows) {
        if (ok) {
            int kept = 0;
            float ks = 0.f, mass = 0.f;
            for (int i = 0; i < a.topk; ++i) {
                if (p[i] < 0) continue;
                if (a.cap > 0 && keep[i] && rank[i] >= a.cap_min &&
                    (p[i] >= TF_DS_PRUNE_EXPERTS || !((ins[p[i] >> 5] >> (p[i] & 31)) & 1u)))
                    keep[i] = false;
                if (keep[i]) { ++kept; ks = __fadd_rn(ks, w[i]); }
                else if (stats) mass = __fadd_rn(mass, f[i]);
            }
            if (kept < live) {
                if (a.apply) {
                    const float g = __fdiv_rn(tot, ks);
                    for (int i = 0; i < a.topk; ++i) {
                        if (p[i] < 0) continue;
                        if (keep[i]) { if (a.renorm) w[i] = __fmul_rn(w[i], g); }
                        else { pk[i] = -1; w[i] = 0.f; }
                    }
                }
                if (stats) {
                    atomicAdd(&acc[0], (unsigned)(live - kept));
                    atomicAdd(&acc[2], (unsigned)(mass * 1e6f + 0.5f));
                }
            }
        } else if (a.cap > 0 && live > a.cap_min) {
            // a row with no usable weight sum keeps its first cap_min live slots (route's heaviest): every row then
            // keeps its picks in the cap's set and at most cap_min more, which bounds the window's places
            int seen = 0;
            for (int i = 0; i < a.topk; ++i) {
                if (p[i] < 0) continue;
                keep[i] = seen < a.cap_min;
                ++seen;
                if (!keep[i] && a.apply) { pk[i] = -1; w[i] = 0.f; }
            }
            if (stats) atomicAdd(&acc[0], (unsigned)(live - a.cap_min));
        }
        if (stats) {
            atomicAdd(&acc[1], 1u);
            for (int i = 0; i < a.topk; ++i)
                if (p[i] >= 0 && keep[i] && p[i] < 512) atomicOr(&ma[p[i] >> 5], 1u << (p[i] & 31));
        }
        if (dead && dead[r]) {
            for (int i = 0; i < a.slots; ++i) { pk[i] = -1; w[i] = 0.f; }
        }
    }
    if (stats) __syncthreads();
    if (stats && r == 0) {
        int before = 0, after = 0;
        for (int i = 0; i < 16; ++i) { before += __popc(mb[i]); after += __popc(ma[i]); }
        atomicAdd(stats, 1ull);
        atomicAdd(stats + 1, (unsigned long long)before);
        atomicAdd(stats + 2, (unsigned long long)after);
        atomicAdd(stats + 3, (unsigned long long)acc[0]);
        atomicAdd(stats + 4, (unsigned long long)acc[1]);
        atomicAdd(stats + 5, (unsigned long long)acc[2]);
    }
}

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Slot { float share; int index; bool valid; };
__device__ __forceinline__ void compareSlots(Slot &a, Slot &b) {
    if ((b.valid && !a.valid) || (a.valid == b.valid &&
        (b.share > a.share || (b.share == a.share && b.index < a.index)))) {
        const Slot tmp=a; a=b; b=tmp;
    }
}

__global__ __launch_bounds__(64) void pruneSix(int* picks, float* weights, int rows, const unsigned char* dead) {
    int row=threadIdx.x;
    if (row >= rows) return;
    picks += 7*row;
    weights += 7*row;
    if (dead && dead[row]) {
        #pragma unroll
        for (int i=0;i<7;++i) { picks[i]=-1; weights[i]=0; }
        return;
    }
    float total=0, w[6];
    Slot s[6];
    #pragma unroll
    for (int i=0;i<6;++i) {
        w[i]=weights[i];
        s[i]={0,i,picks[i]>=0};
        if (s[i].valid) total=__fadd_rn(total,w[i]);
    }
    if (!(total>0) || isinf(total)) return;
    #pragma unroll
    for (int i=0;i<6;++i) if (s[i].valid) s[i].share=__fdiv_rn(w[i],total);
    compareSlots(s[0],s[5]); compareSlots(s[1],s[3]); compareSlots(s[2],s[4]);
    compareSlots(s[1],s[2]); compareSlots(s[3],s[4]);
    compareSlots(s[0],s[3]); compareSlots(s[2],s[5]);
    compareSlots(s[0],s[1]); compareSlots(s[2],s[3]); compareSlots(s[4],s[5]);
    compareSlots(s[1],s[2]); compareSlots(s[3],s[4]);
    float prefix=0;
    unsigned drop=0;
    #pragma unroll
    for (int i=0;i<6;++i) {
        if (s[i].valid) {
            if (i>=3 && prefix>=0.85f) drop |= 1u<<s[i].index;
            prefix=__fadd_rn(prefix,s[i].share);
        }
    }
    if (!drop) return;
    float kept=0;
    #pragma unroll
    for (int i=0;i<6;++i) if (picks[i]>=0 && !(drop & (1u<<i))) kept=__fadd_rn(kept,w[i]);
    float scale=__fdiv_rn(total,kept);
    #pragma unroll
    for (int i=0;i<6;++i) if (picks[i]>=0) {
        if (drop & (1u<<i)) { picks[i]=-1; weights[i]=0; }
        else weights[i]=__fmul_rn(w[i],scale);
    }
}

__global__ void pruneTile(int* picks, float* weights, int rows, const unsigned char* dead) {
    const int tid=threadIdx.x, lane=tid & 7, row=blockIdx.x*4+tid/8;
    if(row>=rows) return;
    const unsigned mask=0xffu << (tid & 24);
    int pos=row*7+lane;
    if (dead && dead[row]) {
        if (lane<7) { picks[pos]=-1; weights[pos]=0.f; }
        return;
    }
    bool valid=lane < 6 && picks[pos]>=0;
    float w=valid?weights[pos]:0.f;
    float total=0.f;
    #pragma unroll
    for(int i=0;i<6;++i) total=__fadd_rn(total,__shfl_sync(mask,w,i,8));
    if(!(total>0) || isinf(total)) return;
    float share=valid?__fdiv_rn(w,total):0.f;
    unsigned live=__ballot_sync(mask,valid)>>(tid & 24);
    int rank=0;
    #pragma unroll
    for(int i=0;i<6;++i) {
        float val=__shfl_sync(mask,share,i,8);
        rank += ((live>>i)&1u) && (val>share || (val==share && i<lane));
    }
    float prefix=0.f;
    #pragma unroll
    for(int k=0;k<5;++k) {
        unsigned winner=__ballot_sync(mask,valid && rank==k);
        int src=(__ffs(winner)-1)&7;
        float val=__shfl_sync(mask,share,src,8);
        if(k<rank && winner) prefix=__fadd_rn(prefix,val);
    }
    bool drop=valid && rank>=3 && prefix>=0.85f;
    if(!__any_sync(mask,drop)) return;
    float kept=0.f, v=drop?0.f:w;
    #pragma unroll
    for(int i=0;i<6;++i) kept=__fadd_rn(kept,__shfl_sync(mask,v,i,8));
    if(drop){ picks[pos]=-1; weights[pos]=0.f; }
    else if(valid) weights[pos]=__fmul_rn(w,__fdiv_rn(total,kept));
}

#define CHECK(call) do { cudaError_t err=(call); if(err != cudaSuccess){ std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(err)); std::exit(2); } } while(0)
int main(int argc, char** argv){
    const int deadPct=argc==1 ? -1 : std::atoi(argv[1]);
    constexpr int nodes=256;
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t start,end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    for (int rows: {1,7,28,64}) {
        int n=nodes*rows*7;
        std::vector<int> hp(n), refp(n), gotp(n);
        std::vector<float> hw(n), refw(n), gotw(n);
        unsigned seed=51;
        auto next=[&](){seed=1664525u*seed+1013904223u;return seed>>8;};
        for(int j=0;j<n;++j){
            hp[j]=j%7==6?384:int(next()%384);
            hw[j]=j%7==6?1.f:(next()%12+1)*0.03125f;
            if(j%7!=6 && (next()%19==0)) hp[j]=-1;
        }
        int *dp; float *dw; unsigned long long *stats;
        CHECK(cudaMalloc(&dp,n*sizeof(int))); CHECK(cudaMalloc(&dw,n*sizeof(float))); CHECK(cudaMalloc(&stats,48)); CHECK(cudaMemset(stats,0,48));
        unsigned char* mask=nullptr;
        if (deadPct>=0) {
            std::vector<unsigned char> hm(nodes*rows);
            for (auto& byte:hm) byte=next()%100 < unsigned(deadPct);
            CHECK(cudaMalloc(&mask,hm.size())); CHECK(cudaMemcpy(mask,hm.data(),hm.size(),cudaMemcpyHostToDevice));
        }
        cudaGraphExec_t execs[4]; 
        for(int mode=0;mode<4;++mode){
            cudaGraph_t graph;
            CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int k=0;k<nodes;++k){
                int offset=k*rows*7;
                auto dead=mask ? mask+k*rows : nullptr;
                TfDsPrune a={0.85f,3,1,1,6,7,rows,0,0};
                if(mode==3) pruneTile<<<(rows+3)/4,32,0,stream>>>(dp+offset,dw+offset,rows,dead);
                else if(mode==2) pruneSix<<<1,64,0,stream>>>(dp+offset,dw+offset,rows,dead);
                else tf_ds_expert_prune_kernel<<<1,64,0,stream>>>(dp+offset,dw+offset,a,mode==0?stats:nullptr,dead,0);
            }
            CHECK(cudaStreamEndCapture(stream,&graph)); CHECK(cudaGraphInstantiate(&execs[mode],graph,0)); CHECK(cudaGraphDestroy(graph));
        }
        for(int pass=0;pass<13;++pass){
            for(int j=0;j<4;++j){
                int mode=(pass+j)%4;
                CHECK(cudaMemcpyAsync(dp,hp.data(),n*sizeof(int),cudaMemcpyHostToDevice,stream));
                CHECK(cudaMemcpyAsync(dw,hw.data(),n*sizeof(float),cudaMemcpyHostToDevice,stream));
                CHECK(cudaEventRecord(start,stream)); CHECK(cudaGraphLaunch(execs[mode],stream)); CHECK(cudaEventRecord(end,stream)); CHECK(cudaEventSynchronize(end));
                if(!pass){
                    CHECK(cudaMemcpy(gotp.data(),dp,n*sizeof(int),cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(gotw.data(),dw,n*sizeof(float),cudaMemcpyDeviceToHost));
                    if(!mode){refp=gotp;refw=gotw;}
                    else if(gotp!=refp || std::memcmp(gotw.data(),refw.data(),n*sizeof(float))){std::fprintf(stderr,"mismatch rows=%d mode=%d\n",rows,mode);return 3;}
                } else {
                    float ms; CHECK(cudaEventElapsedTime(&ms,start,end));
                    std::printf("%d,%d,%d,%.5f\n",rows,pass,mode,1000*ms/nodes);
                }
            }
        }
        for(auto exec:execs) CHECK(cudaGraphExecDestroy(exec));
        if(mask) CHECK(cudaFree(mask));
        CHECK(cudaFree(stats)); CHECK(cudaFree(dw)); CHECK(cudaFree(dp));
    }
}
