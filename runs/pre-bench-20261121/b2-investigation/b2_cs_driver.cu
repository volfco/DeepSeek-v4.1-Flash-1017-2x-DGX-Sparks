// Minimal CUDA Driver API harness for compute-sanitizer (initcheck, racecheck, memcheck)
// Launches tf_ds_block_max_kernel + tf_ds_idx_pick_kernel and tf_ds_compress2x_norm512_kernel
// across WP2 (b28310e) and tip (d0c9fb1) cubins with pdl=0 and pdl=1. Total VRAM < 4 MiB.

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CU(call)                                                                 \
    do {                                                                               \
        CUresult _err = (call);                                                        \
        if (_err != CUDA_SUCCESS) {                                                    \
            const char* _s = nullptr;                                                  \
            cuGetErrorString(_err, &_s);                                               \
            fprintf(stderr, "CUDA error at %s:%d: %d (%s)\n", __FILE__, __LINE__,      \
                    (int)_err, _s ? _s : "?");                                         \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static double mem_available_gib() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    long kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    }
    fclose(f);
    return (double)kb / (1024.0 * 1024.0);
}

static void launch_kernel(CUfunction fn, unsigned gx, unsigned gy, unsigned bx, unsigned by,
                          unsigned shared_bytes, CUstream stream, void** params, int pdl) {
    if (shared_bytes > 49152) {
        CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shared_bytes));
    }
    if (!pdl) {
        CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, shared_bytes, stream, params, nullptr));
    } else {
        CUlaunchAttribute attr;
        memset(&attr, 0, sizeof(attr));
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.gridDimX = gx;
        cfg.gridDimY = gy;
        cfg.gridDimZ = 1;
        cfg.blockDimX = bx;
        cfg.blockDimY = by;
        cfg.blockDimZ = 1;
        cfg.sharedMemBytes = shared_bytes;
        cfg.hStream = stream;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        CHECK_CU(cuLaunchKernelEx(&cfg, fn, params, nullptr));
    }
}

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_normalish(uint32_t* state) {
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) {
        s += (float)(xorshift32(state) & 0xFFFFu) / 65535.0f - 0.5f;
    }
    return s;
}

static void run_idx_suite(const char* tag, CUmodule mod_ops, CUstream stream) {
    CUfunction fn_bmax = nullptr, fn_pick = nullptr;
    CHECK_CU(cuModuleGetFunction(&fn_bmax, mod_ops, "tf_ds_block_max_kernel"));
    CHECK_CU(cuModuleGetFunction(&fn_pick, mod_ops, "tf_ds_idx_pick_kernel"));

    const int shapes[2][2] = {{1, 8192}, {7, 8192}};
    uint32_t rng = 197u;

    for (int si = 0; si < 2; ++si) {
        int rows = shapes[si][0];
        int nb = shapes[si][1];
        int bsize = 8;
        int width = 8 * nb - 3;
        long long ss = 8LL * nb;
        long long os = nb;
        int k = 2048;

        size_t score_bytes = (size_t)rows * (size_t)ss * sizeof(float);
        size_t vis_bytes = (size_t)rows * sizeof(long long);
        size_t bmax_bytes = (size_t)rows * (size_t)nb * sizeof(float);
        size_t cblk_bytes = (size_t)rows * (size_t)k * sizeof(int);

        float* h_score = (float*)malloc(score_bytes);
        long long* h_vis = (long long*)malloc(vis_bytes);
        long long* h_every = (long long*)malloc(vis_bytes);
        int* h_cblk = (int*)malloc(cblk_bytes);

        CUdeviceptr d_score = 0, d_vis = 0, d_every = 0, d_bmax = 0, d_cblk = 0;
        CHECK_CU(cuMemAlloc_v2(&d_score, score_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_vis, vis_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_every, vis_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_bmax, bmax_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_cblk, cblk_bytes));

        for (int dist = 0; dist < 3; ++dist) {
            for (size_t i = 0; i < (size_t)rows * (size_t)ss; ++i) {
                if (dist == 0) h_score[i] = rand_normalish(&rng);
                else if (dist == 1) h_score[i] = (float)((int)(xorshift32(&rng) % 7u) - 3);
                else {
                    uint32_t sp[6] = {0u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x3f800000u};
                    uint32_t bits = sp[xorshift32(&rng) % 6u];
                    memcpy(&h_score[i], &bits, 4);
                }
            }
            for (int r = 0; r < rows; ++r) {
                h_vis[r] = (rows == 1) ? 0LL : (long long)r * width / (rows - 1);
                h_every[r] = 1LL << 40;
            }
            CHECK_CU(cuMemcpyHtoDAsync_v2(d_score, h_score, score_bytes, stream));
            CHECK_CU(cuMemcpyHtoDAsync_v2(d_vis, h_vis, vis_bytes, stream));
            CHECK_CU(cuMemcpyHtoDAsync_v2(d_every, h_every, vis_bytes, stream));

            for (int pdl = 0; pdl <= 1; ++pdl) {
                // Note: deliberately do NOT pre-initialize d_bmax or d_cblk on dist==0 so initcheck verifies
                // the kernels never read uninitialized output/scratch elements!
                void* p_bmax[] = {&d_score, &ss, &width, &bsize, &d_vis, &d_bmax, &os, &nb, &pdl};
                launch_kernel(fn_bmax, (unsigned)((nb + 255) / 256), (unsigned)rows, 256, 1, 0, stream, p_bmax, pdl);

                void* p_pick[] = {&d_bmax, &os, &nb, &k, &d_every, &d_cblk, &pdl};
                launch_kernel(fn_pick, (unsigned)rows, 1, 1024, 1, (unsigned)(nb * 4), stream, p_pick, pdl);

                CHECK_CU(cuMemcpyDtoHAsync_v2(h_cblk, d_cblk, cblk_bytes, stream));
                CHECK_CU(cuStreamSynchronize(stream));

                int valid_cnt = 0;
                for (int i = 0; i < rows * k; ++i) {
                    if (h_cblk[i] >= -1 && h_cblk[i] < nb) valid_cnt++;
                }
                if (valid_cnt != rows * k) {
                    fprintf(stderr, "FAIL idx %s R=%d nb=%d dist=%d pdl=%d valid=%d/%d\n",
                            tag, rows, nb, dist, pdl, valid_cnt, rows * k);
                    exit(2);
                }
            }
        }
        printf("OK idx %s R=%d nb=%d (3 dists x pdl=0,1) cblk[0]=%d cblk[k-1]=%d\n",
               tag, rows, nb, h_cblk[0], h_cblk[k - 1]);

        CHECK_CU(cuMemFree_v2(d_score));
        CHECK_CU(cuMemFree_v2(d_vis));
        CHECK_CU(cuMemFree_v2(d_every));
        CHECK_CU(cuMemFree_v2(d_bmax));
        CHECK_CU(cuMemFree_v2(d_cblk));
        free(h_score);
        free(h_vis);
        free(h_every);
        free(h_cblk);
    }
}

static void run_compress512_suite(const char* tag, CUmodule mod_torch, CUstream stream) {
    CUfunction fn_c512 = nullptr;
    CHECK_CU(cuModuleGetFunction(&fn_c512, mod_torch, "tf_ds_compress2x_norm512_kernel"));

    const int row_list[4] = {1, 7, 32, 48};
    const int d = 512;
    uint32_t rng = 230u;

    for (int ri = 0; ri < 4; ++ri) {
        int rows = row_list[ri];
        long long src_rows = (long long)rows * 2;
        long long stride = d;
        long long ldo = d + 16;
        float eps = 1e-20f;

        size_t ckv_bytes = (size_t)rows * (size_t)stride * sizeof(float);
        size_t slot_bytes = (size_t)rows * sizeof(long long);
        size_t gi_bytes = (size_t)rows * 2 * sizeof(long long);
        size_t ring_bytes = (size_t)src_rows * (size_t)d * sizeof(float);
        size_t w_bytes = (size_t)d * sizeof(uint16_t);
        size_t out_bytes = (size_t)rows * (size_t)ldo * sizeof(uint16_t);

        float* h_ckv = (float*)malloc(ckv_bytes);
        float* h_cgate = (float*)malloc(ckv_bytes);
        long long* h_slot = (long long*)malloc(slot_bytes);
        long long* h_gi = (long long*)malloc(gi_bytes);
        float* h_rk = (float*)malloc(ring_bytes);
        float* h_rg = (float*)malloc(ring_bytes);
        uint16_t* h_w = (uint16_t*)malloc(w_bytes);
        uint16_t* h_out = (uint16_t*)malloc(out_bytes);

        for (size_t i = 0; i < (size_t)rows * (size_t)stride; ++i) {
            h_ckv[i] = rand_normalish(&rng);
            h_cgate[i] = rand_normalish(&rng);
        }
        for (size_t i = 0; i < (size_t)src_rows * (size_t)d; ++i) {
            h_rk[i] = rand_normalish(&rng);
            h_rg[i] = rand_normalish(&rng);
        }
        for (int r = 0; r < rows; ++r) {
            h_slot[r] = r * 2;
            h_gi[2 * r] = (r % 2 == 0) ? (r * 2) : (r * 2 + 1);
            h_gi[2 * r + 1] = (r % 2 == 0) ? (r * 2 + 1) : (((r + 1) % rows) * 2);
        }
        for (int c = 0; c < d; ++c) {
            float wf = rand_normalish(&rng);
            uint32_t wb;
            memcpy(&wb, &wf, 4);
            h_w[c] = (uint16_t)(wb >> 16);
        }

        CUdeviceptr d_ckv = 0, d_cgate = 0, d_slot = 0, d_rk = 0, d_rg = 0, d_gi = 0, d_w = 0, d_out = 0, d_flag = 0;
        CHECK_CU(cuMemAlloc_v2(&d_ckv, ckv_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_cgate, ckv_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_slot, slot_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_rk, ring_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_rg, ring_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_gi, gi_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_w, w_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_out, out_bytes));
        CHECK_CU(cuMemAlloc_v2(&d_flag, sizeof(uint32_t)));

        CHECK_CU(cuMemcpyHtoDAsync_v2(d_ckv, h_ckv, ckv_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_cgate, h_cgate, ckv_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_slot, h_slot, slot_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_rk, h_rk, ring_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_rg, h_rg, ring_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_gi, h_gi, gi_bytes, stream));
        CHECK_CU(cuMemcpyHtoDAsync_v2(d_w, h_w, w_bytes, stream));
        CHECK_CU(cuMemsetD32Async(d_flag, 0u, 1, stream));

        for (int pdl = 0; pdl <= 1; ++pdl) {
            int d_arg = d;
            void* params[] = {&d_ckv, &stride, &d_cgate, &stride, &d_slot, &d_rk, &d_rg, &d_gi,
                              &src_rows, &d_w, &d_out, &ldo, &rows, &d_arg, &eps, &d_flag, &pdl};
            launch_kernel(fn_c512, (unsigned)rows, 1, (unsigned)d, 1, 0, stream, params, pdl);
            // Copy every row's D active channels so initcheck verifies all rows*D outputs were initialized!
            for (int r = 0; r < rows; ++r) {
                CHECK_CU(cuMemcpyDtoHAsync_v2(h_out + (size_t)r * d,
                                              d_out + (size_t)r * (size_t)ldo * sizeof(uint16_t),
                                              (size_t)d * sizeof(uint16_t), stream));
            }
            CHECK_CU(cuStreamSynchronize(stream));
        }
        printf("OK compress512 %s R=%d D=%d (pdl=0,1) out[0]=0x%04x out[511]=0x%04x\n",
               tag, rows, d, h_out[0], h_out[d - 1]);

        CHECK_CU(cuMemFree_v2(d_ckv));
        CHECK_CU(cuMemFree_v2(d_cgate));
        CHECK_CU(cuMemFree_v2(d_slot));
        CHECK_CU(cuMemFree_v2(d_rk));
        CHECK_CU(cuMemFree_v2(d_rg));
        CHECK_CU(cuMemFree_v2(d_gi));
        CHECK_CU(cuMemFree_v2(d_w));
        CHECK_CU(cuMemFree_v2(d_out));
        CHECK_CU(cuMemFree_v2(d_flag));
        free(h_ckv);
        free(h_cgate);
        free(h_slot);
        free(h_gi);
        free(h_rk);
        free(h_rg);
        free(h_w);
        free(h_out);
    }
}

int main(int argc, char** argv) {
    const char* bin_dir = (argc > 1) ? argv[1] : "bin";
    double avail = mem_available_gib();
    if (avail < 3.5) {
        fprintf(stderr, "Refusing to run: MemAvailable %.2f GiB < 3.50 GiB\n", avail);
        return 3;
    }
    printf("b2_cs_driver starting: MemAvailable=%.2f GiB\n", avail);

    CHECK_CU(cuInit(0));
    CUcontext ctx = nullptr;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_CU(cuCtxSetCurrent(ctx));
    CUstream stream = nullptr;
    CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    char path[512];
    CUmodule wp2_ops = nullptr, wp2_torch = nullptr, new_ops = nullptr, new_torch = nullptr;
    snprintf(path, sizeof(path), "%s/wp2-ops.cubin", bin_dir);
    CHECK_CU(cuModuleLoad(&wp2_ops, path));
    snprintf(path, sizeof(path), "%s/wp2-torch.cubin", bin_dir);
    CHECK_CU(cuModuleLoad(&wp2_torch, path));
    snprintf(path, sizeof(path), "%s/new-ops.cubin", bin_dir);
    CHECK_CU(cuModuleLoad(&new_ops, path));
    snprintf(path, sizeof(path), "%s/new-torch.cubin", bin_dir);
    CHECK_CU(cuModuleLoad(&new_torch, path));

    run_idx_suite("wp2(b28310e)", wp2_ops, stream);
    run_idx_suite("tip(d0c9fb1)", new_ops, stream);
    run_compress512_suite("wp2(b28310e)", wp2_torch, stream);
    run_compress512_suite("tip(d0c9fb1)", new_torch, stream);

    CHECK_CU(cuModuleUnload(wp2_ops));
    CHECK_CU(cuModuleUnload(wp2_torch));
    CHECK_CU(cuModuleUnload(new_ops));
    CHECK_CU(cuModuleUnload(new_torch));
    CHECK_CU(cuStreamDestroy_v2(stream));
    CHECK_CU(cuDevicePrimaryCtxRelease(0));
    printf("b2_cs_driver ALL DONE\n");
    return 0;
}
