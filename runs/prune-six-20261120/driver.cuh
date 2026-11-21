#include <array>
#include <cmath>
#include <functional>
#include <limits>
// Modes: 0 generic with stats, 1 previous six-slot prototype without stats,
// 2 recipe specialization without stats, 3 recipe specialization with stats.
void launchMode(int mode, cudaStream_t stream, int *p, float *w, int rows, unsigned long long *st,
                const unsigned char *dead) {
    TfDsPrune a = {0.85f, 3, 1, 1, 6, 7, rows, 0, 0};
    int threads = (rows <= 32) ? 32 : 64;
    if (mode == 0)
        tf_ds_expert_prune_kernel<<<1, 64, 0, stream>>>(p, w, a, st, dead, 0);
    else if (mode == 1)
        pruneSix<<<1, 64, 0, stream>>>(p, w, rows, dead);
    else
        tf_ds_prune_six_kernel<<<1, threads, 0, stream>>>(p, w, a, mode == 3 ? st : nullptr, dead, 0);
}
int main(int argc, char **argv) {
    const bool verify_only = argc > 1 && !strcmp(argv[argc - 1], "--verify-only");
    const int positional = argc - 1 - int(verify_only);
    const char *shape = positional ? argv[1] : "mixed";
    if (positional > 1 || (strcmp(shape, "mixed") && strcmp(shape, "full") && strcmp(shape, "sorted") &&
                           strcmp(shape, "masked") && strcmp(shape, "edges"))) {
        std::fprintf(stderr, "usage: %s [mixed|full|sorted|masked|edges] [--verify-only]\n", argv[0]);
        return 2;
    }
    const bool edges = !strcmp(shape, "edges"), ordered = !strcmp(shape, "sorted"),
               masked = !strcmp(shape, "masked") || edges;
    constexpr int nodes = 256, modes = 4;
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    for (int rows : {1, 7, 28, 32, 33, 64}) {
        int n = nodes * rows * 7;
        std::vector<int> hp(n), refp(n), gotp(n);
        std::vector<float> hw(n), refw(n), gotw(n);
        std::vector<unsigned long long> refs(nodes * 6), gots(nodes * 6);
        std::vector<unsigned char> hm(nodes * rows);
        unsigned seed = 51;
        auto next = [&]() {
            seed = 1664525u * seed + 1013904223u;
            return seed >> 8;
        };
        for (int r = 0; r < nodes * rows; ++r) {
            hm[r] = next() % 100 < 25;
            for (int i = 0; i < 7; ++i) {
                const int j = r * 7 + i;
                hp[j] = i == 6 ? 384 : int(next() % 384);
                hw[j] = i == 6 ? 1.f : (1 + next() % 12) * 0.03125f;
                if (i < 6 && !ordered && strcmp(shape, "full") && next() % 19 == 0)
                    hp[j] = -1;
            }
            if (ordered)
                std::sort(hw.begin() + 7 * r, hw.begin() + 7 * r + 6, std::greater<float>());
            if (edges) {
                int *p = &hp[7 * r];
                float *w = &hw[7 * r];
                switch (r % 13) {
                case 0:
                    w[0] = std::numeric_limits<float>::infinity();
                    break;
                case 1:
                    w[0] = std::numeric_limits<float>::quiet_NaN();
                    break;
                case 2:
                    std::fill(w, w + 6, 0.f);
                    break;
                case 3:
                    for (int i = 0; i < 6; ++i)
                        w[i] = std::numeric_limits<float>::denorm_min() * (i + 1);
                    break;
                case 4:
                    for (int i = 0; i < 6; ++i)
                        p[i] = 400;
                    break;
                case 5:
                    p[1] = 512;
                    p[4] = 511;
                    break;
                case 6:
                    for (int i = 0; i < 6; ++i) {
                        w[i] = i < 3 ? 17.f : 3.f;
                        p[i] = i;
                    }
                    break;
                case 7:
                    std::fill(p, p + 6, -1);
                    break;
                case 8:
                    for (int i = 0; i < 6; ++i)
                        w[i] = std::nextafter(1.f, i & 1 ? 0.f : 2.f);
                    break;
                case 9:
                    for (int i = 1; i < 6; ++i)
                        p[i] = -1;
                    break;
                case 10:
                    for (int i = 0; i < 6; ++i)
                        w[i] = i == 2 ? -0.0f : i == 5 ? -0.25f : w[i];
                    break;
                case 11:
                    for (int i = 0; i < 6; ++i) {
                        p[i] = i;
                        w[i] = i == 0 ? 1e30f : i == 1 ? -1e30f : i == 5 ? 1e-30f : 0.f;
                    }
                    break;
                }
            }
        }
        int *dp;
        float *dw;
        unsigned long long *ds;
        unsigned char *mask = nullptr;
        CHECK(cudaMalloc(&dp, n * sizeof(int)));
        CHECK(cudaMalloc(&dw, n * sizeof(float)));
        CHECK(cudaMalloc(&ds, refs.size() * 8));
        if (masked) {
            CHECK(cudaMalloc(&mask, hm.size()));
            CHECK(cudaMemcpy(mask, hm.data(), hm.size(), cudaMemcpyHostToDevice));
        }
        auto restore_inputs = [&]() {
            CHECK(cudaMemcpyAsync(dp, hp.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(dw, hw.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
        };
        auto reset = [&]() {
            restore_inputs();
            CHECK(cudaMemsetAsync(ds, 0, refs.size() * 8, stream));
        };
        cudaGraphExec_t execs[modes]{};
        for (int mode = 0; mode < modes; ++mode) {
            // The old standalone baseline used >= in the drop test, which differs for a NaN prefix.
            if (edges && mode == 1)
                continue;
            reset();
            for (int k = 0; k < nodes; ++k)
                launchMode(mode, stream, dp + k * rows * 7, dw + k * rows * 7, rows, ds + k * 6,
                           mask ? mask + k * rows : nullptr);
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaMemcpy(gotp.data(), dp, n * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(gotw.data(), dw, n * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(gots.data(), ds, refs.size() * 8, cudaMemcpyDeviceToHost));
            if (!mode) {
                refp = gotp;
                refw = gotw;
                refs = gots;
            } else if (gotp != refp || std::memcmp(gotw.data(), refw.data(), n * sizeof(float)) ||
                       ((mode == 3) && refs != gots)) {
                std::fprintf(stderr, "mismatch dataset=%s rows=%d mode=%d picks=%d weights=%d stats=%d\n",
                             shape, rows, mode, gotp != refp,
                             !!std::memcmp(gotw.data(), refw.data(), n * sizeof(float)), refs != gots);
                return 3;
            }
            cudaGraph_t graph;
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            for (int k = 0; k < nodes; ++k)
                launchMode(mode, stream, dp + k * rows * 7, dw + k * rows * 7, rows, ds,
                           mask ? mask + k * rows : nullptr);
            CHECK(cudaStreamEndCapture(stream, &graph));
            CHECK(cudaGraphInstantiate(&execs[mode], graph, 0));
            CHECK(cudaGraphDestroy(graph));

            // All nodes and both replays accumulate into the same six device counters.
            // Seed close to a 32-bit carry and derive totals from the generic eager windows.
            std::array<unsigned long long, 6> expected{}, actual{};
            for (size_t i = 0; i < expected.size(); ++i)
                expected[i] = ((i + 1ull) << 32) - 1;
            CHECK(cudaMemcpyAsync(ds, expected.data(), sizeof(expected), cudaMemcpyHostToDevice, stream));
            for (int replay = 0; replay < 2; ++replay) {
                restore_inputs();
                CHECK(cudaGraphLaunch(execs[mode], stream));
                CHECK(cudaStreamSynchronize(stream));
                if (mode == 0 || mode == 3)
                    for (size_t i = 0; i < refs.size(); ++i)
                        expected[i % expected.size()] += refs[i];
                CHECK(cudaMemcpy(actual.data(), ds, sizeof(actual), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(gotp.data(), dp, n * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(gotw.data(), dw, n * sizeof(float), cudaMemcpyDeviceToHost));
                if (gotp != refp || std::memcmp(gotw.data(), refw.data(), n * sizeof(float)) ||
                    actual != expected) {
                    std::fprintf(stderr, "graph mismatch dataset=%s rows=%d mode=%d replay=%d\n",
                                 shape, rows, mode, replay);
                    return 3;
                }
            }
        }
        if (!edges && !verify_only) {
            for (int pass = 0; pass <= 16; ++pass)
                for (int j = 0; j < modes; ++j) {
                    int mode = (pass + j) % modes;
                    reset();
                    CHECK(cudaEventRecord(start, stream));
                    CHECK(cudaGraphLaunch(execs[mode], stream));
                    CHECK(cudaEventRecord(end, stream));
                    CHECK(cudaEventSynchronize(end));
                    float ms;
                    CHECK(cudaEventElapsedTime(&ms, start, end));
                    if (pass)
                        std::printf("%s,%d,%d,%d,%.5f\n", shape, rows, pass, mode, 1000 * ms / nodes);
                }
        }
        for (auto exec : execs)
            if (exec) CHECK(cudaGraphExecDestroy(exec));
        if (mask)
            CHECK(cudaFree(mask));
        CHECK(cudaFree(ds));
        CHECK(cudaFree(dw));
        CHECK(cudaFree(dp));
    }
    std::fprintf(stderr, "%s verified\n", shape);
}
