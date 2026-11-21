# Repeated dense projections and routed-expert coverage

The first observed call at each matrix size explains little of the dense prefill gap in the saved 9K request. After excluding those calls, the head achieved **34.9%** of its isolated same-shape cuBLAS throughput and the worker **66.9%**. The head’s long tail persists across repeated operations. This was one eager run with synchronized profiling, so the timings cannot establish the benefit of replacing a kernel in graph-on service. [Recomputed cohorts and per-shape observations](followthrough02/dense-repeats.json) · [Analysis script](dense-repeats.py)

## First and repeated dense dimensions

The 12.706 s prefill interior excludes 250 ms at each request boundary. On each rank, the analysis takes the first fully enclosed call at each distinct `(M,K,N)` and compares the remaining calls to the median of three isolated, same-board cuBLAS measurements of that size. Reference time and observed time are summed over the same calls. The 17 shapes all repeat. [Request and interval construction](stage-windows.py) · [Head reference](followthrough02/head-math.csv) · [Worker reference](followthrough02/worker-math.csv)

| Rank and calls inside the interior | Calls | Engine ms | Reference ms | Reference throughput achieved |
|---|---:|---:|---:|---:|
| Head, first per shape | 17 | 52.79 | 19.87 | 37.6% |
| Head, later at the same shape | 849 | 1,191.15 | 415.18 | 34.9% |
| Worker, first per shape | 17 | 11.97 | 8.81 | 73.6% |
| Worker, later at the same shape | 848 | 438.64 | 293.44 | 66.9% |

Five shapes had already appeared before the interior on each rank. When every earlier call in the saved trace counts as an observation, the repeated cohorts are 854 head and 853 worker calls, achieving 33.8% and 66.9% of reference. Repetition demonstrates prior use of the dimensions. Each call can still use another layer’s weights, output type or stride. The trace omits weight identity, kernel specialization and device-cache residency. The reference used four warm-ups and three CUDA-event batches per shape. The engine separately timed weight unpack and input rotation, while its dense interval includes host dispatch, fused output transforms, and stream synchronization. The cuBLAS test uses FP16 inputs and output, FP32 accumulation and no model epilogue. [Arithmetic reference implementation](math-reference.cpp) · [Engine operation boundary](phase-shapes.patch) · [Rank-specific wall-clock tracing](phase-windows.patch)

### Where the repeated time went

Only 23 of the 849 repeated head calls exceeded three times the median of the repeats at their own shape. They consumed 705 ms, 59.2% of repeated dense time. No repeated worker call crossed that descriptive threshold. On the head, **506 ms of the 776 ms repeated gap** came from one shape, `(2048,1280,16384)`. Its 78 repeats had a median of 1.566 ms and a maximum of 70.870 ms. Twelve intervals alone consumed 475 ms. The 78 worker calls at that shape had a median of 1.332 ms and a maximum of 1.930 ms. The head’s isolated reference itself varied from 0.924 to 4.888 ms across its three warmed batches. Across all 17 shapes, substituting the minimum or maximum observed reference for each shape puts head repeated throughput at 32.3–63.9% of reference and worker at 66.0–70.9%. This sensitivity range is descriptive. [Per-shape times, reference repetitions and tails](followthrough02/dense-repeats.json)

For that shape, the 12 long head intervals averaged **3.75 GB/s** of calibrated L2 read misses across 99.9% of their wall time. The other 66 repeated head intervals averaged **53.29 GB/s** at 97.6% coverage, and all 78 worker repeats averaged **59.12 GB/s** at 97.3% coverage. Only complete device-wide samples at least 5 µs inside the interval were included. Longer intervals had higher median sampled bytes as well as much lower rates. Host descheduling, device waits, dispatch and kernel time remain pooled in this measurement. A focused replay would record host launch and synchronize timestamps alongside CUDA events around the same projection and epilogue, with rank-local weights, strides and types fixed. [Counter alignment and raw inputs](followthrough02/stages/meter/raw.sha256) · [Paired shape and counter analysis](dense-repeats.py)

## What the routed trace can resolve

Each rank captured 98 gate/up and 98 down calls in the same interior. In each stage, 79 calls had 14,336 token–expert assignments and 19 had 5,565, corresponding to 2,048 and 795 tokens at seven slots per token. The gate/up dimensions were `(assignments,5120,1152)` and the down dimensions were reversed on K and N. These counts support the observed useful-throughput figures of **12.86/20.32 TFLOP/s** head/worker for gate/up and **12.38/19.87** for down. [Routed shape counts](followthrough02/dense-repeats.json) · [Measured phase times](followthrough02/cold-compute-efficiency.csv)

The binary’s shape record stores only the phase and three aggregate dimensions. In the pinned source, `exl3_experts.prompt` constructs per-expert counts, member lists and 64-row gate/up and 32-row down work lists in device scratch. The saved trace never exports them. Each row includes the shared expert once. Among the remaining 12,288 assignments of a full chunk, six experts selected for every row would need 192 routed gate/up tiles. An even distribution across 384 experts would need 384 such tiles, with the same total assignments and the same fixed shared expert. The recorded dimensions cannot distinguish those placements. The separate hot-expert telemetry operates on decode rounds and cannot recover the chunk’s prompt routing. [Shape fields and expert phase boundaries](phase-shapes.patch) · [Prompt tile geometry and expert count](../../docs/benchmark-gaps-and-moe-review.md) · [Hot-plan telemetry](hot-device-plan.patch)

There is consequently no measured arithmetic reference matched to these grouped, quantized prompt operations in the saved artifacts. A useful calibration would capture per-layer, per-chunk counts and expert membership, then replay the same assignment and relevant EXL3 weights and layout. Time the complete gate/up and down phases, including the routing-work-list preparation, rotation and epilogues charged to each. This also allows the arithmetic kernels to be compared separately. [Pinned engine build](../../zig/Dockerfile) · [Serving profiler instrumentation](phase-shapes.patch)

From the recipe root, reproduce the JSON using the saved two-rank traces and sampled counters.

```sh
python3 runs/overnight-opt/dense-repeats.py runs/overnight-opt/followthrough02
```
