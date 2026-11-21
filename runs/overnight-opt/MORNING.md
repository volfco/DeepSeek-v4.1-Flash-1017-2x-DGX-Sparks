# Two-Spark inference benchmark report, November 18 at 08:00 Pacific

## Performance against the repository baseline

**Use the pinned deployed settings for short four-user decode. Keep the 32 MiB tokenizer cache as the tested option for repeated long chat histories.** The full eight-switch bundle trades some decoding throughput for first-token latency. In the earlier external cohort, decode accounted for 86.3% of summed request time and prompt prefill for 10.2%. An identical 500K conversation prefix saved 445–508 ms on each tested append size with the token cache. [Runtime configurations](../../best-runtime.md) · [Workload weights](traffic-stages.json)

Both production controls and the eight-switch candidate ran on the same pinned image, with serving graphs and speculative decoding active. All 20 warm responses matched in input, cache cut, output length and token hash. Start occupancy matched and no placement was refused. Output rates below are aggregate rates during the all-active window. Paired changes compare each fixed seed with its two production controls.

| Context × active requests | Deployed output tok/s | Eight switches tok/s | Paired change | Median first-token change on each seed |
| --- | ---: | ---: | ---: | ---: |
| 3K × 1 | 66.7–77.4 | 66.5–76.5 | −0.9% | −4.4%, −5.4% |
| 3K × 4 | 136.5–138.4 | 130.4–134.2 | **−3.5%** | −4.9%, +0.1% |
| 300K × 1 | 71.8–79.7 | 69.5–81.1 | −0.8% | +11.6%, −22.4% |
| 300K × 4 | 131.5–137.5 | 128.7–136.9 | −1.2% | **−4.2%, −11.2%** |

Cold 300K prefill reached **1,910 input tok/s with eight switches versus 1,913 and 1,931 deployed**. The later 3K zero-cut prime reached 1,322 versus 1,653 and 1,661 tok/s, one observation per arm. Earlier 8K/32K/100K gains of 6.9%/5.3%/4.1% came from a different image and prompt rotation. [Current paired measurements](followthrough02/baseline-comparison.json) · [Historical combination study](../combo/RESULTS.md)

## How close each stage is to its measured limit

The two boards have independent memory systems. Their sustained contiguous GPU-read references were 253.2 and 252.1 GB/s. Scattered GPU reads reached 231–236 GB/s. These references address different access patterns. The supplied 273 GB/s specification describes the external interface. The available GPU counter measures sectors leaving L2, which can hit the 16 MB system cache. Physical LPDDR controller utilization, including writes and CPU traffic, is unavailable. [GB10 reference measurements and topology](RESULTS.md#what-the-hardware-reference-can-measure)

| Component | Measured work | Fraction of the relevant measured reference |
| --- | --- | --- |
| Graph-on decode, whole engine | 146–162 GB/s per board across short/deep and one/four streams | **58–64%** of local contiguous reads, or 54–59% of the 273 GB/s specification. At fixed bytes and acceptance the read-only ceiling is 199–209 tok/s at 300K × 4. |
| Routed expert decode | 220.4/220.9 GB/s, 56% of the four-stream eager window on each rank | **93–96%** of scattered reads. Closing that bandwidth gap alone would save about 2.4–3.7% of this eager window. |
| Eager output head, attention and indexer | Four streams, head/worker. Output head 184/183 GB/s, attention 167/173, indexer 18.6/19.0 | About 73%, 66–69% and 7–8% of contiguous reads. Attention counter coverage was 41–45%. Low read traffic alone cannot price compute or latency overhead. |
| Cold and near-cold 300K prefill | Separate long histories at 1,755–1,886 input tok/s during bandwidth capture | **31–34%** of independent contiguous reads. The latest direct baseline above used a different fresh-process crossover. |
| Cold dense projections | 22.1/61.0 useful TFLOP/s across 17 matched shapes | **35% head, 67% worker** of isolated same-shape GEMM throughput. Matching it would remove about 6.4%/1.2% of the respective traced cold window. |
| Cold routed projections | Gate/up 12.9/20.3 useful TFLOP/s, down/combine 12.4/19.9 | A same-shape quantized grouped-expert reference is still needed to quantify the arithmetic gap. |
| Engram storage | At 32K, 210–221 MB/s in 4.25 KiB reads with 251–288 outstanding on average | **23–24%** of the pair-local 0.929 GB/s native 4 KiB, depth-256 reference over the whole prefill. |
| Tensor-parallel and KV traffic | About 1.99 GB/s per direction during cold 300K prefill and roughly 0.8 GB/s at deep decode | About 8% and 3% of the measured 24.5 GB/s per-direction bulk RDMA reference. Short-message latency was 2.02 µs on one rail. |
| Speculation | Target/draft forward 45.4/4.17 ms in a 50.5 ms eager solo round, 74.9/6.99 ms in an 82.7 ms four-stream round | Drafting took about 8% of wall in both. An isolated equivalent drafter reference is unavailable. |
| Tokenization and checkpoint restore | 500K CPU encode 444 → 7 ms with the cache. A 296,960-token restore took 161–178 ms | One linear read of the tokenizer input would take 0.052–0.077 ms. The owner snapshot's logical bytes would take 112 ms at its 2.42 GB/s sequential rate. Actual CPU and NVMe bytes for these paths are unknown. |

Streaming CPU reads on three spare cores per board delivered 23.0–26.7 logical GB/s per rank during the active trials. With all four streams generating, round time rose **8.4% at 3K and 6.9%/16.8% at 300K** relative to the bracketing cache-resident CPU controls. At 300K solo, +2.4%/+3.0% fell below the predeclared gate. Outputs remained identical while some speculative work changed. This supports protecting decode from competing memory traffic. [Controlled interference results](RESULTS.md#decode-sensitivity-to-concurrent-cpu-memory-reads) · [Per-rank stages](RESULTS.md#both-ranks-at-matched-operation-shapes) · [Fabric and storage references](RESULTS.md#pair-local-fabric-and-storage-ceilings)

## Configuration decisions

- **Checkpoint pressure.** On the trim-capable build, 84 histories fit in 299 MiB. At the same 512 MiB cap, trim eliminated 12 evictions and eight durable restores. Twelve revisits took 5.350 versus 5.625 s. All 96 outputs matched, in addition to the earlier 39 frozen cases and eight saved replies. Enabling trim requires a fresh compatible spill namespace. [Pressure crossover](followthrough02/trim-pressure/pressure-summary.json)
- **Admission.** Keep ten-second held-prefix patience and fixed five-second finish grace for general serving. With fully backed histories, short-held fixed grace kept a naturally stopping lead at 3.48 s and cut maximum queue to 4.77 s. The same short-held setting with adaptive grace took 10.82 s for the lead. Early eviction without a committed peer copy can cost approximately 150 s on return. [Natural stops and missing-copy trials](RESULTS.md#snapshot-availability-under-300k-pressure)
- **Other switches.** Retain L2 prefetch, fetch-mode KV, the original index bucket, and uncapped T13 pruning. The faster expert cap failed two independent quality panels. AIO batch 64, the widened CPU placement and final-chunk head skipping failed their performance screens. An exact 100K full-end resend saved 1.224 s against ordinary warm reuse, with a small opportunity in the available traffic trace. [Full experiment record](RESULTS.md)
- **Source validation.** The rebuild includes rank, checkpoint and build corrections. A mode-verified rerun exposed three test-fixture defects hidden by a build rule that compiled every test in Debug. Their corrected complete suites passed **769/769 in Debug and 769/769 in ReleaseSafe**. [Clean replay and verification](build-audit.md)

The corrected distributed greedy kernel matched all 36 direct GPU cases, all 12 same-image serving outputs, 39 frozen references and eight saved replies. Four-way greedy throughput changed **+0.35%**. The source correction is included for correctness; keep the optional exchange off in the general runtime. [Serving and GPU evidence](RESULTS.md#distributed-greedy-argmax)

BF16 reduced-precision prefill also completed its repaired bulk path. All 40 item scores changed, but the average difference was +0.00524 nats/token across 20 sources (95% interval −0.00354 to +0.01402), within the +0.020 margin. The two 8K gains were 3.14% and 2.64%, and the two 32K gains were 2.27% and 2.62%. Neither depth cleared the 3% gate twice. Keep FP32 for general serving. [Timing gate](followthrough02/bf16-quality/prefill-signal.json) · [Quality panel](followthrough02/bf16-quality/quality-vs-control.json)


Both experimental hot-expert settings published plans for 40 layers on each rank, matched all 12 outputs per setting and passed eight saved replies each. Hot:2 reduced greedy throughput 9.9%; hot:4 changed greedy/sampled/mixed throughput +0.4%/−1.8%/−1.7%. Keep extra expert prefetch off and ordinary L2 prefetch on. [Validated pilots](RESULTS.md#hot-expert-prefetch-under-serving-graphs)

Across five disjoint observation windows totaling 2 h 20 m 48 s, the restored production service completed zero requests. The last two lock-free windows covered just 5.1 seconds. The November 13 cohort supplies workload weights; the available overnight samples cannot estimate current repeat or eviction frequencies. [Coverage and counters](kv/quiet-summary.json)


## Status at the reporting cutoff

The baseline, bandwidth intervention, trim, corrected greedy exchange, BF16 and hot-prefetch groups finished. The last group restored the original pinned image, environment and spill mounts on both ranks and reproduced all eight saved production replies. The next reservation began with the clean 128-patch recipe staged and verified on both boards. The supported unsharded, unsliced, memory-only exact-resume comparison is running. The pinned tokenizer-only, Engram and GPU switch comparisons, including keyed drafts and conditional 32K/100K confirmation, follow that reservation. Those results are pending at the cutoff. [Completed reservation](followthrough02/released.json) · [Original pair restoration](followthrough02/hot-series/restored.json) · [Restored replies](followthrough02/hot-series/production.validate.log) · [Clean recipe for the next run](followthrough02/isolated-resume/recipe-proof.json)

After the cutoff, the supported 0121 comparison failed its cold-output gate. The pinned configuration screen and 32K/100K confirmation completed. [Exact resume](RESULTS.md#supported-0121-resume-against-independent-cold-appends) · [Full crossover](RESULTS.md#full-32k-and-100k-confirmation) · [Dense operation follow-up](dense-routed-prefill.md)
