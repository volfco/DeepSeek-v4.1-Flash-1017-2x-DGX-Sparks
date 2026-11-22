# Follow-ups (out of scope of the current campaign)

The [current evaluation order](to-benchmark.md#current-evaluation-order) and [runtime record](best-runtime.md) track the November 18 decisions. The AIO, CPU-affinity, index-bucket, pruning-quality and identical-resend rows below have subsequent measured outcomes. The [overnight report](runs/overnight-opt/RESULTS.md) records their gates and remaining limits.

## Fused-next serving and kernel follow-ups (2026-11-21)

Pre-restart harness, host-dispatch, B2, `dec8` and `0254` tied-score follow-ups completed without touching live serving (`runs/pre-bench-20261121/`). Next without restart (needs idle `/health`): capture `0215` guarded crossover timings. Next requiring an engine restart: rebase/build `0213–0258` (`84806bb`) and run the paired two-Spark replay.

- [ ] **Give `fixed_replay` a prompt that decodes.** `"Reply with the single word: ok"` ends in 1 round/1 token at 49,967/99,904/200,000 tokens, so its token-hash equality is a one-token comparison, its decode ms/round is vacuous, and the 8,192/16,384-block decode buckets were never traced in `s05`. Ask for ~200 tokens at each depth (fixed text, same seed) so the deep decode shapes are actually measured; the prefill half already works. [Results](runs/suite-fused-next2-20261121/RESULTS.md#trace-s05-128-lines-per-rank-both-ranks-equal-in-shape-coverage)
- [x] **Rebase and export `0213–0258` onto `zig/patches/` (`c3b3859`) — done 2026-11-22 after the operator adopted `84806bb`.** 19 patches in `zig/patches/` (`0213–0217`, `0230–0231`, `0240–0242`, `0250–0258`), regenerated from `~/Workspace/tfref-fnext-rebase` (`c3b3859` + the cherry-picked round) by `scripts/export-fnext-series.sh` (or `just export-fnext`); numbers come from the commit subjects because the round has gaps. Two conflicts resolved by hand: `0217` (keep `0182`'s `topk_fast` const + take the `const o: Ops` dynamic-shared opt-in) and `0256` (constructor takes `topkIndicesArm` dispatch and keeps `.topk_indices_fast`; the launch keeps `0182`'s precedence and adds the trace note). `0254`/`0255` replace the hand-ported pair and are content-equal to their commits. Verified: the chain applies in filename order onto a clean `TF_REF` and reproduces the rebased tip; `zig build test` passes (one run hit the known rank-1 spill-scan flake); a `TF_REF + patches` checkout type-checks with `zig build install native -Dfatbins=<dummy dir>`. Not done: an image build from the recipe and its deploy gate. [Lineage audit](runs/pre-bench-20261121/topk-and-patches/RESULTS.md) · [Gate](runs/suite-fused-next2-20261121/RESULTS.md)
- [x] **Measure patches 0254–0258 on both Sparks (suite `runs/suite-fused-next2-20261121/`, ran 2026-11-22 02:34–04:31 UTC).** 5/5 blocks PASS: exact 39/39 per block, probe 11/11 reply hashes equal with identical rounds/accepted, `fixed_replay` input bytes/prompt tokens/token hash equal, `dec8_staggered` 3/3 waves equal on every block, validation rc 0. Speed neutral: next-vs-base dec8 pure decode `-1.35%`/`+1.71%` against a base-base control of `-2.23%` (`+2.36%` true_active ms/round), `fixed_replay` prefill mean `+0.02%`. Trace (`s05`) engaged `cand_fast8` (decode, width 1,024), `topk_indices_cut` (rows 128, width 21,875), `idx_fused` (32,768-block prompt shape) on both ranks. Not covered: 8,192/16,384-block *decode* shapes, because `fixed_replay`'s "reply with the single word: ok" finished in 1 round/1 token at all three depths. [Results](runs/suite-fused-next2-20261121/RESULTS.md). Original plan: Compare `d5ecba0` with `84806bb` (`0254` `topk_indices_cut`, `0255` `cand_fast8`, `0256/0258` thread-safe `fused_trace`, `0257` `planKvSource2`) using identical serving switches, greedy decoding and identical input token IDs. Start with `bash runs/suite-fused-next2-20261121/preflight.sh` (builds/ships `bench-fused-next-84806bb`, waits for 30 idle minutes, snapshots production env, captures the guarded `0215` timings, runs ABBA `s01–s04` + `s05` traced). The suite's `fixed_replay` cell replays unsalted fixed-input prompts at ~49,967 / ~99,904 / ~200,000 tokens (8,192/16,384-block buckets + fallback decode/deep prefill) comparing token hashes, prefill tok/s and decode ms/round (`table2.py`); `dec8_staggered` locks admission order (`dec8_replay_audit.py`); `s05` runs `TF_DS_FUSED_TRACE=1` on both ranks to capture `topk_indices_cut`/`cand_fast8`/`idx_small` engagement. Gate: `python3 -I runs/suite-fused-next2-20261121/gate.py` (fixed: suite path, flags-based lead banner, drafting-rule only for drafting arms). [Suite README](runs/suite-fused-next2-20261121/README.md) · [Serving coverage](runs/suite-fused-next-20261121/RESULTS.md#caveats) · [Standalone kernel results](runs/suite-fused-next-20261121/kernel-review/RESULTS.md)
- [ ] **Capture guarded `0215` small-row crossover timings when `/health` is idle (no restart needed; 36/36 parity done).** Byte parity passed `36/36` (`0` diffs across `R=1,7` × `nbk=2,049..16,384` random/ties and 4 `0254` distributions); guarded timing (`timing_valid: false`) held back while production served continuous live traffic. Keep `small_nbk = 6,144` until `~/tf-bench/pre-bench-20261121/crossover/run_timing_loop.py` captures `timing_valid: true` during an idle window. **Tried 2026-11-22 02:31 during the suite window and still not captured** (`0/36`, `BUSY connections=None waiting=None counters=(None,None)`): the harness polls a live `/health`, so a *stopped* production reads as no server, not as idle. It needs production running and quiet. The launch step's results rsync also targets a missing head directory (`crossover-0215`). [Ceiling](runs/suite-fused-next2-20261121/RESULTS.md#incidents-and-fixes-this-window) [Parity and staged runner](runs/pre-bench-20261121/crossover-0215/RESULTS.md)
- [x] **Investigate both intermittent byte mismatches from review B2.** Closed as a `tools/fused_bench/bench.py` harness bug, not a kernel defect: `Buffer.__init__`/`read` used legacy stream-0 `cuMemcpyHtoD_v2`/`cuMemcpyDtoH_v2` (asynchronous for `<=64 KiB` pageable buffers) while kernels launched on `CU_STREAM_NON_BLOCKING` (`flags=1`). Reproduced on GB10 (`2/220` stream-0 failures: `4,012` and `4,009` diff bytes matching WP2's `3,984`/`3,707`) vs `0/220` and `0/14,400` with stream-ordered `cuMemcpyHtoDAsync_v2`, and `0` errors across 12 `compute-sanitizer` (`initcheck`/`racecheck`/`memcheck`) suites. Fixed in `bench.py`. [B2 report](runs/pre-bench-20261121/b2-investigation/RESULTS.md)
- [x] **Account for the eight-stream throughput dip.** Closed as non-kernel scheduling/metric variance on median Wave 0 (`+175.87 ms` Pair A, `+204.65 ms` Pair B): (1) random HTTP socket arrival order under `TF_DS_ROUND_ROWS=16` gave straggler `s=7` fewer draft slots (`+4`/`+5` physical rounds = `+146.99`/`+138.84 ms`, `83.6%`/`67.8%` of dip), and (2) `min(t_first)` excluded longer `ord=0` prefills in `base` (`+33.24`/`+20.90 ms`) despite `next` prefilling all 8 prompts faster (`-20.68`/`-16.24 ms`). True active decode `ms/stream-round` is `+0.003%` (Pair A), `-0.51%` (Pair B) and `-0.30%` (PDL). Built `dec8_replay_audit.py` (`--admission-mode staggered`) for post-restart replay. [Forensic report](runs/pre-bench-20261121/dec8-audit/RESULTS.md)
- [x] **Repair the legacy indexer timing labels before reusing those results.** Fixed all four 3-chain/4-name callers in `tools/fused_bench/bench.py` (`case_idx`, `case_compress_prompt`, `case_rope`, `case_prompt_source`), added pre-capture `ValueError` validation and `row["bytes"]` on `0254`/`0255` cases, and audited all 98 JSONs (`10` files / `100` rows in `runs/fused-next-20261121/{wp1,wp3,wp4}` had `floor` mislabeled as `newnf`; `0` in `suite-fused-next-20261121`). [Harness audit](runs/pre-bench-20261121/harness-audit/RESULTS.md)

### Additional investigations

- [x] **Make timing validity explicit for each benchmark case.** `bench.py` now lazy-loads `libcuda.so.1`, requires `live.connections == 0` and non-bool non-negative integer token counters before and after each block, suppresses contaminated medians (`timing: {}`, `timing_valid: false`) when retries exhaust, and tracks `timing_retries` separately (`9/9` CPU unit tests passing in `tools/fused_bench/test_bench_guard.py`). [Benchmark harness](tools/fused_bench/bench.py)
- [x] **Exercise the real 0231 prompt-compressor dispatch.** Extracted `planKvSource2` (`0257` / `9ece091`) called directly by `prompt.zig`, pinned multi-chunk progressions (`full == 0`, odd trailing rows, 64-slot ring wrap, partial-fusion staging, `ChunkTooLong`) in Zig host tests (`630/632`), and verified 4-chunk stateful GB10 GPU byte parity (`0` byte diffs across all 7 ring/latent/packed buffers). Live next-token check rides the post-restart Spark replay. [0231 dispatch report](runs/pre-bench-20261121/p0231-dispatch/RESULTS.md)
- [x] **Explain the remaining top-k cost on tied scores.** At `128 × 32,768 (k=512)`, `~82–86%` (`~31.5 µs`) of `100.717 -> 138.788 µs` is `+2.01` extra row-scan passes (`+32.1 MiB` + 6 barriers) and `~14–18%` (`~6.1 µs`) is cross-warp single-bank contention (`sm_121` already coalesces intra-warp uniform shared `atomicAdd`). Cutoff-straddling `-∞` rows are `<0.03%` of served wall time (`applyPool` L24–36 retains `16,384 >> 512` finite columns and finishes in `2.89` passes; `0253` bypasses `len <= 512`). Closed with no kernel change. [0254 tie audit](runs/pre-bench-20261121/topk-and-patches/RESULTS.md)

## Follow-up: fused-kernel operational overhead, patches 0195+ (2026-11-21)

**Items 1-4, 8-11, 13, 15 and 17-19 landed as `zig/patches/0200`-`0212`** (2026-11-21; one patch an item, each riding the
switch its parent patch already has, so nothing new is on by default). **Collapsed the same day:** 0200, 0201, 0202,
0209, 0210 and 0212 are now folded into `0195`; 0203, 0204, 0205, 0206, 0207, 0208 and 0211 stay separate (0207 because
it measured 13-17% slower on the served decode shape, 0206/0208 because they cost the key writes they add and changed
four long-prompt replies in the gate). 137 patches total; the collapsed tree reproduces the pre-collapse tip byte for
byte. [The README's rows](zig/patches/README.md) say
what each changed and which host test pins it; the item-to-patch map is 1 = 0200, 2 = 0201, 4 = 0202, 15 = 0203 (the new
`TF_DS_CHECK_INVALID`), 17 = 0204, 18 = 0205, 3 = 0206, 8 = 0207, 9 = 0208, 10 = 0209, 11 = 0210, 13 = 0211, 19 = 0212.
Locally: `zig build test` passes every suite and a clang stub-header syntax check covers the three hand-written `.cu`
files (the recipe chain 0001-0212 applies to `TF_REF` and reproduces the work tree byte for byte).

**The gate ran on 2026-11-21; the standalone half passes and `TF_DS_IDX_FUSED` fails on the pair.**
[runs/fused-followup-20261121/RESULTS.md](runs/fused-followup-20261121/RESULTS.md) holds the evidence:

- Byte parity (all pass): norms at both widths, both compressors, all five rope-store chains, the 0206 key fold's keys
  and tile maxima against `scoreKey` at seven widths, 8,516 pair-softmax cases, 4,111 six-slot drop vectors, 8,257,536
  scale pairs, and the buffers 0200 stopped writing (still at their pre-launch bytes). Every kernel keeps `LOCAL=0
  STACK=0` (no spills).
- SASS (what the items asked for): `fused_prompt_idx_key` 6 -> 4 barriers (0202, with the fold's barrier left inside the
  `L > 32` branch), `fused_idx_key` 5 -> 3; the shared scale array and second `srow` read are gone with 0209 (LDS 26 ->
  4); 0210's `compress2_norm` inlines five `ITEMS` specialisations (static counts rise, executed path is shorter); 0212's
  prompt key runs two y lanes at the served shape with identical bytes.
- Timings: prompt compressor 28-67% faster, rope-store chains 14-45% faster, decode compressor 13-17% slower (0207's
  `isfinite` branch, ~0.03% of a round), and 0206's fold costs the key writes (R=128 width 32,768: 316.6 µs with the fold
  against 38.5 without; the previous tree is 41.0 there).
- Paired arms: `base`, `TF_DS_FUSED_COMPRESS2_NORM`, `TF_DS_FUSED_ROPE_STORE`, `TF_DS_FUSED_ROUTE` and the previous fused
  bundle are 39/39 identical to production. **`TF_DS_IDX_FUSED=1` differs on 4 of 39** (`len_20000`, `len_30000`,
  `len_50000`, `len_64000`; completion length and token hash).

**Open: which of the thirteen changed the pruned selection, starting with 0206's key encoding.** The fold writes
`idx_fused.zig:scoreKey`'s key, which XORs the top bit; the kit's `_score_keys` does not (`0x80000000ffffffff` vs
`0x00000000ffffffff` on the same cell, every column of a probe row). If the kit's `topk_select_pruned` compares signed,
that flip reverses negative-vs-positive scores. It is a suspect, not a proof: the previous bundle never produced keys at
those widths, and `len_100000` still matches where the fold also engages, so pin the width window first (instrument
`keys_done` at the pruned site) before changing either side. Then re-run the `TF_DS_IDX_FUSED` arm and the fused bundle;
`all`/`allpdl` carry the same four differences. PDL itself is now measured bit-identical on the fused bundle (`allpdl`
vs `all`: 39/39), only its speed is unmeasured.

**Not measured:** the kit's `_score_keys` launch cost at the fold's widths (it faults for `width % 4 != 0`; the harness
never priced the launch the fold replaces), throughput on the two-rank 1,024-token hashes beyond the serving probe, and
`TF_DS_CHECK_INVALID`'s diagnostic arm. The `Additional 0195+ performance findings` below are unaffected and still
unbuilt.

The rest of that review is measurement-first and stays unbuilt:

1. **Profile-gated: skip RoPE partner shuffles on wholly unrotated warps.** `dsv41_torch.cu:647,721` shuffles before testing whether an element rotates. At served shapes, 87.5% of latent and 50% of query shuffles are source-level unnecessary. Guard only entire unrotated warps and retain full-mask shuffles for mixed/rotated warps. Inspect SASS first because NVCC may already sink them; then compare executed SHFL counts and decode/prompt microbenchmarks.
2. **Profile-gated: cache the two compressor exponentials only if SASS shows duplicates.** `tf_ds_compress2_pair` (`dsv41_torch.cu:369-377`) has four `expf` calls but two distinct operands. Reusing `e0/e1` is bit-exact and removes duplicate evaluation only if NVCC has not already common-subexpression-eliminated it. Inspect SASS before editing; if duplicated, measure decode R=1/7/32 and prompt R=128/1024, otherwise close this item with no change. **(0207's finite-row half already removed the duplicate `expf`s; this one is only the SASS check that NVCC had not common-subexpression-eliminated them.)**
3. **Profile-gated: warp-aggregate 0195 histogram updates for ties.** The per-element shared `atomicAdd` in `tf_ds_pool_kth` (`dsv41_ops.cu:698-700`) contends on equal-digit prefixes. Test `__match_any_sync` plus one leader `atomicAdd` of each group's count, which preserves integer selection exactly. Measure random, tied, and flat inputs with SASS/NCU shared-atomic evidence; random data may lose from the extra match instruction. Sweep 256/512/1024 threads separately rather than assuming a smaller CTA is faster.
4. **Specialize and vectorize aligned served-shape loads.** For 0195 `bsize == 8`, use two aligned `float4` loads per full block with a scalar tail and the existing comparison/NaN order; generic row strides require explicit 32-byte alignment guards (`dsv41_ops.cu:752-765`). Prompt torch norm can use aligned eight-byte four-BF16 loads while preserving conversions and arithmetic order (`dsv41_torch.cu:325-362`). Require bit parity and SASS proving actual wide LDG/LDS instructions, intact scalar fallbacks, no spills, and a timing win—source casts alone are insufficient.
5. **Specialize 0195's block size at compile time.** `tf_ds_idx_fused_kernel` takes `bsize` as a runtime argument (`dsv41_ops.cu:752-765`), so the inner `for (j < bsize && …)` loop cannot be fully unrolled. Add a template on the served `bsize == 8` (full unroll, static trip count, pairs with the aligned `float4` loads of the vectorized-loads item above) while keeping the runtime path for other sizes and every `fuseOk` fallback. Check SASS for the unrolled 8-wide body and unchanged comparison/NaN order.
6. **Re-test the 0197 decode compressor at 128 threads with four channels a thread.** The current kernel launches one thread per channel (`block.x = d`, `dsv41_torch.cu:376-452`) and, at D=512, must exchange all 512 values through shared `row[512]` with an extra barrier just to give each lane four adjacent channels for the kit's `squares4` order (`:426-435`). A 128-thread form where each thread loads four adjacent BF16 channels, pairs and squares them in registers, and writes four outputs reproduces the same reduction order without the exchange, the barrier or the shared array, and turns scalar accesses into 16-byte coalesced ones. The recorded sweep that rejected 128 threads (4.116 µs vs 2.710 at 512 threads, [audit](runs/fused-kernel-audit-20261120/RESULTS.md)) measured the *pre-optimization* variant, before rounded channels were kept in registers and current-row loads reused; that ranking has not been re-tested on the final kernel. Sweep 128/256/512 again at R=1/7/32/48 for both widths with the current arithmetic, and require byte-identical output, rings, stride padding and the invalid flag.

0199 needs no further kernel work now (0211 took the last item). Stats-on minus stats-off is 0.728 µs a layer at seven rows, only 0.069% of a 42 ms round across 40 layers, and the earlier generic-kernel serving A/B was neutral. Use existing `TF_DS_T13_PRUNE_STATS=0` only if the counters are operationally unused.


## Follow-up research: larger fusions and shared-memory/occupancy changes (2026-11-21)

- **Larger fusions:** With 0200-0212 the dead stores, the prompt query launch, the 0195 key pass and the 0198 rotation pass are folded; the next question is whether further folds—such as compressor + latent RoPE/FP4 or more of the indexer chain—remove measured work rather than merely launches. Preserve every live intermediate and each call site's exact arithmetic; for example, `rd.lat` still feeds the index-key projection before latent packing. Require a profiler-shown bottleneck, standalone live-output parity, the 39-case exact suite, long-reply hashes, PDL coverage, and a graphs-on serving A/B. Engram `add2Bf16 + _engram_gate` remains parked for its three-reduction bit-identity risk.
- **Shared memory and occupancy:** `red[512]`, `srow`, and `sc[16]` are oversized for some runtime shapes, but unused shared storage executes no instructions. First use NCU/ptxas to establish whether shared allocation actually limits occupancy in `tf_ds_compress2_norm_kernel`, `tf_ds_fused_prompt_idx_key_kernel`, or 0195. Only then test dynamic or shape-specialized `red`/`srow` allocations and occupancy against kernel time. Do not assume shrinking arrays or using a smaller CTA helps, and do not replace warp-uniform arithmetic with lane-0 arithmetic plus an extra shuffle without measurements.
- **CUDA FP4 conversion intrinsics:** As an opt-in experiment, test `__nv_cvt_float2_to_fp4x2` for paired E2M1 packing and `__nv_cvt_fp4_to_bf16raw` for query dequantization. Pre-clamp with `tf_ds_clamp6`: the [documented conversion](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html) maps NaN to positive MAXNORM while the kit requires -6. The latent path currently avoids normalized division with input-unit thresholds, so do not assume `v / scale` plus an intrinsic is either equivalent or faster. Exhaustively compare all 65,536 BF16 inputs at every reachable scale, ties, signed zero, infinities, NaNs and nibble order; check all 16 dequant codes and scaled-BF16 output, then accept only on SASS/timing evidence.
- **Extend 0195 to 100K+ contexts with bounded larger shared allocations:** At 100K scored keys, `nbk` is about 12,500 and the existing high-key/maxima cache needs about 50 KB. Test the existing opt-in function-attribute support in `zig/src/cuda/module.zig:72-81` and raise `max_nbk` only within the verified device limit and **65,536 blocks**. The radix shortcut that skips constant column bytes fails at 65,537, which must retain the unfused fallback. Gate parity at 8,192/8,193 and 65,536/65,537, actual 100K shapes, successful launches, occupancy, and fused-versus-three-kernel timings. Hierarchical selection is a separate later design experiment beyond that boundary, not the first change.
- **Split PDL entry only around proven independent work:** `pdl_enter` currently performs `griddepcontrol.wait` then `launch_dependents` at kernel entry. The [programmatic dependent launch model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) permits successor overlap before dependent data is ready. Test launching dependents earlier and prefetching only fixed norm weights or provably initialized RoPE storage before the wait; every dependent input read/write must remain after `griddepcontrol.wait`. Device `pos` indexed RoPE loads are not automatically pre-wait-safe, so use per-kernel provenance rather than a global relaxation. Require delayed-producer canaries, graph and non-graph bit parity, SASS load/wait placement, and measured end-to-end overlap without occupancy loss.
- **Decide adoption and then delete the superseded path:** every fused kernel currently ships beside a full unfused fallback, and 0197 supersedes 0127's kernel at widths 128/512. 0200 took the dead scratch arguments and 0203 gave `ch.invalid` its reader, so what is left is the pair of implementations itself. After a switch clears the exact gate and a graphs-on serving A/B, pick the default per switch and shape, then remove what the default makes unreachable: 0127 at 128/512, the fused-path scratch parameters, and the second implementation of each chain. Until then keep both paths, but record the adoption decision per switch so the code does not carry two implementations indefinitely. Removing the loser also shrinks the surface 0203/0204's guards must keep honest.
- **Measure the decode launch floor before spending on micro-optimization:** The repaired 0197 decode compressor measures 1.563/1.806 µs at R=1 and only 2.004/2.263 µs at R=48 for widths 128/512 ([decode.log](runs/fused-kernel-audit-20261120/decode.log)), so at decode row counts these nodes are dominated by graph-node and launch cost rather than kernel body. Treat 0201, 0207, 0209-0211 and the RoPE/softmax items as prompt/large-batch work and judge them there; expect little from them at R≤7. 0195 is the exception — 9.804 µs at 4,096 blocks is genuinely work-bound, so 0206/0208 and the histogram/vectorization items are the decode-relevant ones. Re-measure the floor with an empty kernel node on the same graphs before claiming any decode-side win.
- **Small-row parallelism for 0195:** `tf_ds_idx_fused_kernel` is one CTA per row (`dsv41_ops.cu:738-800`), so a seven-row decode round launches seven CTAs over the whole device while each reads 32K scores, which plausibly explains part of the 2x gap to a memory-bound estimate. Research a two-phase form — per-CTA partial histograms and compaction merged by total key order — or a row-per-warp-group layout. Selection ties are a total order on `(score bits, inverted column)`, so any merge must reproduce that order exactly, and the `nbk <= 8192` shared-memory bound has to survive the split. Gate on byte-level `mask`/`cblk` parity, actual 4,096- and 8,192-block rows, and a real R=1/7 timing win over the current single-CTA form.
- **Make fused-vs-unfused parity durable and re-runnable:** the evidence for every kernel's bit identity currently lives in scratch runs (`runs/fused-kernel-audit-20261120/build.sh`, `probe.py`, `decode.py`) rather than the project's own tooling. Promote a trimmed version into a Justfile target that rebuilds both CUDA variants, runs the focused parity cases per kernel (including the fallbacks 0206-0212 changed), and prints pass/fail, so a later edit cannot silently break the contract. Pair it with a short table in the docs: kernel, reference implementation, rounding points and guarded shapes. This is the cheapest way to keep 0206-0212 honest.
- **Measure 0197's duplicated partner resolution before changing it:** `tf_ds_compress2_partners` (`dsv41_torch.cu:410-425`) runs in every warp of the block, so sixteen warps repeat the same `rslot` reads and ballots per group at D=512. The data is L1-resident and the loop is one or two iterations at served row counts, so treat it as a measurement question (executed ballots/loads in SASS or NCU), not an assumed win.
- **Prove engagement and cover the real shape matrix before declaring optimality:** Use Nsight kernel names/traces to establish which fused paths execute at each row count, width, and PDL/graph configuration; add only opt-in counters if tracing is insufficient. Include 0195's 16K-65K window and the larger-allocation 100K research arm, prompt query reuse, decode R=1/7/32/48, prompt R=128/1024, and tied/random/flat selection data. Keep always-on counters and service overhead out of production.

## Additional 0195+ performance findings

These 17 proposals extend the review of `tfref-fusedbench@d66e663`. Byte counts and operation counts below come from the audited shapes. Their effect on GB10 latency and serving throughput still needs measurement.

### Prompt indexing and workspace

- **Skip compaction when 0195 returns only a mask.** Once the radix cutoff is known, each thread can write its block's mask bit directly. The prompt call passes no compact block list. Now that 0200 removed the dead `pidx` store, this path can also drop the ballots, prefix sums and compaction barriers (the decode arm still needs them for `cblk`). Check the newest visible block, ties, NaNs and negative infinity against the existing mask. [Kernel after the cutoff](zig/patches/0195-dsv41-tf-ds-idx-fused.patch#L105)
- **Resolve 0195 cutoff ties during decode block-list compaction.** Stop radix selection after the score digits. If a tie straddles the cutoff, take all higher score keys and fill the remaining slots from tied columns in ascending order during compaction. At 4,096 blocks and `k = 2,048`, the current selector used column-ID passes on all 64 tied test rows (both passes on 63) and on zero of 64 random rows. The host variant matched a full sort on all 192 random, tied and mostly negative-infinity rows. Time both distributions on GB10 to price the extra tie counting against the saved passes. [Radix loop](zig/patches/0195-dsv41-tf-ds-idx-fused.patch#L29) · [Ordered compaction](zig/patches/0195-dsv41-tf-ds-idx-fused.patch#L111)
- **Score only the retained blocks in later prompt indexers.** The candidate source at layer 20 builds a per-row pool. Layers 24, 28, 32 and 36 currently score the full context and then call `applyPool`. Score directly from retained block IDs and map picks back to original columns. At eight keys per block and 2,048 candidate blocks, each row needs at most 16,384 scored columns. Compare the final ordered picks, including ties, invalid entries and partially visible blocks (`prompt.zig:1435–1450`). Compare this list-producing variant with the mask-only path above.
- **Bypass prompt scoring when every compressed key fits.** For `0 < n_comp_end <= index_topk` (512 here), emit the visible IDs with `iotaVis`, set `sh.kk`, and skip the indexer query projection, RoPE, quantization, scoring and selection. Decode already uses this identity. Check 511, 512 and 513 columns and rows whose visibility ends inside the chunk (`prompt.zig:1373–1390`, `round.zig:1185–1190`).
- **Size the prompt candidate mask to the rows that reach the decoder.** Bounded replay can pass only 128 rows to the candidate source while `ch.pool` reserves space for 2,048. At 1,048,576 compressed columns and eight columns per block, allocating 128 mask rows saves 240 MiB per rank. Grow before a larger replay or full-prefill call needs the buffer (`prompt.zig:498–504, 2438–2455`).
- **Share prompt scratch between keyed and dense scoring.** Allocate one aligned workspace large enough for either `ch.keys` or the simultaneously live `ch.score`, `ch.bmax` and `ch.pidx`. The reviewed configuration could release about 104 MiB per rank. Include the unfused pool fallback and the split-index all-gather, which reuses `ch.keys` after selection, in the lifetime check (`prompt.zig:490–505, 1431–1465`).
- **Reuse the prompt visibility vector across layers.** Its values depend on chunk start, row count and compression ratio. Reuse `ch.vis` while that tuple is unchanged and refresh it after a replay cut or ratio change. The checkpoint's 18 ratio-2 encoder layers can then share one upload per encoder-only chunk (`prompt.zig:1095–1101`).

### Compression and FP4 packing

- **Fold prompt raw-ring writes and position setup into the fused paths.** The ratio-2 source uploads ring indices and launches separate scatters for raw KV and scores. Write both tails by modular position and derive the group ID and RoPE position from chunk start, ratio and row in the fused stores. Across the four KV-source layers, a full chunk currently incurs 11 metadata uploads and six scatter launches. Preserve ring wrap, an odd trailing row and the one-row case with no completed pair (`prompt.zig:1296–1306, 1317–1347`).
- **Skip 0198 latent packing on nonowner decode rows.** Sharded decode redirects those rows to a trash destination. Use row ownership to skip latent rotation, packing and stores after the replicated index-key projection has consumed `rd.lat`. Check mixed-owner rounds on both ranks (`round.zig:1160–1177`).
- **Compute a repeated ratio-2 pair once per speculative window.** Rows with the same ordered pair of resolved source slots get the same softmax and normalized latent. One contiguous seven-row window has four distinct pairs. Compute each once and deliver its BF16 result to every corresponding output row. Every input row still writes its own raw-ring slot. Check separate requests, ring wrap and partners read from an earlier round. [0197 decode kernel](zig/patches/0197-dsv41-tf-ds-fused-compress2-norm.patch#L213)
- **Derive the 0198 key and query E8M0 scale directly from BF16 bits.** The group maximum comes from BF16 values. Its exponent and a significand comparison with 1.5 determine the power-of-two scale and can eliminate the FP32 division by six. The host comparison covered all 65,536 BF16 encodings. Check the minimum-scale clamp, infinity and exact scale bytes on GB10. [Scale calculation](zig/patches/0198-dsv41-tf-ds-fused-rope-store.patch#L98)
- **Multiply by the exact reciprocal when applying the 0198 scale.** Construct the reciprocal once per 32-element group from its power-of-two exponent, using zero for an infinite scale. This can replace 28,672 per-element divisions in a seven-row, 32-head, 128-wide query. Across all 65,536 BF16 encodings and 254 reachable scales (16,646,144 pairs), the host check matched key codes and dequantized query values. Compare output bytes and graph timings on GB10. The current query baseline is 1.681 µs at that shape. [Key packing](zig/patches/0198-dsv41-tf-ds-fused-rope-store.patch#L121) · [Query quantization](zig/patches/0198-dsv41-tf-ds-fused-rope-store.patch#L247) · [GB10 baseline](runs/fused-kernel-audit-20261120/gb10-repeat.log#L25)
- **Use a warp integer maximum for the 0198 FP4 scale groups.** Clear the BF16 sign bit and map NaNs to zero for the maximum. A masked `__reduce_max_sync` can replace four shuffle/maximum stages for 16 lanes or five for 32. A host check covered all 65,536 encodings and both group widths. Compare emitted GPU instructions and packed bytes for keys, queries and latents. [Current reduction](zig/patches/0198-dsv41-tf-ds-fused-rope-store.patch#L90)

### Sharded KV fetch and expert launches

- **Reuse fetched KV and renumbered picks between decode layers.** In fetch mode, the served checkpoint has 38 compressed attention layers and eight stretches with the same KV source and pick set. Fetch once per stretch and reuse the codes, scales and indices for each layer's attention. At seven rows and 512 picks, eliminating 30 full fetches represents 29.53 MiB of logical row payload per round. Invalidate on a new round, KV update or pick update, and keep both ranks' ring schedules in agreement (`round.zig:1329–1374`, `config.zig:287–290`).
- **Gather a single sharded prefill row's picks directly.** When the peer lacks a cache copy and one row reaches attention with more visible columns than picks, gather into pick order on the owner and send a fixed-size payload (144 KiB at 512 slots). The receiver can address each valid slot by its pick position and preserve invalid picks as −1. This removes the context-sized clear, four mark/scan launches, stream synchronization and four-byte host read from each rank per affected fetch (`shard.zig:345–347`, `prompt.zig:1219–1236, 1258–1269`).
- **Use a bitset for the union of multi-row sharded prefill picks.** Mark each valid pick with an atomic OR and scan the population counts of the 32-bit words. A word's prefix count plus the number of set bits below the pick gives its compact index in ascending order. At 1,048,576 positions the bitset and per-word prefixes occupy 256 KiB per rank, down from 4 MiB for the current array, and the scan covers 32,768 entries. A host check matched union counts, indices and invalid-pick handling in 30 cases. Benchmark long prefill chunks with realistic overlap between rows to measure contention on shared words. [Current union scan](zig/patches/0015-dsv41-kv-shard-attn.patch#L225) · [Pick renumbering](zig/patches/0015-dsv41-kv-shard-attn.patch#L283)
- **Tighten the uncapped 0199 downstream expert grid.** Each row has six routed picks and the same shared expert. Use `min(expert_count, 6 * rows + 1)` as the upper bound on distinct experts, retaining any tighter applied-cap bound. At seven rows on TP2, 49 places become 43, eliminating 432 empty gate/up blocks and 240 empty down blocks per MoE layer per rank. The earlier capped test measured only a 0.0–0.5% decode-time change for a larger grid reduction, so this is a lower-priority efficiency experiment (`exl3_experts.zig:393–395, 430–433`). [Capped comparison](runs/review-round/RESULTS.md#applied-cap-t15)

## Fusion candidates outside the indexer and compressor

Source locations below use the applied TensorFold recipe tree at `ffdce65`. Byte and launch counts describe the current paths. GB10 speedups for these proposals remain unmeasured.

### Drafting and text prefill

- **Pack the DSpark confidence input in one kernel.** Gather each draft token's embedding, widen it and the corresponding hidden row to FP32, and write both into the final input for the existing cuBLAS confidence projection. At four streams, 11 preparation kernels become one, excluding the validity-flag clear and cuBLAS call. Host packing matched byte for byte across 48 stream, block and depth combinations. Compare GPU confidence values and accepted draft counts (`draft.zig:692–703`).
- **Use the decoder's fused attention output in DSpark.** On supported split-attention shapes, let the merge perform inverse RoPE and prepare the rotated input to the output projection. This could remove `ropeHeads` and `rotMany` from each of the three draft blocks, six launches per pass. Compare the projection inputs byte for byte and retain the separate path for one-split shapes (`draft.zig:749–786`, `round.zig:1325–1372`, `tri_attn.zig:118–130`).
- **Fold the prompt expert activation into the gate/up GEMM.** Compute paired gate and up tiles so the epilogue can apply the activation and down-input rotation before storing. At 2,048 rows, seven slots and TP2 width 1,152, the FP32 gate/up intermediates occupy 126 MiB per rank per layer. The whole gate/up stage occupied 17.9% of the head rank's chat-prime wall window. Time the standalone epilogue to determine how much is recoverable before changing the tile layout (`exl3_experts.zig:249–262`). [Stage timings](runs/overnight-opt/followthrough02/stage-windows.csv#L27)
- **Fuse DSpark Markov bias and scoring on greedy draft steps.** On a cache miss, round the computed bias tile to FP16, add the logits and emit the tile winner with the same tie rule. Writing and rereading 64,640 staged FP16 values costs 252.5 KiB per missed stream, step and rank. Measure hits and misses separately. The sampled path consumes an intermediate between these launches (`tri_markov.zig:124–141`).
- **Decode dense EXL3 prefill weights inside the GEMM.** The large-row path expands a complete FP16 matrix for each projection. On TP2, `wq_b` alone materializes 40 MiB per rank per call. Compare 256- and 2,048-row chunks because several row tiles can otherwise decode the same weights repeatedly. Standalone unpack accounted for 1.0% of the head rank's chat-prime wall window (`exl3_prefill.zig:98–111`). [Stage timings](runs/overnight-opt/followthrough02/stage-windows.csv#L25)
- **Choose text or image expert bias in the first routing pass.** Route each prompt row once with its own bias and write picks and weights directly to their destination. This removes one repeated routing launch and two device copies per overlapping image span per MoE layer. Check text-only, image-only and mixed chunks against the existing picks and weights (`prompt.zig:1597–1607, 1615–1632`).

### Sampled verification, drafter state and vision

- **Exchange rank-local top-k candidates during sampled verification.** A plain TP2 device-draw round gathers 64,640 FP32 logits per rank per row, then copies the shards into vocabulary order. Each rank could select its best 56 pairs of FP32 logit and global token ID. Merging the two lists before the existing draw would reduce each rank's collective input from 258,560 to 448 bytes per row for every supported positive-temperature `top_k` from 1 to 56. A host check matched the ordered global candidates in 96 cases, including cross-rank ties, signed zeros, infinities and NaNs. Benchmark selection plus exchange plus draw with graphs on. Check seeded picks and mixed greedy rows, carry the exchange mode to both ranks and the graph key, and keep full logits for scoring and host speculative sampling (`lanes.zig:1449–1467, 1512–1523`, `round.zig:2009–2027`, `dsv41_ops.cu:1390–1422`).
- **Write prompt taps directly in DSpark's input layout.** Each of three tapped layers writes a separate plane. At chunk completion the serving path copies all three into the row-major layout that `absorb` consumes. Decode already writes taps side by side, and prompt's `hcMean4` accepts the same output stride. At 128 rows and 5,120 channels this removes three copy launches per rank and 3.75 MiB copied, or 7.5 MiB of logical reads and writes. A host comparison matched the packed bytes in seven full and partial chunk cases. Check the prompt tap probes that currently expect contiguous planes (`round.zig:1978`, `exact.zig:355`, `prompt.zig:2582–2590`, `model.zig:1700–1704`, `layer_main.zig:892`).
- **Batch the three drafter KV ring stores after projection.** Both `absorb` and `absorbMany` group the stage projections, then launch one normalization, RoPE and ring-write kernel per stage. A single grid over stages and rows can read each stage's weights and write its ring directly. The extra BF16 output, `dr.ao`, has no reader in either path. At seven rows and 512 channels this takes three launches to one and removes 21 KiB of scratch writes per rank. Compare ring bytes after wraparound and after batched absorbs into different slots (`draft.zig:475, 487–506, 510–542`, `tri_norm.zig:67–72`).
- **Fuse the vision tower's residual adds with the following RMSNorm.** Each of 32 blocks has two residual adds. The first feeds the block's second norm. The second feeds the next block's first norm or the final tower norm. Combining each pair could remove 64 launches per encoded image and one 1,024-channel BF16 residual reread per pair, totaling 128 KiB of logical reads per image patch across the tower. The kernel must write the rounded BF16 residual for the next skip connection and normalize that rounded value in the existing reduction order. Compare both the per-block residual probes and the final normalized features (`vit.zig:17–19, 273–278, 305–345`, `dsv41_ops.cu:1218`).
- **Derive image-span token types and source rows inside the embedding kernel.** The CPU builds two arrays from the image grid. Each upload waits for the tower stream to synchronize. The span row index and grid width determine the start, newline, end or aligner row directly. Computing that mapping in the existing span kernel removes two host-to-device copies and two explicit stream synchronizations per encoded span. A host check reproduced every type and source row across 4,093 rectangular grids within the 4,096-token buffer limit. Measure image preparation latency and compare the final BF16 span embeddings (`picture.zig:41–68`, `vit.zig:349–352, 366–393`, `vision.zig:131–139`).

### Drafter projections, prompt attention and replay

- **Project vocabulary logits only for the draft steps the pass will score.** The three attention blocks need every row. After the final block, however, the TP2 head still projects `N * blockRows(steps)` rows while Markov and confidence read only the first `steps` per stream. Pack those rows while collapsing the hidden state, retaining the pre-normalization values for confidence. With four streams, five block rows and three requested steps, the head would project 12 rows instead of 20, avoiding 517,120 FP32 outputs (1.97 MiB) per rank. A host row-selection check covered 114 layouts, including 64 shallow passes. Measure how often the served policy asks for fewer steps than the block width. Compare the retained logits, seeded drafts and confidences across the GEMM shape change, and account for the distinct strides used by the noise and speculative-score hooks (`draft.zig:151–175, 558–562, 669–703`, `tri_markov.zig:124–135, 160–165`, `lanes.zig:1779–1788`).
- **Write the prompt window's KV ring tail from the normalization and RoPE kernel.** Outside ring mode, attention first copies any required history to `wsrc`. It reads that private snapshot and the new KV rows from `wsrc`, then uploads destination indices and scatters the last new rows into the ring. The KV producer can also write the last `min(n, 144)` rows to their modular ring slots after the history copy. That removes one index upload, one scatter launch and up to 144 KiB of KV rereads per layer and rank. Only the retained tail has unique destinations when the chunk exceeds the ring. A host comparison matched the attention input and final ring in 432 cases covering wraparound, long chunks and replay floors (`prompt.zig:1065–1071, 1085–1107, 1142–1165`).
- **Reuse the first decoder layer's attention mix when replay keeps the same rows.** At layer 20, `replayCut` computes the mix for the KV-source update. The main loop then computes it again. When the cut drops and prepends zero rows and no intervening operation changes the hidden state, the normalized input and the pre, post and combination weights are already in their destination buffers. Reusing all four would remove `_hc_partial` and `_hc_finish` once per eligible chunk. Cold prompts of up to 128 tokens are a direct test case. Compare all four outputs between the first mix with a pending MoE gather and the second mix over the posted residual (`model.zig:1617–1634, 1665–1670`, `prompt.zig:1002–1012, 1318–1322, 2539–2550`, `tri_basic.zig:18–25`).
- **Use the decoder's joint query and KV normalization in drafter attention.** After the input projections and any LoRA updates, query normalization and rotation run separately from KV normalization and RoPE. The decoder's `qKvNorm` already accepts both inputs. The launch fixture records the matching query width 1,280, KV width 512, RoPE width 64 and one input rotation. This could save one launch per draft block, three per pass. Verify kit dispatch at actual draft row counts, byte-compare both outputs and the rotated query input, and confirm the drafter's −1 slot sentinel leaves its ring untouched (`draft.zig:729–749`, `tri_norm.zig:43–55`, `round.zig:1182–1183`, `fixtures/launches.json`).
- **Assemble resumed decoder rows and positions in one kernel.** On an exact-resume prepend, the current path copies the fresh residual and its four mixing weights separately, restores each from one or two ring segments, and uploads a new position vector. One kernel can choose the tail or fresh row by output position, copy both fields and write aligned positions. This takes four copy launches, or six when the restored prefix wraps, plus a host upload down to one launch. Keep the subsequent tail stash ordered after the restore so it cannot overwrite an unread slot. Host row layouts matched in 150 wraparound and short-append cases (`prompt.zig:183–189, 2498–2512, 2544–2576`).
- **Select image embeddings inside prompt embedding initialization.** The host already prepares one embedding ID per row. It can encode each overlapping image's source-row index there, allowing the initialization kernel to select the image or text table and broadcast the selected row to the four hidden streams. Today every image row gets the placeholder embedding first, followed by four strided copy launches per overlapping span. Direct selection removes those launches and 40 KiB of overwritten BF16 stores per image token per rank. Preserve the original sequence IDs for Engram. A host check matched the four output streams across 125 chunk layouts, including multiple and partially overlapping spans at chunk boundaries (`model.zig:1559–1567`, `prompt.zig:974–989`, `tri_basic.zig:38–42`).
- **Widen the large-prefill router weights once per layer.** Above the decode-row threshold, each call converts the same 384 × 5,120 FP16 gate matrix to FP32 before cuBLAS. The drafter already widens its static router weights at load. Applying that pattern to selected prompt layers avoids one conversion launch and 11.25 MiB of logical conversion traffic per cached layer, rank and subsequent chunk. Each resident FP32 matrix costs 7.5 MiB. All 40 would occupy 300 MiB per rank, so time a hot subset first. The entire routing stage accounted for 0.90% of the head rank's chat-prime wall window, which bounds the opportunity there (`prompt.zig:479, 1597–1605`, `draft.zig:393–396`, `config.zig:280–286`). [Stage timings](runs/overnight-opt/followthrough02/stage-windows.csv#L28)

## TensorFold upstream performance candidates

Source references use the applied TensorFold tree at `ffdce65`. The counts below describe work each change could remove. GPU latency and throughput are unmeasured.

### DeepSeek V4.1 on CUDA

- [ ] **Read contiguous verify-round taps directly in the drafter.** When participating windows occupy consecutive rows of `rd.taps`, pass the first row and total count to `mainKeys`. Keep packing windows that have gaps. Seven rows with three 5,120-channel BF16 taps avoid a 210 KiB copy per rank. Also skip the three tap launches in ordinary verify rounds when the drafter is disabled, retaining taps for explicit probes. Compare drafter ring contents for a single window, adjacent windows and windows with gaps. See `dsv41/round.zig:1976`, `dsv41/model.zig:1760` and `dsv41/draft.zig:512` under `zig/src/families/`.
- [ ] **Pad prompt attention picks once per selection.** Reuse the padded list until an indexer changes the picks or a replay cut changes the rows. For a cold 127-token prompt, layers 2–7 share the same 63 picks per row. Padding to 64 once saves five buffer fills and five copy launches per rank. Preserve the compact row stride used by sharded fetch and check the −1 padding at the attention input. See `zig/src/families/dsv41/prompt.zig:1122` and `:1251`. [Layer map](docs/fused-c-d-scoping.md#L14)
- [ ] **Pass image rows through the prompt Engram gate.** For an image row, write its original hidden streams directly to the gate output. This removes one restore launch per overlapping span per Engram layer. Preserve image positions in the token history used to hash neighboring text. Compare hidden rows at both ends of a span, including one that crosses a chunk boundary. See `zig/src/families/dsv41/prompt.zig:2433`.

### Nemotron, Qwen3.5 and GLM

- [ ] **Nemotron CUDA — end an absorb-only draft after its KV write.** Level zero currently runs attention and its output projection before returning. The prompt absorb path already ends at the cache write. Moving the level-zero return there would save the attention chunk and merge launches plus one output projection per call. Compare the resulting cache and the first draft that reads it. See `zig/src/families/nemotron/cuda_mtp.zig:144`, `:166` and `:295`.
- [ ] **Qwen3.5 Metal — save only the final convolution snapshot during ordinary prefill.** The convolution kernel writes state for every row. With recording disabled, the cache commits only the final row. At 128 rows across 18 recurrent layers, suppressing the first 127 snapshot writes avoids 80.37 MiB of stores per chunk. Retain per-row snapshots during speculative verification so partial acceptance can restore any kept prefix. See `zig/kernels/metal/qwen3_5/gdn_pre.metal:71` and `zig/src/families/qwen3_5/state.zig:64`.
- [ ] **Qwen3.5 Metal — run the vocabulary head only on the final prefill chunk.** Every chunk commits recurrent state, but the first-token sampler reads only the final chunk's logits. Skip the vocabulary projection and the 485 KiB logits copy on each earlier chunk. Compare final logits and cache state across different chunk boundaries. See `zig/src/families/qwen3_5/backend.zig:74` and `zig/src/families/qwen3_5/forward.zig:103`.
- [ ] **GLM Metal — reuse the projection and indexer weights from draft absorb.** Batched absorb computes both for every row. The draft path then gathers the last row of each stream that will draft and recomputes both outputs. Gather the needed query inputs and indexer weights with those rows. Time the extra gather traffic against the saved projection and scaling dispatch, and compare drafts for batches with mixed depths. See `zig/src/families/glm/slots.zig:357`, `zig/src/families/glm/forward.zig:220` and `zig/src/families/glm/mtp.zig:52`.

## Follow-up: why a seven-row drafter pass costs the first six drafts (2026-11-19)

With `TF_DS_DRAFT_BLOCK=7` and the seventh draft never offered, the round loop verified 1.6% more drafts and ran
0.4% more rounds than the six-draft block on 21 code-like prompts, with identical replies. That cost cancels the
seventh draft's gain. If it comes from the wider window's pricing and not from the drafts, seven would be about
1.3% faster than six. The logged arm and the replay are in
[runs/draft-ladder-20261119](runs/draft-ladder-20261119/RESULTS.md); the switch and its log are in the bench image.

## Owed at the next full benchmark round: the Engram lead reader (2026-11-19)

`TF_DS_ENGRAM_LEAD=1` is on in production since November 19, 02:22 UTC, ahead of its full test. The soak, the
long-lived deep-session run, 300K prompts, the warm tail and the exact panel are still owed; the protocol, the
keep rule and the rollback are in [to-benchmark.md](to-benchmark.md#current-evaluation-order).

## Parked: round-wide expert caps (future idea, 2026-11-19)

- `TF_DS_T13_EXPERT_CAP=32,1` and `24,1` are fast at four and eight streams and pass the single-stream likelihood line, but replies vary with load. Not adopted. Reopening needs the eight-stream SQL and tool panels with reply text saved, `32,1` first, and a quality gate in place of the concurrent-equals-solo check.
- A per-stream cap and a reduced-precision first pass for choosing experts were read against the source and are not worth building at these sizes. [Screen and source reading](runs/screen-20261118/RESULTS.md)
- The lead reader's full round is queued in the [evaluation order](to-benchmark.md#current-evaluation-order).

## Follow-up: the C/D fusions are built and queued behind the Spark gate (2026-11-19)

- `0197` `TF_DS_FUSED_COMPRESS2_NORM` (ratio-2 compressor chain + trailing RMSNorm in one launch, decode and prompt) and
  `0198` `TF_DS_FUSED_ROPE_STORE` (decode indexer key/latent/query and prompt key/latent rope-store chains in one launch
  each) are built, CPU-emulation verified and exported. Both are off by default and queued after 0193/0194 -> A -> B.
  [Scope](docs/fused-c-d-scoping.md) · [Report](../p0197/report.md)
- Expected gain is small (C ~3, D ~20 launches a round, 0.05-0.2%): keep or drop on the PDL gap profiler (`TF_DS_PDL`,
  continue.md item 2) and the Spark exact suite, never on the eager profile alone.
- Parked: Engram's `add2Bf16` + `_engram_gate` fold (two launches a pass for a three-`tl.sum` reduction-order risk);
  `applyPool` and `candFast` (already single launches). Revisit only if the measured gap justifies it.

## Follow-up: fuse the kernels that choose experts (lossless, small; 2026-11-19)

- Choosing experts is three launches a layer in a decode round: the gate projection and the top-six routing (both kit kernels) and the prune (`tf_ds_expert_prune_kernel`, the engine's own). Together they are 2.4–3.2% of a solo round and about 2% at four streams in the eager profile, roughly 41 µs a layer, reading at 6–9 GB/s, so launch latency and not memory bandwidth bounds them. [Stage windows](runs/overnight-opt/followthrough02/stage-windows.csv)
- Fusing the prune into the routing launch, or all three into one, might recover under 1% of a round. That figure is an estimate; nothing was built.
- The gate and routing kernels have no source in the image or any local checkout. A fused kernel must reproduce their arithmetic bit for bit, so start by obtaining the reference source or by proving a replacement against the 39-case exact suite. [2026-11-19 audit](docs/kernel-kit-provenance.md): the source is public (kit @ abe07c7 + `bertholomus/TensorFold` @ `deepseek-v41-tp2`); a rebuilt kit is a different binary, so the Spark exact gate decides, not source access. The fused route (0193/0194) and candidates (a)/(b) in `continue.md` item 4 are the live work here.
- Measure with graphs on before building: the eager profile overstates launch costs that a captured graph hides.

## Follow-up: a rebuilt kit would be a new artifact, not a drop-in (2026-11-19)

The kit (`deepseek-v4.1-tensorfold-tp2-2xgb10` @ `abe07c7`) is public, manifest-pinned, and rebuilt only by
`tfref-next`'s `tools/dsv41-zig/build_kit.sh` (nvcc/Triton build-to-build differences are expected). If a kit
rebuild or a from-source kit ever becomes attractive (e.g. EXTRA_ROWS coverage), the path is: rebuild → `verify_kit.sh`
→ build a Spark image → 39-case exact suite (+ the 8 saved replies) → adopt or drop; the published kit stays the
record. Until then the untouchable rule stands (see [provenance audit](docs/kernel-kit-provenance.md)). Reference
sources for the Triton AOT set and the EXL3 extensions are `bertholomus/TensorFold` @ `deepseek-v41-tp2`
(`kernels.py`, `cuda/exl3/`) and the Python engine's `tools/dsv41/`.

## November 17 work packages (historical)

- Use the [current evaluation order](to-benchmark.md#current-evaluation-order) for the next pair experiments. The older plans below preserve design details and patch numbers.
- W1/0136 passed the 1.5M-token regression soak. Leave the row cache off. W6/0129 mapped warm-up did little to reduce the roughly six-second first-request cost. W5/0143 lost its screen. W4 compiled and ran. Its eager MoE bandwidth estimate deprioritizes M5/0169 and its 0196 follow-up. [Combo2](runs/combo2/RESULTS.md) · [W4](runs/w4/RESULTS.md)
- W3/W9 are open. Extend the 64-item generated prune-off self-consistency set with independent held-out references and concurrent task outcomes. The [applied-cap T15 trial](runs/review-round/RESULTS.md#applied-cap-t15) found at most a 0.5% median decode-time reduction at eight fixed rows. W7 concerns rank 0's weight file on /nfs1. 0163 subsequently implemented suffix scanning; its 6,000-token throughput screen was neutral. [Score set](tools/score_sets/README.md) · [Combo3](runs/combo3/RESULTS.md)
- Keep broken experiments out of performance rankings. Resolve 0132 CPU IDs, 0133 AIO splitting and 0170 full-hit lookup/KV placement before repeating those rows. For the later image fix 0169 initialization, 0186 compilation, default-path 0190, and the armed 0184/0185/0186/0188 failures before their respective timings. [Odds](runs/combo4/RESULTS.md)

## Original framing of the same review (kept for the rules)

- Findings, ceilings and work packages W1-W10: `docs/benchmark-gaps-and-moe-review.md`. The following list records the original November 15 ordering.
- First, in this order: W1 (the Engram row cache `0114` evicts from a std hash map and never rehashes it, the bug `0119`
  fixed in the token cache; it fits `runs/overnight/RESULTS.md`'s progressive regression, so `0114`'s "keep" is withdrawn
  until it is retested past the onset), W2 (fold the overnight verdicts into `to-benchmark.md`; production's switches in a
  file), W3 (score sets and a prune-off arm: the base of every row runs `TF_DS_T13_EXPERT_PRUNE=1`).
- Then W4, one instrument run that decides the rest: whether the decode MoE is limited by the memory bus (its time
  tracks expert bytes at ~200-220 GB/s) or by trellis decode, at 8 streams and at real prefill chunk sizes.

## Baseten-ultrafast series (0119–0122)

- Step 0 found (2026-11-16): phase A is LOSSY, not bit-identical — every off-grid
  cut moves bits, on both prompt sizes tested (`runs/matrix/step0-cold-shape/`; `--logits-sha 1` prints nothing on this
  build; teacher-forced NLL was the instrument). `0121` stays `TF_DS_EXACT_RESUME=1` opt-in, never default; see
  `decisions.md` 2026-11-16. Remaining gates, in this order: `0119`'s `encode_s` share of TTFT at
  100K and 250K have measured gains on multi-message histories. The 500K case and the production mix remain open; `0122`'s `tools/samp_test.py`
  distribution gate (5,000 seeds, DRAFTS=0 against the switch on) decides whether the accept rule is right; `0120`'s
  self-consistency set (`tools/score_sets/items.jsonl`, generated 2026-11-16 — 64 items, 24,497 tokens; gate PASSED:
  argmax_match 1.000 on all 3,072 continuation rows on `TF_DS_T13_EXPERT_PRUNE=0`; see `tools/score_sets/README.md`)
  decides `argmax_match`. Then the standing hardware smoke
  (`start.sh restart` + one chat completion) and one `to-benchmark.md` row an arm, 3x alternation for anything kept.
- `0121` phase B: keep the reply's decode-written rows (`TF_DS_EXACT_RESUME=reply`), LOSSY by construction. Needs a tap in
  `round.zig` at the decoder's first layer into the same 128-row tail and a Snap at stream end; gate it with `0120`'s
  scoring of the same continuation cold and reply-resumed.
- `0121`: `snapBytesFn` charges the tail for every state while only end states allocate it (~5 MB over-budget each, so
  earlier eviction); budget by `keepKind` when the numbers matter. The tail's start/end are not carried in
  `snap_save`/`snap_restore`: rank 1 keeps its own pass's span per kept state (followPair's `kept_tail`).
- `0122`: runnable on the TP2 pair since the fix round below (no longer refused at start). A speculated stream's depth
  now follows the core's per-depth rates (the head's chances read the drafts, which would bias the rule), so the head's
  confidences are unused for it; recalibrating them for `sum min(p, q)` is a follow-up (`0050`'s mechanism). Drop the
  switch if it does not beat `TF_DS_T26_KEYED_DRAFT`. Benchmark arm: `TF_DS_SPEC_SAMPLE=1 TF_DS_T26_KEYED_DRAFT=0`
  (`.env` sets T26=1 and the environment only wins when set, so it must be exported as `0`; on is exactly `1`).
- `0120`: a score stream runs alone both ways now (nothing is admitted beside it: `error.ContextFull`, requeued);
  `validate.py`'s concurrent check is the gate before lifting that.
- `0119`: the cache's 32 MiB is outside `prepare_gate`'s reservation (small against the 2.5 GiB floor, but it is the one
  memory figure not yet measured on the head).
- `zig/patches/README.md` gained rows for `0117` (energy) and `0118` (LoRA) as well, which were landed without one.
- Review round (2026-11-14), fixed in the patches: the series did not compile for the image (`hf_text.zig` and
  `ops.zig` passed `std.c.getenv`'s `[*:0]u8` where a slice is taken; a host `zig build` never analyses `Model.open`
  past `kernels.available`, so type-check with `zig build install native -Dfatbins=<dir of dummy *.fatbin>`, never
  host tests alone). `0122`: the Drafter's new fields were left undefined with the switch off (drafts on: a memset and
  an extra NCCL gather on stack garbage, the 0114 class). `0121`: `fillBegin` zeroed the tail's device pointer (rank 1
  faulted at the first keep), the prepend wrote over the chunk's rows and read its positions from before `pos0`,
  rank 1's restore cleared the tail's span (it failed the resume chunk rank 0 ran), and the both-rank compare never ran
  (the features ride only the spill/KV hello) - a `switches` frame (kind 15) now goes first, only when a switch is on.
  `0120`: the score's NLL had its sign flipped (PPL came out as 1/PPL), other streams could join a score stream's lane,
  and stop/grammar/thinking_budget/messages are now refused with `tensorfold_score`. `0119`: per-entry overhead
  charged (160 B) and the map rehashed after an eviction (tombstones), and `encode_s`/`render_s` time both renders.
- `0122` fix round (2026-11-15), no longer refused: the six blockers are fixed in the patch. (1) The verify's q gather
  now runs on both ranks: a `spec_q` frame (kind 16) names the rows and candidate ids after the verify, and both ranks
  run the same gather + all-gather (`Drafter.specGather`; rank 0 `Lanes.speculateRound`, rank 1 `followPair`'s
  `.spec_q`); the pass's per-step Zq gather is gone (the drafter pass gains no collective). (2) The normaliser's
  partials ride that gather at `r * wide` with `wide` derived from the frame on both ranks. (3) q includes the Markov
  bias: `tf_ds_spec_clean_kernel` runs between `_bias` and `_score` and writes `logit + bias`. (4) Max-shifted partials
  folded in double, log-space accept. (5) A rejection's residual gives x no mass (q(x) is the candidate's own number);
  the token-0/1 write is gone. (6) An empty residual keeps x when p has it, else draws from p (measure zero). Also: a
  speculated stream's allocation no longer reads its drafts' confidences (lookahead bias), T26 is refused only when on
  as ops reads it (`1`), and a tripwire counts drafts that are not the argmax of the gathered clean + noise.
- `0122` still open, in order of what to watch on the pair: (a) the `spec_q` gather is a new eager collective between
  the verify and the next pass (ring path with `TF_RDMA_DEVICES`): a hang there shows as both ranks stuck after a
  sampled verify; (b) `_score`'s arithmetic is assumed (logit + fp16 bias as fp32): a nonzero "tripwire misses" on the
  `dsv41 spec-sample:` line means q is wrong and the distribution gate will fail; (c) the .cu changes compiled only
  under clang with stubs: the image build is their first nvcc; (d) `samp_test.py` (5,000 seeds, DRAFTS=0 against the
  switch on) is still the first distribution gate; (e) host-drawn sampled rounds cost the device draw's saving
  (T27) and one sync + small D2H a round; (f) world 4 / 2D drafter head refused (`SpecSampleWorld`), not built.
  Benchmarking it: `tools/bench.py` sends temperature 0 only, so a `bench_matrix.sh` arm with the switch on measures
  greedy streams (the no-regression half) and never speculates; the sampled numbers (accept, tokens/round, decode
  tok/s at temperature 0.7 and 1.0, 1 and 4 streams, against `TF_DS_T26_KEYED_DRAFT=1`) need a sampled load, which
  no tool here sends yet. `validate.py` is greedy too, so its gate should stay IDENTICAL with the switch on.
- **`0121` is LOSSY (step 0 done 2026-11-16).** The full-run finding stands: turn 6 (1-token append, resumed at
  8,228 of 8,281) gives `96d0fd7dcd8e` vs cold `9c5a438c3fb3`, deterministically in 3/3 passes. The cause is any
  chunk-shape change, not the 64/128 kernel classes — bits move for every off-grid cut measured
  (DEBUG_SPLIT 8228/9300/9380/9440/9480; NLLs move both directions; t3k bit-stable). Phase 1 (resume only when
  shapes match cold) is vacuous; phase 2 cannot fix bits baked into the restored parent-pass state. Verdict and
  reasoning in `decisions.md` 2026-11-16; artifacts `runs/matrix/step0-cold-shape/`. Keep exists only as an opt-in,
  never on by default; phase B stays unstarted. Also: prompts of 121-2,727 tokens keep nothing to resume in either
  arm, and with the switch on turn 5 resumes 2,048 rather than the previous prompt (2,727).
- `0121` still open: `snapBytesFn` charges every state a tail with the switch on and rank 1 allocates one for grid
  states too (5.2 MB each); off-grid starts for the ratio-4/128 compressor layers are not addressed (only ratio 2);
  `TF_DS_DEBUG_SPLIT` is not refused with spill/KV shard. A pre-existing leak: an error after `snaps.append` in
  `snapSaveFn` destroys a Snap still listed. `0120` still open: `tensorfold.drafts`/`sampling` in the reply are computed
  before the score override; a score waiting at the queue's front holds back admissions behind it.

## Decode-step structure (plan A–E and original gates)

An external agent implemented all six items on `decode-step-structure` of `../tfref-next` (from
`series-0119` = TF_REF fbbc971 + 0001-0122). The six patches landed. The runtime switch group is included in the [combo campaign](runs/combo/RESULTS.md). Export + how to apply: `/home/colum/Workspace/tfref-next-handover/`
(`git am --3way` onto `series-0119`; verified by tree hash). Six patches, `0123`-`0128`:

1. `0123` `TF_DS_ROUND_PROFILE=1|2` — the served process's round profile at `/metrics` (the batch driver's
   line, key-equality equal against `tf-dsv41-lanes --profile` apart from `step_ms: -1`; a trailing
   `profile_note` line carries the attribution-only caveat). Unblocks the profile without stopping the pair.
2. `0124` `TF_DS_DEVICE_ACCEPT=1` — item A reduced to its buildable core: a chain window's keep computed on
   the device (`tf_ds_accept_chain_kernel` after the pick kernel); the one-frame verify+pass stretch stays
   unbuilt (the depth rule, the cuts and the T26 keys are rank-0 host state; see `docs/device-accept.md`
   (untracked) and `decisions.md`).
3. `0125` `TF_DS_COMPRESS_SIDE=1` — the compressor chain (a_in fork) on its own stream.
4. `0126` `TF_DS_PDL=1` — `griddepcontrol` at every hand-written kernel's entry + the PDL attribute on their
   launches; plus `tools/nsys_breakdown.py` (untracked), the gap measure, modeled on Baseten's; `l2_prefetch`
   deliberately excluded; `dsv41_shard.cu` is the known next candidate. Whether nsys works on GB10 is unchecked.
5. `0127` `TF_DS_FUSED_COMPRESS2=1` — the ratio-2 compressor chain (scatter x2/memset/gather x2/compress2) in one
   kernel, byte-identical (the review-caught cross-block race in drafted verify windows is closed by resolving
   partners against `rslot` in-kernel; see the commit).
6. `0128` `TF_DS_ENGRAM_BOTH=1` — both Engram layers' rows land at the round's first Engram cut; three stretches
   a round become two. Item E's full staging was **not** built: without cuFile the libaio completion is only
   observable on the host (see `docs/engram-staging.md`); option B (pinned host CQ poll) rejected on risk/reward.
   The drafter's blocks already group their stages' `wkv` into one `grouped` call at absorb (`draft.zig:505`), so
   the `mtp_wkv` analogy is spent too.

**Measured 2026-11-15 (image `exl3-tp2-b40ea899e367`, gate `validate.py --compare` against the campaign's base
save — IDENTICAL 8 of 8 with switches off and with `TF_DS_ENGRAM_BOTH=1`).** One bench block per arm: base
1771/2095/2140 tok/s prefill 2K/8K/32K, decode 62.4, multi.1 65.7, multi.4 122.4; `TF_DS_ENGRAM_BOTH=1`:
1760/1998/2103, decode 62.8, multi.1 61.7, multi.4 120.7 — nothing outside the noise band; **no measurable
improvement**. The `0123` serve profile works end-to-end on the live server. `exact.py` still owed before any of
the six is called done. Artifacts: `runs/matrix/b40-*.bench.json`, `b40.validate.json`, `b40-both.validate.json`.
Also: the pair's first request after a restart now deterministically fails the `start.sh` smoke (pre-existing on
the old image too; see `decisions.md` 2026-11-15) — `bench_matrix.sh`'s restart gate aborts every arm while it
persists; arms must be driven manually until this is root-caused or the smoke is revisited.

Gates before any of these is called done (in order, per patch): the image rebuild add, `start.sh restart` + one
chat completion with the switch on (host tests are not a substitute for the hardware smoke), `tools/exact.py`
IDENTICAL (0124 is the one to watch — the kernel re-implements the chain accept), `tools/validate.py --compare`,
then the keep/drop rows in `to-benchmark.md` ("Decode-step structure" section). The rank-0 profile
(`runs/ultrafast/prof/`) is the sizing evidence: host path ~1% of a round, MoE 60-70% of GPU time — items A, C, D
are small wins by their own gates; `0123` is the instrument. `0128` is the one plausible keeper.

## Hit-rate patches 0110-0116 (landed, uncommitted; one pair pass done)

- One signal-grade pass on 2026-11-13: all 9 gates IDENTICAL (cross-image), 0114 signed + (−57% Engram read
  time), 0116 + (zero-cost counters). The other five: no regression, wins unproven — run their targeted probe
  rows (`to-benchmark.md`: per-agent-tool fan-out, early-history edit, 8-stream pressure, 6-10 turn rewind),
  then 3x alternation for any toggle kept. Results table in `to-benchmark.md` "Pass results".
- HARDWARE SMOKE IS PART OF LANDING ANY PATCH SERIES (the 0114 segfault taught this: no host test runs
  `Model.open`, and diffs cannot show `= undefined` construction). Minimum: `start.sh restart` + one chat
  completion on the pair before a series is called done.
- `Model.open`'s `m.* = undefined` + field-by-field init makes all `Model` field defaults dead code — every
  future patch adding a defaulted `Model` field must add it to the init list (0114's bug class). Consider a
  literal init or a lint over the list.
- Review test gaps not covered by the fix round: 0114 mixed hit/miss AIO batches, concurrent readers and a
  `SCALES=map` end-to-end; 0115 tier-on (`rules.grid` set) + `shared_plan` combo; 0116 `publish()` under the host
  lock (test-only exposure). Ref: `docs/prefix-cache-hitrate-review.md` + the review artifacts in `/tmp/cache-review/`.
- Finding 4's shared-cut `min_tokens` exemption (needs `0099`+) after its `to-benchmark.md` rows measure; findings
  8-11 and 14-17 remain unimplemented picks (budget accounting, `min_prompt`, `presence_refused` repair,
  compaction attribution, prefetch, join match, restore economics) — see the review's ranked list.
- Pre-existing flake: `lane_host.test.prefill interleave` fails intermittently under plain `zig test` (reproduces at
  the pre-patch base 3/6; passes under `zig build test`). Fix or quarantine it.
- `start.sh`'s lock (`flock -n 8`, start.sh:122) is inherited by child processes: killing a `start.sh` leaves its
  orphans (e.g. `prepare.sh`'s image build) holding the lock, and every later `start.sh` dies with "another
  ./start.sh is already running" until the orphans exit. Drop the lock fd in spawned children (`8>&-`, as
  `bench_matrix.sh`'s mem_sample does with fd 9). (Bit us during the 0110-0116 benchmark pass, 2026-11-13.)
- `just matrix-start arms="a b"` landed `arms=a` as a bogus first argument (just substitutes unquoted into the
  shell line) and silently skipped the first arm — worked around in `bench_matrix.sh` (args strip a leading
  `arms=`), so both invocation forms work now. SAME trap bit `restart-head flags="…"` twice (once it glued
  `TF_DS_ENGRAM_LEAD=1` into `HF_HUB_OFFLINE` and the switch never armed — caught by checking the engine log
  for `engram-lead: on`): the recipe now takes `flags` first and strips `flags=`/`offline=` prefixes (all four
  call forms tested). Rule of thumb: after any `just <target> param="…"` call, verify the switch in the
  container env / engine log, not just the exit code.
- LOSSY proposals stay proposals (append-only compaction, image-row skip, canonical tool renderer): they can change
  reply bits and need opt-in flags + `exact.py` if ever built.

## Prefix-cache hit-rate review (docs/prefix-cache-hitrate-review.md)

- Ranked hit-rate improvements (18 findings, all code-validated) await the operator's pick. Top candidates:
  shared-cut discovery past the first user message; `history_len/2` edit-recovery gate; eviction protection for
  shared/branch states in both layers; `SPILL_MIN_TOKENS` shared-cut exemption; strided spill retention;
  process-lifetime Engram row cache keyed `(layer, row id)` incl. the rounds' AIO path. Ref: the review's
  ranked list and its "Validated / not verified" section.
- Before implementing findings 3/7/8: capture the finding-12 counters (memory CacheStats, resume-depth) on the
  live pair as a baseline; most of the review's frequency claims are code-traced projections.
- `to-benchmark.md` Engram corrections (row-cache scope/keying/hook, both-ranks metric, image-pad context) —
  fold in when the matrix rows are next edited.
- Stale refs to fix when touched: `tools/prefix_fanout.py` cites missing `PORT-PREFIX-SHARING.md`; `model.zig`
  and patch `0012` cite missing `docs/port/dsv41-state.md`.

- Per-switch `exact.py` records (the doc's full identity gate). The signal pass gates each arm with
  `tools/validate.py --compare` only (operator decision: no deep testing). `runs/exact-oldimg.json` is the
  cross-image base record. Ref: `to-benchmark.md` "Before the first run".
- TRIM row (`TF_DS_ENGRAM_TRIM=1`): skipped (operator decision) - punches ~101 GB per Spark irreversibly
  (nlink=1 both sides). Run it when the disk is actually needed; tables stay whole meanwhile.
- **Engine bug: `TF_DS_ENGRAM_DEDUPE=1` + `TF_DS_ENGRAM_LEAD=1` breaks the rank pair during decode**
  (NCCL communicator aborted, rank 1's link closes; all decode streams fail; reproduced 2/2 runs). Find and
  fix in the shared dedupe/lead path; add a pair test. Ref: `runs/matrix/*/dedupe-lead.*`. **Root cause found
  (2026-11-16, code read of `second-pass` @ 44b2cbb):** the dedupe scratch (`eh.d_uniq/d_back/d_w/d_s/d_keys/d_vals`,
  `prompt.zig:1701-1721` `useDedupe`) is one unguarded set of buffers, but LEAD runs two reader threads at once
  (`Lead.after`, `engram_lead.zig:70-80` spawns layer j+1's read of this chunk and layer j's of the next together;
  both call `engramReadRows(..., true)`). `dedupeIds` + the back-copy (`prompt.zig:1902-1915`) race; `back[]`
  corruption makes `eh.d_w[back[i]*row_w ..]` slices fault or serve garbage rows -> EngramRankFault -> the pair
  breaks. The main-thread fallback `engramRows` (dedupe on, `prompt.zig:2270`) can also run beside a lead reader.
  Fix shape: per-reader dedupe scratch (or a mutex around `engramReadRows(..., true)`); rounds are unaffected
  (they pass `dedupe=false`, `round.zig:1930`).
- `TF_DS_ENGRAM_LEAD=1` is the one reproducible prefill win (+5.0%, +6.2% two passes) - confirm with the strict
  3x base/arm alternation before claiming, then consider on by default.
- `TF_DS_ENGRAM_SCALES=map`: the cost is the page-fault path (cold -11%, warm -5%). The proposed `map-warm`
  row (`0099`+) decides keep/drop; do not enable map meanwhile.
- Proposed rows in `to-benchmark.md` (not implemented; need `zig/patches/0099`+): map-warm, decode threads,
  pool threads, CPU placement, row cache, read depth, host timing.
- "The quality set" is undefined in this repo (upstream author's 40-item set, PR #108). Sampled arms are
  scored on accept/tokens_per_round + greedy identity only. Port the upstream set if depth results need it.
- Strict 3x base/arm alternation for any toggle that lands within noise or is worth keeping (the current
  pass is one block per arm).
- Draft-depth pins break concurrent bit-equality (`concurrent == solo`) while keeping sequential identity -
  either document the caveat in the patches or find the shape-dependent path that changes sampled/batched
  greedy output under a pin.

## Runtime LoRA for the DSpark drafter (operator direction, not started)

- Goal: apply a trained PEFT LoRA to the checkpoint's DSpark blocks (`mtp.0`-`mtp.2`) at serve time, so a
  drafter fine-tune does not mean re-quantizing and splicing the 196 GiB pack. Operator picked the runtime
  adapter over the offline merge+requantize route and over training a fresh block drafter.
- Design: additive epilogue on the projection's output (`y += (alpha/r)*B*(A*x)`) through the family's
  glue/buf mechanism; the EXL3 trellis kernels in `projection.zig` are untouched. PEFT folder behind a new
  `TF_DS_LORA` switch, off by default (the series' rule).
- Hard parts: TP=2 sharding of A/B (the drafter is on both ranks, `start.sh:103`); CUDA-graph capture and
  the AOT fill set (`tf-dsv41 aot-needs`, `triton_fill.py`) must include the new op or replay drops it;
  draft rounds are priced by bytes read, not launches, so the two GEMMs can eat the acceptance gain.
- Gates: host tests (zero adapter bit-identical, unadapted module no-op), hardware smoke, `exact.py` /
  `validate.py` (drafted replies must equal plain - a drafter adapter may never change reply bits), then
  `acceptance_rate`/`tokens_per_round` per class in `to-benchmark.md`.
- Step 0: a dev checkout of `bertholomus/TensorFold` @ `TF_REF` + the `zig/patches` chain (the repo only
  builds it inside `zig/Dockerfile`), plus Justfile targets for build/test/smoke. The fork's family dir is
  `zig/src/families/dsv41/`.
- Reference: upstream `docs/recipes/training-a-drafter.md` (loss, taps, acceptance scoring), upstream
  `docs/quantization.md` "Merging a LoRA" (the offline route, MLX affine 4-bit only), `zig/src/core/lora_plan.zig`
  (PEFT validation to mirror for EXL3 layouts).

## Energy per request (0117 design and follow-ups)

The patch and energy tool exist. Request-level instrumentation cost about 7–17% decode in the screen. Leave it off during ordinary performance comparisons. [Combo2](runs/combo2/RESULTS.md)

- Goal: measure the pair's energy use and report it as a metric, and - if the engine allows it - put the
  request's own figure in its reply so a caller sees what its request cost.
- Already there: `nvidia-smi --query-gpu=power.draw` reads on GB10 (51-72 W recorded in `runs/overnight/RESULTS.md`),
  `scripts/bench_overnight.py`'s `Sampler` polls clocks/temp/power every 3 s on both Sparks (max only, no
  integral, no per-request attribution), the engine already dlopens NVML for P2P topology
  (`zig/src/cuda/p2p_nvml.zig`), and every reply already carries a non-standard `tensorfold` block built at
  `zig/src/server/chat_generation.zig:313` (`tokens_per_second`, `prefill_seconds`, `time_to_first_token`, draft
  counters, `gpu_seconds` under `TF_Q27_PROFILE`) - that block is the natural home for `energy_joules`, not the
  OpenAI `usage` object that strict clients parse.
- Verified on the pair 2026-11-14 (head `gx10-eb0e`, worker `gx10-235f`, driver 580.159.03): both GB10s answer
  `nvmlDeviceGetTotalEnergyConsumption` with rc 0 - a cumulative millijoule counter (head 12,619,882,529 mJ rising
  ~58 J a second, worker 13,293,409,885 mJ). So a request's energy is an exact counter delta at start and finish:
  no sampling thread, no integration bias, and no 3 s-sampler problem for a 133 ms TTFT. A read is one driver call
  outside the captured graphs; `nvmlDeviceGetPowerUsage` (mW) is there too if a cross-check is wanted.
- Decided (operator, 2026-11-14): energy covers **both Sparks' GPUs**, and phase 1 is **metrics + the per-rank log
  line only** - no reply field until real numbers exist. So the rank-1-to-rank-0 exchange is deferred, and the reply
  keeps carrying rank-0-only figures like the rest of its block.
- Phase 1 (the work): `TF_DS_ENERGY` switch (off by default); counter delta around each request; the delta appended to
  that request's `done` line (patch `0070`'s line, after its existing fields so old parsers still match) on **each
  rank** - the worker's number comes from its own log, or its own `/metrics` if rank 1 serves one (check that); a
  `joules_total` counter beside `generation_tokens_total` in `zig/src/server/metrics.zig` for rank 0. Then
  `tools/energy.py` (behind `just energy`) that joins both ranks' `done` lines by request and reports the pair's J per
  request, J per 1K prompt tokens and J per 1K decode tokens at 1 and 4 streams, next to the accept rate and
  tokens/round - drafting and energy trade off, so they belong on the same row.
- Phase 2 (only if the numbers are worth showing a caller): the pair total in the reply's `tensorfold` block, which
  needs one small exchange at request end so rank 1's delta reaches rank 0.
- Still open: gross window energy against marginal over a measured idle baseline. The head also runs a desktop and
  ~18 containers, so a gross figure attributes their draw to the request. Phase 1 can report both (counter delta at
  the window, and the delta over an equal idle window just before it) and let the numbers say which is worth keeping.
- Implementation note: `p2p_nvml.zig`'s symbol table is all-or-nothing (`orelse` fails the whole `Api`), so
  `nvmlDeviceGetTotalEnergyConsumption` (and `nvmlDeviceGetPowerUsage`, mW, for a cross-check) need their own
  optional lookup rather than a new required entry, or a driver without one breaks the P2P survey.
- Gates: replies bit-identical with the switch on and off (an NVML read must not perturb the round), a hardware
  smoke, then a measurement row in `to-benchmark.md` - J per 1K prompt tokens, J per 1K decode tokens and J per
  request at 1 and 4 streams, next to the accept rate and tokens/round, since drafting and energy trade off.
- Reference: `docs/measurements.md` (how the tools measure), `zig/patches/0069`/`0116` (the `/metrics` pattern),
  `zig/patches/0070` (the per-request log line).

## Ideas from basetenlabs/deepseek-ultrafast (plan written, not started)

- Plan, code references, tests and gates: `docs/baseten-ultrafast-plan.md`. Four items, in order: tokenization
  cache (`0118`, bit-identical), teacher-forced scoring + `tools/score.py` (`0119`, fills the "quality set"
  column), exact-position resume (`0120`/`0121`), speculative sampling for sampled streams (`0122`, LOSSY).
- Each item has a step 0 that can stop it: time render against encode before the cache; the off-grid chunk
  identity experiment before exact resume (it decides whether review finding 18's "identity-bound" holds).
- The plan's "locate first" items were not read (the session lost its shell): `Model.snapCopy`/`snapBytes`,
  `Store.marks`/`min_gap`, the hello/feature compare, `HfText.encode`, the core lane `Window` struct, the `done`
  line builder. Look them up before starting the item that names them.
- Also a no-code benchmark row: Baseten's Tetris prompt on the pair, for a like-for-like tokens-a-round figure.
- To explore (same doc, last section; not planned work): Baseten issues a whole decode step as one graph with no
  host in the loop, runs the compressor/indexer and drafter prep on parallel streams, and chains kernels with
  programmatic dependent launch. Step 0 for all three is one block with rank 0's `--profile` serve argument (never
  run): rank 1's line shows `enqueue_ms` 14.3 against `forward_ms` 54.6 a round, attribution only, and what
  `enqueue` covers is unchecked.
- Also there (D, E): a launch-by-launch table of a layer against Baseten's fused kernels. Most of this round's
  kernels are the kit's, so only the ratio-2 compressor's scatter/gather/pool chain is patchable here. Measure
  first with `tf-dsv41-lanes --profile 2` (phase times and launches a phase, eager rounds only).
- Second pass (same doc, "Second pass" section): both engines compared subsystem by subsystem. No large algorithmic
  gap; this engine is ahead on memory, pool-only scoring and cut-ins. Leads, all unmeasured: absorb off the round's
  critical path, Engram hash/submit before the round, the device accept chained into the draft pass, the serial
  expert read at start (87.8 s of a 101.5 s load), whole-text reprocessing on streamed replies, prompt gathers on
  NCCL at fp32. Instruments to add first are listed there. Also corrected in the plan: an odd chunk start is
  refused today (`prompt.zig:1167`), so exact resume is even positions only; `enqueue_ms` includes the Engram landing.

## Second-pass follow-ups (`zig/patches/0160`-`0171`, landed 2026-11-16)

- **Numbering: the next patch is `0191`** (`0187` was already covered by T15, `0189` is a host test). 0176-0181 are not free slots - they are the named set's *plan* numbers,
  which landed inside this series as `0166`-`0171` (the map is `runs/bench-prep2/env-vars.md`). TF_REF + `zig/patches/*`
  in filename order equals `second-pass-int` `432d800` (verified 2026-11-16). The next export runs `--base 432d800
  --first 0191`.
- **Built and exported (2026-11-16), off by default, pair outcomes in [combo4](runs/combo4/RESULTS.md)** - the plan's odds 0172-0175 as `0182`-`0185`,
  in plan order, each with a host test and a `to-benchmark.md` row under "Ready to benchmark: the odds":
  - `0182` `TF_DS_TOPK_FAST`: the indexer's top-k in three reads of the row (a sampled threshold, one compacting pass,
    the exact k-th among the candidates, the eight-digit select as the fallback). The algorithm is checked host-side
    against a brute force on 51 rows (`runs/bench-prep2/topk-fast/`); the CUDA later compiled for the odds screen.
  - `0183` `TF_DS_PREFILL_GATHER_OVERLAP`: the chunk's MoE gather on its own stream, joined where the pending post reads
    it - the window is the two Engram layers' host read (~70 ms a layer) against an 8.31 ms exchange.
  - `0184` `TF_DS_IDX_BUCKET_FINER`: the round's bucket rounded to a multiple of 1024 instead of the next power of two
    (70,000 positions: 70,656 keys against 131,072); a non-pow2 `nb` leaves the fused `_topk_sel` for `topk_i64` +
    `_topk_finish`, so measure it paired with `0182`. `exact.py` is its gate.
  - `0185` `TF_DS_BF16_PARTIALS` (**LOSSY**, the series' first): wo_b writes bf16, the gather is bf16 (half the payload),
    the mixes widen back to fp32. `exact.py` differs by design; the gate is `validate.py` plus the recorded delta.
  - `0188` `TF_DS_HEAD_LAST_CHUNK`: a prompt chunk that is not the pass's last skips the head (its logits were thrown
    away); `moreRowsAfter` is the test, the chunk still returns `.head`.
  - `0189` (plan 0166's evidence, no switch): the draft chain's granted+slack stop already ships - `depth.Rule.budgets`'
    node-probability branch, which the served engine takes - and this patch pins it with a host test. Baseten's in-graph
    confidence stop needs conditional nodes both ranks agree on: not built.
  - `0190` `TF_DS_SNAP_TRIM`: a kept state carries no ring for the decoder's last half (2.81 MiB a state the replay
    rebuilds); `LayerShape.ring` puts it in the layout, so planes, sizes and digest follow. Refused without
    `TF_DS_REPLAY_FLOOR=1`; no `format.version` bump (the layout digest already separates the stores).
  - `0186` `TF_DS_GREEDY_ARGMAX`: a greedy round's head exchanges each rank's (value, index) a row (8 B) instead of its
    vocabulary half (258,560 B) and reduces them to the same pick; the graph key carries the round's choice. The pair +
    reduce are checked host-side against a brute force on 58 rows (`runs/bench-prep2/argmax-map/`) - which caught the
    rank-offset bug before the export; the kernels compiled with 0186z; the armed smoke failed.
- **`0187` expert grid sized by picks (plan 0164) is CLOSED - it already exists, and its rest is kit-bound.** The decode
  half is `TF_DS_T15_PLACE_BOUND` (`zig/patches/0053`): under an applied prune cap the experts' grids and work lists stop
  at `expert_prune.placeBound` instead of every slot of the window. The prompt half cannot use that bound (the expert
  pruner runs on decode windows only, `round.zig`'s `moe`, so a prefill chunk has no applied cap to bound), and the
  distinct count the picks produce lives on the device while the kit's `rot_in`/`gateup_epilogue` grids index a place by
  `(row, slot)` - a smaller grid needs a place list, which is the kit's half. Applying the cap to prefill first would be a
  LOSSY behaviour change of its own (dropped expert slots on prompt rows), not a grid tweak. The original screen omitted the cap. The later [T15 trial](runs/review-round/RESULTS.md#applied-cap-t15) held cap 12,1 fixed on both ranks and measured the bounded grid at eight rows.
- **The unbuilt odds** (one item, no code - no switch of it exists in `native.zig` and no branch carries a commit for it):
  `0191` restore lookahead (plan 0169, `TF_DS_RESTORE_AHEAD`): split `lane_host.park` into `prepare(job)` and unchanged
  admission so a queued request's disk restore starts before a lane is free; it can evict good states if the reservation
  guard is wrong, so it wants its own design pass. Facts: `runs/bench-prep2/SERIES.md`.
- **M7's batching step** (`0192`; `0171`'s commit body designs it): several streams' warm tails as one decode-style
  round over forced rows on full end states. The two missing pieces: the forced path must suppress emission (tail
  rows are prompt rows) and the round's window validation must accept rows before `cache_len`. Then
  `rows_max`/`round_rows` sizing for k*T rows.
- **M3's probe study** (`0167`; measurement, no patch number): can a linear probe at layer 13 or 19 predict "the next
  draft is rejected" at 95 % precision with 50 % recall? If not, drop the candidate. `conf:` is the stand-in;
  `file:` refuses until the weights exist (the format is the patch's body).
- **M5's decoded cache, deferred after W4** (landed `0169`; follow-up `0196`): the kit contract (decode-to-buffer kernel + a decoded
  grouped_cp variant) is the patch's body; `decoded...` fails closed until someone recaptures it. W4's pure-read
  probe (`0142`) decides whether the candidate lives at all.
- **`0166`'s substring splice** (numbered `0193`): the recorded spelling matches whole pieces only; a template that
  merges plain text around a block misses. The substring splice (encode the piece's prefix/suffix around the record)
  is the follow-up.
- **`0162`'s pinned staging + `uploadAsync`** (numbered `0194`): read overlapped with upload (double-buffered spans),
  the plan's option (b) after the parallel readers.
- **`0163`'s per-round arena** (numbered `0195`): the reply-side allocations still accumulate in the per-request
  arena until the reply ends (the plan's one robustness item) - the scans' share shrank, the token decodes and
  `Shown` copies remain.
- **`0168`'s decision** (measurement, no patch number): run the check (`engram_rowcache_*` on `/metrics`) and drop
  the filter if the lock does not show.
- **The zstd check**: `just zstd-check` on the head (one 1 GiB slice of the rank cache; under 3 % saving closes the
  trellis-recompression idea). The script is built and smoke-tested (`runs/zstd-recompress/check.sh`); still not run
  on the pair.
- **`0164`'s arm is a machine-with-room arm**: on the pair's unified memory the 70 % formula has no room and stays at
  the 0.5 GiB floor.
- **`native.test.a follower killed mid-round: the link watch stops the collectives and the pair ends within a second`
  is flaky** (seen once while gating `0186`, clean on the re-run; its own "within a second" bound makes it
  timing-sensitive). Not caused by the odds; if it repeats, give it a looser bound or quarantine it.
- **Two host tests of the landed series fail on a clean `second-pass-int`** (`zig build test`, 2026-11-16, nothing of
  `0182`): `round.test.TF_DS_EXPERT_HOT_CACHE: hotPick takes the top counts' ranges, ties to the lower id` (expects 2,
  gets 0; `0169`) and `prompt_cache.test.0178 (TF_DS_PROMPT_ENDSTATE): eligible offers at == len only with the flag`
  (`0170`). Either a test bug or a real defect in the pair - find out before either switch is trusted. (The landed
  series' "merged tree `zig build test` green" claim in `continue.md`/`env-vars.md` was wrong.)
- **Independent reviews of the named set are owed**: `0160`/`0161` were subagent-reviewed (their findings are folded
  in); `0162`-`0171` landed numbers have host tests but no independent review pass - run `reviewer` children over
  `zig/patches/0163`,`0165`-`0171` (the dense ones: the M3 accept rule, M4's end-state frames, M7's policy) before
  the first production consideration.

## WP4 follow-ups (from `runs/fused-next-20261121/wp4/`)

- **The prompt half's `compress2_norm` can take the same PDL split as the decode compressor** (launch_dependents, read the fixed `comp_norm` weight, wait, body). Left out of `0250` because the prompt path's bodies are at the bandwidth roofline, so the floor is not the bound there; worth a prompt-chain measurement before landing. [WP4 results](runs/fused-next-20261121/wp4/RESULTS.md#item-2--split-pdl-entry-around-proven-independent-work)
- **Exercise `TF_DS_FUSED_TRACE=1` on a live round.** `0251`'s trace is host-side and covered by the served-matrix host test, but production holds the head's memory, so no live round printed it; a pair run would confirm the tuples the call sites record. [Trace](runs/fused-next-20261121/wp4/engagement-trace.txt)
