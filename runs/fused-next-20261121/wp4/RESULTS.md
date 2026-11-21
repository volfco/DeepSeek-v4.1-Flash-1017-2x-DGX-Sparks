# WP4: PDL overlap and the cross-kernel boundaries

Run 2026-11-21 on the GB10 head (`sm_121`, unified LPDDR5X), engine tree `~/Workspace/tfref-fused-next` branch
`fused-next`, CUDA 13.0, `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false` (never fast math). Every timing block held
`~/tf-bench/gpu.lock`, waited for `health.live.connections == 0` and verified the served-token counters did not move
across it (`timing_clean: true`, `timing_retries: 0` in each JSON); timings are CUDA-graph medians of 20 repetitions of
100 replays (µs per replay) with the IQR. No commit in this package changes a default: every change rides
`TF_DS_PDL`, which stays opt-in and off.

Harness (recipe repo, synced to `~/tf-bench/fused-next/tools`): `tools/fused_bench/chain.py` (new: chain, parity and
canary modes over the WP0 harness, which is imported unchanged except for the PDL launch attribute, an `on_stream`
argument and the `delay_copy` canary kernel in `build.sh`'s `floor.cu`). `bench.py` gained `pdl=` (cuLaunchKernelEx with
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, the engine's `cuda/launch.zig` path) and `native(...,
pdl=1)`; its default behaviour is unchanged.

## Item 1 — the decode chain's launch floor (research item "Measure the decode launch floor")

One decode layer's fused launches in serving order, in one graph, 20 x 100 replays, old = new = `9d4232a` (the WP3
tip; this item changes nothing). Serving order and geometry from `round.zig`/`exact.zig`: 0197 `compress2x_norm512`
(grid = rows, 512 threads), 0198 `fused_idx_key` (rows, 128), 0198 `fused_rope_store` (rows, 256), 0198
`fused_rope_qd_pair` (rows x 4, 256) at R>=32 or `fused_rope_qd` (rows x 32, 128) below, 0195 `idx_fused` (rows, 1024,
nb*4 shared) or at R<=7 with nbk>=6144 the engine's small-row dispatch (`block_max` + `idx_pick`), 0199 `prune_six`
(1 block, 32/64 threads). The empty arm is the same number of `empty_kernel` nodes at each launch's grid/block/shared,
plain and with the PDL attribute.

| rows | nb | k | chain, no PDL | chain, PDL | empty nodes | empty nodes, PDL attr | bodies above the empty floor (no PDL / PDL) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 8 | 9.612 | 8.116 | 2.688 | 2.322 | 6.92 / 5.79 |
| 7 | 8 | 8 | 10.500 | 8.915 | 2.776 | 2.421 | 7.72 / 6.49 |
| 32 | 8 | 8 | 11.764 | 10.414 | 2.756 | 2.402 | 9.01 / 8.01 |
| 48 | 8 | 8 | 12.393 | 10.879 | 2.769 | 2.408 | 9.62 / 8.47 |
| 7 | 4096 | 2048 | 15.175 | 13.605 | 2.769 | 2.402 | 12.41 / 11.20 |
| 32 | 8192 | 2048 | 21.344 | 20.567 | 2.762 | 2.402 | 18.58 / 18.17 |
| 48 | 8192 | 2048 | 24.014 | 22.492 | 2.952 | 2.709 | 21.06 / 19.78 |

Reading: a decode layer's six fused nodes cost 9.6-24.0 µs, of which 2.7-3.0 µs is graph-node/launch floor (28% at
R=1 nb=8, 11% at R=48 nb=8192). PDL (`griddepcontrol` at entry + the launch attribute, the tree's current form) already
removes 1.5-1.9 µs of that chain; the remaining floor with PDL is 2.3-2.7 µs. The bodies are 5.8-21.1 µs, and they are
what the WP0 ranking prices (0195 at nb=8192 dominates: 18.4 µs of the 18.6 µs body total at R=32 nb=8192 is the
indexer, at its issue/latency bound).

At 40 layers a round, this layer's fused nodes are 0.39-0.96 ms of a 42 ms round (0.9-2.3%), and the launch floor
inside them is 0.09-0.12 ms (0.2-0.3%). That is the ceiling for anything that only removes launches; the bodies are
where the remaining time is, and WP0 already found most of them at a bound.

## Item 2 — split PDL entry around proven independent work

**What is provably independent pre-wait, per kernel** (`round.zig`'s producers in brackets):

| kernel | first dependent byte | pre-wait candidate |
|---|---|---|
| 0197 `compress2x_norm{128,512}` | `ckv`/`cgate` rows, `rslot`, `gi`, the ring (the scatter of this round's rows) | `w` = `lay.comp_norm`: a model weight, loaded at startup, no writer in the round |
| 0198 `fused_idx_key` | `x` = `rd.ikp` (the grouped projection), `cos`/`sin` at `pos[r]` | `w` = `lay.idx_k_norm`: a model weight |
| 0198 `fused_rope_store`, `fused_rope_qd{,_pair}` | `rd.lat` / `rd.iq`, `cos`/`sin` at `pos[r]`, `pos[r]`/`rows_idx[r]` | nothing fixed: the rope table is indexed by a device position |
| 0195 `idx_fused`, `idx_pick`, `block_max` | `rd.score`, `vis`, `every` | nothing fixed |
| 0199 `prune_six` | `pick`, `wts` | nothing fixed |

So only the two norm kernels have work that can move before `griddepcontrol.wait`; every other kernel keeps the
tree's back-to-back wait-then-launch entry.

**Measured candidates** (chain, µs; the arms are old cubin/no-PDL, old/PDL, new/no-PDL, new/PDL, and the empty floor;
IQRs in the JSONs are 0.002-0.10 throughout, so every delta below is outside the spread):

| candidate | what it does | R=1 nb=8 | R=32 nb=8 | R=48 nb=8 | R=7 nb=4096 | R=32 nb=8192 | R=48 nb=8192 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `launch_dependents` before `wait` in every kernel (no prefetch) | 8.243 | 11.085 | 11.453 | 13.601 | 21.137 | 23.652 |
| B | A + hold the norm weight in a register across the body, both kernels | 7.849 | 10.122 | 10.717 | 13.376 | 20.230 | 22.323 |
| D | A + `prefetch.global.L2` of the weight (no register held) | 8.005 | 10.285 | 10.902 | 13.542 | 20.401 | 22.536 |
| E | B, but the compressor's non-PDL path keeps the old body (two instantiations) | **7.930** | **10.124** | **10.572** | **13.380** | **20.298** | **22.246** |
| — | old cubin, PDL (the tree's current form) | 8.105 | 10.472 | 10.817 | 13.563 | 20.486 | 22.464 |

A alone is **slower** than the current form at every shape above nb=8 (+0.6 µs at R=48 nb=8192): a successor scheduled
with nothing to do before its wait only takes SM slots from the running kernel. D's L2 prefetch recovers part of that
but not the load latency the body pays at its end. B is the fastest chain but its held register costs the 512-wide
compressor on the non-PDL path, which is the default configuration: `compress-decode R=48 D=512` measured 2.095 (old)
vs 2.278 (B) with IQR 0.002-0.004, a served-shape regression. E keeps B's pre-wait read on the PDL path and the old
body on the default path, so neither is paid for the other. `wp4-a.json`, `wp4-b.json`, `wp4-d.json`, `wp4-e.json`.

Landed as commit `0250` (item 2), `zig/kernels/cuda/dsv41_torch.cu` only:
- `pdl_launch` / `pdl_wait` / `pdl_enter` (enter = wait then launch, unchanged behaviour) next to the existing helpers;
- `tf_ds_compress2x_norm{128,512}_kernel`: `pdl` path = `launch_dependents`, read `w[threadIdx.x]`, `wait`, body with the
  held weight; non-`pdl` path = the old body with `w[threadIdx.x]` read where it always was;
- `tf_ds_fused_idx_key_kernel`: `launch_dependents`, read `w[threadIdx.x]`, `wait`, body with the held weight (faster on
  both paths: 1.621/1.625/1.644/1.645 -> 1.583/1.583/1.604/1.607 at R=1/7/32/48).
- `tf_ds_triton_rmsnorm_row` takes an optional preloaded weight (`PRELOAD`), default off for its other callers.

Chain effect (E vs the current form, PDL on, µs): R=1 8.105 -> 7.930, R=7 nb=8 8.951 -> 8.730, R=32 nb=8 10.472 ->
10.124, R=48 nb=8 10.817 -> 10.572, R=7 nb=4096 13.563 -> 13.380, R=32 nb=8192 20.486 -> 20.298, R=48 nb=8192 22.464 ->
22.246. Non-PDL is unchanged within the spread (9.550/10.474/11.802/12.332/15.130/21.386/23.984 against
9.627/10.513/11.812/12.292/15.107/21.387/23.985), which is the point of the two-instantiation form.

**SASS placement** (`cuobjdump -sass`, `bin-cand/new-torch.cubin`): `tf_ds_compress2x_norm512_kernel` — `@P0 PREEXIT`
(`launch_dependents`) at 0x00b0, `LDG.E.U16` of the weight at 0x00c0, `ACQBULK` (`griddepcontrol.wait`) at 0x00f0, first
dependent `LDG.E.64` (the row's `rslot`) at 0x0110. `tf_ds_fused_idx_key_kernel` — `PREEXIT` at 0x00b0, weight `LDG.E.U16`
at 0x0100, `ACQBULK` at 0x0120, the dependent row load after it. The non-PDL path branches over all three (`@!P0 BRA`
from 0x00a0 to the first body instruction), so the default path's SASS is the old one. Resources: the compressor goes
30 -> 32 registers (the union of the two instantiations), `STACK=0 LOCAL=0`, shared unchanged at 3088/1040 bytes;
512 threads x 32 registers = 16,384 registers per block, 4 blocks/SM by registers and by the 2048-thread cap either
way, so occupancy is unchanged. `fused_idx_key` stays at 29 registers.

**Static SASS and occupancy of the two touched kernels** (`cuobjdump -sass`/`-res-usage` through `tools/sass.sh`,
`wp4/sass.json`): the compressor's kernel now carries two body instantiations, so its *static* counts double —
`tf_ds_compress2x_norm512_kernel` 440 -> 760 instructions (LDG 15 -> 30, LDG.64 4 -> 8, LDG.U16 1 -> 2, LDS 5 -> 10,
STS 2 -> 4, SHFL 10 -> 20, BAR 2 -> 4, MUFU 8 -> 14, BRA 31 -> 44), `..._norm128` 392 -> 656 — while the executed path
is one of them (~6 KB contiguous, no spills, `STACK=0 LOCAL=0`). Registers 30 -> 32 (the union of the two paths);
occupancy is unchanged (512-thread kernel: 3 blocks/SM, 48 warps, 100% theoretical; 128-thread: 12 blocks/SM, 48 warps,
100%). `tf_ds_fused_idx_key_kernel` stays at 29 registers and 344 -> 336 instructions with an identical opcode
histogram (LDG 8, LDG.64 2, LDG.U16 2, STG 3, LDS 4, STS 3, SHFL 11, BAR 3, REDUX 2, MUFU 2) and 12 blocks/SM at 100%.
The measured cost of the extra static code is none: the non-PDL timing of both kernels is unchanged within the spread.

**Delayed-producer canaries** (`chain.py --mode canary`, 200 µs `delay_copy` producers into every input, `wp4-canary-e.json`):
for all six kernels at R=48 nb=8192, `pdl=0` and `pdl=1`, the output is byte-identical to the no-producer reference
(`canary_diff=0`), and the negative control (the same producers on a side stream with no dependency) differs
(`race_control_diff` 2042-317,690 bytes), so the canary does observe pre-landing reads. Every dependent read is after
the wait on both paths.

**Chain parity** (`chain.py --mode parity`): the whole chain run four ways (old/new cubin x PDL on/off) and both as a
captured graph and as direct stream launches, every live output of every node compared after each arm: 0 differing
bytes at R=7 nb=8 and R=48 nb=8192, including graph-vs-stream. **Kit parity** (`bench.py --parity-only --kernel
compress-decode,key`): 30 cases, 0 differing bytes against the kit chain (`compress2x` + `_rmsnorm`, `_rmsnorm` +
`_rope_heads` + `_fp4_store`) over R=1/7/32/48 D=128/512 mixed, unaligned, cached, cross, invalid and stress layouts,
and the key at R=1/7/32/48 plus NaN/+-inf/+-0 (`wp4-i2-parity-e.json`). Host suite: see the package summary.

Reproduce:

```bash
# build old (9d4232a) and candidate cubins on the head
rsync -a --include='dsv41_ops.cu' --include='dsv41_torch.cu' --exclude='*' zig/kernels/cuda/ head:tf-bench/fused-next/src-cand/
ssh head 'cd ~/tf-bench/fused-next && bash tools/build.sh src-old src-cand bin-cand'
# chain floor + candidates
ssh head 'cd ~/tf-bench/fused-next && python3 tools/chain.py --old bin --new bin-cand \
  --kit /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --mode chain --json wp4-e.json'
# canaries, chain parity, kit parity
ssh head 'cd ~/tf-bench/fused-next && python3 tools/chain.py --old bin --new bin-cand --kit ... --mode canary \
  --shape "R=48 nb=8192 k=2048" --json wp4-canary-e.json'
ssh head 'cd ~/tf-bench/fused-next && python3 tools/chain.py --old bin --new bin-cand --kit ... --mode parity \
  --shape "R=48 nb=8192 k=2048"'
ssh head 'cd ~/tf-bench/fused-next && python3 tools/bench.py bin bin-cand <kit> --kernel compress-decode,key --parity-only'
```

## Item 3 — engagement: which fused arm runs at each row count, width and PDL setting

Commit `0251`, `zig/src/families/dsv41/fused_trace.zig` plus the launch sites it is called from. `TF_DS_FUSED_TRACE=1`
turns on a once-per-distinct-tuple record `dsv41 fused trace: <arm> rows=<n> width=<d> pdl=<0|1>`; the `note()` calls sit
where the kernel is chosen: `exact.zig`'s seven fused launchers (`compress2xNorm`, `compress2Norm`, `idxKeyFused`,
`ropeStore`, `ropeQd` with its pair/single split, `promptIdxKey`, `promptLatent`), `ops.zig`'s `idxFused`, `idxPick` and
`expertPruneDead`, and the unfused arms at `round.zig`'s and `prompt.zig`'s fallback branches. It is host-side only: the
launchers hand over the shape they already computed, nothing is read back from the device, no bytes or fingerprint bits
move, and with the variable unset the cost is one cached bool per launch decision.

The engine itself cannot be run on the head in this session (production holds ~115 of 121 GiB, ~6 GiB free), so the
matrix is exercised by the engine's own host test harness: `zig build test` now runs
`fused_trace.test.the served matrix selects the recorded arms, once per tuple`, which walks the served matrix
(rows 1/7/32/48/128/1024 x bucket widths 61 and 65,533 x D 128/512 x PDL on/off x the prune slot width) through the same
gate functions the call sites use and asserts the arm each shape takes, that the tuple is recorded once, and that the
switches and bounds fall back (`fused_trace.test.the arms switch off with the switches`). Its printed table is
`engagement-trace.txt` (98 lines: 96 matrix tuples + 2 dedup cases). What it shows:

- decode rows 1 and 7 with nbk 8192 take the small-row arm (`idx_small`: block_max + idx_pick); rows >= 32 take the
  one-launch `idx_fused`; nbk 8 (the small pool) always takes `idx_fused`.
- the compressor takes `compress2x_norm` at D=128 and 512 for every decode row count; D=256/1024 fall back.
- the key is `key_fused` at D=128; the latent `latent_fused` at D=512; the query is `query_fused` below 32 rows and
  `query_pair` from 32 (the 0241 dispatch).
- `prune_six` at every row count, `pdl` recorded as a separate tuple, and the prompt arms (`compress2_norm`,
  `prompt_key_fused`, `prompt_latent_fused`, `query_pair`) at rows 128/1024.

`TF_DS_FUSED_TRACE` is opt-in and off, so no default-on behaviour changed; `zig build test` is 625/627 with the same two
pre-existing failures (`prompt_cache.test.0178`, `round.test.TF_DS_EXPERT_HOT_CACHE`) before and after this commit.

## Item 4 — a durable parity target for the WP1-WP3 kernels and their fallbacks

`tools/fused_bench` now covers every kernel WP1-WP3 changed plus the unfused arms their switches fall back to, and
`just fused-bench` runs it as one command that prints the per-case PASS/FAIL tally and exits non-zero on any difference:

```
PARITY=1 OUT=runs/fused-next-20261121/wp4 JSON=wp4-parity.json \
  OLD=<tree with the WP3 tip sources> NEW=~/Workspace/tfref-fused-next just fused-bench
```

What changed in the harness (recipe repo; no engine change, so no engine commit for this item):

- `bench.py --kernel fallback` (part of `all`): the unfused arms the gates refuse to. `fallback idx` runs
  block_max + topk_indices + pool_pick at nbk 16,385 (past `idx_fused.max_nbk`), which is what the engine runs past
  131,072 columns, and compares the new tree's bytes with the old tree's; `fallback compress-decode` runs `compress2x`
  at D=256 and D=1024 (widths `decodeOk` does not take) on identical ring bytes in both trees and compares the pair sum
  and the scattered ring rows. The unfused arms of the 0198 and 0199 switches are the kit's own launches, which are
  already the reference arms of the `key`/`latent`/`query`/`prompt-*` and `prune` cases.
- the Justfile recipe takes `PARITY=1` (`--parity-only`: no timing, no idle-production wait, so it can run while
  production serves as long as 3 GiB is free), `OUT=` for the artifact directory and `JSON=` for the result file.
- `tools/fused_bench/README.md` gains the coverage table the todo item asks for: kernel and switch, the reference the
  harness compares against, the rounding points the fused arm must keep, and the guarded shapes (so a later edit that
  widens a gate without a new case is visible).

The three full parity runs with OLD = the WP3 tip `9d4232a` and NEW = this package's tip are recorded below (item 5 of
the hand-back: three sequential runs of `--parity-only` over every kernel).

## The per-layer budget after this package

One decode layer's six fused nodes in one graph, 20 x 100 replays, PDL on (the switch's own configuration), against
the same number of empty nodes and against the six kernels' standalone medians (WP0's `BASELINE.md`, one node a graph):

| rows | nb | chain (PDL) | empty-node floor | six standalone medians, summed | 0195 alone (WP0) | what is left |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 7.930 | 2.321 | 10.14 | 1.85 | 5.6 µs of bodies across five small kernels, each ~1.0-1.5 above its own floor |
| 7 | 8 | 8.730 | 2.422 | 11.00 | 2.03 | as above |
| 32 | 8 | 10.124 | 2.400 | 12.49 | 2.04 | as above (the query is the largest at 2.98 standalone) |
| 48 | 8 | 10.572 | 2.418 | 13.84 | 2.05 | as above |
| 7 | 4096 | 13.380 | 2.408 | 19.09 | 10.12 | the indexer, at its issue/latency bound (WP1's items) |
| 32 | 8192 | 20.298 | 2.402 | 30.96 | 20.51 | the indexer is the whole layer: the chain costs its standalone time |
| 48 | 8192 | 22.246 | 2.502 | ~32 | ~20.5 | as above |

- **launch/dependency floor: 2.3-2.5 µs a layer** (29% of the chain at R=1 nb=8, 11% at R=48 nb=8192). PDL already
  removes 1.5-1.9 µs of the chain; item 2's split removes a further 0.17-0.35 µs. What remains is graph-node cost no
  kernel body can reduce: 0.09-0.10 ms of a 42 ms round at 40 layers (0.2%).
- **kernel bodies: 5.6-19.7 µs a layer**, and at nb >= 4096 the layer is the indexer (the chain measures the indexer's
  own standalone time, 20.5 µs at R=32 nb=8192, with everything else hidden behind it under PDL). WP0 put 0195 at
  96-610 GB/s against a 954 GB/s cache roofline with one CTA a row, eight barriers and scalar/64-bit loads; WP1's
  block-size, vectorization, mask-only and small-row items are the remaining work there.
- **what the PDL entry buys and costs**: at R=32 nb=8192 the chain is 10.5 µs below the sum of the six standalone
  medians (31.0), i.e. the current entry already hides about a third of the standalone time. At the small nb=8 shapes
  the hidden part is 2.2-3.3 µs of the 10.1-13.8 µs sum, which is the launch floor, not body overlap.
- **nothing else is left to fuse at these boundaries**: WP3 closed the compressor+latent fold (one launch floor plus
  ~0.05 µs of L2 traffic at decode rows); every remaining cross-kernel boundary carries a dependent read, so fusing it
  would only remove a launch floor (0.2-0.3% of a round).

## Harness fix found while running the durable target

The shared `bench.py` query case launched its "old" arms with the pre-0241 flat grid (`rows * heads` blocks) while
`tf_ds_fused_rope_qd_kernel` reads `blockIdx.y` as the head (`r = blockIdx.x, h = blockIdx.y`), so any tree from 0241
on reads past its rows: at R=7 with heads=32 the grid is 224 blocks where the kernel expects (7, 32), and the launch
faults (`cuStreamSynchronize: 700 illegal memory access`). It went unnoticed because WP0's old arm predates 0241 and
WP3's own runs used its variant harness (`wp3/vbench.py`). Fixed in `bench.py` to launch the tree's own dispatch on
both arms (pair geometry at rows >= 32, one warp a head below it); the query cases then pass at R=1/7/32/48/128/1024
against the kit chain, old = `9d4232a`. Without this the durable target could not run against the WP3 tip at all.

## Full parity runs, OLD = the WP3 tip `9d4232a`, NEW = this package's tip

Three sequential runs of `PARITY=1 OUT=runs/fused-next-20261121/wp4 JSON=wp4-parity-N.json just fused-bench` (the
durable command; every kernel of WP1-WP3 plus the fallbacks, 150 parity cases each):

| run | cases PASS | cases FAIL | tally line |
|---|---:|---:|---|
| 1 (`wp4-parity-1.json`) | 150 | 0 | `PASS` (exit 0) |
| 2 (`wp4-parity-2.json`) | 150 | 0 | `PASS` (exit 0) |
| 3 (`wp4-parity-3.json`) | 150 | 0 | `PASS` (exit 0) |

## Hand-back summary

| WP4 item | verdict | commit |
|---|---|---|
| 1. measure the decode chain's launch floor | done (research item; no code change): a layer's six fused nodes cost 7.9-24.0 µs with a 2.3-2.5 µs PDL-on launch floor (11-29% of the chain); PDL already removes 1.5-1.9 µs | none |
| 2. split PDL entry around proven independent work | landed: `launch_dependents` then the fixed norm weight then `wait` in the decode compressor (PDL path only) and the indexer key; chain 0.17-0.35 µs faster per layer at every decode shape, non-PDL path unchanged, canaries exact | `f013282` (0250) |
| 3. prove engagement | landed: `TF_DS_FUSED_TRACE` opt-in arm trace at the launch sites, with the served matrix as a host test (`engagement-trace.txt`) | `59e71a8` (0251) |
| 4. durable parity target | landed in the recipe repo (no engine change): `bench.py --kernel fallback`, `PARITY=1/OUT=/JSON=` on `just fused-bench`, the coverage table in `tools/fused_bench/README.md`, and the query-geometry fix that lets the harness run against the WP3 tip; three full runs 150/150 PASS, 0 fail | none (harness) |

Left open:

- **Prompt-side PDL split.** The prompt half's `compress2_norm` reads `w` (fixed) too, but the prompt path is not
  launch-bound (its bodies are at the bandwidth roofline, WP0), so nothing was changed there. Worth revisiting only if
  a prompt-side chain measurement shows a floor worth attacking.
- **No hardware counters.** Every decision here was made from SASS placement, resource usage and chain timings. If a
  counter is wanted for item 2, the request is in `NCU_REQUESTS.md`.
- **The engine's own trace was not exercised on a live round** (production holds the head's memory); the arm matrix is
  covered by the host test instead. A future run with `TF_DS_FUSED_TRACE=1` on the pair would print the same tuples
  from the call sites.
- **Item 1's floor is graph-node cost**: nothing in the kernel bodies can reduce it, and the remaining per-layer
  bodies are 0195's (WP1's open items).
