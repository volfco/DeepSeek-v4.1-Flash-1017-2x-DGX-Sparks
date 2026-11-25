# Kernel pattern catalogue (sweep 2026-11-23)

Built from the record, not from memory. A pattern is listed only with a **measured** number, and the number is quoted
with its shape and its source. Every timing is standalone GB10 (`sm_121`, CUDA graphs, `-O3 --fmad=false --ftz=false`)
unless the line says "serving". Nothing in this sweep was timed: this workstation has no GPU.

Paths are relative to the recipe repository (`runs/…`, `docs/…`, `continue.md`, `zig/patches/…`). Engine line numbers
are `~/Workspace/tfref-kstack4` at `2bb052f`.

## How to read the gains

The record's own verdict frames every pattern below (`continue.md`, "Audit review and re-score of rounds 0259–0279"):
the fused rounds worked inside about 8% of a decode round, a 2 µs saving a layer over 40 layers is about 0.08 ms of a
42–87 ms round, and the only serving-visible kernel gain so far is 0259 (about 1% of decode wall). A standalone win
is a per-launch fact. It becomes a serving fact only when multiplied by launches a round.

## Patterns with a measured gain

| # | Pattern | Where it was done | Measured | Source |
|---|---|---|---|---|
| P1 | **Stop a radix select at a complete bin.** After the bins above the cutoff are consumed, if the remaining rank equals the cutoff bin's population, every key at or above the partial prefix (low digits zero) wins: exactly k keys, no further digit pass. | 0254 `tf_ds_topk_indices<EarlyCut>` (`dsv41_ops.cu:147–251`), reused by 0195's `tf_ds_pool_cut` (`:974–1015`) | 237.0 → 100.7 µs (128 × 32,768, k=512, random); 77.96 → 30.86 (128 × 4,096); all −∞ 227.2 → 138.8; mixed specials 7 × 65,537 142.8 → 120.0; deep fallback chain 61.26 → 37.62 (7 × 16,385 blocks) | `runs/suite-fused-next-20261121/kernel-review/RESULTS.md` |
| P1b | **Skip key digits that are constant** (the inverted column's high bytes below 65,536 columns), and **find the cutoff bin with one warp** instead of one thread walking 256 bins. Part of the same 0254 numbers; not separated. | 0254 | (inside P1's figures) | same |
| P2 | **Fuse a chain of tiny per-row kernels into one launch at small row counts.** The floor of an in-graph node is 0.457 µs; a chain of 3–6 launches at 1–7 rows is mostly floor. | 0197 compressor + RMSNorm; 0199 six-slot prune; 0195 indexer selection; 0260 confidence pack | 3.48 → 1.75 µs (1 × 512), 3.99 → 2.00 (7 × 512), 6.74 → 2.01 (32 × 512); 9.782 → 2.547 µs (7 rows, counters on); 44.77 → 9.80 µs (4,096 blocks); 0260: 4 launches × 0.457 µs floor, not timed alone | `runs/fused-kernel-audit-20261120/RESULTS.md`; `runs/prune-six-20261120/RESULTS.md`; `continue.md` register |
| P3 | **Reuse a fetched or computed result across layers that share its inputs.** | 0259 `KV_FETCH_REUSE` (30 of 38 exchanges a round) | serving: decode rounds 0.5–2.5% shorter (mean −1.7%), on beats both controls in 11 of 12 pairings; wall −0.3 to −1.6% in all six cells (re-score: "about 1%") | `runs/kvreuse-ab-20261122/RESULTS.md`; `continue.md` re-score |
| P5 | **Aligned, vectorized loads and a static trip count.** Two 16-byte loads a block of eight fp32 in place of eight scalar loads; the block size a template constant. | 0255 `tf_ds_cand_fast8_kernel` (`dsv41_ops.cu:889–908`); 0213 `tf_ds_block_max_t<8>` (`:1046–1075`) | 0255: 1.257 → 1.051 µs (7 rows, 128 blocks); 1.689 → 1.318 (7 × 2,048 blocks); 3.369 → 2.849 (48 × 2,048); 15–23% at all 12 aligned shapes; registers 31 → 22. 0213: −6% to −20% at the work-bound decode/prompt shapes | `runs/suite-fused-next-20261121/kernel-review/RESULTS.md`; `runs/fused-next-20261121/wp1/RESULTS.md` item 2 |
| P6 | **Exact bit-level replacements for float work.** Scale byte from the bf16 exponent bits, multiply by an exact power-of-two reciprocal instead of `__fdiv_rn`, whole-warp `__reduce_max_sync` over sign-cleared integer bits. | 0240 (`dsv41_torch.cu`) | 0198-key 1.772 → 1.623 µs (R=1), −8.0 to −8.4% at R=1/7/32/48; prompt key 4.270 → 1.909 (R=128, −55%); all 65,536 bf16 encodings 0 differing bytes | `runs/fused-next-20261121/wp3/RESULTS.md` item 2 |
| P7 | **Skip provably no-op work.** | 0242 unrotated warps skip the partner shuffle (latent chain only); 0214 the mask-only arm skips ballot, prefix and compaction; 0216 a chunk whose keys all fit skips projection, scoring and selection; 0211 prune prefix early-out on finite nonnegative shares | 0242: 1.540 → 1.401 µs (R=1), −9% at every decode row, prompt latent 4.209 → 1.664 (R=128). 0214: mask arm −10% to −13% vs 0213. 0216: six launches a layer a short chunk removed (count, not timed). 0211: **parity only** (4,111 vectors identical to the complete rule), no timing on record; the already-ordered short path of 0199, which predates it, measured 1.274 vs 1.811 µs (7 rows, stats off) | `runs/fused-next-20261121/wp3/RESULTS.md` item 1; `wp1/RESULTS.md` items 4, 6; `runs/fused-followup-20261121/RESULTS.md` (0211 parity); `runs/prune-six-20261120/RESULTS.md` (0199) |
| P8 | **Spread wide work over the grid; never one block a row over a wide input.** | 0215 small-row form: `block_max` over the grid, then one selection CTA a row | 12.14 → 10.53 µs (1 row), 12.44 → 11.80 (7 rows) at 8,192 blocks; **loses** 8.46 → 8.67 at 7 rows / 4,096 blocks; 0217: 2.5–3.4× faster than the unfused chain at 100K–260K tokens | `runs/fused-next-20261121/wp1/RESULTS.md` items 5, 7; `runs/pre-bench-20261121/crossover-0215/RESULTS.md` |

## Patterns in the brief's list that the record does **not** support with a measured gain

| # | Pattern | What the record says | Disposition in this sweep |
|---|---|---|---|
| P4 | Write in the consumer's layout; drop the gather or copy launches | 0278 `TAPS_PACKED`: about 44 µs of work removed a prompt, measured TTFT −63.5 ms at 32K, "about 1,400 times the work removed … not the stated mechanism" (`continue.md`, "Unexplained"). 0260 `CONF_PACK`: adopted and neutral; best case 1.8 µs a pass. 0269 `DRAFT_TAPS_DIRECT`: neutral. | Work removed is real (launches, bytes); **no measured gain attributable to the pattern**. Surveyed and listed, never cited as a gain. |
| P7 (grid part) | Stop a launch grid past the last row | 0264 `PLACE_BOUND_ALL` re-scored **neutral**, not −0.41%; adopted on a round-count-confounded number (`continue.md`, "Corrections to earlier records") | Work removed, no measured gain. |
| P6 (one case) | Warp integer maximum | Holds only for a **whole-warp** reduction. The half-warp masked form is an anti-pattern (below). | Apply only where the group is a whole warp. |

## Anti-patterns the record paid for

| # | Anti-pattern | Measured cost | Source |
|---|---|---|---|
| A1 | **One block a row serializing wide work** | 0193/0194 fused route: 18–19% slower single-stream, 11% at 4 streams (one block a row recomputes the 384 × 5,120 gate). 0279 wide pick: 64 serial tiles a CTA at 65,536 maxima, neutral at 300K/600K. 0267: see A3. | `runs/fused-bench-20261119/RESULTS.md`; `docs/fused-kernel-rebenchmark-audit.md` |
| A2 | **Lane stores 64 bytes apart** (eight int64 keys a block written from one lane) | 0206 key fold 316.6 vs 38.5 µs (R=128, width 32,768) | `runs/fused-followup-20261121/RESULTS.md` |
| A3 | **A scalar CUDA-core dot where a tensor-core dot existed** | 0267 `MARKOV_FUSED`: −5.9 to −10.2% greedy tok/s; pass 6.33 vs 4.20 ms | `runs/overnight-20261122/k267/RESULTS.md`; `continue.md` re-score |
| A4 | **A half-warp masked `__reduce_max_sync`** | latent 1.541 → 1.644 µs, +0.10 µs at every served row; ptxas duplicated the masked REDUX into three paths | `runs/fused-next-20261121/wp3/RESULTS.md` item 2 |
| A5 | **`isfinite` (or any extra predicate) in a hot per-element loop** | 0207: decode compressor 13–17% slower (1.81 vs 1.61 µs at R=1, 2.13 vs 1.87 at R=7) | `runs/fused-followup-20261121/RESULTS.md` |
| A6 | **Unconditional full passes** | 0261's selector: eight radix digits over 64,640 logits a rank whatever the row, nine logical reads of each row, about 2.22 MiB a row a rank (a count of load requests; never timed) | `docs/fused-kernel-rebenchmark-audit.md`, "0261" |
| A7 | **Fusion where the buffers are distinct / the shape is large** | 0198 at 128 rows: fused 5.80 vs split 5.02 µs; split wins one invocation by 2.56–3.30 µs; fused wins with eight distinct buffers in one graph and at 48 and 1,024 rows | `runs/suite-fused-next2-20261121/query128/RESULTS.md` |
| A8 | **Per-call host uploads or synchronizes a caller could hoist** | 0269: two host-to-device uploads on every nonempty absorb; ceiling 0.01–0.05%, measured neutral | `docs/fused-kernel-rebenchmark-audit.md`, "0269" |
| A9 | **A guard that costs more than the work it skips** on a latency-bound chain | 0242's guard on the key path 1.623 → 1.686 µs, on the one-element query neutral (1.336 vs 1.338); reverted there | `runs/fused-next-20261121/wp3/RESULTS.md` item 1 |
| A10 | **Warp-aggregated histograms** (`__match_any_sync`) for the radix count | 2.9–4.4 µs slower at 1,024 threads; `sm_121` already coalesces intra-warp uniform shared `atomicAdd` | `runs/fused-next-20261121/wp1/RESULTS.md` item 3 |

## Bit-exactness rules that held (each one was a real mismatch first)

- A partial radix prefix is an inclusive threshold **only as a set**. Wherever the exact k-th key is consumed as a
  value, an order or an estimate, the early exit changes bytes (this sweep's survey sorts the P1 sites by that test).
- 0254's constant-column-byte skip holds only while every column's inverted id shares its high bytes: below 65,536
  local columns. A rank-1 global id range starting at 64,640 straddles 65,536
  (`docs/fused-kernel-rebenchmark-audit.md`, "0261" item 3).
- The decode RMSNorm had to spell out the kit's approximate PTX (`MUFU.SQRT` then `MUFU.RCP`), nine bf16 bytes
  differed otherwise; torch's prompt norm keeps `rsqrt`. Triton contracts `fma(xe, cs, -(xo*sn))`. No change here may
  touch a rounding point.
- Kernels keep `LOCAL=0 STACK=0`; a new per-thread array is a hardware question, not a source one.
- `-0` is `+0` in every score key; a NaN's place differs by selector (`tf_ds_score_key` orders its bits,
  `tf_ds_draw_key` puts it lowest, `sample.cu`'s `key` orders its bits). A port keeps the site's own key.

## What "transfers" means in the survey

A site takes a pattern when (a) the same work is present, (b) the result is bit-identical by construction and a CPU
emulation of old and new source agrees over the edge matrix, and (c) no anti-pattern is introduced. Call frequency
decides the order, not whether it is applied.
