# report-0261 — TF_DS_TOPK_MERGE

**Commit:** `33091fb` (parent `e06735c`). **Switch:** `TF_DS_TOPK_MERGE` (off by default; `1`/`0`/`""`/unset; anything
else refuses startup with `TF_DS_TOPK_MERGE: expected 1 or 0`). It only takes effect together with
`TF_DS_T27_DEVICE_DRAW=1`, because eligibility requires `path == .device_draw`.

## Files touched

- `zig/src/families/dsv41/topk_merge.zig` (new, pure): switch, `max_k` (56), `addRow` (kmax), the `works`
  eligibility predicate, the frame-header helpers, host references (`drawKey`, `localTop`, `merge2`, `greedyPick`,
  `sampleCandidates`) and tests.
- `zig/kernels/cuda/dsv41_ops.cu`: `tf_ds_topk_select_kernel`, `tf_ds_topk_merge_kernel`, `tf_ds_draw_rows_topk_kernel`
  inserted after `tf_ds_draw_rows_kernel`, which is unchanged.
- `zig/src/families/dsv41/ops.zig`: `topk_merge` bool + three `?cuda.Function` (looked up only when on);
  `topkSelect`, `topkMerge`, `drawTopk`.
- `zig/src/families/dsv41/round.zig`: `Round.topk` (the round's kmax, 0 = off) and `Round.tpairs`; the head arm
  after the TF_DS_GREEDY_ARGMAX arm; `Graphs.Key.topk`.
- `zig/src/families/dsv41/model.zig`: `topk_merged` + `rd.tpairs` allocated beside `draw_tab` under the switch;
  `launchPick` arm (merge, then draw).
- `zig/src/families/dsv41/lanes.zig`: `verifyFn` decides kmax and `rd.topk`, puts kmax in the verify header, resets
  it; the follower's `.verify` decodes and checks it; `pairSwitches` bit 3.
- `zig/src/families/dsv41/native.zig`: the `PairSwitchesDiffer` hint names the new switch.
- `zig/src/families/dsv41/sampling.zig`: `testRow` made `pub` (the brief asks to reuse it). Nothing else changes.
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll`.

## Launch counts and bytes (one verify round, R rows, TP2)

| | Off (today, device draw) | On |
| --- | --- | --- |
| Head, both ranks | `gatherF32` (R x 258,560 B a rank) + 2x `copyRows` | `topkSelect` + `allGatherInPlace` (R x kmax x 8 B a rank) |
| Pick, rank 0 | `drawRows` | `topkMerge` + `drawTopk` |
| Kernels + collectives | 2 kernels (rank 1) / 3 (rank 0) + 1 collective | 1 / 3 + 1 collective |

At R=16, kmax=56: 4,136,960 B -> 7,168 B a rank. Merge and draw stay separate kernels, as the brief lists them. Fusing
them into one would save one small launch on rank 0 and is a possible follow-up.

## Design decisions and deviations

- **Selection:** a copy of `tf_ds_draw_rows_kernel`'s radix select (8 digits over the 64-bit draw key) and its rank
  sort, with the global id `col0 + c` (`col0 = rank * hn`; the `copyRows` layout puts rank k's columns at
  `k*hh`, and `hh == hn` under TP2 without the 2D split). One 1024-thread block a row. It is the same shape as the
  qualified draw kernel, which already selects over 64,640+ columns.
- **Pairs carry raw value bits**, not key-normalised ones, so -0 and NaN reach `double(v)/t` unchanged. Rank 0's
  `row[0]` NaN flag rides bit 31 of its first id, mirroring `tf_ds_argmax_pair_kernel`. The merge moves it to the merged
  first id and clears bit 31 everywhere else. Greedy rows honour it: pick 0, exactly `tf_ds_block_argmax`'s rule.
  Sampled rows do not, because today's sampled path has no `row[0]` rule either.
- **Merge is two-list only** (`shortcut:` comment). A generic world-way merge would need a dynamically indexed
  per-rank cursor array, which lands in local memory (the LOCAL=0 rule). `works` requires world == 2.
- **kmax** = the largest `top_k` over rows with `temp > 0`. Greedy rows' `top_k` is unbounded, so it is ignored.
  Eligibility also requires `world * hn == vocab` and `hn >= kmax`, so no padding or limit logic is needed in the
  kernels.
- **Carry to the follower:** the follower never calls `setDraw`/`roundArgmax` and gets the rules only via rank 0,
  so kmax rides the verify frame header:
  - layout: bits 24..29 hold kmax, bit 30 must be clear, bit 31 is the existing map flag, and the window count is
    `& 0x00ffffff`;
  - off path: with kmax 0 the header bytes are unchanged;
  - the follower returns `BadFrame` if kmax != 0 and its switch is off or kmax > 56 (the `argmax_map` precedent).
- **Graph key:** `Key.topk = rd.topk` only on the stretch that runs the head (`end == layers.len`), because kmax is
  a baked kernel argument and the all-gather's size there. Other stretches' launches don't depend on it, so they keep
  their graphs. Without this, each distinct kmax would recapture every stretch.
- **Cross-rank consistency (deviation from the pointer):** the warning the brief names
  (`kv_lanes.zig:370`, "rank 1's TF_DS_KV_SHARD_CARRY/_TWINS ... differ") lives in the shard-layout reply, which
  only runs under `TF_DS_KV_SHARD`, and extending it needs a rank_wire schema change. The check that runs in the
  served pair config is `pairSwitches` / `sendSwitches` (lanes.zig: exact resume / spec sample / bf16 partials),
  which compares the ranks' bits whenever world >= 2 and refuses with `PairSwitchesDiffer`. TF_DS_TOPK_MERGE is
  bit 3 there. An all-off pair still sends no switches frame. On top of that, the per-frame refusal above always holds.
- **Probe:** `works(..., probe=false)` from lanes, because `Model.verify` passes a null probe. The head arm also
  requires `probe == null`; probe tools call `round.forward` on Rounds whose `topk` stays 0.
- `rd.logits` is not written in merged mode. Its only readers on the served path are `argmaxRows`/`drawRows` (not
  taken) and `roundLogits` (host_draw only, which is never merged).

## Gate

1. `zig build test` on the final commit: **exit 0, 27/27**. The final dsv41 test binary run directly 5 times: 489/489
   each time. Intermittent failures were seen only on the pre-amend binary:
   - First gate run: `rank_wire.test.rank 0 and a follower agree, restore, and fail closed over a loopback link`
     failed, and `parts.test.every mutant is killed by its own check` crashed (segfault in a loopback thread at
     `rank_wire.zig:1974`).
   - One of three direct runs: `native.test.a follower killed mid-round: the link watch stops the collectives and
     the pair ends within a second` failed (`native.zig:1204`, the link-close reason).

   All three are loopback-link/timing tests in code this patch does not touch, and none reproduced on rerun. The
   base binary passed 4/4 direct runs here, so they are **unattributed intermittents**, not a proven base flake.
   The amend added only the `Round.init` field initialisation (see "Amend" below). `zig test -lc topk_merge.zig`:
   17/17, including sampling's own tests.
2. CUDA syntax check (round-local stubs, see report-0260): ok.
3. CPU harness `h0261/` (`build.sh`, `extract.py`, `topk_run.cpp`, `run.log`). It extracts the three new kernels
   **and the untouched `tf_ds_draw_rows_kernel` + `tf_ds_block_argmax(_v)`**, and runs them with one std::thread per
   CUDA thread, real `std::barrier` `__syncthreads` and warp-barrier shuffles. 110 rounds, 325 picks, **0
   failures**. Each round checks:
   - (a) each rank's list against an independent per-shard key sort, plus the flag;
   - (b) the merged list against a whole-row key sort;
   - (c) every pick against the full-row draw kernel run on the concatenated rows, and on NaN-free rows also
     against a C++ transcription of `sampling.sampleRow`.

   The rounds cover:
   - k = 1..56 fuzz rounds with mixed greedy/sampled rows (top_k < kmax prefix rows);
   - cross-rank ties at the cutoff, and all-equal rows;
   - +-0, +-inf, NaN (incl. rank 0's `row[0]`, rank 1's first column, all-NaN rows), FLT_MAX and subnormals;
   - two rounds at hn = 64,640 (vocab 129,280), k = 56 and 20.

   Mutants: dropping `col0` from the key (crash on a wrong-id read), dropping the merged flag (58 failures),
   ignoring the flag on greedy rows (18), not masking bit 31 in the draw (22).

Host tests (`topk_merge.zig`):
- ≥96 candidate cases (56 fuzz k-values, 8 tie cases, 40 special-value cases): merged equals the full key sort on every
  row, and equals `sampling.topCandidates` on NaN-free rows (prefixes 1, kmax/2, kmax).
- The greedy pick against the device argmax rule.
- 288 seeded draws over mixed rounds: `sampleRow(full row)` equals `sampleCandidates`/`greedyPick` on the merged list.

## Amend (found in final review)

`Round.init` builds `rd` from `undefined` and assigns each field (`round.zig`, `var rd: Round = undefined;`), so the
`topk: u8 = 0` / `tpairs: u64 = 0` defaults never applied. Off, a probe tool's or a layer check's `round.forward`
(`layer_main.zig:674/946/1093`) could have read garbage in the head arm. Fixed by assigning `rd.topk = 0; rd.tpairs =
0;` beside `rd.map = false`. `Model.init` sets `rd.tpairs` after `Round.init` under the switch.

## Bit-identity claims

- **By construction:**
  - The candidate set and order (unique keys, exact radix select, same key function).
  - The values (raw fp32 bits copied from `rd.hl`, the bytes `copyRows` would have placed in `rd.logits`).
  - The greedy pick.
  - The sampled draw's arithmetic: the same source lines as `tf_ds_draw_rows_kernel`'s tail, fed the same (value,
    id) sequence in the same order, so the same device exp/log run on the same inputs.
- **By CPU emulation only:** the kernels' indexing, barriers and merge logic.

## What only the hardware gate can settle

- End-to-end picks, on vs off, for sampled TP2 rounds with mixed greedy rows, under CUDA graphs on both ranks. The
  token streams should be identical: the only device-math operations are shared verbatim.
- The all-gather in place on `rd.tpairs` inside the captured head stretch, and a varying kmax across rounds: graph key
  hits, a recapture per new kmax on the head stretch only.
- **Transport change, not only size:** today's `gatherF32` (`prompt.zig:151`) rides the RDMA ring when the payload
  fits `r.s.max_bytes`, else NCCL. The merged exchange uses `e.comm.allGatherInPlace` (NCCL), as the
  TF_DS_GREEDY_ARGMAX precedent does. The gate should time the head stage on vs off. A tiny NCCL all-gather may cost
  more than the bytes saved if the full gather was already on the ring. Moving the pairs onto the ring would be a
  follow-up (pairs are u32x2, so `count % 4 == 0` holds whenever R*kmax is even).
- The follower's frame decode with real rank-0 headers. Pair startup with mismatched switches should refuse with
  `PairSwitchesDiffer`.
- ptxas `LOCAL/STACK` for the new draw kernel. Its tail copies the qualified kernel's `double probs[64]`/`r[8]`, so it
  should report whatever `tf_ds_draw_rows_kernel` reports.
