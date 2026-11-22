# BRIEF-0261 — Rank-local sampled top-k selection and merge

**Switch:** `TF_DS_TOPK_MERGE` (off by default). **Module:** `zig/src/families/dsv41/topk_merge.zig` (pure).
**Commit subject:** `0261 dsv41: TF_DS_TOPK_MERGE (off by default): sampled verification exchanges rank-local top-k pairs`

## What it replaces

A TP2 verify round's sampled rows today gather the **full** vocabulary halves: the head stage in
`zig/src/families/dsv41/round.zig` (locate by content: "the head on every row: the logits, the ranks' columns in
rank order") runs `e.gatherF32(rd.hl, rd.hg, R * hn)` and then `copyRows` × world into `rd.logits` `[R,
world*hh]` (fp32, the ranks' halves in rank order), and `Model.roundArgmax` (`model.zig`) launches
`tf_ds_draw_rows_kernel` (`dsv41_ops.cu`) over the full rows with the `TfDsDrawRow` table that `lanes.zig`'s
`verifyFn` uploads (`setDraw`). That collective is 258,560 bytes per row per rank (64,640 fp32).

New mode: each rank selects its best `(f32 value, u32 global id)` pairs from **its own** head output
(`rd.hl`), the two ranks exchange those lists (448 bytes per row per rank at k=56), the lists merge in the
existing tie order, and the existing seeded draw runs on the merged candidates.

## Contracts (all bit-level)

1. **Candidate order** = `tf_ds_draw_key(v, id)` descending (`dsv41_ops.cu`): value order-preserving bits with
   `-0` as `+0` and NaN lowest, ties to the **lower id**. For rows without NaN this is exactly
   `sampling.zig`'s `before` order (value desc, id asc).
2. **Global id** of local column c on rank k = `k * hh + c` (the `copyRows` layout; TP2: `hh == hn`). Verify
   against the actual layout before coding.
3. **Only the global top `top_k` per row matters** — the draw kernel's own comment says the margin is never
   read: "The first top_k candidates by (value desc, id asc) are the first top_k of the host's top_k + 8". So
   each rank emits its local top `kmax` pairs, where `kmax = max(top_k)` over the round's rows (1..56); the
   merged top-`kmax` in order contains every row's true top-`top_k` as a prefix. A row with `top_k < kmax`
   uses its prefix.
4. **Sampled-row draw** = the tail of `tf_ds_draw_rows_kernel` verbatim (from `const double t = fmax(rule.temp,
   1e-6);` on: `scaled = double(value)/t`, the keyed splitmix64 uniform `tf_ds_mix64` chain, `score = scaled -
   log(-log(u))`, the top_p pairwise-sum prefix rule, the min_p rule, the argmax of score). **Do not modify
   `tf_ds_draw_rows_kernel`** (it is hardware-qualified; the new candidate-list kernel carries its own copy of
   the tail with a one-line comment saying why it is duplicated).
5. **Greedy rows** (`temp <= 0`) inside a mixed round: take the merged top-1 — with the **row[0] NaN rule**
   handled exactly as the existing greedy rank reduction does: see `tf_ds_argmax_pair_kernel` /
   `tf_ds_argmax_reduce_kernel` in `dsv41_ops.cu` ("rank 0's row[0] NaN rule as pick 0"; the flag rides the
   high bit of the local index). This precedent already solved the tie rules (earlier rank wins a tie; each
   rank's own tie went to its lower column) — mirror it.
6. **Eligibility** (host side, one predicate like `sampling.argmaxMapWorks`): switch on, `path == .device_draw`
   (`RoundRules`), every sampled row drawable with `1 <= top_k <= 56` (already implied by
   `deviceDrawable`), `world == 2`, `e.two == null` (no 4-node split), and no probe reading full logits
   (`Probe.check(probe, .logits, ...)` — force the old path under a probe). Anything else takes today's path
   **unchanged** (greedy-only rounds, `host_draw`, score streams, `TF_DS_SPEC_SAMPLE`, world 1).

## Implementation shape (follow the codebase's own precedents)

- **Selection kernel** (one row a block is fine at these sizes): top-`kmax` of the rank's `rd.hl` row by
  `tf_ds_draw_key`, emitting `(value bits, global id)` pairs. Reuse the `tf_ds_draw_key` device function and,
  if useful, the radix-select shape of `tf_ds_draw_rows_kernel` (it selects over 64,640 columns already) or
  `tf_ds_argmax_pair_kernel`'s simpler reduction — your call, report it.
- **Exchange**: the pairs all-gather. The greedy path already exchanges rank-major pairs `[world][rows][2]`
  (`tf_ds_argmax_pair_kernel`'s output feeds a collective — find how it is wired and reuse that mechanism;
  `e.comm.allGather` accepts typed payloads). Same launches on both ranks, graph-capturable.
- **Merge kernel**: per row, merge the world sorted lists into the global top-`kmax` in key order (world=2 is a
  two-list merge; keep it generic if it costs nothing).
- **Candidate draw kernel** (`tf_ds_draw_rows_topk_kernel`): per row, greedy rule or the duplicated draw tail
  over the merged list.
- **Model/lanes wiring**: buffers (pairs `[world][rows][kmax]` + merged `[rows][kmax]`) in the model arena
  beside `draw_tab`, a `roundArgmax` variant that draws from merged candidates, `lanes.zig` path selection.
- **"Carry the exchange mode to both ranks and the graph key"**: add the switch to the cross-rank consistency
  check (find the code behind the warning "rank 1's TF_DS_KV_SHARD_CARRY/_TWINS (1/0) differ from rank 0's") and
  to the graph capture key (find the capture-key code near `graph_cache.zig` / `Model`'s graph arms) so a
  captured graph never mixes modes.
- Keep `rd.logits` full materialization **only** on the fallback path; in merged mode skip the `gatherF32` +
  `copyRows` (that is the point). Full logits stay available for scoring and host speculative sampling — those
  are `host_draw`/probe paths and unaffected.

## Host reference + tests (`topk_merge.zig`)

The historical evidence: "A host check matched the ordered global candidates in 96 cases, including cross-rank
ties, signed zeros, infinities and NaNs." Recreate it as tests:

- ≥96 cases: per-rank `sampling.topCandidates` on the shards vs `topCandidates` on the concatenated row
  (ordered global candidates must be identical), covering cross-rank ties **at the cutoff**, signed zeros,
  ±inf, NaN placement (incl. rank 0's row[0] NaN), k = 1..56, and rows with `top_k < kmax` (prefix rule).
- Seeded draw equality: `sampling.sampleRow(full row)` vs a `sampleCandidates` reference (the served
  arithmetic on the merged list) for mixed greedy/sampled rows — the picks must match.
- Reuse `sampling.zig`'s `testRow` generator for fuzz rows.

## CPU-emulation harness (`runs/kernel-round-20261122/h0261/`)

Extract and execute the select, merge and candidate-draw kernels' sources on CPU against independent
references over the same matrix (small synthetic `hn` for speed; one case at hn=64,640). Include the greedy
NaN-first case and the draw tail's top_p/min_p arithmetic (compare against a direct transcription of
`sampling.zig`'s `sampleRow` in C++).

## Out of scope

The host-draw path, score streams, `TF_DS_SPEC_SAMPLE`, the 2D/four-node head, greedy-only rounds
(`argmaxMapWorks`' arm), and any change to `tf_ds_draw_rows_kernel` or `sampling.zig`'s served semantics.
