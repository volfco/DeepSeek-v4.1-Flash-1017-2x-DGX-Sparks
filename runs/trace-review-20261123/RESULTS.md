# Trace review — results (2026-11-23)

Read-only analysis of the saved traces on the workstation. Nothing ran on the pair, no engine change, no commit.
Scripts are beside this file, their outputs in `out/`. Engine source read at `~/Workspace/tfref-kstack4` (`2bb052f`);
`file:line` below is under `zig/src/families/dsv41/` unless a path is given.

**Each block is marked Measured / (M), from a trace by a named script, or Inferred / (I).** Shares are of the 42.8 ms
one-stream cycle. The mech arms are greedy, short-context (under 1K tokens), image `10c9866`, trace 1 in 4, round
profile off on both ranks; their own steady cycle is 39.86 ms (M, `closure2.py`), so absolute times are theirs and the
shares are approximate.

## Summary

1. **The two slow cells that have a trace are slow at one place.** In `tr16a` the 32K × 1 and 128K × 1 cells run
   0.94 and 0.62 ms a round slower than the same cells in `tr16b`, with no code change. All of the first and about
   80% of the second sits at the first transfer of the forward's second graph stretch, the Engram layer-1 gather,
   which follows the host's Engram landing. Rank 1 is the late side; rank 0's GPU clock absorbs it as ring wait. (M)
   That the landing itself is what runs late, and that the other slow cells on record are the same thing, is
   inferred: none of them has a trace.
2. **The drafter pass now has a breakdown.** A one-stream, 6-step pass is about 3.9 ms of GPU time: the vocabulary
   head 1.08 ms, the three MoE stages 1.29 ms, the three attention stages 0.68 ms, Markov steps 0.49 ms, its twelve
   ring transfers 0.23–0.26 ms. Another 0.39 ms sits between the forward's last transfer and the pass (absorb, pick,
   uploads) and 0.10 ms between the pass and the forward. (M)
3. **A Markov bias miss costs 133–153 µs**, one 33 MB head read at 217–250 GB/s, whatever the stream count. The
   interval is cleanly bimodal and the slow mode is on the same ring sequence in all four arms and on both ranks
   (906 of 906 steps), so it is the token, not noise. 3.2 misses a pass, 0.44–0.49 ms, 1.0–1.15% of a cycle. (M)
4. **The ranks' per-gather skew is random, and it costs about four times the B2 figure.** The signed median B2 used
   is 2–5 µs; the mean absolute difference is 17–23 µs a hidden gather, uncorrelated between two runs of the same
   tokens (r = 0.02–0.07). The pair pays half of it at every gather: the 71 plain attention and MoE gathers of a
   forward take 0.58 ms more than the standalone protocol floor, against 0.35 ms with `PREFETCH=0`. (M)
5. Four smaller items, each 0.2–0.9%: the decode KV fetch sends every window row's 512 picks separately, the
   top-k exchange of a sampled round goes over NCCL and that round's tail is longer than a greedy round's that ships
   whole logits over the ring, the keyed draft noise costs 24 µs a Markov step, and the hand-offs hold 0.14–0.21 ms
   nobody has attributed.

**The forward does not start cold after the pass**, as far as these traces can show. By layer 1 every attention and
Engram interval equals later layers of its kind, and the first ones do not depend on whether the preceding Markov
step missed (M). Layer 0's MoE is the slowest of the forty; its slope with rows is high and its intercept is not
(4.4 distinct expert places a row against 1.5 at layers 18 and 20–22), so that is routing, not cache state (I).

## Ranked findings

Ranked by the largest figure in each row; § is the section below with the evidence. The ceilings do not add: rows 2,
4 and 5 are all time spent waiting at ring transfers and were separated by transfer class, and row 1 is ranked on a
cell that has the stall, not on the suite average.

| # | Finding | Ceiling (share of 42.8 ms) | Basis |
|---|---|---|---|
| 1 | Stalls at the Engram layer-1 cut, the slow cells (§1) | 0.5–0.95 ms a round in an afflicted cell (1.2–2.2%); 0.06–0.10 ms averaged over a mixed suite (0.15–0.24%) | position and rank M (two cells of one arm against its repeat, both ranks); cause I |
| 2 | Random rank skew at the gathers (§2) | 0.58 ms a forward plus 0.07–0.09 ms a pass (1.5%); the part tied to the prefetch stream 0.23 ms (0.5%) | M, four arms; how much is recoverable is I |
| 3 | Markov bias misses (§3) | 0.44–0.49 ms a pass (1.0–1.15%); `0281` at the offline hit rates 0.20–0.33 ms (0.46–0.78%) | cost M, eight rank-arms + four more files; gain I |
| 4 | Decode KV fetch (§4) | 0.39 ms a forward on the critical path (0.9%); deduplicating the window's rows up to 0.22–0.32 ms (0.5–0.75%) at short context | cost M; overlap at depth unknown |
| 5 | Hand-offs (§6) | 0.45–0.52 ms (1.05–1.2%), of which 0.14–0.21 ms (0.3–0.5%) is not absorb or pick | M |
| 6 | L2 prefetch inside the pass, `0282` (§5) | at most 0.18 ms (0.43%) | I, by analogy with the forward |
| 7 | Top-k / argmax pairs over the ring instead of NCCL (§8) | sampled rounds 0.1–0.2 ms (0.25–0.45%); greedy rounds 0.09 ms (0.2%) | difference M but confounded; size of the NCCL part I |
| 8 | Keyed draft noise (§8) | 0.146 ms a sampled pass (0.34%); about 0.5 ms a four-stream sampled pass | one stream M; four streams I |
| — | Drafter vocabulary head (§5) | 1.08 ms (2.5%) | M; no exact lever found |
| — | Drafter MoE stages (§5) | 1.29 ms (3.0%) | M; memory-bus bound like the forward's |

## How the records were placed

A record carries only its key, and the drafter's attention and MoE gathers of a stage share one key. Position comes
from the ring sequence instead (`recon.py`).

- **Forward template (M).** 91 transfers in a fixed order: per layer `[engram at 1, 14] [kv at 2, 8, 14, 20, 24, 28,
  32, 36] attn_gather moe_gather`, then the head gather on a full-logit round (`round.zig:2087–2197`). A TOPK_MERGE
  round has 90 (the pairs travel over NCCL, `round.zig:2191`). Every same-round pair of records agrees with the
  template in all twelve trace pairs: 0 anchor conflicts, 0 forward records off the template (`out/recon_all.txt`).
- **Pass (M).** Six hidden gathers (stage j attention, then stage j MoE: `draft.zig:663, 670`) and one gather a
  Markov step (`draft.zig:194`), between two forwards. Steps a pass = gap between forward anchors − 91 − 6. In the
  mech arms 644 of 756 passes have 6 steps, 101 have 5. In `tr16a/b` the forward before a pass can have 90 or 91
  transfers, so the step index of 3,050 of 5,240 pass records is ambiguous by one; pass positions below come from
  the greedy arms only.
- **`round` counts work units**, a pass and a forward each (`model.zig:1743, 1955`), not decode rounds.
- **Closure (M, `closure2.py`).** `prev` is the previous transfer's copy-done, so the tiles `[prev, copy]` cover the
  stream with no gap. In steady decode, N × the sampled tiles equals the exact elapsed time to 0.01–0.17% in all
  eight rank-arms.
- **The trace's own drain** runs every four work units and always lands before a forward (`rdma.zig:853–866`). It
  adds 125 µs to that forward's first interval (M, `drains.py`: 464 against 338 µs). Position-0 numbers below
  exclude drained boundaries.
- **The file is not in sequence order** where the 4,096-record ring wraps; `tracelib.py` sorts.

## 1. Stalls at the Engram layer-1 cut: the slow cells

**Measured** (`slowcell.py`, `slowcell_spans.py`, `engram_stalls.py`, `stall_classes.py`).

- The k272 arms `tr16a` and `tr16b` ran the same deterministic workload cells. Saved GPU-clock forward a round: 32K × 1
  33,523 against 32,556 µs; 128K × 1 33,056 against 32,462; 4K × 1 33,503 against 33,522 (control). The cells were
  found in the traces by forward count (97 / 114 / 126); the cell wall from the trace differs by +937 µs a round
  (32K) and +624 (128K), and by 0 in the control. Of the audit's list of slow cells only `tr16a` 128K × 1 has a
  trace; `tr16a` 32K × 1 is not on that list but is slow by the same metric.
- Class by class, the 32K difference is in one position:

  | 32K × 1, tr16a against tr16b | rank 1 | rank 0 |
  |---|---|---|
  | engram, producer interval (mean, n 13) | 673.9 against 202.0 µs, × 2 a round = **+944** | 225.4 against 226.6 = −2 |
  | engram, ring time | −49 | 487.5 against 15.7 µs, × 2 = **+944** |
  | every other class, summed | +17 | −69 |
  | cell difference | +937 | +937 |

  Rank 1 arrives late at the Engram gather; rank 0 arrives on time and waits in its collect kernel. In the 128K cell
  the same position gives +526 (rank 1 producer) and +505 (rank 0 ring) of the +624, about 80%; the remainder is
  inside the noise of the MoE class there (±0.4 ms a round, as the control cell shows).
- Counting every round of the cell, not only the sampled Engram records (exact elapsed time between consecutive
  sampled records, spans that contain the layer-1 gather against spans that do not): in the slow 32K cell 16–17 of
  20 spans with the gather run long, median excess 1.72 ms against 0.40 ms for spans without; in `tr16b` 3 of 20
  and 0.31 against 0.38 ms. So the stall is in most rounds of the cell, about 1.3 ms each, for the cell's whole
  4–5 s. 114 rounds × 0.8 × 1.3 ms ≈ 119 ms against the cell's 107 ms. In the slow 128K cell 14 of 23 spans run long.
- Over a whole arm, stalls above 0.5 ms in the forward are almost all at this one position: 31 sampled events at the
  layer-1 gather against 3 in the other 88 forward transfers (`tr16a`), and 16 against 3 (`tr16b`). Pair cost
  63–102 µs a forward averaged over the suite. Layer 14, the other Engram layer, has 1–4.
- Three patterns: rank 1 alone late (clusters in particular cells), rank 0 alone late, both late by a similar 1–3 ms.

**Source.** A forward is issued as three graph stretches, cut before each Engram layer (`round.zig:2044–2079`). The
round's Engram reads are hashed and submitted after stretch 0 is issued (`round.zig:2075`, `2271–2339`), then
`engramLand` blocks on them before stretch 1 (`round.zig:2068`, `2355`: `eh.aio.?.wait`). Stretch 0 is layer 0 alone,
about 1.1 ms of GPU time, and the host's hashing of the round's n-grams, the submit and the reads all have to fit
inside it. The reads are Linux AIO on `O_DIRECT` descriptors, and the module's own header gives "~0.4–0.8 ms for a
6-row window" and mentions a touch "only to wake the drive" (`engram_aio.zig:1–7`). Layer 14's reads have ten more
milliseconds, which is why it almost never stalls.

**A saved host figure that fits (M, not from the traces).** Rank 1's exit print in the November 22 phase runs
(`runs/overnight-20261122/phases/*.rank1.txt`, eager, profile on): `engram_hash_ms` 0.459 and `engram_read_ms` 0.381
a round at one stream (14 rounds), 0.712 and 0.698 at four streams (147 rounds). So the hash alone takes about
0.46 ms of stretch 0's 1.1 ms before the reads go out, and hash plus read is already 0.84 ms on an ordinary round.

**Inferred.**
- Rank-local late arrivals at this cut are the Engram landing (hash, submit, read, decode, upload) overrunning
  stretch 0. The trace shows which rank is late and where, not which host step took the time; another host delay
  at the same cut would look the same.
- Both-late events are a candidate for graph captures on a new key (the trace cannot tell a capture from a replay,
  report-0272 "Known gaps").
- This explains why the slow cells show on rank 0's GPU clock (`tail`) and not as host wait (`wait`): the stalled
  host is rank 1's, and rank 0 spends the time inside its stretch waiting on the ring. Rank 0's own stalls would show
  as `wait`.
- Why a whole cell is slow and its repeat is not is open. Candidates: the drive's power state between reads 38 ms
  apart, other I/O on rank 1's drive (spill writes after a deep prefill), or rank 1's host thread running slower
  (the same cpuset holds 81 engine threads). None is tested.

**Next step: experiment, no build.**
- Arms, each the 32K × 1 and 128K × 1 cells repeated five times with `TF_DS_RING_TRACE=16`, and the round profile
  either off on both ranks or on for both: (a) control; (b) `TF_DS_ENGRAM_SUBMIT_EARLY=1` (0161, exists, off in
  production: the hash moves to the pass's return and the reads go out before stretch 0, which roughly doubles the
  read's slack; combo3's `s03` found it neutral on average, which says nothing about stall cells); (c) NVMe
  power-state latency tolerance set to 0 on both nodes for the run (a runtime host setting; record and restore it).
- Score with `slowcell_spans.py`'s count (spans with the layer-1 gather that run long) and `stall_classes.py`.
- Also collect both ranks' Engram host timers (`engram_hash_ms`, `engram_read_ms`; rank 1's need its profile on),
  the read histogram (`TF_DS_ENGRAM_METRICS`) and `iostat` on both nodes during the cells.
- If the landing is the cause and (b) does not remove it, the patch brief is: submit the pending row's reads at the
  pick, before the drafter pass (4 ms earlier; that row's token is known then), and the draft rows' reads at the
  pass's return as 0161 does. Switch off by default. Parity: the same rows are read into the same buffers before the
  same `engramLand`; only the submission time moves.

## 2. Random rank skew at the gathers

**Measured** (`ring_cost.py`, `skew_jitter.py`; pairs both ranks by sequence).

| arm | mean \|p0 − p1\| a gather, attn / MoE | signed mean | 71 plain attention + MoE gathers: ring time | above their 1.08 ms floor | half the absolute difference, summed |
|---|---|---|---|---|---|
| fx1L (production) | 17.4 / 23.2 µs | −3.2 / −4.6 µs | 1.66 ms | 0.58 ms | 0.77 ms |
| fx1Lb (repeat) | 17.9 / 23.5 | −4.4 / −4.0 | 1.66 | 0.58 | 0.79 |
| ns1L (`SIDE=0`) | 13.1 / 16.6 | +0.1 / −1.7 | 1.59 | 0.50 | 0.56 |
| np1L (`PREFETCH=0`) | 9.5 / 13.2 | −0.2 / +0.3 | 1.44 | 0.35 | 0.43 |

- `p` is a rank's producer interval; ring time is `copy − stage` averaged over the two ranks; the floor is the
  standalone bench's 9.4 µs + 1.16 µs a row (`runs/sched-20261122/RESULTS-A.md`) at each transfer's bytes.
- Only the plain-layer attention gathers and the MoE gathers are counted. The attention gather after a KV exchange
  waits for the exchange (§4) and the forward's first gather for the hand-off (§6); with those and the
  kv, engram and head transfers a whole cycle's ring time is 2.64–2.75 ms against a 1.95 ms floor.
- The pass's twelve gathers add 0.07–0.09 ms above their floor in every arm.
- The difference is not content: on the same sequence in `fx1L` and `fx1Lb` it correlates at +0.02 (attention) and
  +0.04 (MoE), while each rank's own producer time correlates at 0.88–0.97 between the arms and the two ranks within
  an arm at 0.86–0.96.
- Half the mean absolute difference, 8.7–11.6 µs a gather, is somewhat above the measured excess over the floor
  (7.6–8.5 µs); the excess is the figure used.

**Correction to the record.** B2's "≈0.14–0.16 ms a round" is the cost of the *systematic* part. The pair waits for
the later rank at every gather whichever rank it is, so the cost follows the absolute difference: 0.58 ms a forward
with production settings. k272 RESULTS calls the 19.6 µs balanced wait "the transfer floor"; the standalone bench
puts the whole gather at 15–17 µs.

**Inferred.** The side streams cause part of it: with prefetch off the excess is 0.35 ms, with the HC side stream
off 0.50 ms. Both are large net wins (cycle 39.86 ms against 42.02 without prefetch and 40.27 without the side
stream), so the question is whether their gain can be kept with less jitter. `PREFETCH=0` still leaves 0.35 ms, which
nothing here explains.

**Next step.** Round 0283 (`TF_DS_L2_DELAY_NS`, `_RATE_GBPS`, `_BUDGET_MIB`) is already built to sweep the prefetch.
Run each arm with `TF_DS_RING_TRACE=4` and `TF_DS_ROUND_PROFILE=0` and score it with `ring_cost.py` as well as round
time: an arm that lowers a rank's producer time but raises the absolute difference can lose. SCHEDULING.md's row 1
(fork the prefetch just after the stage publication, so it runs under the ring wait on both ranks alike) is the
patch to try if the sweep shows the jitter follows the pacing.

## 3. A Markov bias miss

**Measured** (`markov.py` on the four mech arms, `markov_other.py` on `tr16a/b` and the rowsplit arms).

- Steps 1–5, one stream: fast mode 9.7–9.8 µs (p10–p90 8.6–10.8), slow mode 142–162 µs, nothing between
  (n 906 a rank-arm). Gap of the medians by rank-arm: 132.5, 141.7, 145.7, 152.7, 146.2, 139.6, 146.1, 140.8 µs.
- The slow mode is on the same sequence in every arm and on both ranks: 906 / 906 agree in six comparisons, 905 in
  the seventh.
- Slow share 51.1% for steps 1–5, 66–67% for step 0 (the stream's real last token; step 0's interval also holds the
  vocabulary head, 1,089 against 1,222–1,244 µs). The served counter reads 51.7–53.3%. `tr16a/b`, a mixed suite
  with sampled traffic: 51.3–52.5%, gap 137–149 µs.
- Four streams (greedy rowsplit arms, n 42–46): a step is slow 96–100% of the time and costs 153–176 µs, about the
  same as one miss. One head read serves every missing row of the step.
- The miss cost moves between restarts and ranks by ±7% (133–153 µs: 217–250 GB/s over 33.1 MB).

**Source.** A step is `_bias`, `_score`, `_finish`, the gather, `_pick` (`tri_markov.zig:124–166`); `_bias` recomputes
a row against `markov_head` for a token with no cached slot (`tri_markov.zig:92–97`), 256 cached rows
(`tri_markov.zig:23`).

**Pricing `0281` (inferred from the offline hit rates in continue.md, natural text).** A 6-step one-stream pass has
0.67 + 5 × 0.51 = 3.2 misses, 0.44–0.49 ms. Raising the hit rate from 49% to 72% (1,024 rows) saves about 0.20 ms a
pass, to 81% (2,048) 0.27 ms, to 88% (4,096) 0.33 ms. At four streams the gain is smaller than the hit rate suggests:
a step is fast only when all four rows hit (0.88⁴ = 60%).

**Next step.** The idle-window microbenchmark is no longer needed to decide whether `0281` deserves an arm; it does.
Run it greedy and sampled at one and four streams, and compare rounds and accepted / drafted counts per seeded
request against control (a wrong cached row cannot move output bits).

## 4. The decode KV fetch

**Measured** (`ring_cost.py`, `layer_profile.py`).

- Eight `kv` transfers a forward, at layers 2, 8, 14, 20, 24, 28, 32, 36 (what `KV_FETCH_REUSE` left of 38).
- Every one carries rows × 147,456 B, in the mech arms and in `tr16` at 4K–128K alike: 512 picks × 288 B for each
  window row (`shard_ring.zig:217–224`, `round.zig:1481–1535`).
- With one stream the holder (rank 0 here) sends everything and rank 1 sends nothing. Rank 1's ring time is
  33–37 µs + 5.8–6.7 µs a window row (three production-setting arms); rank 0 then waits a median 47.6–48.7 µs at the attention gather that follows.
  That wait, × 8, is 0.39 ms a forward the pair spends on the exchange.
- KV-layer attention at 6 rows: 290–341 µs against 222–227 µs at plain layers.

**Inferred.** The window's rows are consecutive positions of one stream, so their pick sets overlap. Below 512
compressed rows of context they are identical and the payload is rows × the same bytes. The union kernels exist for
the prompt path only (`shard_ops.zig:129–175`); the decode fetch has no deduplication. Sending the union would save
up to (rows − 1) × 6.7 µs a KV layer: 0.22 ms a forward at the mean 5.1 rows, 0.32 ms at 7.

**The traces cannot say** how much the pick sets overlap at depth.

**Next step.** First a counter, not a patch: distinct picks across a window's rows per fetch, by depth, under the
existing profile (one line a run). If the union is under half of rows × 512 at 32K and 128K, the patch brief is a
decode union fetch behind a switch: the owner packs the window's distinct picks, the other rank renumbers its picks
to them, the attention reads through the index it already takes (`round.zig:1535–1539`). Parity argument: each
(row, pick) reads the same bytes in the same pick order, as the prompt path's union already proves in its tests
(`shard.zig:1142`). The count must stay on the device (the ring's row kernels already send a device-listed count).

## 5. The drafter pass, segment by segment

**Measured** (`pass_breakdown.py`; 6-step passes, 6 block rows, n 159–162 a position a rank-arm; producer interval,
mean µs, `fx1L` rank 0 / `fx1Lb` rank 0; the other six rank-arms agree within 3%).

| Segment (what the interval holds, `draft.zig:642–714`) | producer | ring |
|---|---|---|
| A0: forward tail, host hand-off, embed, mix, stage-0 attention | 620 / 613 | 33 / 32 |
| M0: mix + stage-0 MoE | 472 / 476 | 29 / 28 |
| A1: mix + stage-1 attention | 232 / 221 | 27 / 23 |
| M1: mix + stage-1 MoE | 444 / 443 | 25 / 22 |
| A2: mix + stage-2 attention | 225 / 225 | 27 / 28 |
| M2: mix + stage-2 MoE | 377 / 382 | 21 / 18 |
| K0: post, collapse, norm, **vocabulary head on all block rows**, Markov step 0 | 1,190 / 1,208 | 36 / 19 |
| K1–K5: pick, bias, score, finish (each) | 66–86 / 72–94 | 11–14 / 11–13 |
| Sum of the twelve tiles | 4,213 / 4,217 | (of which ring 261 / 227) |

- **By component (I, from the table):** head 1.08 ms (K0's hit mode less a 10 µs step); MoE 1.29 ms; attention
  0.68 ms (taking stage 0 as the mean of stages 1 and 2); Markov steps 0.49 ms including step 0's; ring 0.23–0.26 ms;
  hand-off into the pass 0.39 ms.
- **The MoE stages vary 2× between passes at fixed rows** (p10–p90 296–587 µs), as the forward's do with routing.
- **Against `pass_ms`.** The tile sum starts at the forward's last transfer, so it includes absorb and pick, which
  `pass_ms` does not, and excludes the pass's tail, which it does. Net, a one-stream greedy pass is about 3.9 ms,
  which matches the rowcost arms' 3.75–3.90 ms. The suites' 4.15–4.26 ms is a mix: sampled passes carry the noise
  tap (+0.15 ms) and 1.43 streams a pass (a four-stream 5-step pass is about 7.2 ms: `pass_x4.py`, n 19–76 a
  position).
- **Five-step passes** (5 block rows, n 19–30 a position): 3.81–3.88 ms of tiles; the MoE stages drop to 411 / 363 / 310 µs.

**The head.** 1.08 ms is the largest single item of the pass and equals the forward's own head interval
(1,100 µs, `layer_profile.py`). It is a full read of the rank's half of the head a second time a cycle
(`draft.zig:679–682`). No exact lever was found: every Markov step needs its row's logits. A smaller head for the
drafter alone would keep output bits and change acceptance; that is an operator decision, and nothing here prices it.

**`0282` (inferred).** In the forward the prefetch is worth 41 µs an attention layer and 20 µs a MoE layer at 6 rows
(`layer_fit.py`, `fx1L` against `np1L`). The pass has none (`draft.zig:10`). Three stages by analogy: at most
0.18 ms.

**Next step.** Round 0280's pass timer is already built; this table is what to check it against (a pass total near
3.9 ms at one stream greedy, the head near 1.08 ms, a miss step near 0.14 ms). For `0282`, score the arm with
`pass_breakdown.py` on a 1-in-4 trace as well as `pass_ms`: the attention and MoE stage intervals are where a gain
would show, and the head and Markov intervals should not move.

## 6. Hand-offs and the cold-start question

**Measured** (`handoff.py`, medians, three arms × two ranks).

- Into the pass: A0 588–617 µs against A1/A2 218–226 µs. The boundary holds 369–396 µs. `absorb_ms` 0.26 and
  `logits_ms` 0.05 (profile lines) are inside it; about 0.06–0.09 ms is uploads, the graph launch and the embed.
- Into the forward: position 0 318–338 µs against plain-layer attention 216–220 µs, so 98–121 µs. With
  `PREFETCH=0`, 75–89 µs: layer 0's attention is the one attention with no prefetch before it, worth about 30 µs.
  The rest is the pass's tail (pick, confidence head, synchronize, two downloads), the host and the graph launch.
- Rank 0 reaches the forward's first gather after rank 1: rank 1 waits a median 30–43 µs there, and rank 0's first
  stage kernel takes 8.5–15 µs against 3.4 µs everywhere else (`ring_cost.py`).

**Next step.** None proposed as a patch. The absorb and the pick are already known and switchable (0160). The
unattributed 0.06–0.09 ms before the pass and about 0.10 ms before the forward need a host-side stamp to split
(uploads, graph launch, the pass's tail, the frame to rank 1); 0280's pass timer covers the pass side. `0282` mode 2
(a hand-off prefetch of the forward's first reads) is worth at most the 30 µs layer 0 lacks.

**Cold start (M, `handoff.py`, `cold_moe0.py`).** With one record in four, the visible Markov step of the preceding
pass depends on the phase, so each early interval is conditioned on a different step. After a miss against after a
hit: forward position 0 −8 to +2 µs, Engram layer 1 −1 to +2 µs, attention layer 1 −3 to −1 µs (n 41–92 a side, six rank-arms). Attention
at layer 3 onward equals later plain layers (214–229 against 212–228 µs). MoE at layer 0 is inconclusive: a miss
token goes with fewer verified rows, and within a row count the difference is +16 to +101 µs at 3–5 rows and −8 to
−70 µs at 6–7 (n 6–37 a side), the same pattern in every arm because they replay the same tokens. Weighted,
+11 to +26 µs.

**k267's +0.25 ms.** No trace was taken with `0267` on, so these traces cannot test it. They do rule out a general
"the forward is slow after a heavy pass" effect beyond layer 0's missing prefetch.

## 7. Per-layer producer profile

**Measured** (`layer_fit.py`: least squares on rows 2–7, ranks pooled; full table `out/layer_fit.txt`, row-standardized
per-position table `out/layer_profile_mech.txt` and `.csv`).

- **Attention at 6 rows.** Plain layers 221–228 µs, flat from layer 3 to 39. KV layers add 66–118 µs (layer 20 the
  most, 341 µs). Engram layers 558 (layer 1) and 571 µs (layer 14, also a KV layer), of which the Engram gather's
  own interval and ring are 290 and 217 µs. Layer 0 354 µs including the hand-off. Slope 3–4 µs a row, 8–9 at KV
  layers.
- **MoE at 6 rows.** 401–840 µs, slope 40–121 µs a row. At runs/w4's 27.2 µs a routed place that is 1.5 distinct
  places a row at layers 18, 20, 21, 22, about 2.2–2.6 through the middle, 2.9–3.4 at layers 12–13 and 31–39, 3.3–3.6 at
  layers 1–2 and 4.4 at layer 0. The profile repeats to 2% between `fx1L` and `fx1Lb`.
- **Totals at 6 rows.** Attention 10.3 ms, MoE 24.3 ms. One more verified row costs 2.96 ms of MoE and 0.19 ms of
  attention, 3.15 ms (7.4% of a cycle).
- **Prefetch by layer** (`np1L` − `fx1L`, at 6 rows). Attention: 35–42 µs at plain layers, 43–67 µs at KV layers
  (37 at layer 14), about 21 µs at layer 0 and layer 1. MoE: 11–24 µs. No layer where it hurts.
- **Head.** Producer 1,100 µs, ring 77–100 µs for 1.3–1.8 MB of logits.

**Inferred.** Layer 0's MoE excess is routing: its intercept is no higher than other layers', only its slope. The
"43 layers, 40 windows" left unexplained in runs/w4 is the 40 target layers plus the drafter's three stages.

## 8. The other transfers

- **Sampled against greedy round tails (M, `head_exchange.py`, `tr16a/b`).** From the last MoE gather to the pass's
  first gather: 1,753–1,818 µs after a ring-head (greedy) forward, 1,857–1,897 µs after a TOPK_MERGE forward, so
  +79 to +122 µs, though the greedy path moves 1.3–1.8 MB over the ring (85–100 µs) and the other moves about 3 KB
  over NCCL (`round.zig:2191`). The comparison is confounded: the sampled side also runs the top-k select, the device
  draw and a noise-table upload, and the two kinds of round come from different requests.
- **Inferred.** A ring transfer of a few bytes costs 10–17 µs (the Markov gathers, 16 B). The same holds for
  `GREEDY_ARGMAX`, which the record found exact and not faster: it replaces a 100 µs ring exchange with an NCCL one
  (`round.zig:2180`). Both payloads fit a ring slot (4,456,448 B, `model.zig:100`).
- **Next step.** Run the staged `runs/k280/topk-run.sh` (0261 on against off on sampled rounds) first. If on is not
  clearly faster, the patch brief is one switch that sends `rd.tpairs` / `rd.amap` through `gatherF32` when the ring
  is up. Parity: `gatherF32` delivers the same bytes as the all-gather (`prompt.zig:150–156`). One trap: sent in place,
  the buffer is both the send source and the destination of the rank's own slice, which the stage kernel declares
  `__restrict__` (`rdma_gather.cu:57–60`), so the brief must either skip the own-slice copy for this call or stage
  from a separate send buffer.
- **Keyed draft noise (M, `markov_other.py`, `slowcell_spans.py`).** The fast Markov mode is 8.9 µs after a greedy
  forward and 33.1–33.6 µs in passes around TOPK_MERGE forwards: 24.4 µs a step for `keyedNoise` over 64,640 columns
  (`draft.zig:153–159`). Six steps, 0.146 ms a sampled pass. At four streams sampled steps read 245–258 µs against
  153–176 greedy (I: about 80 µs a step). Next step: none proposed. The noise covers the whole vocabulary by design,
  and folding it into the `_score` launch would touch a kit kernel.
- **Engram gather.** 512–717 KB, ring 33 µs (rank 0) and 57 µs (rank 1), twice a round.
- **Proxy (M, `proxy.py`, `fx1L` and `fx1Lb`, both ranks).** The drafter's transfers and the head look like the
  rest: doorbell to last write posted 0.69–0.74 µs at p50 and under 0.9 µs at p99 (kv 1.0–1.65 µs at p99), last
  completion to flag 16–32 ns, and one `woke` record a rank an arm (the first). Nothing for the proxy to give back.

## Tails and periodicity (lead 5)

- **Tails (M, `tails.py`).** Excess over the (position, rows) median: producer p99 170–176 µs, p99.9 294–332 µs;
  ring p99 140–158 µs, p99.9 211–278 µs. The MoE tail is routing (both ranks slow on the same sequence in 443–450 of
  about 800 events above 100 µs).
- **No period.** The per-forward mean excess has autocorrelation 0.02–0.05 at lags 32 and 64 and 0.14–0.17 at lag 1.
  The hot plan is not armed in production (`rd.hot` is null without `TF_DS_EXPERT_HOT_CACHE`), so nothing rebuilds
  every 32 rounds.
- **Millisecond stalls** are at the stretch boundary of §1 and almost nowhere else.
- **Request boundaries (M, `dark.py`).** 123–134 ms between a request's last forward and the next request's first
  pass (short prompts, mech arms), and one 1.1–1.3 s gap after the first forward of each arm (I: the restart's smoke
  request).
- **The 70–100 ms host stalls** of the re-score do not appear in any trace as a single interval. The largest
  in-request tile in the mech arms is 2.7–3.2 ms, the first Engram landing of the first request (`dark.py`).

## Inventory of trace artifacts

| Artifact | Where | What it records |
|---|---|---|
| Ring traces (0272) | `overnight-20261122/k272/tr16{a,b}.trace.r{0,1}.jsonl` (1 in 16, full suite, rank-0 profile on, 47K gpu records a rank); `sched-20261122/b2/mech/{fx1L,fx1Lb,np1L,ns1L}` (1 in 4, 8 × 384 greedy tokens, 19,457 gpu records a rank, identical keys in all four); `sched-20261122/b2/rowsplit/{k0x1,k0x4,k6x1,k6x4,fx1,fx4}` (1 in 16, 1.0–4.8K records) | per sampled transfer: GPU-clock `prev, stage, doorbell, collect, flag, copy`, key, plus `proxy` (host clock) and `drain` records |
| GPU clock and temperature samples | 68 `*.clocks.jsonl` under `engine-followup/`, `admission-grace/` | one line every ~5 s: MHz, °C, throttle word per rank |
| Stage windows and meters | `overnight-opt/followthrough02/stage-windows.csv` (130 rows), `serving-meter01/windows.csv`, `*/meter/` | per-stage phase ms and read GB/s per rank (Nov 18, stage-instrumented image); pair read bandwidth and RDMA / NVMe counters |
| Profile lines | 85 non-empty `*.profile.txt` | lifetime means: `forward_ms`, `pass_ms`, `absorb_ms`, `logits_ms`, `round_clock_us` `wait` / `tail` |
| Phase runs | `overnight-20261122/phases/{d32x4,p128}.rank{0,1}.txt` | eager `phases_ms` by stage, with the profile's sync every round |
| Fused-arm engagement | `fused-next-20261121/wp4/engagement-trace.txt` (98 lines), suite `s05` | first-engagement shape of each fused kernel (rows, width, pdl); no times |
| SASS counts | `fnext-sass-20261122/opcounts.txt` | per kernel `EX2 / RCP / SHFL / ATOMS / BAR` counts |
| nsys attempt | `sched-20261122/b2/img/` (wrapper and Dockerfile only) | no capture kept; it recorded OS-runtime calls only (B2) |
| 0272 harness | `kernel-round-20261122c/h0272/` | synthetic traces for the summarizer's tests |

**Stack traces and crash dumps in saved logs** (`grep -ril` over `runs/` outside `sessions/`):
- `combo4/bisect-to0184.rank0.log`, `bisect-to0185b.rank0.log`: a real segfault with a Zig stack, address
  `0xaaaaaaaaaaaaaac2` (undefined-fill pointer) in `moe` at that tree's `round.zig:1573`, on the first request. The
  suite's `0184z-bench-hot-null-fixup.patch` is the fix; not a live issue.
- `w4/*.log` (16 files): `free(): invalid pointer` at exit with `TF_DS_EXPERT_BYTES=1`, recorded in
  `runs/w4/RESULTS.md` as a teardown bug of 0140.
- `overnight-opt/bf16-fix/host-safe-sandbox.log`: unit tests segfaulting and one panic in a host sandbox build
  (loopback-link tests of `kv_lanes` / `rank_wire`). Test environment, not serving.
- `overnight-opt/engineering02/engram-cpus/bench.restart.log`: a refused start (`InvalidEngramCpus`) with allocator
  leak reports and empty stack traces.
- No panic or stack trace in any arm of the November 22–23 suites.

## Already known, not re-reported

The signed producer skew and its prefetch cause (B2), the proxy's cost and scheduling (A2, k272), the ring's
standalone price (A1), the aggregate prefetch gain, the MoE being bus bound with 27.2 µs a place (runs/w4), the KV
fetch reuse (0259), `absorb_ms` and the absorb reorder (0160), the pass's structure and the cache's miss rate (the
audit review). Corrected here with evidence: B2's cost of skew and k272's "transfer floor" (§2).

## What the traces cannot answer, and what would

| Question | Why not | Extra trace |
|---|---|---|
| Which kernel inside a producer interval is slow | only gathers are stamped | 0280's pass phase timer; an eager phase run for the forward |
| What rank 1's host does during an Engram stall | GPU-side only | `TF_DS_ENGRAM_METRICS` and the Engram host timers on both ranks, `iostat` |
| Capture against replay at a stretch start | the graph flag is baked at capture | log the capture with its ring sequence |
| Pick overlap across a window's rows at depth | only bytes are recorded, and they are fixed | a distinct-pick counter per fetch |
| The NCCL exchange's cost | not a ring transfer | a GPU-clock stamp before and after the all-gather, or the staged 0261 on / off run |
| k267's forward shift | no trace with `0267` on | one mech arm with `TF_DS_MARKOV_FUSED=1` and the trace |
| Greedy or sampled, per record | not in the key | a flag bit in the record |
| Four-stream behaviour in detail | 1 in 16, 15–76 samples a position | a 1 in 4 trace of a four-stream run |
| The 70–100 ms host stalls | none fell in a traced run | a per-round `wait` histogram instead of a mean |

## Unverified

- That the rank-local stretch-boundary stalls are the Engram landing, and which of its steps (hash, read) is slow
  (the position, the rank and the source fit; the host was not observed). The drive power-state, spill-write and
  host-thread explanations are guesses.
- That the both-ranks-late events are graph captures.
- That the side streams cause part of the jitter (one run each for `np1L` and `ns1L`), and that any of the 0.58 ms
  is recoverable.
- The size of the NCCL share of the sampled-round tail; the sampled and greedy rounds compared are different requests.
- The four-stream noise cost (80 µs a step) and the four-stream pass total, from 19–76 samples a position.
- `0281`'s gains: the hit rates are offline, on natural text, not model output.
- `0282`'s 0.18 ms: an analogy, not a measurement; the drafter's stages need not respond like the target's layers.
- The KV deduplication saving beyond short contexts.
- All absolute times are greedy short-context runs on image `10c9866`; production is `2bb052f`.
- The NVMe power-state latency setting proposed in §1 (the controller's `pm_qos_latency_tolerance_us` attribute) was
  not checked on the nodes.
