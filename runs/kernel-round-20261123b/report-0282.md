# report-0282 — TF_DS_DRAFT_PREFETCH

**Commit:** `92dc42f` on `kernel-stack-0280` (`~/Workspace/tfref-kstack5`, parent `b0715eb`). 0283 (`1421187`) then
made the sites take `Ops.l2`'s pacing and budget, so in the built image the drafter's prefetch follows
`TF_DS_L2_DELAY_NS` / `_RATE_GBPS` / `_BUDGET_MIB` like the forward's.
**Subject:** `0282 dsv41: TF_DS_DRAFT_PREFETCH (off by default): the paced L2 prefetch inside the drafter pass`
**Switch:** `TF_DS_DRAFT_PREFETCH`: unset, `""` or `0` off; `1` the three sites a stage; `2` those and the hand-off.
Anything else prints `TF_DS_DRAFT_PREFETCH: expected 0, 1 or 2` to stderr and refuses startup
(`error.BadDraftPrefetch`, at `Ops` init).
**Nothing here ran on a GPU.**

## Premise

Holds. Every linear of the drafter's three stages is an EXL3 trellis in the served checkpoint (the rank cache's index
lists `mtp.{0,1,2}.attn.*|tr` and `mtp.*.ffn.shared_experts.*|tr`, 4 bits a weight; no `.weight` float linears), so
every site has real ranges. Why the kit's own forks in the pass "move no bytes" (`draft.zig:10`) is not recoverable
from anything here.

## The sites and what each launch touches (TP2 rank part, unset 16 MiB budget)

| Site | Where it forks | Ranges, in order | Bytes |
|---|---|---|---:|
| 1 | `attention`, behind q's linears (`wq_a`, `wkv` and `wq_b` issued) | the four `wo_a` groups (2,097,152 each), `wo_b` cut to 8,388,608 of 10,485,760 | 16,777,216 |
| 2 | `attention`, behind `wo_b`; half the budget | `hc_ffn` fn 1,966,080; the gate; shared gate 2,949,120; shared up, cut; shared down not reached | 8,388,608 |
| 3 | `body`, behind the experts, stages 0 and 1 | next stage's `hc_attn` fn 1,966,080, `wq_a` 3,276,800, `wkv` 1,310,720, `wq_b` cut to 10,223,616 of 10,485,760 | 16,777,216 |
| 3 | the same, last stage | the vocabulary head, 16,777,216 of 248,217,600 | 16,777,216 |
| 4 (`2` only) | `body`, behind the Markov loop's last `_pick` | target layer 0: `hc_attn` fn 1,966,080, `wq_a` 4,915,200, `wkv` 1,966,080, `wq_b` cut to 7,929,856 of 13,107,200 | 16,777,216 |

A pass touches 120 MiB a rank with `1` and 136 MiB with `2`. The byte counts are pinned by the host test
`the served drafter's sites: bytes a launch at the unset 16 MiB budget`, from the shapes in the rank cache's index.

**Launches a pass:** 9 on the drafter's prefetch stream with `1`, 10 with `2`, at one stream or four (the count
depends on the stages, not the rows). Off: none. Each fork also records two events and one stream wait, which a
captured graph holds as dependencies, not launches. The pass graph's key is unchanged.

**Banner**, once a process on each rank, at the first body that forks (the first capture, with graphs on):
`dsv41: TF_DS_DRAFT_PREFETCH=<1|2>: the drafter pass forks the paced L2 prefetch, <9|10> launches a pass (first at <N> streams, <S> steps)`.

## Contradictions with the brief, and findings about the drafter's weights

1. **Site 1 sits behind `wq_b`, not behind `wq_a` / `wkv` alone.** The brief's parenthesis says "(`wq_a`, `wkv`
   issued)". The forward's first site, which the brief asks to mirror, is after `wq_b`'s launch
   (`round.zig:1216`, just before the `a_q` mark; its comment reads "behind q's linears"). I mirrored the forward.
   Moving the fork up to the `attn_in` mark is a three-line change if the earlier fork is what was meant; it would
   run the prefetch beside `wq_b`'s own 10 MiB read.
2. **At the served six-row block the gate the pass reads is not `gate_w`.** The kit has drafter gates only at 5, 10
   and 15 rows (`tri_norm.rowmmGateFits`); a pass of 6 rows (one stream, `TF_DS_DRAFT_BLOCK=6`, six steps) takes the
   cuBLAS gate and reads the fp32 copy `dr.gate_f[j]` (2,621,440 bytes), not the fp16 `gate_w` (1,310,720). Site 2
   prefetches whichever the pass will read, by the same predicate (`Drafter.kitGate`, which `moe` now calls too).
3. **Site 2 never reaches the shared expert's down trellis.** After the mix, the gate and the shared gate, 851,968
   bytes of the half budget are left at 6 rows (2,162,688 at 5), all spent on the shared up. `TF_DS_L2_BUDGET_MIB=32`
   would cover up and down.
4. **The head site covers 6.8% of what the head reads** (16 of 236.7 MiB a rank), as the forward's own last site
   does. The trace review prices the pass's head at 1.08 ms, its largest item.
5. **Stage 0's input linears have no site.** In the forward a layer's inputs are prefetched behind the layer before
   it. The pass starts at stage 0, so its `hc_attn` fn, `wq_a`, `wkv` and `wq_b` are read cold in both modes.
6. **Routed experts are not prefetched**, as in the forward (three routed experts a row at 8.4 MiB each).

## Costs to know before reading the arms

- **The body-end join makes the pass wait for the last fork.** With `2` that is the hand-off launch, forked behind
  the last `_pick` with only the confidence head after it: up to about 76 µs (3 µs delay, then 16 MiB at 230 GB/s).
  With `1` the last fork is the head's, far earlier, and has long finished. So `pass_ms` in the `dpf2` arms can rise
  even if the cycle falls. Score them on forward + pass, not on `pass_ms` alone.
- **The hand-off (`2`) has no supporting evidence in the traces.** `runs/trace-review-20261123/RESULTS.md` section 6
  found no sign that the forward starts cold after the pass and bounds a hand-off prefetch at the roughly 30 µs that
  layer 0's attention lacks. Section 5 bounds sites 1–3 at about 0.18 ms a pass by analogy with the forward
  (41 µs an attention layer, 20 µs a MoE layer at 6 rows). Both values are kept as briefed.

## Why the bits cannot move

`tf_ds_l2_prefetch_kernel` writes nothing. Off: `dr.pf == null`, so no stream, no event and no fork exist, `prefetch`
and `joinPrefetch` return at once, and the pass's launches and graph are unchanged; `moe` chooses its gate by the
same predicate as before. Both ranks alike under one env; rank-local in principle.

## Capture safety

The body joins the prefetch stream after the confidence head's GEMV, so a capture ends with every stream joined
and `Drafter.pass`'s synchronize covers the forks. A body that fails inside its capture joins what it forked, ends
the capture and clears the flag (`round.zig`'s `leaveCapture` order); an eager body that fails clears the flag. No
host sync and no pageable transfer were added inside the body.

## Files

`draft_prefetch.zig` (new, pure: switch, `plan`, `trellis`, `hcFn`, `launches`, 3 tests), `draft.zig` (the stream and
events at `Drafter.init`, `dropPrefetch`, `prefetch`, `joinPrefetch`, `kitGate`, the four sites, the join, banner),
`ops.zig` (`Ops.draft_prefetch`), `model.zig` (drop at close and on a failed open), `layer_main.zig` (drop in the
exact tooling's two drafter sites), `dsv41.zig` (export).

## Gate

- `zig build test`: exit 0. `zig build`: exit 0. The test log's last line is a `failed command: …/test …` label:
  Zig 0.17's build runner prints it with any test step that wrote to stderr, base `2bb052f` prints the same (three
  on a cold cache, one once the unchanged modules' runs are cached) and the exit code is 0. The family's test binary
  run directly ends `All 520 tests passed.` with exit 0, the three `draft_prefetch` tests among them.
- `cuda-syntax-stubs/check.sh`: the same 9 errors as base `2bb052f`, output identical after path normalization.
- Host tests: parse matrix and refusal; `plan` (order, empty ranges skipped, the cut, the 16-range cap); the served
  sites' bytes above.

## What only hardware can settle

- Whether any site pays: the stage intervals of the pass (`pass_phases_ms` `attn_*`, `gate`, `experts`; or
  `pass_breakdown.py` on a 1-in-4 ring trace) are where a gain would show; the head and Markov intervals should not
  move.
- Whether two-stream capture of the pass works on this driver. The forward's stretches do exactly this, but a pass
  graph has never held a second stream.
- Whether the prefetch disturbs the gathers' staging kernels beside it (the 3 µs delay exists for that in the
  forward).
- Drafts unchanged: by construction; the arm should still compare per-request rounds and accepted / drafted with
  control.
