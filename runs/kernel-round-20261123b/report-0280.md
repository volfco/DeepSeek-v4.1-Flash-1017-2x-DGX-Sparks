# report-0280 — TF_DS_PASS_PROFILE and the pass counters

**Commit:** `cb8a4ca` on `kernel-stack-0280` (`~/Workspace/tfref-kstack5`, parent `2bb052f`).
**Subject:** `0280 dsv41: TF_DS_PASS_PROFILE (off by default): the drafter pass's phases, steps and bias misses under the profile`
**Switch:** `TF_DS_PASS_PROFILE`: unset, `""` or `0` off; `1` on; anything else prints `TF_DS_PASS_PROFILE: expected 1 or 0`
to stderr and refuses startup (`error.BadPassProfile`, at `Model.open` before the weights load).
**Nothing here ran on a GPU.** The local gate is compile, host tests and the stub syntax check.

## What it does

1. **Phase timer** (`pass_profile.Timer`, the switch). `round.PhaseTimer`'s shape: a CUDA timing event recorded on the
   pass's stream at each phase's end, read after the pass's own synchronize. With the switch on `Drafter.pass` runs
   `body` eagerly and never through `PassGraphs` (the condition is now `graphs != null and probe == null and pt == null`).
2. **Pass counters** (`pass_profile.Counts`, no switch beyond a round profile: `Model.prof`, i.e. `TF_DS_ROUND_PROFILE=1|2`
   on rank 0, `--profile` / `RANK1_PROFILE` on a follower). Counted in `Model.pass` after the pass returns, from the
   pending tokens and the drafts the pass already read back. No device work, so a graph pass counts the same.

## Env for an attribution run

```
TF_DS_ROUND_PROFILE=2 GRAPHS=0 TF_DS_PASS_PROFILE=1 RANK1_PROFILE=2
```

`GRAPHS=0` is the recipe setting that becomes `TF_DS_GRAPHS=0`; it is what makes the forward's `phases_ms` valid. The
pass timer does not need it: the switch alone makes the pass eager, with graphs on or off. `TF_DS_*` goes to both
ranks, so rank 1's passes are eager too; rank 1 prints only with `RANK1_PROFILE`.

## Output lines

Rank 0, at each `/metrics` scrape, through `std.log.info` after `phases_ms` and before `profile_note`
(`native.printRoundProfile`); cumulative since the process started:

```
{"rank": 0, "markov_cache": {"hits": H, "misses": M}}
{"rank": 0, "pass_counts": {"passes": P, "rows": N, "steps": {"1": a, "2": b, "3": c, "4": d, "5": e, "6": f, "7": g, "8": h}, "hits": H, "misses": M, "miss_steps": S}}
{"rank": 0, "pass_phases_ms": {"upload": x, "embed": x, "mix": x, "attn_in": x, "attn_q": x, "attn_kv": x, "attn_core": x, "attn_out": x, "gather_a": x, "gate": x, "experts": x, "gather_m": x, "post": x, "head": x, "mk_bias": x, "mk_score": x, "mk_gather": x, "mk_pick": x, "conf": x, "read": x}, "passes": P}
```

- `markov_cache` keeps 0267's shape (the lead's `table_gpu.py` reads it). `hits` + `misses` = step rows (streams ×
  steps, summed). The counters start at zero (`m.pass_counts = .{}` in `Model.open`; 0267's started at `0xAA…`).
- `pass_counts.steps["k"]` = passes that ran k Markov steps. `rows` = block rows summed (streams × `blockRows(steps)`),
  so `rows / passes` is the mean. `miss_steps` = steps in which at least one stream's row missed.
- `pass_phases_ms` values are mean ms a pass (4 decimals), summed over the three stages and over the steps. The line
  is printed only with the switch on. The first two lines print under any round profile.
- A follower prints the same three lines with its own rank at exit (`lanes_main.zig`, the `--profile` block), and the
  batch driver prints them for rank 0 at the end of its run.
- Banner, once a process on each rank at the first profiled pass:
  `dsv41: TF_DS_PASS_PROFILE: the drafter pass runs eagerly under phase events (first at <N> streams, <S> steps)`.

## Phases (each mark charges the time since the mark before it)

| Phase | What ends there |
|---|---|
| `upload` | the index rows, pick lists, `out`'s first column (and the keyed table) |
| `embed` | `embedInit` |
| `mix` | both `hcPre2` mixes a stage |
| `attn_in` | `wq_a` + `wkv` (one group; LoRA terms if any) |
| `attn_q` | q's RMSNorm + `wq_b` |
| `attn_kv` | the block rows' `kvNormRope` |
| `attn_core` | `sparseAttn` + `ropeHeads` |
| `attn_out` | `wo_a` (rotMany + glinear) + `wo_b` |
| `gather_a` | the attention partial's cross-rank gather |
| `gate` | the gate (kit `rowmmGate` or the cuBLAS fallback) + `route` |
| `experts` | the routed experts and the shared one |
| `gather_m` | the MoE partial's gather |
| `post` | the last `hcPost` |
| `head` | collapse + RMSNorm + the vocabulary head |
| `mk_bias` | a step's `_bias` |
| `mk_score` | `_score` + `_finish`; also the noise hook before `_bias` and the clean-score hook after it, when a pass has one |
| `mk_gather` | a step's gather of the ranks' bests |
| `mk_pick` | `_pick` |
| `conf` | the confidence input pack and its GEMV |
| `read` | from the last kernel's end to after the pass's synchronize and its two device-to-host copies |

## Launches at the served shape

None added or removed, switch off or on, at one stream (six block rows, six steps) or four. On: 64 event records a pass
at three stages and six steps (3 + 11 × 3 + 2 + 4 × 6 + 2; 70 with a noise hook), and the pass's launches are issued
eagerly instead of replayed. The timer holds 91 events (four stages, eight steps, a hook mark each).

## Why the bits cannot move

Off: `dr.pt == null`; every `mark` is a null check, `stepsCut(..., null)` issues exactly `stepsTapped`'s launches in
the same order, and the pass takes the graph as before. On: the body is the function the graph captures, called
directly, as `probe != null` already does in the layer tooling; events write nothing. The counters read host arrays.

## Files

`pass_profile.zig` (new: switch, `Phase`, `Timer`, `Counts`, the line printers, 5 tests), `draft.zig` (the marks, the
eager condition, banner), `tri_markov.zig` (`stepsCut` / `Cuts`; `stepsTapped` and `localBestWith` delegate with no
cuts), `model.zig` (parse, timer, counters), `native.zig` and `lanes_main.zig` (printing), `markov_tokens.zig`
(`has`, ported from 0267), `dsv41.zig` (export).

## Gate

- `zig build test`: exit 0. The log's last line is a `failed command: …/test …` label: Zig 0.17's build runner
  prints it with any test step that wrote to stderr. Base `2bb052f` prints the same (three on a cold cache, one once
  the unchanged modules' runs are cached) with exit 0. The family's test binary run directly ends
  `All 512 tests passed.` with exit 0; the log holds the 12 `TF_DS_PASS_PROFILE: expected 1 or 0` refusals of the
  parse test.
- `zig build`: exit 0.
- `cuda-syntax-stubs/check.sh`: base `2bb052f` 9 errors, this tree 9 errors; the two outputs are identical once the tree
  path is normalized (no `.cu` touched).
- Host tests: parse matrix and refusal; `charge` (each gap to the phase of the mark that ends it, both mixes and all
  steps summed, every gap charged once); the widest pass fits `max_marks`; `Counts.pass` (rows, histogram, hits,
  misses, miss steps, the overflow bucket); the three lines' exact text and that `pass_phases_ms` parses as JSON with
  20 keys.

## Design decisions

- The switch alone makes the pass eager and marks it; printing rides the existing profile outputs. Without a profile
  the timer still runs but nothing prints.
- `markov_cache` stays its own line and its two numbers are repeated inside `pass_counts`, so 0267's readers and the
  brief's one-line reading both work.
- A hook before `_bias` (the keyed noise) is charged to `mk_score`, as the brief asks, by a mark of its own.
- `Timer.begin` resets the mark list, so a pass that failed part-way cannot shift the next pass's phases.

## Deviations from the brief

- The steps histogram has buckets 1..8 (`TF_DS_DRAFT_BLOCK` allows 7 steps; the brief said 1..6).
- `pass_counts` carries `passes` as well, so means need no second line.

## Cross-check for the first hardware run

`runs/trace-review-20261123/RESULTS.md` section 5 measured the pass from ring traces at one stream, six steps: the
vocabulary head 1.08 ms, the three MoE stages 1.29 ms, the three attention stages 0.68 ms, the Markov steps 0.49 ms,
the twelve ring transfers 0.23–0.26 ms, a greedy pass about 3.9 ms in all; section 3 gives a miss step about 0.14 ms
against 9.8 µs for a hit. The timer should reproduce roughly that: `head` near 1.08; `gate` + `experts` + their `mix`
near 1.29; `attn_*` + their `mix` near 0.68; `mk_bias` + `mk_score` + `mk_pick` near 0.49 less the gathers;
`gather_a` + `gather_m` + `mk_gather` near 0.25. `mk_bias` over `pass_counts.miss_steps` should come out near
0.14 ms a miss step. A large disagreement means the eager pass is not a fair proxy for the graph pass, not that the
traces are wrong.

## What only hardware can settle

- Every number. In particular whether the eager pass's phases add up to something close to the 4.1–4.2 ms graph pass:
  eager launches add host time and the events cut PDL chains, so read shares, not totals.
- `gather_a`, `gather_m` and `mk_gather` include waiting for the other rank. With `RANK1_PROFILE` unset rank 1 still
  runs its passes eagerly (the env reaches it) but prints nothing.
- That the events and the eager pass leave drafts unchanged is by construction, not observed: compare per-request
  rounds and accepted/drafted against a control, as for 0281.
