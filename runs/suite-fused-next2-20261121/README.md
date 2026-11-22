# suite-fused-next2-20261121: `d5ecba0` vs `84806bb` (patches 0254-0258), pre-restart checklist done

Paired two-Spark gate for `todo.md` "Measure patches 0254-0258 on both Sparks": base
`tensorfold-dsv41:bench-fused-next-d5ecba0` against candidate `tensorfold-dsv41:bench-fused-next-84806bb`
(0254 `topk_indices_cut`, 0255 `cand_fast8`, 0256/0258 thread-safe `fused_trace` with 0254/0255 arms,
0257 `planKvSource2`) on production's `.env` switches, ABBA s01-s04 plus s05 = candidate with
`TF_DS_FUSED_TRACE=1` on both ranks (traced replays of the 8,192/16,384-block buckets and the 200K fallback).

## Fixes over `suite-fused-next-20261121` (do not run the old folder's gate on these blocks)

- `gate.py` now takes its suite path from its own location (or `$GATE_SUITE`), compares the lead-reader banner
  against each block's `TF_DS_ENGRAM_LEAD` flag (production's `.env` default `1`) instead of its arm name, and
  applies the "drafting must differ" rule only to arms whose flags change drafting switches — a fused-kernel arm
  reproducing base's rounds/accepted drafts is bit-exactness evidence, not `switch NOT APPLIED`.
- New `fixed_replay` cell: unsalted fixed-seed greedy replays at ~49,967 / ~99,904 / ~200,000 tokens
  (identical bytes and token IDs across blocks; `text_sha`/`prompt_tokens` equality is gated, `token_sha` equality
  is the exactness gate, prefill tok/s and client-span ms/round are compared in `table2.py`). This covers the
  8,192- and 16,384-block decode buckets plus the `>16,384` fallback decode and a deep prefill — the exact panel's
  49,967/99,904 cases run too, but unsalted and rate-recording.
- New `dec8_staggered` cell: `runs/pre-bench-20261121/dec8-audit/dec8_replay_audit.py record --admission-mode
  staggered` locks admission order (`adm_ord == [0..7]`), removing the Wave-0 admission-order artifact the dec8
  audit found; cross-checked against s01's own dec8 prompts/hashes from s02 on.
- `table2.py`: fixed_replay identity+rate table (deltas vs s01), `dec8_replay_audit compare` invocations, and
  standard-cell rate orientation.
- `launch.sh`'s 0215 capture uses the crossover harness's real CLI and refreshes the parity state first
  (`measure.py --mode parity` writes the `cases` + `all_parity_passed` that `--mode timing-burst` requires; without
  it the burst exits 1 and the capture would silently skip). Verified against the head's
  `~/tf-bench/pre-bench-20261121/crossover/` tree.

## How to run (nothing runs until the operator starts it)

1. `bash runs/suite-fused-next2-20261121/preflight.sh` **from the workstation** (recipe root): refuses if head
   `MemAvailable < 4.5 GiB`, builds+ships `bench-fused-next-84806bb` via
   `runs/fused-next-20261121/bench/build.sh 84806bb` (no container restart), syncs **this folder only** to the head
   (never `just sync-head`: it would push the recipe's `zig/patches` over the head's and change the production build
   hash), and starts `launch.sh` detached on the head.
2. `launch.sh` (head) waits for 30 external-idle minutes (12h cap), snapshots `production.rank{0,1}.env` from the
   running containers, refreshes the `0215` parity state (`measure.py --mode parity`, ~3 s — `timing-burst` refuses
   to run without `all_parity_passed` in the state JSON), then captures the guarded `0215` small-row crossover
   timings into `runs/pre-bench-20261121/crossover-0215/` via `run_timing_loop.py --cubin bin/ops.cubin --json
   results.json --budget-s 120` (~2 min; on failure or continuous traffic it continues with `small_nbk = 6144`
   unchanged and `timing_valid: false` rather than dirty medians), then starts `start.sh`.
3. `start.sh` runs `run_suite.py` detached; the runner restores production from `.env` on completion, `STOP`, any
   exception, or a non-loopback client. Per block: fresh `SPILL_DIR=/home/colum/.cache/dsv41-suite-fnext2-s0X` on
   both Sparks (cold spill namespace), arm proof, `image_ok` by running image tag.
4. Gates and tables: `python3 -I runs/suite-fused-next2-20261121/gate.py` and
   `python3 -I runs/suite-fused-next2-20261121/table2.py` from the recipe root. CPU tests for the gate:
   `python3 -m unittest discover -s runs/suite-fused-next2-20261121 -p 'test_*.py'`.

## Pass criteria (from `todo.md` "Measure patches 0254-0258 on both Sparks")

- `gate.py` all PASS: exact panel 39/39 (`exact:runs/combo3/blocks/a01.exact.json`), probe/deep replies equal,
  `fixed_replay` `text_sha`/`prompt_tokens`/`token_sha` equal across blocks, `dec8_staggered` prompt/sha streams
  equal, validation equal to the saved references.
- `table2.py` deltas: prefill tok/s and decode ms/round of `fixed_replay` within the base-base spread of the
  standard cells; `dec8_staggered` `true_active_ms_per_round` deltas within control variation (the dec8 dip
  attribution says the barrier-mode dip was scheduling, not kernels).
- s05 `TF_DS_FUSED_TRACE=1` prints `topk_indices_cut` / `cand_fast8` / `idx_small` / `idx_fused` arms on both ranks'
  logs at the 8,192/16,384/32,768-block shapes (`runs/pre-bench-20261121/p0231-dispatch/RESULTS.md` covers the host
  side; the live-round trace is what s05 buys).
