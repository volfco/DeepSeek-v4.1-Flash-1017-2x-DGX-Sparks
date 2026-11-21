# bench-prep — the benchmark plan for the new switches (patches `0129`-`0145`)

Forward-looking plan for the new engine switches: the `0129`-`0145` series of the engine fork (`0130`-`0147` as the
work packages of `docs/benchmark-gaps-and-moe-review.md` number them — confirm each number against its
implementation report, the numbering here is indicative). **Nothing here is measured.** The patches are not in this
repo's `zig/patches/` yet, and the branch that carries them (`bp/bench-prep`, after integration) may not exist or may
change. Names, defaults and semantics in `env-vars.md` are a draft; the implementation reports refine them. One
switch at a time, everything else identical, as in `to-benchmark.md`.

Files here:

| File | What it is |
| --- | --- |
| `README.md` | this plan: branches, transport, per-arm protocol, gate order, restore |
| `run_arm.sh` | the parameterised driver (restarts the pair; **never** run from an agent session) |
| `env-vars.md` | draft table of every new switch: default, rank scope, semantics, gate, keep/drop bar |
| `r8.requests.json` | 8-stream `tf-dsv41-lanes --requests` file for the W4 r8 cells, NEEDS-VERIFICATION |
| `overnight-fold.md` | the W2 deliverable: `runs/overnight/RESULTS.md` folded into one dated section for `to-benchmark.md` |

The W3 (b) remainder is `tools/score_human_text.py` (not here): it cuts held-out human text into candidate
prompt/continuation splits. It mints **no** token ids — those are recorded on the pair through
`tools/score_sets/README.md`'s `/tokenize` flow, which is why its output is `candidates.jsonl`, not `items.jsonl`.

Sources: `to-benchmark.md` (the "Proposed" table, the Engram/matrix protocol), `docs/benchmark-gaps-and-moe-review.md`
(work packages W1-W10; the W4/W6 instruments are what most of the `0129`-`0145` switches serve), and
`docs/baseten-ultrafast-plan.md` (the instrumentation-first list its item 2 maps onto). Read them before an arm;
do not re-derive them from this file.

## Where the patches live and how the pair gets them

- Engine: the fork at `../tfref-next`, on the branch **`bp/bench-prep`** (after integration; created from
  `decode-step-structure` + the `0129`-`0145` series). This repo's `zig/patches/` holds the `*.patch` files the image
  builds from; `../tfref-next` is read-only from here — never edit it from a bench session.
- Transport (the repo's standing rule: hand patches over as `git format-patch` output, no files added by hand in
  `zig/patches/`):
  1. In `../tfref-next`, on `bp/bench-prep`: `git format-patch -o <scratch> series-0119..bp/bench-prep`
     (or the integration base the patches sit on — whatever `git log --oneline` shows as their parent).
  2. The series is already exported into **this** repo's `zig/patches/` (`0129`-`0145`, by
     `scripts/port-patches.sh`); a regeneration is `scripts/port-patches.sh --engine ../tfref-next \
     --branch bench-prep --base decode-step-structure`
     (spell out the exact files; the glob is only shorthand).
  3. Sync and rebuild: `just sync-head`, then on the head (or via `just restart-head flags="..."`) `./start.sh restart`.
     `start.sh` rebuilds the image when the patch series changed (the image's `tf.patches` label is the hash of
     `zig/patches/`; see `scripts/prepare.sh`), so the rebuild is triggered by the `cp`, not by hand.
     First build after a series lands takes as long as `prepare.sh` (~hours for a cold build; ~6 min for the Zig
     builds themselves). Do not run this from an agent session; the pair is the operator's.
- The image's patch hash changes with `zig/patches/`, so `runs/matrix/base.validate.json` (image
  `f3f69b951cd3`-era base save) remains the cross-image identity gate for the bit-safe switches: every switch that
  claims "identical" must compare against it, and the compare must pass on the **new** image with switches off
  before any arm is called done.

## Per-arm protocol: ONE switch at a time

Carried over from `to-benchmark.md`, unchanged:

- One switch per arm; every other setting identical to base (image patch hash, `CONTEXT`, `POOL`, `PARALLEL`,
  `ROUND_ROWS`, `GRAPHS`, `TF_FILL_*`, `SPILL_GIB`, `KV_SHARD`, the prompt salt policy). Two runs that differ in
  more than the switch are not an A/B.
- Arms with a bit-safety claim (`0130`-`0134`, `0136`, `0147` by construction as instruments) still run the identity
  gates; "identical by construction" is a claim to verify, not to assume (the `0124`/`0125` precedent: "if not,
  it is a wrong kernel").
- Bit-safe is not the same as timing-invisible; measure every switch. Neutral switches are kept only as drafts
  with their numbers attached, or dropped.
- `0145` (`TF_DS_ALLOCATE_LEARNED`) is not bit-safe: it changes the allocator's row choices, so round composition
  moves. It is **sequential**: greedy replies keep their bits alone, the concurrent check does not. Gate it with
  both `validate.py`'s sequential and concurrent checks (as the depth-pin rows in `to-benchmark.md` were gated),
  and score it on the quality set, not on identity alone.

## Gate order per arm

Run strictly in this order; stop at the first failed gate. `run_arm.sh` implements exactly this sequence.

1. **Restart with the arm's vars exported** (and nothing else changed): `just restart-head flags="VAR=v ..."`
   (the flags= string goes to both ranks via `start.sh`).
2. **Health**: poll `http://127.0.0.1:8000/health` until `status: ok` and `pair.broken: false`.
   **The first request after a restart deterministically fails** (`cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`,
   `prompt=8 tokens=1`) on this image and its predecessor — it is environmental, not a regression from any
   `0129`-`0145` patch. Retry once: the identical retried request passes. `run_arm.sh` does one warm-up request
   before gating; if it fails twice, the arm aborts (restore base runs regardless).
3. **Identity / smoke gate**: `python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017
   --save runs/bench-prep/<label>/validate.json --compare runs/matrix/base.validate.json`.
   The tool prints `PASS  equal to the saved replies` and `all passed` and exits 0 when the replies equal the base
   save — `scripts/bench_matrix.sh` labels that outcome `IDENTICAL` in its status line, and so does this plan; there
   is no literal `IDENTICAL` string in `validate.py`'s output (it ends `sys.exit(1 if failed else 0)`,
   `tools/validate.py:222`). For `0145` (`TF_DS_ALLOCATE_LEARNED`), both the sequential compare and the concurrent
   check (`concurrent == solo`) are required. Record `GATE-FAILED-rcN` per arm; do not spend a bench block on a
   failed gate.
4. **`tools/exact.py` where demanded** (bit-safe switches making a bit-safety claim beyond validate.py's smoke;
   every switch whose implementation report says "bit-identical" should get the record/compare treatment before
   it graduates). Pattern:
   `python3 -I tools/exact.py record --out runs/exact-base.json --label base` (done once, on the base image),
   then `record --like runs/exact-base.json` on the arm, then `compare`.
5. **Bench block** (natural prompts, fresh salt per run):
   `python3 -I tools/bench.py --suite prefill,decode,multi --streams 1,4 --reps 3 --metrics
   --prefill-sizes 2000,8000,32000 --label <label> --out runs/bench-prep/<label>/bench.json`
   Prefill 2K/8K/32K + decode/multi at 1 and 4 streams, 3 reps — the same block shape as `scripts/bench_matrix.sh`.
   `--metrics` snapshots `/metrics` before/after so the Engram histogram delta survives bench.py's series
   truncation; pair it with the `MemAvailable` low marks on both ranks (`scripts/memavail.sh` or the matrix
   sampler pattern) — `SCALES=map-warm`, pool and thread switches all trade memory or page cache.
6. **Instruments, where the switch is one** (the instrument patches, numbering indicative — see `env-vars.md` for
   the semantics of each): the done-line fields `engram_host_ms` and `queue_s`/`restore_s` (parse rank 0's done
   lines in the bench window only; keep the window clean of other requests), `TF_DS_EXPERT_BYTES` (the per-round
   expert-bytes counter beside `expert_prune`'s line), `TF_DS_PREFILL_PROFILE` (`prefill_phases_ms` per chunk),
   `pass_send_ms` (a profile column), and `tf-dsv41-probe` (a standalone binary; not an arm — run it beside the
   matrix, not instead of it). These have no keep/drop bar by themselves; they exist to size the MoE step (W4) and
   the TTFT decomposition (W10). Do not "keep" an instrument; keep the numbers it produced.

## The restore step (mandatory, after EVERY arm — including failed ones)

1. `just restart-head` (empty flags → `.env` alone, i.e. base production) or on the head `./start.sh restart`.
2. Health + one warm-up request (the first-request failure applies here too).
3. Verify base still matches: `python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017
   --save runs/bench-prep/<label>/restore.validate.json --compare runs/matrix/base.validate.json` must print
   `all passed` and exit 0 (the harness's `IDENTICAL`). If it does not, stop all benchmarking and report; the pair
   is not in a known state.
4. `run_arm.sh` performs all of the above and prints a `restored=yes/no` verdict component.

## The driver

`runs/bench-prep/run_arm.sh <label> [VAR=v ...]` — the parameterised driver implementing steps 1-3 and 5 plus the
restore (step 4, `exact.py`, and step 6's instruments are the operator's to run: `exact.py`'s base record must
outlive an arm, and the instruments print to the log the arm is already writing). It refuses to arm while the pair
is up with any known bench switch set (it greps the running container's env for switch **names** only, never
values), and warns when an arm var's name is not a known switch (a typo would read as a neutral arm). It is **not
to be run from an agent session**; it restarts the operator's pair. Example:

```bash
runs/bench-prep/run_arm.sh map-warm TF_DS_ENGRAM_SCALES=map-warm
runs/bench-prep/run_arm.sh decode-threads TF_DS_ENGRAM_DECODE_THREADS=4
```

Labels, first-request failure and artefacts (`runs/bench-prep/<label>/{status.log,restart.log,validate.json,
validate.log,bench.json,bench.log,restore.restart.log,restore.validate.log,...}`) follow `scripts/bench_matrix.sh`'s
conventions.

## Known limits of this plan

- The `0129`-`0145` series is exported into this repo's `zig/patches/` but has not been built into an image here, and
  nothing in it has run on the pair; every command above is documentation for
  the operator, not a runnable recipe.
- `r8.requests.json` is NEEDS-VERIFICATION. Its token ids are **not** fabricated: they are lifted byte-exact from
  `tools/score_sets/items.jsonl` (the eight prose items, `/tokenize` with `enable_thinking:false`), so they are ids
  this pair's tokenizer actually produced. What it lacks is `expect_sha` (no served reply was ever recorded for
  these ids), so `all_equal` in the lanes log is meaningless for it; for identity checking, rebuild the file on the
  pair with an eight-prompt variant of `runs/ultrafast/uf_profreq.py`. Its prompts are 270-319 ids, where `r1`/`r4`
  use ~21: the r8 cells are new cells, not a continuation of the r4 column.
- The W4 arm (`r8` profile cells) is driven through `runs/ultrafast/uf_prof.sh`'s machinery on the head, not through
  `run_arm.sh` (it needs the `tf-dsv41-lanes` batch driver as both ranks, which is a different restart shape than
  the served pair).
- 3x alternation is the bar for calling anything POSITIVE (`to-benchmark.md`, `runs/overnight/RESULTS.md`); the
  single block per arm implemented by `run_arm.sh` is signal grade. Do not promote a switch to production from one
  pass.
- First-request failure after restart: environmental, pre-existing, not caused by any `0129`-`0145` patch (it was
  observed on the image before the series existed). Retry once and move on; if the retry fails, the arm aborts and
  the restore step still runs.
