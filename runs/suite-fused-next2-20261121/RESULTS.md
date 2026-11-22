# suite-fused-next2-20261121: 0254–0258 pair gate — **5/5 blocks PASS, neutral within the floor**

Paired two-Spark ABBA gate of `tensorfold-dsv41:bench-fused-next-84806bb` (0254 `topk_indices_cut`, 0255
`cand_fast8`, 0256/0258 thread-safe `fused_trace`, 0257 `planKvSource2`) against `bench-fused-next-d5ecba0`
(0213–0253) on production's `.env` switches, plus `s05` = candidate with `TF_DS_FUSED_TRACE=1` on both ranks.
Run 2026-11-22 02:34–04:31 UTC on `gx10-eb0e`/`gx10-235f`, production stopped for the window and restored after
(`bench-prune-six-20261120`, health 200, chat answers).

**Verdict, by the todo's own gate: pass.** Exact 39/39 on every block, every probe reply hash equal with identical
rounds and accepted drafts, `fixed_replay` input bytes/prompt tokens/token hash equal across all five blocks,
`dec8_staggered` prompt/sha streams identical in 3/3 waves on all five blocks, validation rc 0 on every block.
Speed: the two pass1/pass2 comparisons have opposite signs and both sit inside the base-base control spread, so
**0254–0258 change nothing measurable in serving** (they were already −39…−60% / −15…−23% standalone).

## Gate (`gate.py`, 5 of 5 pass)

| Block | Arm | Image | Exact 39 | Probe 11 rows | fixed_replay | dec8_staggered | mem low H/W |
|---|---|---|---|---|---|---|---|
| s01 | base | `bench-fused-next-d5ecba0` | rc 0 | 11/11 hashes, 0 drafting rows differ | 3 clean, cold | 3 waves (re-recorded, see below) | 5.2/6.4 GiB |
| s02 | next | `bench-fused-next-84806bb` | rc 0 | same | 3 clean, cold | 3 waves, 3/3 match s01 | 4.8/6.5 |
| s03 | next | `bench-fused-next-84806bb` | rc 0 | same | 3 clean, cold | 3 waves, 3/3 match s01 | 5.4/6.4 |
| s04 | base | `bench-fused-next-d5ecba0` | rc 0 | same | 3 clean, cold | 3 waves, 3/3 match s01 | 4.9/6.5 |
| s05 | trace | `bench-fused-next-84806bb` + `TF_DS_FUSED_TRACE=1` | rc 0 | same | 3 clean, cold | 3 waves, 3/3 match s01 | 5.2/6.7 |

All blocks: `status=ok`, clean cells 42/42 (s01–s04) and 11/11 (s05), clock max 2392/2385 MHz `valid=True`, lead
banner on both ranks, graphs destroyed 128/128 (s05: 8/8). Binaries hash-identical on both ranks
(`4fd57ef1…` native, `0d60da68…` lanes).

## Speed

**dec8_staggered** (`dec8_replay_audit compare`, admission order locked, all shas equal in every pair):

| Pair | arm change | pure_decode_tok/s | true_active_ms/round | legacy_tok/s |
|---|---|---|---|---|
| s01 → s02 | base → next (pass 1) | −1.35% | +1.26% | −1.21% |
| s04 → s03 | base → next (pass 2) | +1.71% | −1.72% | +1.51% |
| s01 → s04 | **base → base (control)** | −2.23% | +2.36% | −1.93% |
| s01 → s05 | base → next + trace | −0.24% | +0.38% | +0.09% |

The control arm is the widest move in the table (±2.2–2.4%), so the resolution floor of this window is larger than
either next-vs-base delta. Mean next-vs-base: +0.18% pure decode; mean base-vs-base: −2.23%.

**fixed_replay prefill tok/s** (identical input bytes; deltas vs s01):

| Block | 50,405 tok | 99,839 tok | 200,297 tok |
|---|---|---|---|
| s01 base | 2,043.1 | 1,987.4 | 1,877.5 |
| s02 next | 2,018.8 (−1.19%) | 1,970.9 (−0.83%) | 1,877.6 (+0.01%) |
| s03 next | 2,051.2 (+0.40%) | 2,010.5 (+1.16%) | 1,887.9 (+0.56%) |
| s04 base | 2,034.5 (−0.42%) | 1,985.0 (−0.12%) | 1,892.4 (+0.79%) |
| s05 trace | 2,092.3 (+2.40%) | 2,049.9 (+3.14%) | 2,012.8 (+7.21%) |

next-vs-base mean +0.02%; base-base control −0.25% to +0.79%. s05's +2.4…+7.2% is not attributable to tracing
(host-side lines only) and is not claimed as a gain.

**Standard cells** (orientation): prefill 8K/32K/100K 1,955–1,971 / 2,073–2,089 / 2,032–2,044 tok/s and decode
81.2–81.6 (dec1), 116.0–116.3 (dec4), 138.2–138.5 (dec8) tok/s are indistinguishable across arms.

## Trace (`s05`, 128 lines per rank, both ranks equal in shape coverage)

`cand_fast8` 10× (decode, rows 1–16, width 1,024), `topk_indices_cut` 1× (rows 128, width 21,875),
`idx_fused` 2× (rows 6 width 32,768; rows 128 width 21,875), `compress2_norm`/`compress2x_norm` 11× each,
`prune_six` 11×, `key_fused`/`latent_fused`/`query_fused` 11× each, `prompt_key_fused`/`prompt_latent_fused` 23×
each, `query_pair` 3×. All `pdl=0`.

**Coverage gap:** the 8,192- and 16,384-block *decode* shapes were never traced, because every `fixed_replay`
prompt asked for the single word "ok" and finished in **1 round / 1 token** (`rounds=1`, `ms/r=0.0`,
`sha=4b13688173` at all three sizes). Token-hash equality there is a one-token comparison and decode ms/round is
vacuous; only prefill tok/s carries information from those cells. The 32,768-block shape traced here is a prompt
shape, not the deep decode bucket the README promised.

## Incidents and fixes (this window)

1. **`preflight.sh` never ran as written** — two path bugs, fixed here and committed: `cd "$(dirname "$0")/../../.."`
   resolved to `~/Workspace` (one level too high; `runs/fused-next-20261121/bench/build.sh` then "No such file"),
   and the suite rsync target `"$H:$O/"` resolved to `~/runs/...` instead of
   `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/...`. The image build+ship itself succeeded through the script
   before the rsync step failed.
2. **`dec8_staggered` failed in s01 (`rc=2`)** — `runs/pre-bench-20261121/dec8-audit/dec8_replay_audit.py` was not
   on the head; the suite folder never carried it. Synced at 03:01 (before s02 reached the cell), so s02–s05
   recorded cleanly; s01's cell was re-recorded on the base image at 04:36–04:38 UTC and merged into
   `blocks/s01.json` (3 waves, `rc=0`). Both ranks' waves then matched s01 3/3.
3. **`launch.sh`'s 0215 results rsync** targets a head directory that does not exist (`crossover-0215`); non-fatal,
   the capture itself had nothing to copy (item 4).
4. **The guarded 0215 capture stays INVALID** (`0/36`, `BUSY connections=None counters=(None,None)`): the harness
   needs a *live but idle* `/health`, and production was stopped for this window, which it reports as no server.
   `idx_fused.small_nbk` stays 6,144.
5. **Production rank 0 was already stopped when the window opened**: `docker kill` SIGKILL at 02:09:29 UTC (exit
   137), rank 1 exit 1, no kernel OOM lines; consistent with `./stop.sh` after the operator approved the restart.
   This session did not stop it.
6. `start.sh restart` returned 1 on three of the four restarts in this window with the known first-request smoke
   failure (decisions.md 2026-11-15); each time a retried smoke answered and the ranks were healthy.
7. The runner's production restore logs `env_match=False` while listing only the base image's own env
   (`CUDA_*`, `NCCL_*`, `PATH`, …) as missing; the TF_*/TENSORFOLD_* switches matched. Pre-existing, unexamined.

## Reproduce

```
bash runs/suite-fused-next2-20261121/preflight.sh        # fixed: build+ship, sync, detached launch
python3 -I runs/suite-fused-next2-20261121/gate.py       # 5 of 5 blocks pass
python3 -I runs/suite-fused-next2-20261121/table2.py
```

Artifacts: `blocks/` (per-block JSON, rank logs, exact records, validate), `runner.log`, `launch.log`,
`table2.txt`, `gate.txt`, `production.rank{0,1}.env`.
