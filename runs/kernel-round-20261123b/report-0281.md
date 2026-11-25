# report-0281 — TF_DS_MARKOV_CACHE

**Commit:** `b0715eb` on `kernel-stack-0280` (`~/Workspace/tfref-kstack5`, parent `cb8a4ca`).
**Subject:** `0281 dsv41: TF_DS_MARKOV_CACHE (unset: 256 rows): the drafter's cached Markov bias rows`
**Switch:** `TF_DS_MARKOV_CACHE=<rows>`: unset or `""` is 256 (today's); an integer 256..4096 otherwise. Anything else
prints `TF_DS_MARKOV_CACHE: expected a row count from 256 to 4096` to stderr and refuses startup
(`error.BadMarkovCache`, at `Ops` init).
**Nothing here ran on a GPU.**

## Contradiction with the brief: the kit has no ranking

The brief's first preference was the kit's own longer ranking. The served kit holds none. Read-only on the head:

- `/nfs1/tensorfold-dsv41/kit-abe07c7d/kit` (the container's `TF_DS_KIT`) holds `KIT-LICENSES.md LICENSES MANIFEST
  README.md aot cubins engram.json rope-*.f32 rope.json tools vision`. `tools/` is `make_bias_vl.py make_rope.py
  make_topk_sel.py verify_kit.sh`.
- `grep -rli markov` over the kit matches eight AOT cubins and nothing else (the `_bias`, `_score`, `_finish` and
  `_pick` variants). `MANIFEST` and `README.md` contain neither "markov" nor "tokens".
- No copy of the served build's `markov_tokens.py` or `tools/dsv41/markov_tokens_gen.py` exists under `~/Workspace`.

So the second preference applies: a frequency ranking built from general text.

**The served 256 cannot be regenerated either.** `markov_tokens.zig` says the list came from ExLlamaV3's calibration
text. Tokenizing all six of ExLlamaV3's current calibration files (repo HEAD `8c593e4`) with the served tokenizer and
ranking every subset by frequency gives at most 164 of the served 256 ids, and four served ids (`,U`, `variable`,
` LETTER`, ` LAT`) occur in none of the files. The served list is from a different sample (it reads like Python and
reST documentation). The first 256 are therefore kept as they are and the continuation is appended, not re-derived.

## The continuation (`markov_tokens.more`, 3,840 ids)

- **Source:** ExLlamaV3 `exllamav3/conversion/standard_cal_data` at `8c593e4` (MIT), fetched from GitHub; each file's
  git blob hash matched the repository listing. Tokenizer: the served checkpoint's `tokenizer.json`
  (sha256 `c90dfa01…476b`), copied read-only from the head.
- **Rule, fixed before any held-out number was read:** a token's score is half its relative frequency in `code.utf8`
  plus an eighth of its relative frequency in each of `c4`, `technical`, `tiny` and `wiki`; ties to the lower id; ids
  already in the served 256 left out; `multilingual.utf8` unused. No weight was tuned afterwards.
- **Generator and log:** `runs/kernel-round-20261123b/h0281/gen.py`, `run.log`, `ranked.json`, `more.zig.inc`.
  `ranked` sha256 (JSON list) `c72064989bac…457b`.
- The corpus `runs/overnight-20261122/corpus.json` was used only for the held-out check below.

| Rows | Held-out hit rate (145,210 corpus ids) | Source mass covered | Cache bytes a rank | Fill launches at load |
|---:|---:|---:|---:|---:|
| 256 (today) | 49.2% | 41.4% | 33,095,680 (31.6 MiB) | 16 |
| 512 | 54.4% | 53.7% | 66,191,360 | 32 |
| 1,024 | 61.3% | 62.4% | 132,382,720 | 64 |
| 2,048 | 67.7% | 71.1% | 264,765,440 (252.5 MiB) | 128 |
| 4,096 | 76.5% | 79.9% | 529,530,880 (505.0 MiB) | 256 |

The 256-row figure reproduces the audit's 49.2%. The audit's corpus-fitted lists hit 61 / 72 / 81 / 88%; this list is
7–13 points below them at every size, which is the overstatement the brief warned about.

## What a miss costs, and what the list is worth (measured cost, inferred saving)

The cost of a miss step is no longer unknown. `runs/trace-review-20261123/RESULTS.md` section 3 measured it from the
saved ring traces: a hit step takes 9.8 µs and a miss step 142–162 µs, cleanly bimodal, so a miss costs 133–153 µs
(one stream; ±7% between restarts and ranks). A six-step one-stream pass has 3.2 misses, 0.44–0.49 ms.

Scaling those 3.2 misses by this list's held-out miss rates (50.8% today, 32.3% at 2,048 rows, 23.5% at 4,096):

| Rows | Misses a six-step pass | Saved a pass at 133–153 µs a miss |
|---:|---:|---:|
| 256 | 3.2 | — |
| 2,048 | about 2.0 | about 0.16–0.18 ms |
| 4,096 | about 1.5 | about 0.23–0.26 ms |

That is an inference from natural-text hit rates, not a measurement; the review's own 0.27 / 0.33 ms figures used the
corpus-fitted rates.

**Four streams are bounded differently.** One head read serves every missing row of a step, so a step costs about
one miss (153–176 µs) however many of its rows miss, and today a four-stream step is slow 96–100% of the time. A
step is fast only when all four rows hit: at this list's rates about 21% of steps at 2,048 rows (0.677⁴) and 34% at
4,096 (0.765⁴). So the four-stream gain a pass is bounded by the share of all-hit steps, not by the hit rate.

## What changed

- `markov_tokens.zig`: `cached` untouched; `more`; `ranked = cached ++ more`; `rowOf` (binary search over a
  comptime-sorted table) replaces 0280's `has`.
- `markov_cache.zig` (new, pure): `env_name`, `parse` / `fromText`, `tokens(rows)`, `slots(table, rows)`, `hit`,
  `bytes`, `fits`, `inVocab`, `fillLaunches`, 4 tests.
- `ops.zig`: `Ops.markov_cache: ?usize` (null: unset).
- `draft.zig`: `Drafter.init` takes `k` from the switch; the token upload, the cache's arena take, the slot table and
  `Markov.fill` follow it; `Drafter.mk_rows`; the load banner.
- `model.zig`: the profile's hit/miss counter uses `markov_cache.hit(tok, dr.mk_rows)`.

Banner (once a rank at load, only when the variable is set):
`dsv41: TF_DS_MARKOV_CACHE=<N> (rank <r>): <N> Markov bias rows cached at load, <bytes> bytes (<MiB> MiB) a rank, <L> _bias launches`.

## The cap (4,096) and the other refusals

- 4,096 × 64,640 = 264,765,440 cache elements, an eighth of 2^31. The kit's `_score` takes `SLOT` as `*i32` and
  `c_row` as `i32` (read from the kit's `aot/aot.json`) and its index arithmetic is in a cubin, so `Drafter.init` also
  refuses any rows × columns above 2^29 (a byte offset stays under 2^30), whatever the split.
- 505 MiB a rank is already a real cost on these boxes (see "Memory" below).
- The ranking's last id occurs 3 times in 100,000 source tokens; rows past it would buy little.
- A ranked id outside the checkpoint's vocabulary refuses startup (`markov.py` keeps only ids within the vocabulary;
  the old code would have indexed past the table).

## Launches at the served shape

A pass: none added or removed at one stream or four. Every step still issues `_bias`, `_score`, `_finish`, the gather
and `_pick`; the switch changes how many of a step's rows `_bias` has to compute. At load: `fill` issues N / 16
`_bias` launches instead of 16.

## Why the bits cannot move, and the gate that checks it

Unset, `""` and `256`: the same int64 token list byte for byte, the same slot table, the same arena takes in the same
order, the same 16 fill launches. Larger: every served token keeps its row; a new row is written by the same `_bias`
kernel that would stage it during a step, from the same head and embedding. Output tokens cannot move whatever the
cache holds, because drafts only choose how many rows a round verifies.

**So the exact panel does not test this patch.** A wrong cache row would change drafts, and that shows only as a
different number of rounds or different accepted / drafted counts. The hardware gate must compare, for seeded
requests, **per-request rounds and accepted / drafted against control**, not only token hashes.

## Gate

- `zig build test`: exit 0. The log's last line is a `failed command: …/test …` label, which Zig 0.17's build
  runner prints with any test step that wrote to stderr (base `2bb052f` prints it too, exit 0). The family's test
  binary run directly ends `All 517 tests passed.` with exit 0; the log holds the 22 refusals of the parse test.
- `zig build`: exit 0.
- `cuda-syntax-stubs/check.sh`: the same 9 errors as base `2bb052f`, output identical after path normalization.
- The pure modules run standalone too: `zig test zig/src/families/dsv41/markov_cache.zig` ends `All 6 tests passed.`
- **The required proof:** test `unset and 256 give exactly today's token list and slot table` builds today's table
  the way `2bb052f` does (`-1`, then `cached[j] -> j` over 129,280 ids) and requires, for unset, `""` and `256`: the
  token list equal to `markov_tokens.cached` as bytes, the slot table equal element for element, 33,095,680 cache
  bytes and 16 fill launches. `git diff 2bb052f -- markov_tokens.zig` leaves the `cached` array's lines untouched.
- Also tested: a larger cache keeps every served token's row, has exactly N cached tokens, each row holding its own
  token, and the counter's `hit` agrees with the slot table for every id; `rowOf` over all 4,096 ids; bounds.

## Memory (read-only check on the pair, 2026-11-23)

`free -m` with production up: 6.9 GiB available on the head, 7.5 GiB on the worker, of 124.5 GiB. `scripts/config.sh`
defaults `MEMGUARD_GIB=2.5` (a rank's container is killed below it). A 4,096-row cache takes 473 MiB more than
today's a rank, 2,048 rows 221 MiB more. That fits, but it is a tenth of the margin; watch `MemAvailable` in the
`mc4096` arms.

## Out of scope, follow-up only

A per-reply LRU filled on a miss (audit estimate 66–73% with 256 extra rows). It needs device-side slot updates
inside the pass graph.

## What only hardware can settle

- The saving itself. The miss cost is measured (above), so `runs/negative-kernels-20261122/markov.py` no longer has
  to run before the arm; the arm goes straight to the pair. What the traces do not show is the other side:
- whether a cached row read from a 505 MiB table (DRAM, 126 KB a row, no longer L2-resident as 31.6 MiB may be) is
  as cheap as today's 9.8 µs hit step. If it is not, 4,096 rows can lose to 2,048.
- The kit `_score`'s behaviour at slots past 255: by its argument types it should index correctly, but no launch with
  a slot above 255 has ever run here.
- The served hit rate at 2,048 and 4,096 rows (the `markov_cache` / `pass_counts` lines give it per arm).
