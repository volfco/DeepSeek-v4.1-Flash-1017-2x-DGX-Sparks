# Decisions

Short entries: what was decided, what else was considered, and why. Code references are patch numbers under
`zig/patches/`.

## Hit-rate patches 0110-0116: scope, switches, and #4 as benchmark rows

- Scope: major (which review findings become engine patches and how they are gated).
- Considered: (a) top 6 findings + the metrics finding as patches (7); (b) top 6 only; (c) all 14
  implementable findings; (d) top 3 quick wins. #4 (`SPILL_MIN_TOKENS`): patch-unconditional vs always-on
  patch vs benchmark rows only. LOSSY sub-items (append-only compaction, image-row skip, canonical tool
  renderer): opt-in toggles vs skipped.
- Decision: (a) — `0110` shared-cut discovery, `0111` edit-recovery gate, `0112` shared/branch eviction
  protection, `0113` strided spill retention, `0114` Engram row cache, `0115` shared flag from the marks plan,
  `0116` cache metrics — each on a `TF_DS_*` switch. Behaviour switches off by default (the series' rule:
  unset behaves like the pinned engine); `0116` follows `0069` (on, `=0` off). #4 lands in `to-benchmark.md`
  as measurement rows only (measure before changing a default). All LOSSY sub-items skipped (stay in the
  review as proposals).
- Why: the seven carry the review's ranked impact and every one is bit-safe (kept states are exact copies;
  resume is bit-exact by construction); the metrics patch is the measuring stick for the other six. #4 changes
  on-disk behaviour for every deployment, so it gets the matrix's measure-first rule instead. Patches are
  built as one ordered chain over the deployed series (`fbbc9713` + 0001-0098 incl. the working-tree `0097`
  + 0103-0106), exported as consecutive diffs so they apply in filename order; host tests per patch plus a
  clean-tree sequential apply + `zig build test` validation. Implementation by parallel subagent workers in
  isolated worktrees (`0112`/`0115` flagged: adjacent `keepWith` hunks), independent review + validation
  subagents after integration.

## Prefix-cache hit-rate review: scope and deliverable

- Scope: major (what the review covers and what it produces).
- Considered: (a) prompt/prefix cache only; (b) prompt cache plus Engram row reads (Engram de-duplicated against
  `to-benchmark.md`); (c) every caching layer incl. CUDA-graph and page cache. Deliverable: report only vs report
  plus implementation. Workload: agents only vs agents + long chats vs benchmark matrix.
- Decision: (b), report only, optimized for agents + long chats up to the 1M window (incl. eviction/restart).
  Engram findings must not duplicate `to-benchmark.md`'s matrix; only gaps or corrections with code evidence.
- Why: the prompt cache is what the hit-rate metrics (`prompt_tokens_cached_total`, `usage.prompt_tokens_details`)
  measure, and Engram's read volume is directly driven by prompt-cache misses, so both belong in one review; report
  only keeps the running benchmark campaign untouched and leaves implementation picks to the operator. Engine code
  read at `/tmp/tf-engine` (`fbbc9713` + patches 0001-0098, commit `f72c1573`), deployed diff per `zig/patches/`.

## 0097 fix: DecodePool.loop called `work` as a method

- Scope: minor (build fix; no bits, no behavior change).
- Considered: (a) fix `0097-engram-fast.patch` in place; (b) a follow-up patch on top of the series.
- Decision: (a), `p.work(j)` -> `work(j)` in `DecodePool.loop`.
- Why: `work(j: *Job)` is a plain function of the pool's namespace (the sibling call in `decode` already calls it
  bare), so method syntax passed the pool where a Job belongs and the server build failed at `engram_io.zig:414`.
  The series is pre-pair and pre-PR and the image hash changes either way, so a fixing patch would only carry a
  broken intermediate. Found by the first full build of the 0094-0098/0103-0106 series (host tests do not build
  the server).

## Benchmark evidence tools (scripts/bench_matrix.sh, tools/obs_arm_metrics.py, scripts/dedupe_probe.sh)

- Scope: minor (measurement only; no engine behavior).
- Considered: (a) trust bench.py's `--metrics` snapshots for the per-arm engram histogram; (b) scrape
  `tensorfold:engram_lookup_seconds` per arm around the bench block; (c) take per-arm counts from the lab's
  OpenObserve store (60 s scrapes).
- Decision: (b) as the table's source, (c) as a cross-check, (a) rejected. The same-prompt dedupe probe flushes
  the prompt cache's disk state (the spill tier's `rank*/points` trees) between legs and prints `cached_tokens`.
- Why: bench.py truncates raw snapshots at 20k chars, past which the engram series sit; OpenObserve's scrape
  cadence is too coarse for ~4 min arms and cannot see a crash's last counters. The first probe attempt measured
  the prompt cache instead of the switch (114 -> 12 on warm cache vs 114 -> 114 cold): same-salt A/B is only
  valid with the disk cache flushed, which the probe now enforces and asserts.

## Benchmark campaign protocol (tools/bench_matrix.sh)

- Scope: major (what the to-benchmark.md matrix measures and how).
- Considered: (a) the doc's full protocol: medians of 3 alternating base/arm blocks and a per-switch `exact.py`
  record before each row; (b) one bench block per arm (reps 3), `tools/validate.py --compare` as the per-arm
  identity gate, base blocks bookending the phases; (c) single-rep triage only.
- Decision: (b), operator's call: "do not perform deep testing, just enough to get a signal on if the toggle has a
  net positive or negative effect". Sampled arms are scored on accept/tokens_per_round and greedy identity (the
  doc's "quality set" is undefined in this repo), and TRIM is skipped (it punches ~101 GB per Spark irreversibly).
- Why: (a) triples wall-clock (every switch needs a container restart) and its per-switch exact records add little
  to a sign-a-toggle +/- read; (b) keeps one variable per row and one drift bookend per phase. Deferred work is in
  `todo.md`; the cross-image identity gate is one `exact.py` record pair (old image `runs/exact-oldimg.json` vs the
  new image `--like`).

## 0094 Engram trim: punch the unread rows out of each rank's table

- Scope: minor (disk footprint; no speed, no bits).
- Considered: (a) a per-rank repacked table with rebased hash offsets (halves the file but needs a tool, a loader change
  and a new fingerprint); (b) punch holes in place (FALLOC_FL_PUNCH_HOLE) for the row ranges a rank never reads; (c) do
  nothing.
- Decision: (b), opt-in with `TF_DS_ENGRAM_TRIM=1`, and only when the file has one link.
- Why: each rank reads only its own 12 of 24 hash buckets, so ~101 GB a Spark is dead weight, and (b) frees it without
  touching the layout, the offsets, the sizes or the ranks' fingerprint. (a) buys the same bytes for a table tool and a
  loader rebase; (b) needs neither. The nlink guard keeps a hardlinked `HEAD_ENGRAM_DIR` from emptying the HF cache copy
  that a later `VERIFY_SHA` checks.

## 0095 Engram scales from a mapping, not a second I/O

- Scope: minor (read count; no bits).
- Considered: (a) keep the scale tensor resident in RAM (3.07 GB a rank); (b) map it read-only and copy rows from the page
  cache; (c) interleave weight and scale rows in a repacked table (one span a row); (d) do nothing.
- Decision: (b), opt-in with `TF_DS_ENGRAM_SCALES=map`.
- Why: (a) is 3.07 GB of anonymous memory the head cannot spare (its worst point is ~3.1 GiB above the memguard floor);
  (b) is clean page cache, which the guard counts as available and the kernel reclaims, and its worst case is the current
  4 KiB read. (c) buys the same I/O count but needs a 203 GB repack and a new identity; it can be added later on top of
  (b) without changing the switch.

## 0096 Engram dedupe within a prompt chunk

- Scope: minor (read count; no bits).
- Considered: (a) dedupe a chunk's row ids in the read-ahead and lead readers; (b) a decoded-row LRU across chunks and
  requests; (c) do nothing.
- Decision: (a), opt-in with `TF_DS_ENGRAM_DEDUPE=1`; the round path and the main thread's fallback keep the old path.
- Why: repeated n-grams are common in prose and code, and the same id is the same bytes, so dedupe is exact and cheap.
  (b) would also catch repeats across chunks but needs eviction and locking, and a shared scratch with a round's reads
  would race; the prompt cache already skips cached prefixes, which is where most cross-chunk repeats live.

## 0097 Engram host-side economies

- Scope: minor (CPU and hash start latency; no bits).
- Considered: (a) leave the host path alone; (b) reciprocal modulo, a persistent decode pool, the fallback's hash reuse
  and a ring-slot knob, behind one switch; (c) do only the modulo.
- Decision: (b), opt-in with `TF_DS_ENGRAM_FAST=1` (and `TF_DS_ENGRAM_AIO_SLOTS`).
- Why: each is expected under 1 % of a chunk, so the value is in being able to measure them; the reciprocal is tested
  against `@mod` for every prime and 5,000 values each, and every item is bit-identical, so `FAST=0` is the old engine.

## 0098 Strict Engram table validation

- Scope: minor (refusals; no bits on a good checkpoint).
- Considered: (a) make the dtype and scale-row checks unconditional; (b) refuse behind a switch and warn without it; (c)
  warn only.
- Decision: (b), `TF_DS_ENGRAM_STRICT=1`, plus an ungated bounds check on the token map.
- Why: today's checkpoint passes both checks, so an unconditional refusal would only be exercised by a future checkpoint;
  behind a switch it can be A/B'd against the warning path. The token-map bound is a no-op for every id the map was made
  for, so it needs no switch.
