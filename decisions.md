# Decisions

Short entries: what was decided, what else was considered, and why. Code references are patch numbers under
`zig/patches/`.

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
