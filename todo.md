# Follow-ups (out of scope of the current campaign)

## Prefix-cache hit-rate review (docs/prefix-cache-hitrate-review.md)

- Ranked hit-rate improvements (18 findings, all code-validated) await the operator's pick. Top candidates:
  shared-cut discovery past the first user message; `history_len/2` edit-recovery gate; eviction protection for
  shared/branch states in both layers; `SPILL_MIN_TOKENS` shared-cut exemption; strided spill retention;
  process-lifetime Engram row cache keyed `(layer, row id)` incl. the rounds' AIO path. Ref: the review's
  ranked list and its "Validated / not verified" section.
- Before implementing findings 3/7/8: capture the finding-12 counters (memory CacheStats, resume-depth) on the
  live pair as a baseline; most of the review's frequency claims are code-traced projections.
- `to-benchmark.md` Engram corrections (row-cache scope/keying/hook, both-ranks metric, image-pad context) —
  fold in when the matrix rows are next edited.
- Stale refs to fix when touched: `tools/prefix_fanout.py` cites missing `PORT-PREFIX-SHARING.md`; `model.zig`
  and patch `0012` cite missing `docs/port/dsv41-state.md`.

- Per-switch `exact.py` records (the doc's full identity gate). The signal pass gates each arm with
  `tools/validate.py --compare` only (operator decision: no deep testing). `runs/exact-oldimg.json` is the
  cross-image base record. Ref: `to-benchmark.md` "Before the first run".
- TRIM row (`TF_DS_ENGRAM_TRIM=1`): skipped (operator decision) - punches ~101 GB per Spark irreversibly
  (nlink=1 both sides). Run it when the disk is actually needed; tables stay whole meanwhile.
- **Engine bug: `TF_DS_ENGRAM_DEDUPE=1` + `TF_DS_ENGRAM_LEAD=1` breaks the rank pair during decode**
  (NCCL communicator aborted, rank 1's link closes; all decode streams fail; reproduced 2/2 runs). Find and
  fix in the shared dedupe/lead path; add a pair test. Ref: `runs/matrix/*/dedupe-lead.*`.
- `TF_DS_ENGRAM_LEAD=1` is the one reproducible prefill win (+5.0%, +6.2% two passes) - confirm with the strict
  3x base/arm alternation before claiming, then consider on by default.
- `TF_DS_ENGRAM_SCALES=map`: the cost is the page-fault path (cold -11%, warm -5%). The proposed `map-warm`
  row (`0099`+) decides keep/drop; do not enable map meanwhile.
- Proposed rows in `to-benchmark.md` (not implemented; need `zig/patches/0099`+): map-warm, decode threads,
  pool threads, CPU placement, row cache, read depth, host timing.
- "The quality set" is undefined in this repo (upstream author's 40-item set, PR #108). Sampled arms are
  scored on accept/tokens_per_round + greedy identity only. Port the upstream set if depth results need it.
- Strict 3x base/arm alternation for any toggle that lands within noise or is worth keeping (the current
  pass is one block per arm).
- Draft-depth pins break concurrent bit-equality (`concurrent == solo`) while keeping sequential identity -
  either document the caveat in the patches or find the shape-dependent path that changes sampled/batched
  greedy output under a pin.
