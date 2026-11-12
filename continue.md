# Continue

DeepSeek-V4.1-Flash on 2x DGX Spark (TensorFold engine, EXL3 TP2). Read `README.md` for the deployment,
`to-benchmark.md` for the benchmark matrix and its rules, `docs/measurements.md` for how the tools measure,
`decisions.md` for why things are as they are.

## Current state (forward-looking only)

- A signal-grade benchmark campaign of `to-benchmark.md`'s matrix is being run against the live pair
  (head gx10-eb0e / 100.95.49.21, worker gx10-235f / 10.50.0.11). Image `exl3-tp2-8dc2d276e617`
  (patches 0094-0098 Engram switches, 0103-0106 draft-depth pins). Per the operator: one bench block per
  arm (no 3x alternation), `tools/validate.py` as the per-arm identity gate (no per-switch `exact.py`),
  quality column not scored, TRIM skipped (destructive).
- Driving it: `just matrix-start` / `matrix-status` / `matrix-stop` / `matrix-pull` (Justfile), runner
  `scripts/bench_matrix.sh`, report `tools/matrix_report.py`. Cross-image identity: `runs/exact-oldimg.json`
  (old image record) vs the new image's `exact.py record --like` (run right after the rebuild).

## Next

1. Dedupe same-prompt probe (`scripts/dedupe_probe.sh`) in flight - patch its numbers into the "Dedupe
   same-prompt probe" table in `to-benchmark.md`, then commit everything with a `decisions.md` entry.
2. Prefix-cache hit-rate review is done (`docs/prefix-cache-hitrate-review.md`): 18 ranked, code-validated
   improvements. Pick which to implement (top: shared-cut discovery, edit-recovery gate, shared/branch
   eviction protection, `SPILL_MIN_TOKENS` exemption, strided spill retention, Engram row cache), and capture
   the review's finding-12 counter baseline on the pair before implementing 3/7/8. Details in `todo.md`.
3. See `todo.md` for the follow-ups: the DEDUPE+LEAD rank-pair crash fix, LEAD's alternation confirmation,
   `map-warm` for the scales mapping, per-switch `exact.py`, TRIM.
