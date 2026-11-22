# Fused-Bench Harness Audit (`runs/fused-next-20261121` & `runs/suite-fused-next-20261121`)

## Scope & Summary

Scanned all **98** JSON files across:
- `runs/fused-next-20261121/**/*.json` (**73** files)
- `runs/suite-fused-next-20261121/**/*.json` (**25** files)

### Key Findings

1. **`newnf`-is-actually-`floor` arm-label bug (`len(chains) == 3`, `len(names) == 4`):**
   - **Root cause:** In `tools/fused_bench/bench.py` (introduced in `856b5df3` and extended in `c85cd904`), four callers — `case_idx`, `case_compress_prompt`, `case_rope`, and `case_prompt_source` — built **3** arm chains (`ref`, `old`, `new`) plus **1** `empty_kernel` launch-floor graph (`4` `exes` total), while passing `names = ["ref", "old", "new", "newnf"]` (`5` names once `"floor"` was appended). `zip(names, samples)` bound `samples[3]` (the launch floor) to `"newnf"` and truncated `"floor"`.
   - **Impact on saved JSONs:** Exactly **10 files** and **100 timed kernel rows** (all in `runs/fused-next-20261121/`, **0** in `runs/suite-fused-next-20261121/`) carry `"newnf"` with `"floor"` missing. In all 100 rows, `ref`, `old`, and `new` medians still correspond to `chains[0..2]`, while `newnf` (`0.455–7.219 us`) is the empty-kernel launch floor; `tools/fused_bench/report.py` silently skipped any such row because `"floor" not in t`.
   - **Unaffected baselines & latent callers:** `runs/fused-next-20261121/wp0/wp0.json` predates `856b5df3` and has `"floor"` on all 48 non-`kit-scorekeys` timed rows. `case_compress_prompt` and `case_prompt_source` were not timed in post-`856b5df3` saved JSONs, so zero saved `0197-compress-prompt` rows hit the bug in disk artifacts, though both callers had the bug in `bench.py` and are now fixed. `case_idx_fold` and `case_prune` genuinely construct 4 chains (`len(chains) == 4 == len(names)`).

2. **`timing_clean` coverage across the 98 JSON files:**
   - **29** guarded `bench.py` JSON files include `health.timing_clean` (`True` on all 29), and every one of the 29 has valid non-null integer `before` and `after` token counters (`prompt_tokens_total`, `generation_tokens_total`) with `before == after` and `timing_retries == 0` (none hit the `(None, None) == (None, None)` missing-counter loophole).
   - **32** microbenchmark / harness JSON files (`25` `bench`/`vbench` `results` files in `wp2/` and `wp3/`, `6` `chain.py` `rows` files in `wp4/`, and `1` `bw.py` `rows` file in `wp0/`) omit `health` and therefore lack `timing_clean` (`26` of these contain timed rows; `6` are parity/canary-only).
   - **37** non-microbenchmark JSON files (`15` static SASS/occupancy dumps and `22` end-to-end serving-suite block/schedule/restore artifacts in `runs/suite-fused-next-20261121/`) use separate schemas that do not carry `bench.py`'s `health.timing_clean` field.

---

## 1. Files Affected by the `'newnf'`-is-Actually-`'floor'` Bug (10 Files, 100 Rows)

### Summary by File

| File | Total Rows | Timed Rows | Affected Rows (`newnf` = `floor`) | Affected Kernel Tags |
|---|---:|---:|---:|---|
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 15 | 14 | 14 | `0195-idx` |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 21 | 14 | 14 | `0195-idx` |
| `runs/fused-next-20261121/wp3/final-timing.json` | 28 | 22 | 18 | `0198-key`, `0198-latent`, `0198-prompt-key`, `0198-prompt-latent`, `0198-query` |
| `runs/fused-next-20261121/wp3/i0.json` | 7 | 6 | 6 | `0198-query` |
| `runs/fused-next-20261121/wp3/i1b.json` | 17 | 12 | 12 | `0198-key`, `0198-latent`, `0198-prompt-key`, `0198-prompt-latent` |
| `runs/fused-next-20261121/wp3/i2.json` | 16 | 12 | 12 | `0198-key`, `0198-latent`, `0198-query` |
| `runs/fused-next-20261121/wp3/i2b.json` | 16 | 12 | 12 | `0198-key`, `0198-latent`, `0198-query` |
| `runs/fused-next-20261121/wp3/i2c.json` | 6 | 4 | 4 | `0198-prompt-key`, `0198-prompt-latent` |
| `runs/fused-next-20261121/wp3/i4.json` | 10 | 8 | 4 | `0198-latent` |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 30 | 12 | 4 | `0198-key` |

*(Note: In `wp3/final-timing.json` and `wp3/i4.json`, the 4 unaffected timed rows are `0198-latent` `latent-owner` shapes emitted by `case_latent_owner`, which passed 3 names and retained `floor`. In `wp4/wp4-i2-time-e.json`, the 8 unaffected timed rows are `0197-compress-decode` shapes with 4 chains `ref, old, new, newdup` + `floor`.)*

### Exhaustive Per-Case Listing (All 100 Affected Rows)

| File | Row # | Kernel | Shape | `ref` (us) | `old` (us) | `new` (us) | Mislabeled `newnf` = actual `floor` (us) |
|---|---:|---|---|---:|---:|---:|---:|
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 0 | `0195-idx` | `idx R=1 nb=8 width=61 k=8 list ties` | 20.284 | 1.869 | 2.078 | 0.478 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 1 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list ties` | 26.610 | 2.014 | 2.218 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 2 | `0195-idx` | `idx R=32 nb=8 width=61 k=8 list ties` | 26.611 | 2.033 | 2.240 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 3 | `0195-idx` | `idx R=48 nb=8 width=61 k=8 list ties` | 26.649 | 2.033 | 2.256 | 0.497 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 4 | `0195-idx` | `idx R=7 nb=4096 width=32765 k=2048 list random` | 44.704 | 10.123 | 8.424 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 5 | `0195-idx` | `idx R=32 nb=8192 width=65533 k=2048 list ties` | 61.733 | 20.505 | 17.148 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 6 | `0195-idx` | `idx R=128 nb=4096 width=32765 k=2048 mask ties` | 158.994 | 36.459 | 34.228 | 0.825 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 7 | `0195-idx` | `idx R=1024 nb=4096 width=32765 k=2048 mask ties` | 1771.178 | 580.572 | 583.426 | 3.854 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 8 | `0195-idx` | `idx R=2048 nb=4096 width=32765 k=2048 mask ties` | 3566.192 | 1127.808 | 1132.619 | 7.215 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 9 | `0195-idx` | `idx R=7 nb=2049 width=16389 k=2048 list random` | 44.377 | 6.417 | 5.557 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 10 | `0195-idx` | `idx R=7 nb=8192 width=65533 k=2048 mask flat` | 46.773 | 19.237 | 15.367 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 11 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list random` | 21.920 | 2.034 | 2.240 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 12 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list flat` | 15.592 | 2.034 | 2.220 | 0.497 |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 13 | `0195-idx` | `idx R=128 nb=4096 width=32765 k=2048 mask random` | 132.055 | 28.405 | 25.872 | 0.897 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 0 | `0195-idx` | `idx R=1 nb=8 width=61 k=8 list ties` | 20.282 | 1.870 | 2.179 | 0.477 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 1 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list ties` | 26.612 | 2.013 | 2.300 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 2 | `0195-idx` | `idx R=32 nb=8 width=61 k=8 list ties` | 26.631 | 2.034 | 2.341 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 3 | `0195-idx` | `idx R=48 nb=8 width=61 k=8 list ties` | 26.650 | 2.033 | 2.341 | 0.497 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 4 | `0195-idx` | `idx R=7 nb=4096 width=32765 k=2048 list random` | 44.717 | 10.124 | 8.464 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 5 | `0195-idx` | `idx R=32 nb=8192 width=65533 k=2048 list ties` | 61.760 | 20.513 | 17.106 | 0.497 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 6 | `0195-idx` | `idx R=128 nb=4096 width=32765 k=2048 mask ties` | 162.238 | 36.564 | 29.791 | 0.823 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 7 | `0195-idx` | `idx R=1024 nb=4096 width=32765 k=2048 mask ties` | 1793.373 | 593.670 | 590.199 | 3.854 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 8 | `0195-idx` | `idx R=2048 nb=4096 width=32765 k=2048 mask ties` | 3556.043 | 1147.762 | 1148.337 | 7.219 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 9 | `0195-idx` | `idx R=7 nb=2049 width=16389 k=2048 list random` | 44.400 | 6.438 | 5.577 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 10 | `0195-idx` | `idx R=7 nb=8192 width=65533 k=2048 mask flat` | 46.792 | 19.236 | 13.422 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 11 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list random` | 21.879 | 2.014 | 2.318 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 12 | `0195-idx` | `idx R=7 nb=8 width=61 k=8 list flat` | 15.593 | 2.032 | 2.321 | 0.498 |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 13 | `0195-idx` | `idx R=128 nb=4096 width=32765 k=2048 mask random` | 131.192 | 28.331 | 23.252 | 0.881 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 0 | `0198-key` | `rope key R=1 D=128` | 2.955 | 1.788 | 1.626 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 1 | `0198-key` | `rope key R=7 D=128` | 2.934 | 1.767 | 1.607 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 2 | `0198-key` | `rope key R=32 D=128` | 2.935 | 1.771 | 1.623 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 3 | `0198-key` | `rope key R=48 D=128` | 2.977 | 1.808 | 1.647 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 5 | `0198-latent` | `rope latent R=1 D=512` | 2.076 | 1.540 | 1.401 | 0.455 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 6 | `0198-latent` | `rope latent R=7 D=512` | 2.083 | 1.544 | 1.424 | 0.473 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 7 | `0198-latent` | `rope latent R=32 D=512` | 2.074 | 1.522 | 1.399 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 8 | `0198-latent` | `rope latent R=48 D=512` | 2.076 | 1.543 | 1.417 | 0.457 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 10 | `0198-query` | `rope query R=1 D=128` | 2.141 | 1.442 | 1.336 | 0.456 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 11 | `0198-query` | `rope query R=7 D=128` | 2.177 | 1.563 | 1.405 | 0.516 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 12 | `0198-query` | `rope query R=32 D=128` | 2.710 | 2.975 | 2.587 | 0.498 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 13 | `0198-query` | `rope query R=48 D=128` | 2.996 | 4.020 | 2.812 | 0.517 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 14 | `0198-query` | `rope query R=128 D=128` | 5.024 | 9.242 | 5.801 | 0.703 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 15 | `0198-query` | `rope query R=1024 D=128` | 51.767 | 68.900 | 36.942 | 2.747 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 17 | `0198-prompt-key` | `rope prompt key R=128 D=128` | 7.242 | 4.182 | 1.912 | 0.497 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 18 | `0198-prompt-key` | `rope prompt key R=1024 D=128` | 9.272 | 6.181 | 3.086 | 1.110 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 20 | `0198-prompt-latent` | `rope prompt latent R=128 D=512` | 6.179 | 4.209 | 1.664 | 0.497 |
| `runs/fused-next-20261121/wp3/final-timing.json` | 21 | `0198-prompt-latent` | `rope prompt latent R=1024 D=512` | 10.848 | 10.268 | 6.393 | 0.993 |
| `runs/fused-next-20261121/wp3/i0.json` | 0 | `0198-query` | `rope query R=1 D=128` | 2.137 | 1.440 | 1.338 | 0.459 |
| `runs/fused-next-20261121/wp3/i0.json` | 1 | `0198-query` | `rope query R=7 D=128` | 2.158 | 1.584 | 1.421 | 0.519 |
| `runs/fused-next-20261121/wp3/i0.json` | 2 | `0198-query` | `rope query R=32 D=128` | 2.711 | 2.981 | 2.607 | 0.498 |
| `runs/fused-next-20261121/wp3/i0.json` | 3 | `0198-query` | `rope query R=48 D=128` | 3.015 | 4.020 | 2.794 | 0.518 |
| `runs/fused-next-20261121/wp3/i0.json` | 4 | `0198-query` | `rope query R=128 D=128` | 4.963 | 9.202 | 5.782 | 0.704 |
| `runs/fused-next-20261121/wp3/i0.json` | 5 | `0198-query` | `rope query R=1024 D=128` | 51.631 | 68.603 | 37.319 | 2.734 |
| `runs/fused-next-20261121/wp3/i1b.json` | 0 | `0198-key` | `rope key R=1 D=128` | 2.931 | 1.769 | 1.623 | 0.458 |
| `runs/fused-next-20261121/wp3/i1b.json` | 1 | `0198-key` | `rope key R=7 D=128` | 2.915 | 1.768 | 1.624 | 0.458 |
| `runs/fused-next-20261121/wp3/i1b.json` | 2 | `0198-key` | `rope key R=32 D=128` | 2.936 | 1.768 | 1.625 | 0.458 |
| `runs/fused-next-20261121/wp3/i1b.json` | 3 | `0198-key` | `rope key R=48 D=128` | 2.956 | 1.788 | 1.645 | 0.457 |
| `runs/fused-next-20261121/wp3/i1b.json` | 5 | `0198-latent` | `rope latent R=1 D=512` | 2.056 | 1.543 | 1.399 | 0.458 |
| `runs/fused-next-20261121/wp3/i1b.json` | 6 | `0198-latent` | `rope latent R=7 D=512` | 2.054 | 1.523 | 1.399 | 0.459 |
| `runs/fused-next-20261121/wp3/i1b.json` | 7 | `0198-latent` | `rope latent R=32 D=512` | 2.075 | 1.523 | 1.399 | 0.458 |
| `runs/fused-next-20261121/wp3/i1b.json` | 8 | `0198-latent` | `rope latent R=48 D=512` | 2.058 | 1.542 | 1.399 | 0.457 |
| `runs/fused-next-20261121/wp3/i1b.json` | 10 | `0198-prompt-key` | `rope prompt key R=128 D=128` | 7.369 | 4.191 | 1.930 | 0.499 |
| `runs/fused-next-20261121/wp3/i1b.json` | 11 | `0198-prompt-key` | `rope prompt key R=1024 D=128` | 9.255 | 6.181 | 3.071 | 1.112 |
| `runs/fused-next-20261121/wp3/i1b.json` | 13 | `0198-prompt-latent` | `rope prompt latent R=128 D=512` | 6.179 | 4.194 | 1.644 | 0.497 |
| `runs/fused-next-20261121/wp3/i1b.json` | 14 | `0198-prompt-latent` | `rope prompt latent R=1024 D=512` | 10.606 | 10.262 | 6.421 | 0.989 |
| `runs/fused-next-20261121/wp3/i2.json` | 0 | `0198-key` | `rope key R=1 D=128` | 2.954 | 1.784 | 1.623 | 0.457 |
| `runs/fused-next-20261121/wp3/i2.json` | 1 | `0198-key` | `rope key R=7 D=128` | 2.935 | 1.768 | 1.608 | 0.458 |
| `runs/fused-next-20261121/wp3/i2.json` | 2 | `0198-key` | `rope key R=32 D=128` | 2.934 | 1.771 | 1.624 | 0.457 |
| `runs/fused-next-20261121/wp3/i2.json` | 3 | `0198-key` | `rope key R=48 D=128` | 2.976 | 1.790 | 1.664 | 0.456 |
| `runs/fused-next-20261121/wp3/i2.json` | 5 | `0198-latent` | `rope latent R=1 D=512` | 2.078 | 1.541 | 1.644 | 0.457 |
| `runs/fused-next-20261121/wp3/i2.json` | 6 | `0198-latent` | `rope latent R=7 D=512` | 2.075 | 1.542 | 1.645 | 0.458 |
| `runs/fused-next-20261121/wp3/i2.json` | 7 | `0198-latent` | `rope latent R=32 D=512` | 2.074 | 1.542 | 1.645 | 0.459 |
| `runs/fused-next-20261121/wp3/i2.json` | 8 | `0198-latent` | `rope latent R=48 D=512` | 2.073 | 1.542 | 1.648 | 0.457 |
| `runs/fused-next-20261121/wp3/i2.json` | 10 | `0198-query` | `rope query R=1 D=128` | 2.156 | 1.440 | 1.358 | 0.457 |
| `runs/fused-next-20261121/wp3/i2.json` | 11 | `0198-query` | `rope query R=7 D=128` | 2.157 | 1.583 | 1.501 | 0.519 |
| `runs/fused-next-20261121/wp3/i2.json` | 12 | `0198-query` | `rope query R=32 D=128` | 2.729 | 2.976 | 2.853 | 0.928 |
| `runs/fused-next-20261121/wp3/i2.json` | 13 | `0198-query` | `rope query R=48 D=128` | 2.997 | 4.020 | 3.794 | 1.214 |
| `runs/fused-next-20261121/wp3/i2b.json` | 0 | `0198-key` | `rope key R=1 D=128` | 2.954 | 1.772 | 1.623 | 0.457 |
| `runs/fused-next-20261121/wp3/i2b.json` | 1 | `0198-key` | `rope key R=7 D=128` | 2.935 | 1.768 | 1.624 | 0.457 |
| `runs/fused-next-20261121/wp3/i2b.json` | 2 | `0198-key` | `rope key R=32 D=128` | 2.935 | 1.771 | 1.628 | 0.456 |
| `runs/fused-next-20261121/wp3/i2b.json` | 3 | `0198-key` | `rope key R=48 D=128` | 2.957 | 1.788 | 1.645 | 0.457 |
| `runs/fused-next-20261121/wp3/i2b.json` | 5 | `0198-latent` | `rope latent R=1 D=512` | 2.055 | 1.542 | 1.543 | 0.458 |
| `runs/fused-next-20261121/wp3/i2b.json` | 6 | `0198-latent` | `rope latent R=7 D=512` | 2.053 | 1.522 | 1.526 | 0.459 |
| `runs/fused-next-20261121/wp3/i2b.json` | 7 | `0198-latent` | `rope latent R=32 D=512` | 2.092 | 1.524 | 1.541 | 0.458 |
| `runs/fused-next-20261121/wp3/i2b.json` | 8 | `0198-latent` | `rope latent R=48 D=512` | 2.056 | 1.523 | 1.542 | 0.458 |
| `runs/fused-next-20261121/wp3/i2b.json` | 10 | `0198-query` | `rope query R=1 D=128` | 2.136 | 1.440 | 1.379 | 0.458 |
| `runs/fused-next-20261121/wp3/i2b.json` | 11 | `0198-query` | `rope query R=7 D=128` | 2.178 | 1.563 | 1.481 | 0.518 |
| `runs/fused-next-20261121/wp3/i2b.json` | 12 | `0198-query` | `rope query R=32 D=128` | 2.712 | 2.957 | 2.853 | 0.928 |
| `runs/fused-next-20261121/wp3/i2b.json` | 13 | `0198-query` | `rope query R=48 D=128` | 2.998 | 4.041 | 3.796 | 1.215 |
| `runs/fused-next-20261121/wp3/i2c.json` | 0 | `0198-prompt-key` | `rope prompt key R=128 D=128` | 7.278 | 4.270 | 1.909 | 0.496 |
| `runs/fused-next-20261121/wp3/i2c.json` | 1 | `0198-prompt-key` | `rope prompt key R=1024 D=128` | 9.259 | 6.174 | 3.087 | 1.112 |
| `runs/fused-next-20261121/wp3/i2c.json` | 3 | `0198-prompt-latent` | `rope prompt latent R=128 D=512` | 6.180 | 4.327 | 1.711 | 0.496 |
| `runs/fused-next-20261121/wp3/i2c.json` | 4 | `0198-prompt-latent` | `rope prompt latent R=1024 D=512` | 10.708 | 10.267 | 6.698 | 0.995 |
| `runs/fused-next-20261121/wp3/i4.json` | 0 | `0198-latent` | `rope latent R=1 D=512` | 2.046 | 1.521 | 1.563 | 0.457 |
| `runs/fused-next-20261121/wp3/i4.json` | 1 | `0198-latent` | `rope latent R=7 D=512` | 2.074 | 1.542 | 1.583 | 0.459 |
| `runs/fused-next-20261121/wp3/i4.json` | 2 | `0198-latent` | `rope latent R=32 D=512` | 2.073 | 1.543 | 1.565 | 0.458 |
| `runs/fused-next-20261121/wp3/i4.json` | 3 | `0198-latent` | `rope latent R=48 D=512` | 2.075 | 1.543 | 1.584 | 0.457 |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 25 | `0198-key` | `rope key R=1 D=128` | 2.909 | 1.621 | 1.583 | 0.456 |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 26 | `0198-key` | `rope key R=7 D=128` | 2.936 | 1.625 | 1.583 | 0.457 |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 27 | `0198-key` | `rope key R=32 D=128` | 2.955 | 1.644 | 1.604 | 0.457 |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 28 | `0198-key` | `rope key R=48 D=128` | 2.954 | 1.645 | 1.607 | 0.455 |

---

## 2. Files Missing `timing_clean`

### 2A. Microbenchmark / Harness JSONs Missing `health.timing_clean` (32 Files)

These files were produced by variant scripts (`vbench.py`), `chain.py`, `bw.py`, or earlier/unguarded `bench.py` invocations that omitted top-level `health` (or `health.timing_clean`).

| File | Producer / Schema | Total Rows | Timed Rows | Reason `timing_clean` Missing |
|---|---|---:|---:|---|
| `runs/fused-next-20261121/wp0/bw.json` | `bw.py` (`rows`) | 4 | 4 | `bw.py` omits `health` |
| `runs/fused-next-20261121/wp2/cmp128.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/floor1.json` | `bench`/`vbench` (`results`) | 11 | 5 | `health` key absent |
| `runs/fused-next-20261121/wp2/n128.json` | `bench`/`vbench` (`results`) | 26 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/n2.json` | `bench`/`vbench` (`results`) | 14 | 8 | `health` key absent |
| `runs/fused-next-20261121/wp2/par-new.json` | `bench`/`vbench` (`results`) | 46 | 0 | `health` key absent (parity-only) |
| `runs/fused-next-20261121/wp2/parity1.json` | `bench`/`vbench` (`results`) | 47 | 0 | `health` key absent (parity-only) |
| `runs/fused-next-20261121/wp2/parity2.json` | `bench`/`vbench` (`results`) | 46 | 0 | `health` key absent (parity-only) |
| `runs/fused-next-20261121/wp2/sweep512.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/t1.json` | `bench`/`vbench` (`results`) | 14 | 8 | `health` key absent |
| `runs/fused-next-20261121/wp2/t128.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/t128b.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/t128c.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/t512.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/t512b.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/tt1.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/tt2.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/tt3.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/win128.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/win2.json` | `bench`/`vbench` (`results`) | 10 | 0 | `health` key absent (parity-only) |
| `runs/fused-next-20261121/wp2/win3.json` | `bench`/`vbench` (`results`) | 10 | 0 | `health` key absent (parity-only) |
| `runs/fused-next-20261121/wp2/win4.json` | `bench`/`vbench` (`results`) | 10 | 4 | `health` key absent |
| `runs/fused-next-20261121/wp2/win5.json` | `bench`/`vbench` (`results`) | 9 | 3 | `health` key absent |
| `runs/fused-next-20261121/wp3/i3a.json` | `vbench.py` (`results`) | 7 | 4 | `vbench.py` omits `health` |
| `runs/fused-next-20261121/wp3/i3b.json` | `vbench.py` (`results`) | 7 | 4 | `vbench.py` omits `health` |
| `runs/fused-next-20261121/wp3/q7.json` | `vbench.py` (`results`) | 6 | 3 | `vbench.py` omits `health` |
| `runs/fused-next-20261121/wp4/wp4-a.json` | `chain.py` (`rows`) | 7 | 7 | `chain.py` omits `health` |
| `runs/fused-next-20261121/wp4/wp4-b.json` | `chain.py` (`rows`) | 7 | 7 | `chain.py` omits `health` |
| `runs/fused-next-20261121/wp4/wp4-canary-e.json` | `chain.py` (`rows`) | 1 | 0 | `chain.py` omits `health` (canary-only) |
| `runs/fused-next-20261121/wp4/wp4-chain.json` | `chain.py` (`rows`) | 7 | 7 | `chain.py` omits `health` |
| `runs/fused-next-20261121/wp4/wp4-d.json` | `chain.py` (`rows`) | 7 | 7 | `chain.py` omits `health` |
| `runs/fused-next-20261121/wp4/wp4-e.json` | `chain.py` (`rows`) | 7 | 7 | `chain.py` omits `health` |

### 2B. Non-`bench.py` JSON Artifacts Without `timing_clean` (37 Files)

Listed for completeness over the full `**/*.json` glob; these are static SASS/occupancy outputs (`15` files) and end-to-end serving-suite outputs (`22` files) rather than `fused_bench` timing JSONs:

| File | Artifact Type | Top-Level Shape / Keys |
|---|---|---|
| `runs/fused-next-20261121/review/b1-green/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/review/b1-red/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/review/fix-parity/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/review/wp4tip-self-parity/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp0/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp1/sass-fold.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp1/sass2.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp1/sass4.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp1/sass5.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp2/occ.json` | SASS / occupancy static analysis | `list[8]` |
| `runs/fused-next-20261121/wp2/sass-new.json` | SASS / occupancy static analysis | `tf_ds_compress2x_norm128_kernel, tf_ds_compress2x_norm512_kernel` |
| `runs/fused-next-20261121/wp2/sass-old.json` | SASS / occupancy static analysis | `tf_ds_compress2x_norm_kernel` |
| `runs/fused-next-20261121/wp3/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/fused-next-20261121/wp4/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/suite-fused-next-20261121/kernel-review/sass.json` | SASS / occupancy static analysis | `kernels, limits` |
| `runs/suite-fused-next-20261121/blocks/s01.exact.json` | End-to-end serving suite artifact | `cases, checks, meta` |
| `runs/suite-fused-next-20261121/blocks/s01.json` | End-to-end serving suite artifact | `arm, arm_proof_problems, banners, cell_s, cells...` |
| `runs/suite-fused-next-20261121/blocks/s01.probe.json` | End-to-end serving suite artifact | `list[11]` |
| `runs/suite-fused-next-20261121/blocks/s01.validate.json` | End-to-end serving suite artifact | `list[8]` |
| `runs/suite-fused-next-20261121/blocks/s02.exact.json` | End-to-end serving suite artifact | `cases, checks, meta` |
| `runs/suite-fused-next-20261121/blocks/s02.json` | End-to-end serving suite artifact | `arm, arm_proof_problems, banners, cell_s, cells...` |
| `runs/suite-fused-next-20261121/blocks/s02.probe.json` | End-to-end serving suite artifact | `list[11]` |
| `runs/suite-fused-next-20261121/blocks/s02.validate.json` | End-to-end serving suite artifact | `list[8]` |
| `runs/suite-fused-next-20261121/blocks/s03.exact.json` | End-to-end serving suite artifact | `cases, checks, meta` |
| `runs/suite-fused-next-20261121/blocks/s03.json` | End-to-end serving suite artifact | `arm, arm_proof_problems, banners, cell_s, cells...` |
| `runs/suite-fused-next-20261121/blocks/s03.probe.json` | End-to-end serving suite artifact | `list[11]` |
| `runs/suite-fused-next-20261121/blocks/s03.validate.json` | End-to-end serving suite artifact | `list[8]` |
| `runs/suite-fused-next-20261121/blocks/s04.exact.json` | End-to-end serving suite artifact | `cases, checks, meta` |
| `runs/suite-fused-next-20261121/blocks/s04.json` | End-to-end serving suite artifact | `arm, arm_proof_problems, banners, cell_s, cells...` |
| `runs/suite-fused-next-20261121/blocks/s04.probe.json` | End-to-end serving suite artifact | `list[11]` |
| `runs/suite-fused-next-20261121/blocks/s04.validate.json` | End-to-end serving suite artifact | `list[8]` |
| `runs/suite-fused-next-20261121/blocks/s05.exact.json` | End-to-end serving suite artifact | `cases, checks, meta` |
| `runs/suite-fused-next-20261121/blocks/s05.json` | End-to-end serving suite artifact | `arm, arm_proof_problems, banners, cell_s, cells...` |
| `runs/suite-fused-next-20261121/blocks/s05.probe.json` | End-to-end serving suite artifact | `list[11]` |
| `runs/suite-fused-next-20261121/blocks/s05.validate.json` | End-to-end serving suite artifact | `list[8]` |
| `runs/suite-fused-next-20261121/restore.json` | End-to-end serving suite artifact | `banners, chat, env_diff_rank0, env_diff_rank1, env_match...` |
| `runs/suite-fused-next-20261121/schedule.json` | End-to-end serving suite artifact | `list[5]` |

---

## 3. Guarded `bench.py` JSONs With `health.timing_clean` Present (29 Files)

All 29 files with `health.timing_clean` had valid integer `prompt_tokens_total` and `generation_tokens_total` before and after, `before == after`, `timing_retries == 0`, and `timing_clean == True`:

| File | Total Rows | Timed Rows | `before` (`prompt/gen`) | `after` (`prompt/gen`) | `timing_retries` | `timing_clean` |
|---|---:|---:|---|---|---:|---|
| `runs/fused-next-20261121/review/b1-green/b1-green.json` | 11 | 0 | `232891412/1679716` | `232891412/1679716` | 0 | `True` |
| `runs/fused-next-20261121/review/b1-red/b1-red.json` | 4 | 0 | `232891412/1679716` | `232891412/1679716` | 0 | `True` |
| `runs/fused-next-20261121/review/fix-parity/fix-full-parity.json` | 162 | 0 | `232891412/1679716` | `232891412/1679716` | 0 | `True` |
| `runs/fused-next-20261121/review/wp4tip-self-parity/fix-full-parity.json` | 162 | 0 | `232891412/1679716` | `232891412/1679716` | 0 | `True` |
| `runs/fused-next-20261121/wp0/wp0.json` | 88 | 60 | `33761950/327623` | `33761950/327623` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-fold.json` | 38 | 24 | `44865926/411443` | `44865926/411443` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-hist.json` | 12 | 12 | `51790056/502093` | `51790056/502093` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-idx2b.json` | 15 | 14 | `49619029/485119` | `49619029/485119` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-idx4b.json` | 21 | 14 | `54025196/535354` | `54025196/535354` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-small.json` | 7 | 7 | `59105315/559128` | `59105315/559128` | 0 | `True` |
| `runs/fused-next-20261121/wp1/wp1-small2.json` | 7 | 7 | `59403766/561903` | `59403766/561903` | 0 | `True` |
| `runs/fused-next-20261121/wp2/wp2-full-parity.json` | 138 | 0 | `111291974/822831` | `111291974/822831` | 0 | `True` |
| `runs/fused-next-20261121/wp3/final-parity-1.json` | 145 | 0 | `166300223/1063999` | `166300223/1063999` | 0 | `True` |
| `runs/fused-next-20261121/wp3/final-parity-2.json` | 145 | 0 | `166300223/1063999` | `166300223/1063999` | 0 | `True` |
| `runs/fused-next-20261121/wp3/final-parity-3.json` | 145 | 0 | `166300223/1063999` | `166300223/1063999` | 0 | `True` |
| `runs/fused-next-20261121/wp3/final-timing.json` | 28 | 22 | `166670490/1064525` | `166670490/1064525` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i0.json` | 7 | 6 | `138518842/979164` | `138518842/979164` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i1b.json` | 17 | 12 | `141339567/990499` | `141339567/990499` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i2.json` | 16 | 12 | `132658648/959397` | `132658648/959397` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i2b.json` | 16 | 12 | `133772947/964460` | `133772947/964460` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i2c.json` | 6 | 4 | `133999482/965346` | `133999482/965346` | 0 | `True` |
| `runs/fused-next-20261121/wp3/i4.json` | 10 | 8 | `145502972/1007030` | `145502972/1007030` | 0 | `True` |
| `runs/fused-next-20261121/wp4/wp4-i2-parity-e.json` | 30 | 0 | `178968241/1134071` | `178968241/1134071` | 0 | `True` |
| `runs/fused-next-20261121/wp4/wp4-i2-time-e.json` | 30 | 12 | `178658165/1133275` | `178658165/1133275` | 0 | `True` |
| `runs/fused-next-20261121/wp4/wp4-parity-1.json` | 150 | 0 | `199574792/1202205` | `199574792/1202205` | 0 | `True` |
| `runs/fused-next-20261121/wp4/wp4-parity-2.json` | 150 | 0 | `200627667/1203685` | `200627667/1203685` | 0 | `True` |
| `runs/fused-next-20261121/wp4/wp4-parity-3.json` | 150 | 0 | `200627667/1203685` | `200627667/1203685` | 0 | `True` |
| `runs/suite-fused-next-20261121/kernel-review/index-timing.json` | 37 | 37 | `3641860/38502` | `3641860/38502` | 0 | `True` |
| `runs/suite-fused-next-20261121/kernel-review/parity.json` | 194 | 0 | `5047193/57851` | `5047193/57851` | 0 | `True` |
