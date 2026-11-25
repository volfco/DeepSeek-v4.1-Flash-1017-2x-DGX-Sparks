# decisions — 0267 (TF_DS_MARKOV_FUSED)

Scope: major (the fused kernel + its arm rule), minor (counter placement, trace arms).

- **Fuse `_bias`'s miss rows into `_score` only; keep `_finish`/`_pick`/`PV`/`PI`.** Choices: (a) fuse through
  `_finish` and the pick; (b) fuse bias+scoring, keep the partials protocol. Decided (b): the brief names "bias miss
  and tile scoring", the cross-rank tie behavior stays untouched, and `PV`/`PI` remain bit-identical for `_finish`.
  Ref: commit `6158a6f` (`tri_markov.localBestFused`, `dsv41_ops.cu`).
- **Dot = sequential fp32 rank-order sum, not WMMA.** Choices: (a) WMMA mirroring tl.dot's 16-at-a-time tiles;
  (b) explicit `__fadd_rn`/`__fmul_rn` sum. Decided (b): the CPU harness executes it exactly, and the single
  rounding assumption (tensor-core 16-at-a-time == serial order, evidenced by the kit's markov_check.py) is
  documented in report-0267. Trade: possibly slower than the tensor-core original — the hardware A/B decides.
- **Fused arm only for all-greedy passes** (noise/mid taps absent): the brief's contract, and the only case where
  the unwritten staging rows have no reader.
- **Counter host-side in `Model.pass`, gated on `m.prof`**, emitted in `printRoundProfile`'s `markov_cache` line:
  hit/miss is host-derivable exactly (static slot table + already-read-back tokens); zero device work, graph-safe.
- **Shared cuda-syntax stubs left untouched** (they fail on the base; 0260's finding). The fixed stub copy lives in
  `h0267/stubs/`. Ref: report-0260.md, report-0267.md gate 2.
