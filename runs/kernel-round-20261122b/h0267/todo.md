# todo — 0267 follow-ups (short; detail in report-0267.md's "Known gaps")

- Hardware gate: 39-case exact suite + long-reply hashes with `TF_DS_MARKOV_FUSED=1` (rounding points in the report).
- Graphs-on A/B at KEPT=2: the fused kernel's dot is CUDA-core work where `_bias` used tensor cores; drop the switch
  if it loses (the miss counter stands alone). Ref: report-0267.md, runs/kernel-round-20261122b/h0267/run.log.
- Read `markov_cache` (TF_DS_ROUND_PROFILE) on hardware to size the miss rate at KEPT=2.
- If the kernel registers spill (acc[16]+tokv[16] per thread), a u32 token table or WMMA revisit; only on evidence.
- Four-node (2D `lg_w`) coverage: one harness case with the quarter geometry if that deployment ever runs the switch.
