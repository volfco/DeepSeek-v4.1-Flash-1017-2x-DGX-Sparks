# Repaired fused kernels on the pair, November 20 (0195/0197/0198 after the GB10 audit)

**Verdict: identical on every comparable output, and not measurably faster.** With all three switches on, the 39-case exact suite is
identical to off and the 1,024-token greedy Tetris reply is back to `aae67e` / 194 rounds (the 0197 drift `053b9e` is
gone). The 100K replies were salted per arm, so they carry no identity check. Every speed metric — short greedy decode, 4-stream sampled, 100K-token prefill and 100K decode — is inside the
off-vs-off2 spread. Nothing deployed; production (`exl3-tp2-ea522d353626`) restored 22:10 UTC and validated (8/8 equal,
all passed).

## Image

`tensorfold-dsv41:bench-fused-audit-20261120`, built on the head (`build.sh`, `build-context.sh`) from
`~/Workspace/tfref-fusedbench` branch `fused-audit` = Nov 20 bench tree `e3eb746` + the audit's
`runs/fused-kernel-audit-20261120/production-port.patch` (commit `ed8276f`, applied cleanly). Binaries differ from the
Nov 20 image and match on both ranks: tensorfold-native `e8e93bc6…`, tf-dsv41-lanes `aacc4597…`.

## Arms

Production `.env` + the listed switches, interleaved off → all3 → off2 → all3b (`campaign.sh`). `probe.py` = the Nov 20
short probe (greedy Tetris 1024, greedy code 1024, 4-stream sampled Tetris 768, each twice) plus one ~99.75K-token
greedy prompt (256 tokens, per-arm salt at the start so no prefix cache serves it: `cached=0` in every arm).
all3 = `TF_DS_IDX_FUSED=1 TF_DS_FUSED_COMPRESS2_NORM=1 TF_DS_FUSED_ROPE_STORE=1`.

| Arm | Exact (39) | Tetris greedy tok/s | Code greedy tok/s | 4-stream agg tok/s | 100K prefill s | 100K decode ms/round | Tetris sha / rounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off | IDENTICAL vs Nov 20 off | 126.0 / 126.6 | 118.4 / 118.2 | 209.6 / 210.5 | 49.36 | 40.5 (74 rds) | aae67e / 194 |
| all3 | **IDENTICAL vs off** | 126.1 / 126.4 | 118.1 / 118.6 | 210.2 / 209.6 | 49.26 | 39.7 (78 rds) | **aae67e / 194** |
| off2 | — | 126.2 / 126.4 | 118.7 / 118.7 | 206.9 / 207.6 | 48.25 | 40.2 (82 rds) | aae67e / 194 |
| all3b | — | 126.4 / 126.7 | 118.7 / 118.7 | 211.2 / 211.3 | 48.35 | 39.0 (82 rds) | aae67e / 194 |

Every short-probe sha (both passes of greedy Tetris, greedy code and the four sampled streams) is equal across all four arms (checked in the JSONs).

## Readings

- **Correctness: fixed.** The repaired 0197 norm no longer changes the long greedy reply; 39/39 identical.
- **Short decode: no effect.** Paired differences (all3−off, all3b−off2) are −0.05/+0.25 tok/s Tetris, +0.05/0.0 code —
  under the ~0.5% control floor. Expected: the prior scoping put these fusions at ~0.05–0.2% of a ~42 ms round, below
  this probe's resolution.
- **4-stream:** all3b is +1.9% over off2, but off2 is itself 1.3% under off; all3 vs off is −0.1%. Noise.
- **100K prefill: no effect.** −0.2% and +0.2% in the two pairs; drift between pairs (~1 s) is 10× the arm difference.
  This covers the prompt-path 0197/0198 kernels. **It does not cover 0195.** The candidate-source layer is 20, with
  `compress_ratios[20] = 1`, so 99.75K tokens gives ~99.75K score columns. `fuseOk` admits only 16,384 < columns ≤ 65,536
  (nbk ≤ 8192), so the unfused pool ran in all arms. 0195's speed and identity are untested on the pair; they need a
  16K–65K-token prompt.
- **100K decode: no signal.** ms/round reads 0.8–1.2 ms lower with all3. The done-line tok/s points the other way in
  pair 1 (85.8 off vs 82.8 all3; pair 2: 77.9 vs 79.2). Each figure is one 256-token reply that differs per arm (74–82
  rounds). `decode_s` has one decimal, and pair 2's gap is a single last-digit step (3.3 vs 3.2 s). In pair 1, all3
  drafted fewer rows a round (3.15 vs 3.30). At ~4.3 ms a row, that accounts for ~0.6 of its 0.8 ms.
- **Engagement not shown.** No counter or log line says the fused paths ran; with outputs now bit-identical, "neutral"
  and "never ran" are indistinguishable here (the Nov 20 0197 sha change was the only engagement evidence and is gone
  by design). Not run: single-switch arms (nothing moved), PDL, `TF_DS_T13_PRUNE_STATS=0`, a matched (unsalted, same-prompt) long case
  for a depth identity check, and any prompt in 0195's 16K–65K window.
