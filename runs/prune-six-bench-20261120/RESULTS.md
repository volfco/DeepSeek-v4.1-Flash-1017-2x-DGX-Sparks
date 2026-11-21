# 0199 six-slot prune on the pair, November 20 (`TF_DS_FUSED_ROUTE` off/on, triage off)

**Verdict: identical output and a small, consistent single-stream gain of ~0.6–0.7% on greedy decode, which matches
the microbenchmark's ~0.7% projection.** All four "on" Tetris passes (126.9–127.2 tok/s) are above all four "off"
passes (125.9–126.5). Code greedy is +0.4–0.6% with slight overlap. 4-stream sampled and the 100K case show no signal.
Nothing deployed; production (`exl3-tp2-ea522d353626`) restored 22:58 UTC and validated (all passed).

## Image

`tensorfold-dsv41:bench-prune-six-20261120`, built on the head (`build.sh`, `build-context.sh`) from
`~/Workspace/tfref-fusedbench` branch `prune-six` = `fused-audit` (`ed8276f`: production source `2039a60` + reworked
0193/0194 + audited 0195/0197/0198) + standalone 0199 (`d66e663`). Its 0193/0194/0199 content equals the recipe's merged
`zig/patches/0199-dsv41-six-slot-prune.patch`. Binaries identical on both ranks: tensorfold-native `37322478…`,
tf-dsv41-lanes `6dc98b73…`.

## Arms

Production `.env` (`TF_DS_T13_EXPERT_PRUNE=1` = applied 0.85/3/kept, no cap, stats on; triage and PDL off), varying
only `TF_DS_FUSED_ROUTE`, interleaved off → route → off2 → route2 (`campaign.sh`). With triage off, the switch's only
effect is 0199's one-thread-per-row prune kernel (the 0194 dead-row merge needs triage). `probe.py` as
`runs/fused-audit-bench-20261120` (short probe twice + one salted ~99.75K-token prompt, `cached=0` in every arm).

| Arm | Exact (39) | Tetris greedy tok/s | Code greedy tok/s | 4-stream agg tok/s | 100K prefill s | 100K decode tok/s (rounds) |
| --- | --- | --- | --- | --- | --- | --- |
| off | IDENTICAL vs Nov 20 off | 125.9 / 126.1 | 118.2 / 118.3 | 209.7 / 209.4 | 49.12 | 81.6 (79) |
| route | **IDENTICAL vs off** | **126.9 / 127.0** | 118.8 / 119.2 | 210.9 / 209.5 | 49.25 | 78.6 (81) |
| off2 | — | 126.2 / 126.5 | 118.9 / 118.5 | 209.8 / 211.0 | 48.30 | 80.3 (80) |
| route2 | — | **127.2 / 127.2** | 119.1 / 119.2 | 208.7 / 212.1 | 48.29 | 76.8 (85) |

Every short-probe sha (Tetris `aae67e` / 194 rounds, code `b11e56`, the four sampled streams) is equal across all arms.

## Readings

- **Identity: holds.** 39/39 exact and every long greedy hash unchanged, consistent with the GB10 bit-level checks.
- **Tetris greedy: +0.75% (route vs off) and +0.67% (route2 vs off2).** Arm means 126.0 / 126.95 / 126.35 / 127.2. The
  ranges don't overlap, and route beats off2 even though off2 ran later, so upward drift alone doesn't explain it.
  Projection was 0.29 ms of a 42 ms round (0.69%).
- **Code greedy: +0.6% / +0.4%** (118.25 / 119.0 / 118.7 / 119.15). Same direction, but off2's 118.9 overlaps route's
  118.8, so on its own this is borderline.
- **4-stream sampled:** −0.1% / −0.6% by pair mean, with the widest spread (208.7–212.1). No signal either way.
- **100K:** prefill unchanged (decode-only kernel). Decode tok/s is lower with route, but each figure is one salted
  256-token reply with different round counts (79–85), and `decode_s` has one decimal. No signal.
- **Caveats:** one session, two pairs, and server tok/s has one decimal (~0.08% step). Engagement is inferred, not logged:
  production's prune settings match `useSixSlotPrune`'s conditions, and nothing else changes with the switch. Not run:
  triage on (the 0194 dead-row path), `TF_DS_T13_PRUNE_STATS=0`, PDL.
