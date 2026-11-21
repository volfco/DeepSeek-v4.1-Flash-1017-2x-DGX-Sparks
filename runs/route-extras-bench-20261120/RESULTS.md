# Triage, prune stats off and PDL on top of 0199, November 20–21

**Verdict: all three are bit-identical and none adds speed on top of `TF_DS_FUSED_ROUTE=1`.** Every arm passes the 39
exact cases and keeps every long-reply hash. Tetris greedy is inside the three baselines' range for every arm. Code
greedy reads 0.4–1% below its neighbouring baselines in every non-baseline arm, most for triage. Keep route alone.
Nothing deployed; production (`exl3-tp2-ea522d353626`) restored 00:10 UTC and validated (all passed).

## Setup

Image `tensorfold-dsv41:bench-prune-six-20261120` (fused-audit + 0199; same as `runs/prune-six-bench-20261120`).
Every arm: production `.env` + `TF_DS_FUSED_ROUTE=1`, plus the listed switch. Baseline at start, middle and end
(`campaign.sh`); same `probe.py`. Exact suite on every non-baseline arm against `base`.

| Arm | Added | Exact (39) | Tetris greedy | Code greedy | 4-stream | 100K prefill s |
| --- | --- | --- | --- | --- | --- | --- |
| base | — | IDENTICAL vs Nov 20 off | 127.0 / 126.8 | 118.7 / 118.9 | 211.2 / 211.9 | 49.23 |
| tri | `TF_DS_DRAFT_TRIAGE=conf:0.2` | IDENTICAL | 126.8 / 126.8 | **117.3** / 118.7 | 210.5 / 211.7 | 49.18 |
| stats0 | `TF_DS_T13_PRUNE_STATS=0` | IDENTICAL | 126.7 / 127.0 | 118.6 / 118.7 | 208.8 / 209.0 | 49.32 |
| base2 | — | — | 127.4 / 127.2 | 119.5 / 119.4 | 209.0 / 211.8 | 48.42 |
| pdl | `TF_DS_PDL=1` | IDENTICAL | 127.0 / 127.2 | 118.8 / 119.0 | 209.4 / 211.6 | 49.25 |
| all | all three | IDENTICAL | 126.7 / 127.1 | 118.8 / 118.7 | 211.4 / 212.4 | 49.34 |
| base3 | — | — | 127.0 / 127.3 | 119.5 / 119.5 | 208.8 / 212.0 | 48.39 |

Every short-probe sha and round count (Tetris `aae67e` / 194, code `b11e56`, sampled streams) is equal across all seven arms.

## Readings

- **Identity: holds everywhere.** That includes PDL, the gate `continue.md` set for it, and triage with route (0194's
  merged dead-row launch).
- **Baseline spread:** Tetris 126.8–127.4, code 118.7–119.5 (drifting up over the session), 4-stream 208.8–212.0.
- **Triage:** Tetris at the bottom of the baseline range; code 118.0 mean vs 118.8 (base) and 119.45 (base2). Its 117.3
  pass is the lowest number in the run. Consistent with Nov 20 (triage alone ~0.5% slower). Not a win, with or
  without the merged launch.
- **Prune stats off:** Tetris in range; code 118.65 below both neighbours (118.8, 119.45); 4-stream at the low end.
  No gain, matching the earlier generic-kernel stats A/B. With 0199 the counters cost ~0.7 µs a layer
  (`runs/prune-six-20261120`), ~0.07% of a round, so this was below resolution.
- **PDL:** Tetris 127.1, in range. Code 118.9 vs neighbours 119.45/119.5 (−0.5%). Neutral to slightly negative.
- **All three:** Tetris 126.9, code 118.75, 4-stream the highest pair (211.4/212.4, within baseline spread). Nothing
  compounds.
- **100K prefill** splits into ~49.2 s and ~48.4 s groups that follow run position, not switches (the same split shows
  in the two previous campaigns). No reading.
- Caveat: one session, one pass per arm, code-greedy drift of ~0.6% across the baselines. The consistent code-greedy
  dip is at most ~0.5–1% and could be order; none of these switches shows a gain to chase.
