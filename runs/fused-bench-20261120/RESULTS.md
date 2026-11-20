# Fused kernels on the pair, November 20 (quick A/B of the reworked 0193/0194, and 0195/0197/0198 one at a time)

**Verdict: the reworked fused route is bit-identical (39/39) and speed-neutral — not because the original fusion was repaired but because it was dropped: the rework keeps the kit gate/router and only merges the prune and dead-row launches, and does nothing under production's switches (no triage); nothing is faster than the control
spread; 0197 alone is the patch that changes the 1024-token greedy Tetris reply.** Nothing deployed; production
(`exl3-tp2-ea522d353626`) restored 19:09 UTC and validated (`restore-validate.log`: all passed).

## What changed since November 19

Only 0193/0194. The recipe's `zig/patches/0193`/`0194` (edited Nov 19 14:12–14:13 PST, after that day's run) are a
rewrite: the kit's gate/router/`decode_prep` stay, and `TF_DS_FUSED_ROUTE=1` merely folds 0177's dead-row mask into the
T13 prune launch (one launch a layer saved, only when prune **and** `TF_DS_DRAFT_TRIAGE` are on). 0194 also adds
`TF_DS_T13_PRUNE_STATS=0` (skip the prune's counters; not run here). 0196 is gone from the recipe. 0195/0197/0198 are
line-for-line the same changes benched on Nov 19 (only `---`/`-- ` trailer lines differ).

## Image

`tensorfold-dsv41:bench-fused-rework-20261120`, built on the head (`build.sh`, `build-context.sh`) from
`~/Workspace/tfref-fusedbench` branch `fused-rework`: production source `2039a60` + reworked 0193/0194 (applied
cleanly, commit `3e1f002`) + 0195/0197/0198 cherry-picked from `fused-bench` (`0b0305a`, `6cd5cac`, `e3eb746`; conflicts
only on old-route import/field lines, resolved by keeping the patch's own lines; each commit's +/- lines verified equal
to its recipe patch). Same binaries on both ranks: tensorfold-native `6f3baf20…`, tf-dsv41-lanes `7bbb68a7…`.

## Arms

Production `.env` (lead reader, draft block 6, T13 prune, token cache) + the listed switch; `campaign.sh`, `probe.py`
as Nov 19 (greedy Tetris 1024, greedy code 1024, 4-stream sampled Tetris 768, each twice; done-line tok/s).

| Arm | Switches | Exact (39) | Tetris greedy | Code greedy | 4-stream | Tetris / code sha | Tetris rounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off | — | IDENTICAL vs Nov 19 A-off | 126.1 / 126.1 | 118.1 / 118.1 | 206.8 / 210.6 | aae67e / b11e56 | 194 |
| tri | TRIAGE=conf:0.2 | IDENTICAL vs off | 125.4 / 125.4 | 117.4 / 117.7 | 208.7 / 210.0 | aae67e / b11e56 | 194 |
| tri-route | TRIAGE=conf:0.2, FUSED_ROUTE=1 | **IDENTICAL vs tri** | 126.0 / 125.8 | 117.8 / 117.9 | 208.8 / 210.1 | aae67e / b11e56 | 194 |
| idx (0195) | IDX_FUSED=1 | — | 126.0 / 126.3 | 118.4 / 118.5 | 207.3 / 210.3 | aae67e / b11e56 | 194 |
| c2n (0197) | FUSED_COMPRESS2_NORM=1 | — | 125.6 / 126.0 | 118.5 / 118.8 | 209.9 / 208.7 | **053b9e** / b11e56 | 195 |
| rope (0198) | FUSED_ROPE_STORE=1 | — | 125.8 / 126.5 | 118.5 / 118.7 | 210.1 / 211.5 | aae67e / b11e56 | 194 |
| off2 | — | — | 126.3 / 126.6 | 118.8 / 118.8 | 209.3 / 211.0 | aae67e / b11e56 | 194 |

Drift: code greedy climbs 117.4 → 118.8 across the run order; off vs off2 differ ~0.6%, the resolution floor. Control spread (off, off2): Tetris 126.1–126.6, code 118.1–118.8, 4-stream 206.8–211.0. Every arm is inside it or
within ~0.5% below it; all sampled-stream shas equal across arms.

## Readings

- **Reworked route (0193/0194): bit-identical, no speed.** 39/39 identical and the same long-reply shas as triage
  alone, where Nov 19's route was 31/39 different and 11–19% slower. Speed equals tri (+0.2–0.5%, noise). Its ceiling is
  one ~2–3 µs launch a layer of a ~42 ms round. Engagement: `model.setDead` uploads the mask every round triage is on, so `rd.dead != 0` and
  the merged launch ran on every layer of every decode round; how many rows were actually marked dead is unmeasured (no
  counter; identical accept counts are expected either way). Triage is the only arm below both controls (~0.5%), and arm
  order confounds the apparent recovery with the route on.
- **0197 is the one that changes the greedy Tetris reply** (`053b9e…`, 195 vs 194 rounds — the same divergence the Nov 19
  bundle showed). 0195 and 0198 keep every sha. 0197's decode norm replicates the kit's Triton `_rmsnorm` order,
  the risk `docs/fused-c-d-scoping.md` §2 flagged — a hypothesis, not verified, and 0197 also touches the prompt path; the 39-case suite (short replies) does not catch it.
- **0195, 0198: no change observed; engagement not shown.** Both within control spread and every sha unchanged; the
  call sites are gated by `fuseOk` bounds and nothing logs whether the fused path ran, so "neutral" and "never ran" look
  the same here. (off2 is the fastest arm on greedy; 0198 leads only one four-stream sample.)
- Not run: `TF_DS_T13_PRUNE_STATS=0`, PDL, exact suite on idx/c2n/rope (passed as a bundle on Nov 19).
- Production restored on both ranks: image `exl3-tp2-ea522d353626`, no `TF_DS_DRAFT_TRIAGE`/`FUSED`/`IDX` in either env.
