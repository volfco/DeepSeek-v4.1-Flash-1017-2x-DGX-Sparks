# Fused kernels 0193–0198 on the pair, November 19 (short A/B)

**Verdict: none of them is faster, and none is adoptable.** The fused route (0193/0194) is 11–19% *slower* and still
changes outputs. 0196 (the drafter's routing, riding the same switch) was only run inside the everything-on arm, whose
greedy rows equal the route-only arm's exactly; that shows no gain on top of the route, not that 0196 engaged. 0195+0197+0198 together are speed-neutral, pass the 39-case exact
suite, but change a 1024-token greedy Tetris reply. Production was restored at 20:34 UTC and validated (8/8 saved
replies, 8/8 concurrent == solo); nothing was deployed.

## Images

Both built on the head from production source `tfref-draftblock` 2039a60 (`exl3-tp2-ea522d353626`'s tree), layered
over the production image, executables sha-checked equal on both ranks (`build.log`).

| Image | Contents | tensorfold-native / tf-dsv41-lanes |
| --- | --- | --- |
| `bench-fused-route-0193-0194-fixed` | 0193+0194 as re-exported 07:08 UTC Nov 19 (after the first image, built 06:39 UTC, failed the gate 36/39; the re-export normalizes with one shared reciprocal, as the kit's SASS does), `../fused-route-0193-0194/build-context.sh` | 352bb79e… / 87f481a4… |
| `bench-fused-0193-0198` | the above + 0195–0198 cherry-picked from engine branch `fused-route` (`stacked-series/`, worktree `~/Workspace/tfref-fusedbench` branch `fused-bench`) | be2c6c65… / 6a9d581e… |

Port notes: 0195–0198's commits equal their `zig/patches` files line for line. Two conflicts: a trivial test-block
conflict in `ops.zig` (0195), and the drafter's gate in `draft.zig` (0196 vs 0192's draft-block). The merge keeps
0192's rule: the fused in-kernel gate is used only where the kit's `rowmm` gate fits (5/10/15 rows); other row counts
keep production's cuBLAS gate, with the routing fused only on the logits (`99f5aae`).

## Arms and gates

Production `.env` (lead reader, draft block 6, token cache) plus the listed switch; private spill directories;
`runs/draft-block-20261118/arm.sh`. Timing = `probe.py`: greedy Tetris 1024 tok, greedy code 1024 tok, 4-stream
sampled Tetris 768 tok, each twice. Rate = the engine's done-line `tok_per_s` (its `decode_s` is 0.1 s resolution, too
coarse for ms/round).

| Arm | Switches | Exact (39 cases) | Tetris greedy tok/s | Code greedy tok/s | 4-stream tok/s | Tetris / code sha |
| --- | --- | --- | --- | --- | --- | --- |
| A-off | ROUTE=0 | IDENTICAL to production | 125.8 / 125.8 | 117.9 / 118.1 | 206.9 / 207.7 | aae67e / b11e56 |
| A-route | ROUTE=1 | **DIFFERENT 31/39** vs A-off | **102.1 / 102.2** | **96.7 / 96.8** | **184.0 / 185.5** | 4e3f10 / 1ab47a |
| A-off2 | ROUTE=0 | — | 126.0 / 126.0 | 118.4 / 118.1 | 209.7 / 210.6 | aae67e / b11e56 |
| B-off | all off | IDENTICAL to A-off | 125.8 / 125.7 | 117.9 / 118.0 | 207.0 / 209.5 | aae67e / b11e56 |
| B-195 | IDX_FUSED, COMPRESS2_NORM, ROPE_STORE | IDENTICAL to B-off | 122.7 / 125.4 | 117.8 / 118.1 | 207.8 / 207.1 | **053b9e** / b11e56 |
| B-all | ROUTE + the three | **DIFFERENT 31/39** | 101.7 / 101.7 | 96.2 / 96.3 | 180.8 / 180.1 | 4e3f10 / 1ab47a |
| B-off2 | all off | — | 126.1 / 126.2 | 118.5 / 118.3 | 209.2 / 210.5 | aae67e / b11e56 |

Control spread (four off arms): Tetris 125.7–126.2, code 117.9–118.5, 4-stream 206.9–210.6 (~1.8%).

## Readings

- **Fused route (0193/0194): −19% Tetris, −18% code, −11% four streams.** Round time 41.8 → 52.6 ms on greedy Tetris;
  tokens per round rose slightly (5.39 vs 5.28, a different trajectory), not nearly enough. Likely cause, a source reading not a profile (0194's prep and its grid-wide barrier share the
  launch and were not separated):
  `tf_ds_fused_route_kernel` is launched with one 256-thread block per row (`ops.zig` `moeRoute`, `grid.x = rows`), and
  each block computes the full 384×5120 fp16 gate itself — the gate weight (3.9 MB a layer) is read once per row and
  only `rows` SMs work, where the kit's `_rowmm_parts` splits the experts across blocks and reads it once. It is also
  built `--fmad=false`. The launches saved are worth far less than that. The fix would be structural (split the gate
  across expert blocks with a second routing phase or a grid barrier), not a tuning change, and the bit-identity
  problem (31/39 after the DIV.RN → RCP·MUL fix; 36/39 before) is unsolved on top of it.
- **0195+0197+0198: no measurable change.** Code and 4-stream inside the control spread; Tetris greedy's first
  repeat was −2.5% and its second −0.3%. Their estimated ceiling was 0.05–0.2% of a round, below what this run can see.
  They pass the 39-case suite (whose replies are ≤ ~56 tokens) but the 1024-token greedy Tetris reply changed
  (`053b9e…` vs `aae67e…`, 195 vs 194 rounds; the done-line sha hashes the output tokens, `chat.zig:722`). The per-round
  counts first differ at round 188, ~token 984. So at least one of the three engaged and is not bit-identical; this
  run did not isolate which, and did not show that each engaged. Code greedy was unchanged.
- **PDL** (`TF_DS_PDL=1`, continue.md item 2) was not run here.

## Files

`campaign.sh` (head-side driver), `probe.py`, `build.sh`, `build-stacked-context.sh`, `stacked-series/` (the six
engine commits on 2039a60), `*.exact.json`/`*.exact-compare.log`, `*.timing.json`, `*.ranks.txt` (image, env and
binary hashes per rank), `restore-validate.log`, `restored-health.json`.

`arm.sh` gained an opt-in `ARM_ALLOW_PAST_TRAFFIC=1` that skips its "non-loopback requests in the running container's
log" refusal (production had 588 external requests, the last at 17:10 UTC, ~2.5 h before; colum said the Sparks were
unused). Live clients still block a restart.
