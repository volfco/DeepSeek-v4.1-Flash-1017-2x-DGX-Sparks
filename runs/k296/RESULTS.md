# k296: sched lane 0294 / 0295 / 0296 on the pair: all exact, no speed win

Suite 2026-11-24 01:09–02:06 UTC, run from the head (`run.sh` → `arms.sh`, a copy of `overnight-20261122/arms.sh` that
also keeps each arm's `/metrics` and `kv_pick_union` line). Production restored afterwards: both ranks back on
`bench-fused-next-2bb052f`, `restored: validate rc=0`.

**Image** `tensorfold-dsv41:bench-fused-next-87e92a2` (`~/Workspace/tfref-sched` `sched-0293` = 1421187 [2bb052f +
0280–0283, all off] + 0294–0296). It was cross-built on the workstation with the base image's own fatbins (embedded
fatbins byte-identical to the base) and layered over 2bb052f on both ranks. The executables' sha256 inside the image
match `SHA256SUMS` on each rank. The build context is in `~/.cache/k296-stage/ctx` on the head. Note: the head's
tailnet address was offline; it was reached with `ssh -J ai-node colum@10.50.0.10`.

**Arms:** ctl1, conf_a (`TF_DS_CONF_LEADER_ONLY=1`), fit_a (`TF_DS_ROUND_MS_FIT=1`), dm065_a (`TF_DS_DRAFT_MS=0.65`),
ctl2, conf_b. Every arm ran `TF_DS_ROUND_PROFILE=1`, which also arms 0296's counter. Each arm ran four checks:
- the exact panel vs `deploy-10c9866/exact.json`;
- the sampled panel vs `sched2/ctl1.sampled.json`;
- the sampled narrow probe (16 rows, seeds 700/701, plus x2/x4 bursts);
- the kvreuse-size workload.

## Gates

| Arm | Exact (39) | Sampled | Probe sha = ctl1 | Probe rounds = ctl1 | Probe accepted = ctl1 | Rank-1 banner | Errors |
|---|---|---|---|---|---|---|---|
| ctl1 | IDENTICAL | IDENTICAL | — | — | — | 0 | 0 |
| ctl2 | IDENTICAL | IDENTICAL | 18/18 | 18/18 | 18/18 | 0 | 0 |
| conf_a | IDENTICAL | IDENTICAL | 18/18 | 18/18 | 18/18 | 1 | 0 |
| conf_b | IDENTICAL | IDENTICAL | 18/18 | 18/18 | 18/18 | 1 | 0 |
| fit_a | IDENTICAL | IDENTICAL | 18/18 | 10/18 | 2/18 | 0 | 0 |
| dm065_a | IDENTICAL | IDENTICAL | 18/18 | 18/18 | 18/18 | 0 | 0 |

The report's follower-side check (draft IDs, held depths and stashed rows logged identical) was not run separately. The
probe's per-row rounds and accepted counts are identical, which those would have to be.

## Speed (sampled probe: total completion tokens / total `decode_s`, 16 rows)

| ctl1 | ctl2 | conf_a | conf_b | fit_a | dm065_a |
|---:|---:|---:|---:|---:|---:|
| 112.61 | 112.61 | 112.56 | 112.84 | 112.44 | 112.73 |

The workload's GPU clock and walls are in `table-conf.md`, `table-fit.md` and `table-dm065.md`. Control spread is
0.1–0.4% on one-stream cells and 0.9–2.6% on four-stream totals (four-stream rounds differ between identical arms).
No test arm leaves that band consistently.

## Verdicts

- **0295 `TF_DS_CONF_LEADER_ONLY`: exact, neutral; adoptable under the exact-and-neutral rule.**
  - Replies, rounds and accepted drafts are identical in both reps.
  - Probe vs the control mean: −0.04% / +0.20%.
  - Rank-0 `pass_ms` 4.21 / 4.17 vs 4.16 / 4.16.
  - Its saving is follower GPU work, which no rank-0 metric here can see. The follower's skip banner fired once a
    run on rank 1.
- **0294 `TF_DS_ROUND_MS_FIT`: exact, moves scheduling as designed, no win.**
  - Probe −0.15% (one rep).
  - One-stream forward a round is +1.5 to +3.3%, with rounds differing; walls −1.9 / −0.6 / +0.95%.
  - Do not adopt. It would need a second rep only to say "slightly negative".
- **0294 `TF_DS_DRAFT_MS`: inert at the served shapes.**
  - `dm065_a` had identical rounds on every probe row. Probe +0.10%; workload within control spread.
  - The serving readers price measured round costs once they exist (`depth.zig` `roundCost`: the shared table is
    used when `plain_guard` is false, its default), so the knob only reaches depths not yet observed.
  - The chooseK clamp fix has no production caller (replay tools only).
- **0296 `kv_pick_union`: the union-fetch gate fails, so do not brief the union fetch.** `distinct / picks` by depth
  band:

  | Arm | 1K | 2K | 4K | 8K | **32K** | 64K | **128K** | 256K |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | ctl1 | 0.110 | 0.229 | 0.415 | 0.364 | **0.582** | 0.496 | **0.612** | 0.551 |
  | ctl2 | 0.110 | 0.229 | 0.398 | 0.361 | **0.582** | 0.496 | **0.612** | 0.553 |

  - The rule was "brief only if < ½ at both 32K and 128K"; both are above ½ in both controls.
  - Caveat: the 32K and 128K bands hold few fetches (288 and 136). The denominator is the pair-wide `R × kk` (J2).

## Review finding not in REVIEW.md (0294, `TF_DS_DRAFT_MS` units)

The report and the commit message both say to set `TF_DS_DRAFT_MS` to the measured whole drafter pass, at the served
stream count: 3.9 at one stream, 7.2 at four, ~11 at eight. In serving, though, the knob feeds `facts().mtp_step_ms`,
and every reader multiplies that value by a count:

| Reader | Multiplied by |
|---|---|
| `zig/src/core/lanes/depth.zig:119` (`forward + mtp_step_ms × drafts`) | drafts |
| `plan_lanes.zig:58,63` (`head_ms × steps`) | steps |
| `engine.zig:374` | `held_levels` |

So the value is a per-chained-head-step cost: about 3.9 / 6 ≈ 0.65 ms at one stream. The 3.5 const is a per-round
allowance only in `chooseK`, which serving never calls. The staged arms `draftms39` / `draftms72` would price five
drafts at 19.5 or 36 ms, against roughly 4–7 ms real. Their effect would still be small, since measured costs
override, but it would point the wrong way. This suite used 0.65 instead and found it inert.

## Files

- Per-arm records: `<arm>.{armed,banner,cmp,exact,probe,sampled,profile,frames,metrics,pickunion}.*`.
- `probe.md`: per-row probe table; regenerate with `python3 probe_table.py <arms…>`.
- `table-*.md`: `table_gpu.py` output per patch.
- `run.log`: the suite log.
