# M3 draft-triage probe study: FAILS its own killing criterion — close M3

Window 2026-11-25 16:39–17:38 UTC. Production restored and validated after each capture
(`tensorfold-dsv41:bench-fused-next-2bb052f`, `max_batch_size` 8, no probe env).

## Verdict

**A linear probe on a mid-depth hidden state cannot predict "this draft row will be rejected" at 95% precision /
50% recall. M3 fails its step-0 and should be closed.** Two layers, ~40k draft rows, and the probe barely beats a
free positional baseline:

| layer | rounds | draft rows | dead | AUC (round split) | AUC (position alone) | 95% precision reachable? |
|---|---:|---:|---:|---:|---:|---|
| 13 | 6,247 | 15,218 | 40.7% | **0.621** | 0.588 | **no threshold reaches it** |
| 19 | 10,322 | 25,219 | 40.0% | **0.651** | 0.590 | **no threshold reaches it** |

The doc's criterion (`docs/benchmark-gaps-and-moe-review.md` M3) is 95% precision at ≥50% recall. Neither layer gets
within reach: at layer 13 even a *leaky* row split only reached 95% precision at 0.2% recall, and the honest
round split never reaches 95% precision at any threshold.

## Why this was always going to be hard

The target is a sampling outcome at a *different* row: row `r` is dead when row `r−1`'s pick did not match its
draft. Row `r`'s hidden state at layer 13 encodes row `r`'s token and context — it does not encode whether the
model's draw at row `r−1` happened to coincide with the draft. Rows after the first break are dead by construction,
which is why position alone reaches AUC 0.59: most of the "prediction" is just "later rows die more often".

That also explains the already-recorded `conf:` result: the chain-survival confidences are the optimal threshold on
the only signal the drafter has, and that arm nets **−0.5%**. The problem was never the probe's quality; there is no
signal to extract.

## What was built (kept, off by default)

- Engine branch `probe-dump` in `~/Workspace/tfref-kstack4`: `27090ee` + `535b40d`, off unless `TF_DS_PROBE_LAYER`
  is set. It taps one layer's post-attention-mix rows (`tri_basic.tap`, no new CUDA kernel) and appends one record a
  verify round. Reviewed: rank-gated (both ranks verify, one writer), `drafts` is `rows−1` as the format assumes,
  `zig build` 0, the three `hidden_probe` unit tests pass, the image's executables match production's own fatbins
  byte for byte on both ranks.
- Image `tensorfold-dsv41:bench-probe-dump` on both ranks (`runs/m3-20261125/build.sh`, cross-built).
- `capture.sh` / `capture.py` (PARALLEL=1 semantics) and `train.py` (round split by default, `--by-row` for the
  leaky comparison), plus `README.md` with the record format.

## Two traps that cost a window (recorded so they do not again)

1. `start.sh restart` runs a smoke test, so the engine's first verify round — and the writer's `open` — happens
   *before* the workload. A capture script that then `rm -f`s the dump unlinks the file under the open fd and the
   engine writes to a deleted inode: the banner fires, nothing reaches disk. Never delete it after the restart (the
   writer truncates on open).
2. `find / -xdev` skips the bind-mounted `/spill`. `std.Io.Dir.cwd().createFile` with an absolute path is fine
   (verified against the same Threaded Io in a standalone program).

## Recommendation

- **Close M3.** Do not build the `file:` probe path; it would implement a predictor that does not work.
- The `0177` dead-row mechanism stays built and unarmed (it is; `file:` refuses). No change to production.
- The decode lever is elsewhere: acceptance rate is workload-dependent (the 2026-11-24 attribution) and the row
  budget is already optimal (the capacity window). The remaining structural item is contention, not drafting.

## Files

`runs/m3-20261125/` — `build.sh`, `capture.sh`, `capture.py`, `train.py`, `README.md`, `REPORT.md`, and
`capture/{armed.txt,probe.log,capture.log,m3-layer13.bin,m3-layer19.bin,restored.validate.*}` on the head
(the two dumps are 220 MB and 364 MB; they are not in git).
