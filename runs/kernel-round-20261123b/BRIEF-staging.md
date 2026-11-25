# Staging the round (after 0280–0283 are committed)

Goal: everything ready for the operator to launch, **nothing launched**.

## 1. Bench image

- Tag: `tensorfold-dsv41:bench-fused-next-<7-hex of the kernel-stack-0280 tip>`, the two cross-built executables
  layered over production's `tensorfold-dsv41:bench-fused-next-2bb052f`, on both ranks.
- Procedure: the one earlier rounds used. Read `runs/overnight-opt/cross-build.py`, `runs/overnight-opt/fatbins.py`,
  `runs/draft-block-20261118/package.sh`, `runs/fused-next-20261121/bench/build.sh`, and the "Engine source locations"
  notes quoted here: cross-compile on this workstation
  (`zig build -Dtarget=aarch64-linux-gnu.2.39 -Dcpu=cortex_a725 -Doptimize=ReleaseFast -Dfatbins=<dir> native bench-follower`),
  with the CUDA fatbins recovered fresh from the base image's own `tensorfold-native` (no `.cu` changed, so they are
  valid as they are).
- **Allowed on the pair** (head `ssh colum@100.95.49.21`, repo `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`; the worker
  is reached from the head): read-only inspection (`docker images`, `docker ps`, `df`, reading files), extracting a
  file from an image, copying the two executables and a Dockerfile over, `docker build` of the new tag on each rank,
  and a `sha256sum` check inside the new image. Check free disk first.
- **Forbidden on the pair:** `./start.sh` in any form, stopping or restarting any container, editing `.env` or any
  tracked file, `just sync-head`, `arms.sh`, `exact.py`, probes, any request to port 8000, any GPU workload. If a step
  seems to need one of these, stop and write the exact command for the operator instead.
- If the build cannot be completed within those limits, leave the cross-built executables with a `SHA256SUMS` in
  `runs/k280/build/` and write the remaining commands.

## 2. Suite files in `runs/k280/` (model them on `runs/k278/` and `runs/overnight-20261122/`; do not run them)

- `arms.txt`: `ctl1, mc2048a, mc4096a, dpf1a, dpf2a, ctl2, dpf2b, dpf1b, mc4096b, mc2048b` on the new image
  (`TF_DS_MARKOV_CACHE=2048|4096`, `TF_DS_DRAFT_PREFETCH=1|2`). `run.sh` for it, following `runs/k278/run.sh`, with
  `SAMPLED=1 PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts'`, `WORKLOAD_SAMPLED=1` (a new env of the
  workload script, added by the lead session: sampled cells so the GPU clock covers sampled rounds) and the banner
  pattern for the new switches.
- `l2-arms.txt` + `l2-run.sh`: a one-at-a-time screen of 0283 like `runs/nccl-screen-20261123/`: `ctl1`, rate 180 and
  270, delay 0 and 10000, budget 8 and 32, `ctl2`.
- `passprof.sh`: one attribution arm modelled on `runs/overnight-20261122/phases.sh` with 0280's env, at one and four
  sampled streams, saving the `pass_phases_ms` / `pass_counts` lines, then restoring production and validating, as
  `phases.sh` does.
- `README.md`: for each script what it answers, its expected duration, the launch command, and the correctness gate.
  For 0281 the gate must compare per-request rounds and accepted/drafted with control. Note that `markov.py`
  (`runs/negative-kernels-20261122/`) should run first in an idle Spark window, and give its command.
- The lead session owns `runs/k280/topk-*` (the 0261 on/off pair) and `runs/overnight-20261122/table_gpu.py`
  (GPU-clock scoring). Do not write those.

## 3. Final report

`runs/kernel-round-20261123b/REPORT.md`: commits, image tag and where it exists, files staged, the exact next commands
in order, and everything that is unverified.
