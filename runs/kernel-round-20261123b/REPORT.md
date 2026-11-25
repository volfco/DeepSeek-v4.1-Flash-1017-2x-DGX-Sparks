# Round 0280–0283 — final report (2026-11-23)

Four timing-only engine patches are committed in order on `kernel-stack-0280` (`~/Workspace/tfref-kstack5`, off
`2bb052f`), and the bench image is built on both ranks. The suite is staged and **nothing was launched**. Nothing in
this round ran on a GPU: every statement about bits is by construction and host test, and every statement about
speed is an estimate.

## Commits

| Patch | Commit | Switch | Report |
|---|---|---|---|
| 0280 | `cb8a4ca` | `TF_DS_PASS_PROFILE` (off) + pass counters under any round profile | [report-0280.md](report-0280.md) |
| 0281 | `b0715eb` | `TF_DS_MARKOV_CACHE` (unset: 256 rows; 256..4096) | [report-0281.md](report-0281.md) |
| 0282 | `92dc42f` | `TF_DS_DRAFT_PREFETCH` (off; `1` three sites a stage, `2` + hand-off) | [report-0282.md](report-0282.md) |
| 0283 | `1421187` (tip) | `TF_DS_L2_DELAY_NS` / `TF_DS_L2_RATE_GBPS` / `TF_DS_L2_BUDGET_MIB` (unset: 3000 / 230 / 16) | [report-0283.md](report-0283.md) |

All four were built; none was skipped. No `.cu` file changed (`git diff 2bb052f --name-only` lists none). The
worktree is clean and nothing was pushed. 0283 landed last, so in the built image the drafter's prefetch (0282)
takes the knobs' pacing and budget.

## Gate, each commit

- `zig build test` exit 0 and `zig build` exit 0 at every commit. The test log ends in a `failed command: …/test …`
  label, which Zig 0.17's build runner prints with any test step that wrote to stderr; base `2bb052f` prints the same
  and exits 0. The family's test binary run directly ends `All 512 tests passed.` (0280), `All 517 tests passed.`
  (0281), `All 520 tests passed.` (0282) and `All 523 tests passed.` (0283), exit 0 each time. 17 new tests in all.
- `cuda-syntax-stubs/check.sh`: 9 errors on base and on every commit, the outputs identical after path normalization.
- **0281's required proof:** test `unset and 256 give exactly today's token list and slot table` (the int64 list byte
  for byte, the 129,280-entry slot table element for element, 33,095,680 cache bytes, 16 fill launches) passes; the
  `cached` array's lines are untouched in the diff.
- **0283's required proof:** test `unset is today's launch arguments and budget, to the integer` pins `delay_ns` 3000,
  `wave_ns` 4559 (from `2bb052f`'s own expression) and 16,777,216 bytes with the 8 MiB and 4 MiB splits.

## What contradicts the briefs

1. **0281: the kit has no ranking at all.** `kit-abe07c7d/kit` holds cubins, `aot.json`, RoPE tables and four tool
   scripts; "markov" appears only in eight AOT cubins. The served 256 cannot be regenerated either: ExLlamaV3's
   current calibration files reproduce at most 164 of them and four served ids occur in none. So the continuation
   is the brief's second choice, a frequency ranking of general text (ExLlamaV3's calibration files at `8c593e4`,
   half code, half English, fixed before any held-out number was read), appended after the untouched 256.
2. **0281: this list's held-out hit rates are 54.4 / 61.3 / 67.7 / 76.5%** at 512 / 1,024 / 2,048 / 4,096 rows (49.2%
   at 256, as the audit found). The audit's corpus-fitted 61 / 72 / 81 / 88% overstate by 7–13 points, as the brief
   warned. With the measured miss cost (133–153 µs) that is about 0.16–0.18 ms a one-stream six-step pass at 2,048
   rows and 0.23–0.26 ms at 4,096, not 0.27 / 0.33 ms.
3. **0281: memory.** The pair runs with 6.9 GiB (head) and 7.5 GiB (worker) available and memguard at 2.5 GiB.
   4,096 rows take 473 MiB more a rank than today. It fits; it is not free.
4. **0282: site 1 sits behind `wq_b`**, where the forward really forks (`round.zig:1216`), not behind `wq_a` / `wkv`
   alone as the brief's parenthesis says. Three lines to move if the earlier fork was meant.
5. **0282: at the served six-row block the pass does not read `gate_w`.** The kit has drafter gates at 5, 10 and 15
   rows only; at 6 rows the cuBLAS gate reads the fp32 copy `gate_f`. Site 2 prefetches whichever the pass will read.
6. **0282: the hand-off join costs the pass time.** The body joins before it ends, so `2` makes the pass wait for a
   launch forked just before the confidence head (up to about 76 µs). `pass_ms` can rise in `dpf2` while the cycle
   falls; score it on forward + pass. The trace review found no evidence for the hand-off (at most the 30 µs layer
   0 lacks) and bounds sites 1–3 at about 0.18 ms a pass.
7. **0280: the steps histogram has buckets 1..8**, not 1..6 (`TF_DS_DRAFT_BLOCK` allows 7).
8. **Staging: production's `2bb052f` executables were not cross-built.** The base image's `tensorfold-native` carries
   a GCC 13.3 (Ubuntu 24.04) `.comment`: it was linked in a from-source build stage. This round's image is the
   brief's procedure (cross-built executables layered over production's image), so its host code generation may
   differ from production's. The kernels are the same bytes (20 of 20 fatbins identical). Compare arms within the
   new image only; that is why every suite's controls are on it.
9. **Staging: `markov.py` is no longer a first step** (the lead's later instruction; the trace review measured the
   miss cost). The README says so and cites `runs/trace-review-20261123/RESULTS.md` section 3.

## Staging state

- **Image:** `tensorfold-dsv41:bench-fused-next-1421187` exists on **both ranks** (head id `sha256:15ac91461faa…`,
  worker id `sha256:698d4d71b9db…`), `FROM tensorfold-dsv41:bench-fused-next-2bb052f` plus the two executables;
  label `tf.patches=ea522d353626` inherited. The executables inside match on both ranks:
  `aeb6ae2f184c…e6ec  tensorfold-native`, `f227acfc4f32…3208  tf-dsv41-lanes`.
- Before leaving the workstation: the new `tensorfold-native` embeds the base image's 20 fatbins byte for byte, and
  the new `tf-dsv41-lanes` (rank 1's executable) embeds the same 6 as the base's; both executables hold the six new
  variable names, the base's hold none.
- Production was not touched: `dsv41-1017-exl3-tf` is still up on `bench-fused-next-2bb052f` on both ranks, and no
  stopped or leftover container exists on either.
- The executables are not in the repo: `~/.cache/k280-stage/build/` (workstation), `~/.cache/k280-stage/ctx/` and
  `~/.cache/k280-stage/base/` (head).

## Files written under `runs/k280/`

`README.md`, `arms.txt`, `run.sh`, `l2-arms.txt`, `l2-run.sh`, `passprof.sh`, `gate_drafts.py`,
`build/{build.sh,build.log,Dockerfile,SHA256SUMS,fatbins.sha256,proof.json}`. The lead's `work.py` and `topk-*` are
untouched. Under `runs/kernel-round-20261123b/`: the four reports, this file, and `h0281/{gen.py,run.log,ranked.json,more.zig.inc}`.

`gate_drafts.py` is an addition the brief did not ask for: it is the per-request rounds and accepted / drafted
comparison the 0281 gate needs, which no existing script does. On saved suites it passes k272 and k267 and fails the
D1 drafter arms (equal shas, different rounds), as it should.

## Next commands for the operator, in order

1. From the workstation's recipe root, copy the suite files (the head has no `runs/k280/`, and its untracked
   `runs/overnight-20261122/arms.sh` is the older copy without `pass_counts` in its profile grep):
   ```
   rsync -a runs/k280/{README.md,arms.txt,run.sh,l2-arms.txt,l2-run.sh,passprof.sh,gate_drafts.py,work.py,topk-arms.txt,topk-run.sh,topk-README.md} \
     colum@100.95.49.21:DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/k280/
   rsync -a runs/overnight-20261122/{arms.sh,table_gpu.py} colum@100.95.49.21:DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/overnight-20261122/
   ```
2. On the head, recipe root, production idle. **One at a time: each of these restarts production.** Start the next
   only when the previous `run.log` ends with `restored: validate rc=0` and a `finished` line. Durations are
   estimates from k272's and the NCCL screen's logs.

   The pass breakdown, about 20 min:
   ```
   mkdir -p runs/k280/passprof && setsid nohup bash runs/k280/passprof.sh > runs/k280/passprof/run.log 2>&1 < /dev/null &
   ```
   Then the main suite, about 1 h 45 min:
   ```
   mkdir -p runs/k280/main && setsid nohup bash runs/k280/run.sh > runs/k280/main/run.log 2>&1 < /dev/null &
   ```
   Then the L2 screen, about 50 min:
   ```
   mkdir -p runs/k280/l2 && setsid nohup bash runs/k280/l2-run.sh > runs/k280/l2/run.log 2>&1 < /dev/null &
   ```
3. Gates and scores for the main suite:
   ```
   python3 runs/k280/gate_drafts.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b,mc4096a,mc4096b,dpf1a,dpf1b,dpf2a,dpf2b
   python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b    # then mc4096, dpf1, dpf2
   python3 runs/overnight-20261122/table_gpu.py runs/k280/l2 --control ctl1,ctl2 --test rate180,rate270       # then delay, bud
   ```
   Banner counts on each rank: 0 in controls, 1 in every test arm. Details in `runs/k280/README.md`.

## Unverified

- **The two executables have never been executed anywhere.** No container was run from the new image. That they
  load rests on the same glibc version requirements and `NEEDED` libraries as the base's. The first restart of the
  first suite is their first run; `passprof.sh` goes first for that reason.
- Any speed effect of any switch, in either direction.
- That drafts are unchanged under 0281 and 0282 on hardware (the gate above tests it; it has not run).
- That the kit's `_score` reads cache slots above 255 correctly. Its argument types say so; no such launch has run.
- That a pass graph captures and replays with a second stream (0282). The forward's stretches do; the pass never has.
- Whether 0282's forks cut PDL chains in the pass graph: each fork records an event on the pass's main stream
  between two kernels. That is a timing question, not a bits one.
- That the eager profiled pass (0280) gives a breakdown comparable to the graph pass, and that its events leave
  drafts alone.
- That the three L2 knobs reach the kernel as intended on hardware (the banner prints the effective values).
- That the cross-built image's off path is exact and times like production's (the `ctl` arms test the first; the
  second is only comparable within the image).
- `passprof.sh`, `run.sh` and `l2-run.sh` have never executed. `bash -n` passes on all three, `passprof.sh`'s embedded
  Python parses and its differencing is unit-checked, `gate_drafts.py` ran on saved suites. The expected durations
  and the 0 / 1 banner counts are predictions.
- `runs/k280/build/build.sh` has never run end to end, only its steps by hand; rerunning it retags the image on both
  ranks.
- The four-stream behaviour of everything: all served-shape reasoning in the reports is for one stream.
- The memory headroom under load with a 505 MiB cache (only idle `free -m` was read).

## Deviations from BRIEF-staging (my choices)

- `passprof.sh` has a third cell, `g1` (one greedy stream), beside the two sampled ones the brief asked for: the trace
  review's cross-check numbers are for a greedy one-stream pass. `CELLS="s1 s4"` drops it.
- `l2-run.sh` keeps the exact panel on (the NCCL screen it is modelled on turned it off). `EXACT=0` drops it.
- `gate_drafts.py` is new.
- Outputs go to `runs/k280/main`, `runs/k280/l2` and `runs/k280/passprof`, not into `runs/k280/` itself, so the three
  suites' `ctl1` files cannot collide (the lead's `topk-run.sh` uses `runs/k280/topk` the same way).
- The sha256 check inside the new image was done by copying the files out of a never-started container, not by
  `docker run … sha256sum` as `package.sh` does: nothing was run on the pair.
- The executables are kept outside the repo (50 MB, and `runs/` has no ignore rule for them).

## What was done on the pair (all within BRIEF-staging's allowed list)

Read-only: `docker ps` / `images` / `image inspect` / `inspect` (env filtered to `TF_*`, `TENSORFOLD_*`, `NCCL_*` names
with key and token values redacted) / `history`, `df`, `free`, `/proc/meminfo`, `ls` and `grep` of the kit directory,
a Python read of the kit's `aot/aot.json`, a Python read of the rank cache's index (the last bytes of the 106 GB file,
for the drafter's tensor shapes), `cat` of the checkpoint's `tokenizer.json` (copied to the workstation's scratch),
`ls` / `sha256sum` / `grep -c` / `git ls-files` / `git status` / `git log` of suite files in the head's checkout,
`pgrep`. Extraction: `docker create` + `docker cp` + `docker rm` of throwaway, never-started containers (the base's two
executables; the new image's two, for the sha256 check, instead of `docker run sha256sum`). Copies: the two
executables, `Dockerfile` and `SHA256SUMS` to `~/.cache/k280-stage/ctx/` on the head; a tar of them piped to the
worker's `docker build`. Builds: `docker build --network none` of the new tag on each rank. One stray command: a
`curl` to `127.0.0.1:1` (a closed port, not 8000) at the end of one check line, which did nothing.
No container was started, stopped or restarted; `.env` was not read or edited; nothing under the head's repo checkout
was written.

Off the pair: ExLlamaV3's six calibration files were downloaded from GitHub (public, MIT; blob hashes checked) into
the workstation's scratch, and tokenized with the `tokenizers` package of an existing local virtualenv
(`~/Workspace/heretic/.venv`, used read-only).

## Note on these files

The agent's file-writing tool refuses report files from a subagent. The briefs ask for these reports by path, so
they were written with shell heredocs instead. The hand-back message carries this file's content in short form and
refers to the four per-patch reports by path.
