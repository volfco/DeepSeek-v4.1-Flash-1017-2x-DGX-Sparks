# k280 — round 0280–0283 on the pair: staged, nothing launched

Image `tensorfold-dsv41:bench-fused-next-1421187` (branch `kernel-stack-0280` = production `2bb052f` + 0280–0283) is
built on both ranks: the two cross-built executables layered over production's image, sha256 checked inside the new
image on each rank ([build record](build/build.log), [how](build/build.sh)). Patch reports:
`runs/kernel-round-20261123b/report-028{0,1,2,3}.md`. Nothing here has run on a GPU.

The lead session's files in this directory (`work.py`, `topk-*`) are described in [topk-README.md](topk-README.md).

## Before the first launch (operator, from the workstation's recipe root)

The head's checkout has no `runs/k280/`, and its `runs/overnight-20261122/arms.sh` is the older copy whose profile
grep lacks `pass_counts` / `pass_phases_ms` (both files are untracked on the head, so this edits nothing tracked):

```
rsync -a runs/k280/{README.md,arms.txt,run.sh,l2-arms.txt,l2-run.sh,passprof.sh,gate_drafts.py,work.py,topk-arms.txt,topk-run.sh,topk-README.md} \
  colum@100.95.49.21:DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/k280/
rsync -a runs/overnight-20261122/{arms.sh,table_gpu.py} colum@100.95.49.21:DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/overnight-20261122/
```

Every script below runs from the recipe root **on the head**, with production idle. Each restarts the server per arm
and restores production from `.env` alone at the end (`restored: validate rc=0` must be the log's verdict). Run them
one at a time: start the next only after the previous `run.log` shows that verdict and its `finished` line.

## Order

1. `passprof.sh` (20 min): the pass breakdown. It is the cross-check that the new image and 0280 behave.
2. `run.sh` (1 h 45 min): 0281 and 0282 against the image's own off path.
3. `l2-run.sh` (50 min): the 0283 screen.
4. The lead's `topk-run.sh` needs no new image and can go anywhere in the order.

`runs/negative-kernels-20261122/markov.py` does **not** have to run first any more. It was meant to price a Markov
bias miss in an idle window; `runs/trace-review-20261123/RESULTS.md` section 3 has since measured it from the saved
ring traces (a hit step 9.8 µs, a miss step 142–162 µs, so 133–153 µs a miss; 3.2 misses and 0.44–0.49 ms a six-step
one-stream pass). 0281 goes straight to its arms in `run.sh`. The microbenchmark stays where it is, with its launcher
`runs/negative-kernels-20261122/run-queued.sh` (a `docker run --gpus all` of image `bench-fused-next-0f89e5a` beside
idle production); it times `_bias` at 0..N missing rows of a 1, 2 or 4 row step and would not answer the one thing
the traces leave open, whether a hit read from a 505 MiB cache is as cheap as one from today's 31.6 MiB. The `mc`
arms answer that directly.

## 1. `passprof.sh` — where a drafter pass's time goes (0280)

- **Answers:** the pass's GPU time by phase (`pass_phases_ms`), and what a pass was (`pass_counts`: steps histogram,
  bias-cache hits and misses, steps with a miss, block rows), at one sampled stream (`s1`), four sampled streams
  (`s4`) and one greedy stream (`g1`), on both ranks.
- **Launch:** `mkdir -p runs/k280/passprof && setsid nohup bash runs/k280/passprof.sh > runs/k280/passprof/run.log 2>&1 < /dev/null &`
- **Env it arms:** `TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PASS_PROFILE=1`. One server start a cell.
- **Outputs:** `<cell>.json` (`own` = the cell's own figures, differenced from the scrapes before and after it),
  `<cell>.rank0.txt` (the raw lines), `<cell>.rank1.txt` (rank 1's lines at its exit: cumulative for the process,
  the smoke test's and the prefills' few passes included), `<cell>.armed.txt`.
- **Gate:** attribution only, so no speed gate. Check: the banner count is 1 on each rank
  (`TF_DS_PASS_PROFILE: the drafter pass runs eagerly`); errors 0; rank 1 printed its two pass lines; restore
  validates. **Cross-check** `g1` against the trace review's section 5 (one stream, six steps): head 1.08 ms, the
  three MoE stages 1.29 ms, the three attention stages 0.68 ms, Markov steps 0.49 ms, twelve ring transfers
  0.23–0.26 ms, a greedy pass about 3.9 ms; `own.mk_bias_ms_a_miss_step` near 0.14. An eager pass pays launch time
  the graph does not, so compare shares before totals.
- Never quote a figure from this run as a speed: the profile syncs every round and the pass runs eagerly.

## 2. `run.sh` — 0281 `TF_DS_MARKOV_CACHE` and 0282 `TF_DS_DRAFT_PREFETCH`

- **Arms** (`arms.txt`, all on the new image): `ctl1, mc2048a, mc4096a, dpf1a, dpf2a, ctl2, dpf2b, dpf1b, mc4096b,
  mc2048b`. `mc` = cached Markov bias rows (unset: 256). `dpf1` = the pass's L2 prefetch at the forward's three sites
  a stage; `dpf2` = those and a hand-off of the target's layer-0 inputs.
- **Answers:** does caching 2,048 or 4,096 bias rows shorten the pass (expected from the measured miss cost and this
  list's held-out hit rates: about 0.16–0.18 ms and 0.23–0.26 ms a one-stream six-step pass; less at four streams,
  where a step is fast only if all four rows hit), and does a prefetch inside the pass pay (bounded at about
  0.18 ms a pass by analogy with the forward; the hand-off has no supporting evidence in the traces).
- **Launch:** `mkdir -p runs/k280/main && setsid nohup bash runs/k280/run.sh > runs/k280/main/run.log 2>&1 < /dev/null &`
- **Score** (workstation or head):
  `python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b` and the
  same with `mc4096a,mc4096b`, `dpf1a,dpf1b`, `dpf2a,dpf2b`. Both patches live in the pass, which the forward's GPU
  clock cannot see: read **`pass_ms`** in the "Rank 0 lifetime profile" block first (0.01 ms resolution, all of the
  arm's traffic), then forward + pass, then wall. `dpf2` joins its hand-off launch before the pass ends, so its
  `pass_ms` can rise while the forward falls: judge it on forward + pass. The script also prints each arm's
  `markov_cache` miss rate and `pass_counts`.
- **Gates, all required:**
  1. Exact panel `IDENTICAL` and sampled panel `IDENTICAL` in every arm (`<arm>.cmp.txt`, the sampled line in
     `run.log`). The `ctl` arms prove the new image's off path against production's reference.
  2. **Drafts unchanged** — this, not the exact panel, is the test of 0281, and it applies to 0282 as well. Output
     tokens cannot move whatever the cache holds; a wrong cached row would change drafts, which shows only as
     different rounds or accepted / drafted counts for the same seeded request:
     `python3 runs/k280/gate_drafts.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b,mc4096a,mc4096b,dpf1a,dpf1b,dpf2a,dpf2b`
     must end `drafts IDENTICAL to control in …` (exit 0). It compares, request by request, every solo probe row's
     `rounds`, `accepted` (the "accepted/drafted" string) and `sha` in `<arm>.probe.json`, and each one-stream
     workload cell's `per_round.rounds` and reply `sha` in `<arm>.json`, and gates only rows on which the two
     controls agree. Checked on saved suites: k272 and k267 pass (19 and 12 rows), the D1 drafter arms fail with
     equal shas and different rounds, as they should.
     Not covered: the workload saves no per-request accepted / drafted (only the cell's rounds), and the multi-stream
     rows are not gated (their rounds vary between identical arms).
  3. Banner counts on **each** rank (`<arm>.banner.txt`): 0 in `ctl1` / `ctl2`; 1 in every `mc` arm
     (`TF_DS_MARKOV_CACHE=<N> (rank r): <N> Markov bias rows cached at load, …`); 1 in every `dpf` arm
     (`TF_DS_DRAFT_PREFETCH=<1|2>: the drafter pass forks the paced L2 prefetch, <9|10> launches a pass …`). Errors 0.
  4. `markov_cache` in `<arm>.profile.txt`: the miss rate should fall from about 52% in `ctl` to roughly 32%
     (2,048) and 24% (4,096) if served traffic resembles the held-out corpus; `pass_counts.steps` should stay close
     to the controls' histogram (the line is lifetime, so the multi-stream rows' round variation moves it a little).
- **Memory:** with production up the pair has 6.9 GiB (head) and 7.5 GiB (worker) available and memguard kills a rank
  below 2.5 GiB. `mc4096` takes 473 MiB more a rank than today, `mc2048` 221 MiB. Watch `MemAvailable` in those arms.
- **If an arm's restart fails** on `MarkovCacheTooLarge`, `BadMarkovCache` or `BadDraftPrefetch`, the message is on
  the rank's stderr in `<arm>.restart.log`.

## 3. `l2-run.sh` — 0283, the L2 prefetch's pacing and budget

- **Arms** (`l2-arms.txt`, new image): `ctl1`, `rate180`, `rate270`, `delay0`, `delay10k`, `bud8`, `bud32`, `ctl2`.
  One knob each on top of the unset 3000 ns / 230 GB/s / 16 MiB; the pass's prefetch stays off.
- **Answers:** whether the kit's three numbers, never swept on this pair, are near the best for the forward (the
  prefetch is worth 5.3% at one stream in total). A screen: it ranks, it does not confirm.
- **Launch:** `mkdir -p runs/k280/l2 && setsid nohup bash runs/k280/l2-run.sh > runs/k280/l2/run.log 2>&1 < /dev/null &`
- **Score:** `python3 runs/overnight-20261122/table_gpu.py runs/k280/l2 --control ctl1,ctl2 --test rate180,rate270`
  (then `delay0,delay10k` and `bud8,bud32`). Here the forward's GPU clock is the metric: one-stream cells' ms a
  round. The two arms of a `--test` pair are different values, so read each column, not the pair's mean.
- **Gates:** exact `IDENTICAL` in every arm (the kernel writes nothing; `EXACT=0` drops the panel); banner count 1 on
  each rank in knob arms (`… the L2 prefetch waits <d> ns before its first wave, issues a wave … every <w> ns …`),
  0 in controls; the banner's numbers match the arm (`rate180` → every 5825 ns, `rate270` → 3883, unset 4559).
  Seeded one-stream cells' rounds and reply shas equal across arms:
  `python3 runs/k280/gate_drafts.py runs/k280/l2 --control ctl1,ctl2 --test rate180,rate270,delay0,delay10k,bud8,bud32`.
- A value that wins by more than the controls' spread deserves an A/B/B/A confirmation with the probe before any
  `.env` change; then try it per rank (the knobs are rank-local, and the one-stream rank imbalance traces to this
  prefetch).

## Files written by this round's agent

`README.md`, `arms.txt`, `run.sh`, `l2-arms.txt`, `l2-run.sh`, `passprof.sh`, `gate_drafts.py`, and `build/`
(`build.sh`, `build.log`, `Dockerfile`, `SHA256SUMS`, `fatbins.sha256`, `proof.json`). The executables themselves are
not in the repo: `~/.cache/k280-stage/build/` on the workstation, `~/.cache/k280-stage/ctx/` on the head.

## Unverified until these run

The image's two executables have never been executed (no container was run from it): `passprof.sh`'s first restart
is their first run, which is one reason it goes first. Everything about speed; that the pass graph captures with a second stream (0282); that the kit's `_score` reads
slots above 255 correctly (0281); that the eager profiled pass reproduces the trace review's breakdown (0280); that
the cross-built executables' off path times like production's from-source build (the `ctl` arms are on the new image
for that reason). `passprof.sh`, `run.sh` and `l2-run.sh` have not been executed: `bash -n` passes, `passprof.sh`'s
embedded Python parses and its differencing is unit-checked, and `gate_drafts.py` was run on saved suites.
