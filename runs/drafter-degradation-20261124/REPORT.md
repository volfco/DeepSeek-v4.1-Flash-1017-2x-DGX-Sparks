# Drafter degradation attribution (2026-11-24)

Read-only analysis of saved run records. Nothing ran on the pair; **no idle hardware window was taken** (both
claims settle from saved data, and the residual open sizes are below the one-arm provable floor). No engine source
or `zig/patches/` was touched; the only file written is this one.

Everything below is marked **(M)** measured — a figure in a file under `runs/` or a source line — **(D)** derived
(arithmetic on (M) figures, shown), or **(I)** inferred (a hypothesis). `continue.md`, `todo.md`, `decisions.md` were
read as leads only; every figure repeated here was re-read from a run record or a source line. `forward_ms_per_round`
is never used; per-round GPU time is `per_round.tail[0]`, and `pass_ms` is the profile field.

Path shorthand: `k265/`, `k269/`, `k272/` = `runs/overnight-20261122/<suite>/`; `rowcost/` =
`runs/overnight-20261122/rowcost/`; `trace/` = `runs/trace-review-20261123/`; `observe/` =
`runs/observe-review-20261123/`; engine source is `~/Workspace/tfref-kstack4` at `2bb052f`
(`zig/src/families/dsv41/` unless a path is given), the Nov 15 source is `~/Workspace/tfref-next` at `8ff4797`.

---

## 0. Reconciling the four "pass" numbers first

**What `pass_ms` is (M).** A host timer around `m.pass` in the draft frame handler: from just after the pass frame
flush to the pass's return — three or four small uploads, one graph replay (stages + vocab head + Markov loop +
confidence head), a synchronize and two device-to-host copies (`lanes.zig:1824–1827`, `draft.zig:570–714`). It
**excludes** `absorb_ms` (0.26) and `logits_ms` (0.05), which are separate profile fields
(`k265/ctl1.profile.txt` line 2). The timer span is the same in the Nov 15 build (its profile line has the same
`pass_ms` field and the same `lanes` draft handler; `runs/ultrafast/prof/g-r1.rank0.log`).

| Figure | Quantity | Method | Work that produced it | Source |
|---|---|---|---|---|
| 3.43 ms | profile `pass_ms` | host timer (above) | Nov 15, `tf-dsv41-lanes` batch both ranks, image `exl3-tp2-f3f69b951cd3`, 1.00 stream/pass; 7 requests: 5 prose (4 greedy 256 tok, 1 sampled), tetris greedy 1024, tetris sampled 1024; 1,072 rounds, drafted 3,085 (2.88/round), accepted 2,251 (73%), 3.10 tokens/round; block rows **5** (checkpoint block — `TF_DS_DRAFT_BLOCK` did not exist), Markov steps ~3–5 (depth rule `min(5, granted+2)` (`core/lanes/depth.zig:225`) with granted 1.5–4.8 per request; rounds-weighted mean ≈ 4.0 **(D)**); ~32% of passes around sampled requests (keyed noise, see below) | `runs/ultrafast/prof/g-r1.rank0.log`, `runs/ultrafast/prof/r1.json`, `runs/ultrafast/uf_profreq.py:41`, `docs/benchmark-gaps-and-moe-review.md:55` |
| 3.51 (K=0) → 3.75/3.82/3.82/3.86/3.86/3.90 (K=1..6) | profile `pass_ms`, same timer | rowcost `FIX_K` arms, Nov 22, image recipe-0264 (`27baa93`) + `KEPT=2` + production env (`CONF_PACK=1`, fused switches, `T26=1`), 1.00 stream/pass, greedy, frozen corpus 4K/32K | `rowcost/fk{0..6}.profile.txt`, per-cell `rowcost/fk*.json` (fk0 cells 3.42–3.52) |
| ~3.9 ms | **not** `pass_ms`: GPU-clock trace tiles | 1-in-4 ring traces; sum of per-segment producer+ring tiles from the forward's last transfer to the pass's last transfer: **includes** the 0.39 ms hand-off (absorb, pick, uploads) and **excludes** the ~0.10 ms pass tail | `trace/RESULTS.md` §5–6, `trace/out/pass_breakdown.txt` (6-step tiles 4,213 µs = 3,952 producer + 261 ring; 5-step 3,870 µs) |
| 4.11–4.26 (claim's 4.16–4.21 = k265 ctl / k296 ctl) | profile `pass_ms`, same timer | served suites: sampled probe + workload cells, 1.36–1.43 streams/pass, mixed greedy/sampled passes, mostly 6 rows/6 steps | `k265/*.profile.txt` (4.11–4.17), `k269/*.profile.txt` (4.19–4.23), `k272/*.profile.txt` (4.20–4.26), `runs/k296/*.profile.txt` (4.16–4.21) |

**Which are the same quantity.** 3.43, 3.51–3.90 and 4.11–4.26 are all `pass_ms`, one method, directly comparable
*only after normalizing work* (block rows, Markov steps, streams/pass, sampled keying). The trace's ~3.9 ms is a
different span (GPU tiles, boundary above) and is not a `pass_ms` reading; it *agrees* with the greedy one-stream
`pass_ms` once its 0.39 ms included hand-off and 0.10 ms excluded tail are applied (M, `trace/RESULTS.md` §5).

**Work normalization (D on M anchors).**

- *Block rows and steps.* `blockRows(steps) = max(5, steps)` (`draft.zig:562–565`); the pass widens past the
  checkpoint's five rows only when the stream's last round kept ≥ `TF_DS_DRAFT_BLOCK_KEPT` rows (=2 in production)
  and the round has room (`passDrafts`, `lanes.zig:2205–2208`, gate input `r.follow.len`). Traced passes confirm:
  `FIX_K=0` (`k0x1`) runs **5 Markov steps** (forward-anchor gap 102 = 91 forward + 6 hidden + 5 Markov transfers,
  746 of 755 passes; `trace/out/recon_all.txt`, placement rule `trace/RESULTS.md`), i.e. 5 block rows — with one
  row a round nothing widens; `FIX_K=6` (`k6x1`) and production (`fx1L`) run **6 steps / 6 rows** (gap 103:
  137/157 and 644/756; 101/756 production passes at gap 102 = passes that did not widen).
- *The fk1–fk6 gradient (3.75 → 3.90) tracks the widening share, not the pin value (I).* Under the kept-rows
  reading of the gate (a round keeping ≥ 2 rows = ≥ 1 accepted draft), the widened share rises with the pin:
  kept rows a round go 1.71 / 2.20 / 2.35 / 2.50 / 2.56 / 2.60 for fk1–fk6 (`rowcost/fk*.json`, 200 tokens an
  arm ÷ rounds), so more rounds clear `KEPT=2` as K grows and the mean moves from the 5-row cost (3.51) toward
  the 6-row cost. fk1 is a Bernoulli mix (one verified draft, survival 0.71): 0.71 × 3.85 + 0.29 × 3.51 = 3.75,
  the observed value (D). The residual fk2–fk6 rise (3.82 → 3.90, +0.08) sits at the edge of the same-work repeat
  band below. Under the verified-rows reading the six arms are identical work and the gradient is run drift; the
  report does not need the distinction.
- *Same-work repeat spread on `pass_ms` (M).* fk0's ten cells, all 5 rows/5 steps/1.00 stream a pass, read
  3.42–3.52 (`rowcost/fk0.json`); same-suite control arms differ 0.01–0.04 (`k265` ctl 4.17/4.14, `k269` 4.19/4.22,
  `k272` 4.21/4.22, `k296` 4.16/4.16). So **±0.05 ms (≈1.4%)** is the measured repeat band.
- *Markov step cost.* mean 78–88 µs a step at 51% misses (`trace/out/markov_mech.txt`, `trace/RESULTS.md` §3);
  a step is 9.8 µs on a hit, 142–162 µs on a miss (M).
- *Keyed noise.* a sampled pass pays the keyed-noise tap, 24.4 µs a Markov step ≈ +0.15 ms a six-step pass
  (`trace/RESULTS.md` §8; `draft.zig:153–159`). `TF_DS_T26_KEYED_DRAFT=1` is on both then and now
  (`runs/ultrafast/prof/g-r1.rank0.cmd`, `runs/k296/ctl1.armed.txt`); fk arms are greedy and pay none.
- *Streams a pass.* 1.00 in Nov 15 and rowcost (M, profile fields); 1.36–1.43 in the served suites (M). A four-stream
  five-step pass is ~7.2 ms of tiles (`trace/out/pass_x4.py` output per `trace/RESULTS.md` §5) — multi-stream passes
  lift a suite mean well above the one-stream figure.

**The like-for-like Nov 15 → 2bb052f delta (D, corrections I).**
Nov 15 as measured = P(5 rows, ~4.0 steps, 32% sampled passes) = 3.43 ms. Normalize: −0.32 × 0.15 ms keyed noise
= 3.38 ms unkeyed (I); +0.9 step × 0.08 ms ≈ 3.39 ms at 5 steps (I). Today's equal-work figure = fk0 = 3.51 ms
(5 rows, 5 steps, greedy, 1.00 stream — the traced `k0x1` shape). **Delta = +0.05 to +0.08 ms a pass, +1.5 to
+2.3%, i.e. +0.12 to +0.19% of the 42.8 ms cycle ≈ −0.15 to −0.25 tok/s at the probe's 128 tok/s.** That sits at
the edge of the measured same-work repeat band (±0.05 ms, fk0's cells 3.42–3.52 and the same-suite control arms)
and far under the pair-gate floor (0.28–0.64% on the GPU clock, `continue.md` "Identical-code pairs" bullet,
re-derived there from `k265` data). Against it, today's build runs `TF_DS_CONF_PACK=1`, which folds the confidence input from
four launches into one (`draft.zig:697–713`; the Nov 15 source at `8ff4797` has only the four-launch branch) — an
unpriced offsetting *saving* (I). **Verdict: no provable per-pass regression at equal work; if one exists it is
≤ ~0.15 ms/pass (≤0.35% of the cycle), below what any single arm can prove.**

**What the 3.9 (trace) vs 4.16–4.21 (served `pass_ms`) gap is (M + D).** ≈ +0.26–0.36 ms, composition:
1. the trace's 3.9 ms is a *greedy, one-stream* pass; the served suites mix in **sampled passes** (+0.15 ms keyed
   noise each) and **multi-stream passes** (streams/pass 1.36–1.43; the only traced four-stream pass reads 7.2 ms
   of tiles) — `k265/ctl1.profile.txt` (`streams_a_pass: 1.36`, `pass_ms: 4.17`), `trace/RESULTS.md` §5;
2. `pass_ms` includes the ~0.10 ms pass tail (pick, confidence head, synchronize, two downloads) that the tiles
   exclude (`trace/RESULTS.md` §5–6);
3. the tiles include the 0.39 ms absorb/pick/uploads hand-off that `pass_ms` excludes (absorb 0.26 + logits 0.05
   are separate fields) — it cancels against (2) when going tiles → `pass_ms`.
The exact sampled/multi-stream split is not recoverable: `pass_ms` is a lifetime mean and the workload cells save
no per-cell `pass_ms` (`runs/overnight-20261122/kvreuse-size-20261122/workload.py` cell schema: `per_round.tail`
only). The claim's "~3.9 ms GPU plus 0.49 ms of gaps" reads the trace correctly: 0.39 + 0.10 = 0.49 ms of
hand-offs (`trace/RESULTS.md` §2, §6).

**If the "degradation" is a measurement artifact, say so plainly:** **yes, for the pass.** 3.43 → 4.21 is not one
quantity changing; it is (a) a deliberate policy widening of the pass (5 → 6 block rows and 5 → 6 Markov steps
under `TF_DS_DRAFT_BLOCK=6` + `KEPT=2`, +0.24–0.39 ms/pass, M rowcost ladder), and (b) workload composition of the
suite mean (streams/pass, sampled keying). The residual equal-work change is +0.05–0.08 ms — unprovable.

---

## 1. Verdicts

Cycle reference: 42.8 ms served solo cycle = 38.27 forward + 4.17 pass + 0.26 absorb + 0.05 logits
(`k265/ctl1.profile.txt`; the sampled probe reads 42.55 ms a round on the tetris row, `k265/ctl1.probe.json`:
`decode_s 16.0 / 376 rounds`).

### Claim 1 — "live 3.1 tokens/round vs 4.2–6.1 on the sampled probe (~40% under the code-like rows)"

**Verdict: the figures are correct as recorded; as a *degradation* claim it does not hold. It is workload, not
regression.** (M) Live: E2 alone median 3.1 tokens/round (2.8–4.2, n=280) and 77.8 tok/s median against 105–135
on the code-like probe rows, "−40%" (`observe/RESULTS.md` rows 3 and 5); the current build's E7 alone reads
2.89–4.64 (n=5) and its three stretches 2.01–3.72 (`observe/RESULTS.md` §2). Benchmark: 4.2–6.1 tokens/round on
the 14 code-like probe rows (json 4.2–4.3 … go 5.8–6.1), acceptance 0.73–0.87; **the prose rows of the same probe,
same build, same config, read 1.8 tokens/round at 0.51–0.54 acceptance** (`k265/ctl1.probe.json`, 16 rows with
per-row `accepted` and `hist`). The within-config band is 1.8–6.1; live's 3.1 sits inside it
(`observe/RESULTS.md` row 3: "Cannot be compared on content; live sits between the two classes").

| | ms/token | % of 42.8 ms cycle | tokens/s |
|---|---|---|---|
| Live (3.1 tokens/round @ ~40.8 ms a round, M `observe/RESULTS.md` §3) | 13.2 | 95% of cycle per round | 76–79 |
| Benchmark bottom (4.2 @ 42.6 ms) | 10.1 | — | 98.6 |
| Benchmark code-like mean (5.3 @ 42.6 ms, D from `k265/ctl1.probe.json`) | 8.0 | — | 124.5 |
| Gap (3.1 → 4.2 / → 5.3) | +3.1 / +5.2 ms/token | — | **−22.6 (−23%) / −48 (−39%)** |

### Claim 2 — "pass 3.43 (Nov 15) → 3.51–3.90 (now) → 4.16–4.21 (served)"

**Verdict: the recorded figures are correct (the served range is 4.11–4.26 across suites), but the growth is a
measurement artifact plus a deliberate policy change. Like-for-like Nov 15 → 2bb052f: +0.05–0.08 ms/pass
(+1.5–2.3%) — not provable; +0.12–0.19% of the 42.8 ms cycle; −0.15 to −0.25 tok/s at 128 tok/s.**

| Component of the apparent 3.43 → 4.21 (+0.78 ms) | ms | % of 42.8 cycle | tok/s at 128 | Grade |
|---|---|---|---|---|
| Policy widening, 5→6 block rows + 5→6 steps + cuBLAS gate at 6 rows (kit gate exists at 5/10/15 rows only); fk0 → fk6 is the 0%→100% widened envelope, fk1–fk5 the mix | +0.24 … +0.39 | +0.6 … +0.9% | −0.7 … −1.2 | M (rowcost ladder; `runs/draft-block-20261118/RESULTS.md`; gate rule `draft.zig:769–775`) |
| Served suite composition vs greedy one-stream (sampled keying +0.15/pass; streams/pass 1.36–1.43; pass tail in `pass_ms`) | +0.26 … +0.36 | +0.6 … +0.8% | −0.8 … −1.1 | M anchors, split not recoverable |
| Nov 15 baseline normalization (keyed noise share, step mix) | −0.05 … +0.02 | ±0.1% | ±0.1 | I (corrections on M anchors) |
| Equal-work regression | +0.05 … +0.08 (max plausible ~0.15) | +0.12 … +0.19% (≤0.35%) | −0.15 … −0.25 | I, at/below the ±0.05 ms same-work repeat band, far below the 0.5% provable floor |

The widening is a net win where it engages: block 6 raised tokens/round 4.76 → 5.13–5.38 on the Tetris prompt
and tok/s +2.5–3.1% on Tetris/code at 1–2 streams (`runs/draft-block-20261118/RESULTS.md`).

Per-stage baseline of the pass today (M, `trace/out/pass_breakdown.txt`, greedy 6-step one-stream): vocab head
1.08 ms (a second full head read a cycle), MoE stages 1.29, attention stages 0.68, Markov steps 0.49 (of which
0.44–0.49 is the 3.2 bias misses/pass at 133–153 µs each), ring transfers 0.23–0.26 (of which 0.07–0.09 is random
rank skew above the floor), hand-off into the pass 0.39 (holds `absorb_ms` 0.26 + `logits_ms` 0.05), tail 0.10.
Per-stage attribution of the *delta* is impossible from saved data (Nov 15 has no pass breakdown; 0280's pass
profiler has never executed) — and unnecessary, since the delta sits inside the same-work repeat band.

---

## 2. Ranked causes and what is disproven

### A. Acceptance (the 3.1 → 4.2/5.3 tokens/round gap)

| # | Cause | Magnitude | Evidence | Confidence |
|---|---|---|---|---|
| 1 | **Prompt class and mix (content)** | spans 1.8 (prose) – 6.1 (go) tokens/round at identical config; live 3.1 inside; live acceptance 64% between prose 0.51–0.54 and code 0.73–0.87; live drafts ≈3.3 rows/round (D: (3.1−1)/0.64) vs probe code-like ≈5.0–5.9 (D) | `k265/ctl1.probe.json` per-row `accepted`/`hist`; `observe/RESULTS.md` rows 3–5 | High that it dominates (the band contains both values); the exact split is unprovable — content is not in the metrics (`observe/COVERAGE.md`: no per-request series, `requests_total` not stored) |
| 2 | **Context depth** | round time +7.5% (8–64K), +9.5% (64–200K), +12.3% (>200K) vs the short-prompt line; live prompts mean ~155K, 95–98% cached; ≈8–12% of the −40% tok/s (≈10–15 tok/s of the ~46) | `observe/RESULTS.md` §2 row 2, D1 (E2 n=11/155/114) | M for round time; (I) that it does not also move acceptance |
| 3 | **Draft policy (`DRAFT_BLOCK=6`, `KEPT=2` widening)** | measured effect already deployed: +0.37–0.62 tokens/round on code-like, ~0 on prose; remaining headroom `DRAFT_BLOCK=7` +0.19 on Tetris (5.13→5.32); round-cost table refit measured −0.24% round time with tokens/round unmeasured | `runs/draft-block-20261118/RESULTS.md`; `rowcost/RESULTS.md` | M for the deployed gain and block 7; live does produce keeps (accepted 60–92% of ~2–5 drafted at 1 stream ⇒ kept ≥ 2 common, `observe/RESULTS.md` §2/E7 table), so widening engages — but on code-like rows only does the *cap* bind (51% of tetris rounds hit 7 rows, `hist` in the probe JSON) |
| 4 | **Drafter weights (D1)** | none — D1 is *worse*: −2.42% decode, rounds up on **16 of 16** probe rows (e.g. json 360→371, prose 421→442, sql 275→282), both arms identical; D1 was fit on a different quant (LibertAIDAI EXL3 2.9 vs served Mia EAQ v1 + Keys ablit) | `runs/d1-20261123/RESULTS.md`, `runs/d1-20261123/{ctl1,d1a,d1b}.probe.json` | M — **disproven as an improvement and as a gap cause**: live and benchmark run the *same* weights |
| 5 | **Triage / confidence gating** | none | `TF_DS_DRAFT_TRIAGE` unset in production (absent from `runs/k296/ctl1.armed.txt` and `.env`); `TF_DS_CONF_PACK` is byte-identical packing (`draft.zig:701–703`); `TF_DS_CONF_LEADER_ONLY` off and output-invariant by construction (`runs/k280/README.md` gate note) | M — **disproven** |
| 6 | **Markov bias cache (256 rows)** | none on acceptance by construction; it prices pass time only | `markov_tokens.zig:4`: "changes the time a step takes, never a draft" | M — **disproven for acceptance** |
| 7 | **Genuine build regression** | not demonstrated: E7 vs E2 at equal depth +4.0/+5.1 vs median +7.5, +9.9 vs +9.5, +8.0 vs +12.3 (n=4); tokens/round E2 3.1 vs E7 2.89–4.64 (n=5) | `observe/RESULTS.md` D1 | M that no sign exists; (I) bounded above ~10% of the gap by sample size |

**Split of the 3.1 → 4.2 gap (1.1 tokens/round).** Prompt-dependent **0.9–1.1 (≈80–100%)**; policy-tunable
**0.0–0.2 (0–20%)** — `DRAFT_BLOCK=7` and only if live content is code-like; genuine-regression **0.0** (no
evidence anywhere; both sides of the comparison run the same build class and weights). Against the code-like mean
5.3 the same shares hold with the prompt share ≈2.0 of 2.2.

### B. Pass cost (the 3.43 → 3.90/4.21 growth)

| # | Cause | Magnitude | Evidence | Confidence |
|---|---|---|---|---|
| 1 | **Workload composition of `pass_ms`** (streams/pass 1.36–1.43, sampled keyed-noise passes +0.15 ms, 6-row/6-step passes) | +0.26 … +0.36 ms of the served-vs-greedy gap | `k265/…profile.txt`, `runs/k296/…profile.txt`, `trace/RESULTS.md` §5/§8 | M anchors; exact split not recoverable (lifetime mean) |
| 2 | **Policy widening (0192 `DRAFT_BLOCK=6` + `KEPT=2`)** | +0.24 … +0.39 ms/pass at one stream (5 rows/5 steps → 6/6 envelope; the fk1–fk5 gradient is the widened share rising with K), incl. the kit→cuBLAS gate change at 6 rows | `rowcost/fk*.profile.txt` + `fk*.json`; `runs/draft-block-20261118/RESULTS.md`; `draft.zig:769–775` | M — deliberate, net-positive on code content |
| 3 | **Equal-work regression** | +0.05 … +0.08 ms/pass (0.12–0.19% of cycle); plausible ceiling ~0.15 | §0 normalization | I, **below the provable floor** (±0.05 ms same-work `pass_ms` repeat band; pair-gate 0.28–0.64% on GPU clock) |
| 4 | Markov bias misses (cost, not delta) | 0.44–0.49 ms/pass (3.2 misses × 133–153 µs), 1.0–1.15% of cycle; 0281 would recover 0.16–0.26 ms at the held-out hit rates | `trace/RESULTS.md` §3, `trace/out/markov_mech.txt`; cache 256 rows `tri_markov.zig:22–23` (identical at Nov 15 source `8ff4797`) | M for cost, I for the recovery |
| 5 | Collectives in the pass (cost, not delta) | 12 ring transfers, 0.23–0.26 ms of which 0.07–0.09 is rank-skew above the floor | `trace/RESULTS.md` §2/§5 | M |
| 6 | No L2 prefetch in the pass (opportunity, not delta) | ≤0.18 ms/pass by analogy with the forward's 5.3% | `draft.zig:10` ("the served L2 prefetch forks move no bytes and are left out") — **the same line is in the Nov 15 source** (`git show 8ff4797:…/draft.zig`); `trace/RESULTS.md` §5 | M that it is unchanged; I on the size |

**Disproven, explicitly.**
- "No L2 prefetch in the pass" as a *regression*: **disproven** — the pass never had it (identical header and pass
  body between `8ff4797` and `2bb052f` apart from the block-rows/conf-pack additions; diff recorded below).
- k267's +0.25 ms carryover into the next forward as a production cost: **disproven** — `0267 TF_DS_MARKOV_FUSED`
  is not in the `2bb052f` lineage (`git log ~/Workspace/tfref-kstack4`), not armed in production
  (`runs/k296/ctl1.armed.txt`), and was rejected after k267/kall (greedy loss +2.75–3.66 ms/round,
  `continue.md` "Unexplained" — noted as a record only). Its mechanism is untraced (`trace/RESULTS.md` §6).
- "11–12 collectives a pass" and "3.2 bias misses a pass" as *deltas*: **disproven as deltas** (both structures are
  unchanged since Nov 15 — `tri_markov.zig` is byte-identical between `8ff4797` and `2bb052f`), **confirmed as
  standing costs** (0.23–0.26 and 0.44–0.49 ms/pass, M).
- Any pass-code regression from the Nov 15 build: **not demonstrated**. The only pass-path differences in the git
  history are `TF_DS_DRAFT_BLOCK` (policy), the kit→cuBLAS gate at non-5/10/15 row counts (its consequence), and
  `TF_DS_CONF_PACK` (production, a saving) — verified by diffing `draft.zig` at `8ff4797` against `2bb052f`
  (block-rows helper `blockRows`, `dr.bn`, the `rowmmGateFits` gate, the `confPack` branch, ring-trace hooks;
  `tri_markov.zig` unchanged). Patches in `zig/patches/README.md` that touch the pass (`0192`, `0260`) say the
  same. The drafter fused-route experiment (0196) is **not** in the recipe (`continue.md` 2026-11-20 block;
  no `FUSED_ROUTE` reference in `draft.zig` at `2bb052f`).

---

## 3. One priced experiment

Both claims settle from saved data, and the one residual (equal-work delta ≤0.15 ms) is below what one arm can
prove (0.28–0.64% GPU-clock floor; ±0.05 ms same-work `pass_ms` repeat band), so no window is spent on it. The one
window
worth taking prices the two standing pass-cost levers, which the saved traces price only by inference — and only
hardware can answer whether a hit from a 221–473 MiB Markov cache is still as cheap as from today's 31.6 MiB
(`runs/k280/README.md`).

**Experiment: the staged `runs/k280/run.sh` (0281 Markov cache + 0282 pass prefetch), ~1 h 45 min, no new build**
(image `tensorfold-dsv41:bench-fused-next-1421187` = `2bb052f` + 0280–0283, all switches off, built on both ranks;
production is restored and validated at the end — the run must end `restored: validate rc=0`).

On the head, from the recipe root, production idle (after the two `rsync` lines in `runs/k280/README.md`):

```
mkdir -p runs/k280/main && setsid nohup bash runs/k280/run.sh > runs/k280/main/run.log 2>&1 < /dev/null &
```

Score (workstation or head):

```
python3 runs/k280/gate_drafts.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b,mc4096a,mc4096b,dpf1a,dpf1b,dpf2a,dpf2b
python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b
python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test mc4096a,mc4096b
python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test dpf1a,dpf1b
python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test dpf2a,dpf2b
```

Read `pass_ms` in the "Rank 0 lifetime profile" block first (0.01 ms resolution).

Confirm / kill, per hypothesis:

| Hypothesis | Confirm | Kill |
|---|---|---|
| H1: 0281, more cached Markov bias rows recover ~0.16–0.26 ms a pass (2,048 / 4,096 rows) and cache hits stay cheap at 221–473 MiB | `pass_ms` falls ≥0.15 ms against **both** controls with `gate_drafts.py` exit 0 and `markov_cache` miss rate falling ~52% → ~32% (2,048) / ~24% (4,096) | `pass_ms` flat (≤0.05), or drafts move (gate exit ≠ 0), or the miss rate does not fall proportionally, or `MemAvailable` trends to the 2.5 GiB memguard floor in `mc4096` |
| H2: 0282, the pass's missing prefetch is worth up to 0.18 ms | `dpf1` `pass_ms` falls ≥0.08 ms, drafts unchanged | ≤0.03 ms — close the record item; `dpf2` must be judged on **forward + pass** (its hand-off join can raise `pass_ms` while lowering the forward) |
| H3: no equal-work pass regression (the §0 residual is noise) | `ctl1`/`ctl2` `pass_ms` lands within the fk6 same-work band (3.85–3.90 for one-stream greedy 6-row traffic) and the exact/sampled panels are IDENTICAL | not killable at this size in one window; only a multi-arm ABBA on a 5-row configuration could, and its ceiling (0.35%) stays under the 0.5% floor |

Optional, same window (one extra line at the head of `runs/k280/arms.txt`, operator action — this analysis writes
only under its own directory): `db5a|tensorfold-dsv41:bench-fused-next-1421187|TF_DS_DRAFT_BLOCK=5` (and `db5b`
after `ctl2`) would give the equal-work 5-row pass directly against Nov 15's 3.43; replies keep their bits at any
block (`runs/draft-block-20261118/RESULTS.md`), both ranks must carry the value.

---

## 4. What the saved data cannot answer

1. **The content split of the acceptance gap.** Live prompts and outputs are not readable from any stored metric
   (`observe/COVERAGE.md`: `requests_total{key,status}` scraped but not stored; the `done` line and round profile
   are log-only; rank 1 has no `/metrics`). The 80–100% prompt-share is a bound, not a measurement.
2. **The equal-work pass delta below ~0.3 ms.** The same-work `pass_ms` repeat band is ±0.05 ms (≈1.4%) and the
   pair-gate floor is 0.28–0.64% on the GPU clock; Nov 15's step mix and sampled-pass share are derived from its
   log lines, not traced. Which input the widening gate reads (`r.follow.len` = kept rows vs verified rows) is not
   settled — the fk1–fk6 ladder fits the kept-rows reading (widening share), and the mech arms' 101/756
   non-widened production passes fit neither reading cleanly.
3. **The Markov miss rate on Nov 15's prose-heavy workload** (no counter existed before 0280) — it shifts the
   step-mix correction by ~±0.05 ms.
4. **The sampled vs multi-stream split of the 4.16–4.21 suite mean** — `pass_ms` is a lifetime mean; the workload
   cells save `per_round.tail` but no per-cell `pass_ms`.
5. **Whether live rounds keep ≥2 rows (widening engagement) per request** — only aggregate `mtp_accepted/drafted`
   counters per 30 s scrape exist; per-request accepted/drafted is in `done` lines that never reached the store.
   Draft rows/round live (≈3.3) and the keep threshold are inferred.
6. **A hit read from a 505 MiB Markov cache vs today's 31.6 MiB** — needs the `mc` arms (the one priced
   experiment).
7. **What rank 1's pass reads** — profile lines quoted are rank 0's; rank 1 needs `RANK1_PROFILE` at exit
   (`runs/k280/passprof.sh` would print both).
8. **k267's forward carryover mechanism** — no trace exists with `0267` on (`trace/RESULTS.md` "What the traces
   cannot answer"); moot for production, since 0267 is not in the deployed lineage.

## Evidence appendix — commands run (all read-only)

- `python3` one-liners over `runs/overnight-20261122/rowcost/fk*.json` (per-cell `pass_ms` 3.42–3.52 at fk0),
  `rowcost/fk*.profile.txt` (3.51/3.75/3.82/3.82/3.86/3.86/3.90), `k265/{ctl1,ctl2}.probe.json` (16 probe rows:
  per-row rounds, accepted/drafted, hist), `runs/d1-20261123/{ctl1,d1a,d1b}.probe.json` (rounds up on 16/16 rows).
- `grep`/`sed` over `runs/k296/*.profile.txt` (4.16–4.21, `streams_a_pass` 1.36), `k265/k269/k272/*.profile.txt`
  (4.11–4.26), `runs/ultrafast/prof/g-r1.rank0.log` (Nov 15 profile + per-request drafted/accepted),
  `runs/ultrafast/prof/g-r1.rank0.cmd` and `uf_profreq.py` (Nov 15 env and workload), `runs/k296/ctl1.armed.txt`
  (production env), `runs/sched-20261122/b2/rowsplit/arms.txt` (`k0x1` = `TF_DS_FIX_K=0`).
- `git show 8ff4797:zig/src/families/dsv41/{draft,tri_markov}.zig` (Nov 15 pass source) diffed against
  `~/Workspace/tfref-kstack4` at `2bb052f`: pass-path delta = block rows, cuBLAS gate at 6 rows, `confPack`;
  `tri_markov.zig` identical; `draft.zig:10` prefetch note present in both.
- Re-read `trace/out/{pass_breakdown,markov_mech,recon_all,markov_other}.txt`, `trace/RESULTS.md`,
  `observe/RESULTS.md`, `observe/COVERAGE.md`, `runs/d1-20261123/RESULTS.md`, `runs/draft-block-20261118/RESULTS.md`,
  `rowcost/RESULTS.md`, `runs/k280/README.md`, `docs/benchmark-gaps-and-moe-review.md`.
- `git log` over all `~/Workspace/tfref-*` worktrees for image/commit lineage (0265–0268 absent from `2bb052f`).
