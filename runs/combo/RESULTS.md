# Combined-configuration campaign (2026-11-16): results

Image `exl3-tp2-b40ea899e367` (TF_REF + `0001`-`0128`) in every block. Runner `scripts/bench_overnight.py` with
`CAMPAIGN_OUT=runs/combo CAMPAIGN_IMAGE=exl3-tp2-b40ea899e367 CAMPAIGN_PROD={} CAMPAIGN_CLOCK_ABORT=0`; schedule
`runs/combo/schedule.json`; tables `runs/combo/tables.md` (`REPORT_IGNORE_CLOCK=1 python3 tools/overnight_report.py runs/combo`).

**Base of every block** = the head's `.env` as found (production): `TF_DS_T13_EXPERT_PRUNE=1` (lossy, unpriced),
`TF_DS_T26_KEYED_DRAFT=1`, `TF_DS_T27_DEVICE_DRAW=1`, `TF_DS_PREFILL_SLICE=5`, `SPILL_GIB=180`, `KV_SHARD=1`,
`PARALLEL=8`, plus the runner's explicit `LEAD=0 ROWCACHE_MB=0 STRICT=0 SCALES=io DEDUPE=0 FAST=0 T25=0
GRAPH_CAP=128 ENGRAM_AIO=1`. The GPU clock cap (2400 MHz) held in every block (max 2392/2385 MHz H/W).
Corpus: `runs/overnight/corpus` (diverse prose + code + p125), the same as the overnight pass.

Gates per arm: `tools/validate.py --compare runs/matrix/base.validate.json` (8 cases incl. concurrent == solo);
`tools/exact.py record --like runs/combo/blocks/c01.exact.json` + compare (39 cases; **note**: `--like` reuses the
salt, so the candidate's prompts resume from the spill tier: this checks resume + decode identity, not cold
prefill — the cold arms are in phase 3); `sampled_ident` (4 seeded temperature-1 requests, production sampler,
`token_sha` per seed; base repeats bit for bit within and across restarts).

Harness notes: the known first-request `CudaFailed` (decisions.md 2026-11-15) hit every restart **except the
`DEVICE_ACCEPT` arm**; the runner accepts a restart only when that is the single error and a retried smoke answers.

## Phase 1: single switches (missing rows of to-benchmark.md "Decode-step structure")

One block each, bases c01/c04/c07/c10 adjacent. Verdict cells: paired decode (same prompt per rep) against both
adjacent bases, band = max(spread, base-to-base gap). One block = a screen, not a sign-off.

| Block | Arm | validate | exact (resume) | sampled | Prefill 8K / 32K / 100K vs bases | Decode 1 / 4 / 8 (paired) | think t1 / t4 | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c02 | `TF_DS_DEVICE_ACCEPT=1` | IDENTICAL | IDENTICAL | identical | -0.5 / +1.6 / +0.3% (within) | -0.4 / -0.4 / -0.3% (within) | -0.1 / +0.9% | neutral; identity holds greedy and sampled |
| c03 | `TF_DS_COMPRESS_SIDE=1` | IDENTICAL | IDENTICAL | - | **-3.0%** (band 1.5) / +2.3 / +0.3% | -0.0 / -0.5 / -0.4% (within) | +0.3 / +0.4% | neutral; 8K prefill dip in one block |
| c05 | `TF_DS_PDL=1` | IDENTICAL | IDENTICAL | - | +0.8 / +0.3 / -0.3% (within) | +0.3 / +0.0 / +0.1% (within) | +0.1 / +0.1% | neutral |
| c06 | `TF_DS_FUSED_COMPRESS2=1` | IDENTICAL | IDENTICAL | - | +0.5 / +0.8 / -0.2% (within) | **+0.5** / -0.2 / -0.2% | +0.5 / +0.9% | neutral to +0.5% decode (one block) |
| c08 | `TF_DS_ENGRAM_BOTH=1` | IDENTICAL | IDENTICAL | - | +1.7 / 0.0 / -0.6% (within) | -0.2 / -0.6 / **-1.0%** (band 0.9) | -0.3 / **-1.1%** | small decode cost at 4-8 streams; 2026-11-15's block also read multi -1.4/-6% |
| c09 | `LEAD=1 SCALES=map ENGRAM_BOTH=1` | IDENTICAL | IDENTICAL | - | **+7.3 / +1.5 / +4.3%**; p125 8K **+7.5%** | -0.9 / -0.8 / -0.0% | -0.4 / **-4.7%** | LEAD+map's prefill win reproduces; t4 (thinking, 4 streams) -4.7% — BOTH or LEAD+map? phase 2 decides |

Already measured elsewhere, not re-run here: `0119` token cache (+, resumed-turn `encode_s` -52 ms, 2026-11-15);
`0121` exact resume (LOSSY, opt-in only); `0122` spec sample — a pass on 2026-11-15 08:27-08:59Z
(`runs/ultrafast/status.log` on the head, not previously recorded): sampled prose agg tok/s 1 / 4 streams
`SPEC_SAMPLE=1 T26=0` 42.1-48.0 / 88.6-90.6, against `T26=0` 55.9-58.1 / 109.0-109.1 and production `T26=1`
53.5-55.5 / 104.9-106.2: **~20% slower**, and its `samp_test` gate is degenerate (the prompt gives one first-token
id, so the chi-square passes trivially). Kept out of the stack.

## Phase 2: the stack (3 passes, s01-s13, 09:42-11:56Z)

Arms rotated over three passes, a base block between passes: `known` = `TF_DS_ENGRAM_LEAD=1
TF_DS_ENGRAM_SCALES=map TF_DS_ENGRAM_STRICT=1 TENSORFOLD_TOKEN_CACHE_MB=32`; `best` = known + `TF_DS_DEVICE_ACCEPT=1
TF_DS_COMPRESS_SIDE=1 TF_DS_PDL=1 TF_DS_FUSED_COMPRESS2=1`; `full` = best + `TF_DS_ENGRAM_BOTH=1`. Full tables:
`runs/combo/stack.md` (`python3 runs/combo/stack_report.py`). Every stack block: validate IDENTICAL; s03 (full) and
s04 (best) exact IDENTICAL and sampled shas equal base's for all 4 seeds; armed on both ranks (env proof, LEAD banner).

Vs the bracketing bases (median of 3 passes [min..max]; base-to-base gap in the last column):

| Metric | best | full | known | base gap |
| --- | --- | --- | --- | --- |
| prefill 8K | **+6.9%** [+4.4..+7.6] | +6.2% [+3.4..+7.5] | **+7.7%** [+5.9..+7.7] | 1.8% |
| prefill 32K | **+5.3%** [+4.7..+5.3] | +4.7% [+4.0..+5.3] | **+5.4%** [+4.2..+5.9] | 0.7% |
| prefill 100K | **+4.1%** [+3.2..+4.7] | +4.2% [+3.1..+5.1] | **+4.7%** [+3.2..+4.8] | 0.5% |
| TTFT 8K | **+8.6%** better | +7.3% | **+8.3%** | 3.1% |
| code 8K | **+4.7%** [+1.9..+5.6] | +1.0% [-1.0..+4.0] | **+4.0%** | 0.7% |
| p125 8K | **+9.6%** | +9.4% | **+9.6%** | 0.8% |
| decode 1 / 4 / 8 (paired) | +0.5 / +0.5 / **+1.0%** | +0.4 / +0.6 / +0.8% | +0.2 / +0.5 / +0.9% | 0.2% |
| think t1 / t4 | +0.5 / -0.2% | +0.0 / -0.9% | +0.1 / -0.3% | 1.4% (t4) |

Arm vs arm, same pass (median of 3): best/known within ±1.2% on every metric, either sign (pass 1 read +1-1.6% on
prefill, pass 3 the reverse); best/full +1.0% prefill 8K, +2.9% code 8K, +0.7% TTFT, +0.3% dec8, +2.0% t4.

Reading:
- **The stack composes**: nothing in it degrades the known wins, and every arm keeps greedy and sampled bits.
  The gain over production is LEAD + mapped scales (prefill +4-10%, TTFT ~8%, decode ~+1% at 8 streams).
- **`0124`-`0127` add nothing measurable** on top (best = known within the pass-to-pass spread) — as the profile's
  ~1% ceilings predicted. Keep-worthy as harmless, not as wins. `DEVICE_ACCEPT` has one side effect of note: every
  restart with it on (21 of 21 over the campaign) skipped the known first-request `CudaFailed`; every restart
  without it (29 of 29, the restores included) hit it. A pointer for the root cause, not a fix.
- **`ENGRAM_BOTH` is a mild degrader** (full < best on 8 of 12 metrics; -1% dec8 alone in c08): leave it off.
- Cold cost of `SCALES=map` (the first request after a restart; overnight -82% / -41%) is absorbed by the `warm`
  cell and is not in these figures: see `cold.*` in `tables.md`.

## Phase 3: soak, cache-policy cluster, cold identity (p01-p07, 11:59-13:22Z)

**Soak** (p01 base, p02 best): dec1/dec8, then 15 uncached ~100K-token prefills (~1.5M tokens through the spill
tier, `cached_tokens=0` on every one), then the same dec1/dec8 prompts again (`dec1b`/`dec8b`).

| Arm | soak 100K tok/s, reps 1-3 → 13-15 (median) | drift | dec1 before → after (paired) | dec8 before → after (paired) | validate |
| --- | --- | --- | --- | --- | --- |
| base (p01) | 1944 → 1923 | -1.1% | -0.4% | +0.4% | passed |
| best (p02) | 2021 → 1984 | -1.8% | +0.0% | -0.3% | passed |

- best holds its 100K prefill gain under sustained load: **+4.0%** over base across the soak, as phase 2 found (+4.1%).
- Decode after 1.5M tokens matches decode before, in both arms. There is no leak or slowdown from the stack.
- The prefill step at rep 13 (~1% to 2%) shows up in **both** arms at the same point. It is common mode (likely spill
  eviction once the 180 GiB tier fills, or thermal: max 87 °C), not something the stack causes.

**Cache-policy cluster** (p03 = best + `TF_DS_SHARED_CUTS=1 TF_DS_EDIT_MARK=1 TF_DS_KEEP_PROTECT=1
TF_SPILL_STRIDE=8 TF_DS_SHARED_PLAN=1`, beside p04 = best). One pair, so this is a screen:
- validate passed. exact IDENTICAL against c01. It did **not** resume: `cached_tokens` was 47,104, the cold figure,
  against 344,064 in the resumed c/s blocks, because the soak blocks had evicted c01's prompts from the spill tier.
  So the cluster is gated cold only, and resumed identity is still owed. The sampled shas equal base's for all 4 seeds.
- Every prefill cell was cold (`cached_tokens=0` on all 17 requests in both blocks), so the cluster's hit-rate
  purpose is **not** exercised here. This only checks that it costs nothing.
- p03/p04 (computed from the block medians): prefill 8K +0.8%, 32K +1.6%, 100K +1.6%, code +2.5%, p125 +1.5%.
  p03 sits above every phase-2 `best` block, so this needed more passes. Phase 4 overturns it (below).

**Cold identity** (`SPILL_GIB=0`, so nothing can resume from disk; p05 base records a fresh salt, p06/p07 rerun it
with `--like`, which closes phase 2's resume-only caveat): `cached_tokens` = 47,104 of 398,839 prompt tokens in
all three (prefix sharing inside the run only, the same in every arm).
- p06 `best`: exact **IDENTICAL** (39 cases, cold prefill + decode, 1-8 streams, up to 100K). The sampled shas equal
  base's for all 4 seeds, and repeat within the block.
- p07 `best + cache cluster`: exact **IDENTICAL**.
- The restore to production succeeded (`env_match=True` on both ranks, chat smoke ok) at 13:22Z.

## Phase 4: cache cluster and stride, 2 more rotated passes (q01-q09, 13:25-15:13Z)

Order: base, best, best+cache, best+stride, base, best+stride, best+cache, best, base. Each block ran the standard
cells plus `validate` and the hit-rate probes `tools/prompt_reuse.py 12000` and `tools/prefix_fanout.py`. Full
tables: `runs/combo/phase4.md` (`python3 runs/combo/stack_report.py runs/combo q`). The restore to production succeeded at 15:13Z (`env_match=True`).

Each arm against its bracketing bases (median of 2 [min..max]; the base-to-base gap is the noise floor):

| Metric | best | best+cache | best+stride | base gap |
| --- | --- | --- | --- | --- |
| prefill 8K | +8.1% [+7.7..+8.6] | +7.7% | +4.8% [+1.3..+8.4] | 1.2% |
| prefill 32K | +5.6% | +4.9% | +3.4% | 1.5% |
| prefill 100K | +4.6% | +4.0% | +2.5% | 1.2% |
| TTFT 8K | +9.5% better | +9.9% | +4.6% | 2.5% |
| code 8K | +5.5% | +4.6% | +3.1% | 2.0% |
| p125 8K | +10.0% | +8.2% | +7.6% | 1.8% |
| decode 1 / 4 / 8 | +0.4 / +0.5 / +0.9% | +0.2 / +0.3 / +0.8% | +0.6 / +0.5 / +1.0% | ≤0.5% |
| think t1 / t4 | -0.2 / -2.8% | +0.6 / +0.2% | +0.6 / -0.2% | 1.5% (t4) |

- validate passed in all 9 blocks.
- Hit-rate probes, every block (base included). `prefix_fanout` cached 186,368 tokens at every level in every block;
  identical counts mean hit rate is unchanged.
  - `prefix_fanout` PASS (4 of 4 levels).
  - `prompt_reuse` resumes 88% of every turn on every arm. That is base's own figure, the same as 2026-11-13, and it
    fails the probe's 90% bar on every arm.
  - The stack and the cluster leave hit rate where it was. These probes cannot show the cluster's win cases
    (history edit, rewind, pressure).
- Prefill, best against best+cache in the same pass: +0.4 to +0.9%. Code +0.9%, p125 +1.7%. Decode equal.
  - Together with p03/p04 (the reverse sign, +0.8 to +1.6%), that is 3 pairs with a median of about 0.
  - p03's reading was noise. The cluster gives **no cold-prefill gain**.
- best+stride: q06 ran low at every prefill size, including its warm cell (8K 1209 against ~1750). q04 was normal.
  The 3 cluster blocks (also stride 8) were normal too, so this looks like a whole-block state effect. Unresolved, n=2.
- best's t4 (thinking, 4 streams) read -2.8% here; c09 had read -4.7%. Pooled, 5 of 19 LEAD+map blocks fall below
  the base floor (136.4). This was resolved in phase 5.

## Phase 5: thinking at 4 streams (r01-r07, 15:16-15:58Z)

Base and best alternate ×3 with bases on each side. Each block ran `warm` + `t4x`: 6 code-thinking prompts × 4 streams,
768 tokens. The prompts are fixed per rep, so reps pair across arms, and rep 0 is the `think` cell's t4 prompt.

| Block | rep 0 | reps 1-5 | median |
| --- | --- | --- | --- |
| r02 best | -3.4% | +0.8 / +0.3 / +0.4 / +0.6 / +1.0% | +0.5% |
| r04 best | -3.3% | -0.6 / -0.2 / +1.2 / +1.3 / +0.3% | +0.0% |
| r06 best | -0.4% | +0.7 / +0.9 / -1.4 / +0.7 / +0.6% | +0.6% |

There is no t4 loss on the workload: median +0.4%. The low tail in earlier blocks is rep 0, the single prompt the
`think` cell measures, which dips about 3% intermittently under LEAD+map. It is prompt-specific and within ±3.5%.
Production was restored at 15:58Z (`env_match=True`).

## Recommendation

Pooled figures for best against production: phase 2 (3 blocks) and phase 4 (2 blocks).
- prefill: 8K +7 to +8%, 32K +5.3 to +5.6%, 100K +4.1 to +4.6%
- code 8K about +5%, p125 8K +9.6 to +10%
- TTFT 8K 8.6 to 9.5% better
- decode 1 / 4 / 8: +0.4 / +0.5 / +0.9%
- thinking t4: +0.4% (phase 5)

**Use production + the `best` arm: `TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_SCALES=map TF_DS_ENGRAM_STRICT=1
TENSORFOLD_TOKEN_CACHE_MB=32 TF_DS_DEVICE_ACCEPT=1 TF_DS_COMPRESS_SIDE=1 TF_DS_PDL=1 TF_DS_FUSED_COMPRESS2=1`.**
- `best` is the configuration the soak, the cold exact and the sampled identity ran on. `known` (the first four
  flags) performs the same, but it was gated only in phase 2.
- Against production: prefill +4-10% (8K +6.9%, 100K +4.1%, p125 +9.6%), TTFT 8K ~8.6% better, decode +0.5-1.0%.
- Greedy and sampled bits are identical, both cold and resumed.
- No drift over a 1.5M-token soak.
- 0124-0127 cost nothing and gain nothing measurable. `DEVICE_ACCEPT` is the one with a side benefit: every restart
  with it skipped the first-request `CudaFailed`.
- The cache-policy cluster (SHARED_CUTS, EDIT_MARK, KEEP_PROTECT, SPILL_STRIDE=8, SHARED_PLAN) keeps identity and
  passes the hit-rate probes. Cold prefill is a wash over 3 pairs. Add it for its hit-rate purpose (untested), not
  for speed.
- Leave `TF_DS_ENGRAM_BOTH` off.
- Known caveat of `SCALES=map`: the first requests after a restart are slow (cold cost; overnight -82%/-41%).
  `0129` `map-warm` is the fix and is not in this image.
- Not covered: `0129`-`0145`, which were added during this campaign and are not in image `b40ea899e367`.
