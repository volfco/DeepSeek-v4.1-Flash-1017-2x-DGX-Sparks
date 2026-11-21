# combo4: the odds (0182–0190) screen + combination search (2026-11-17)

Image `exl3-tp2-8f494f5632e1` = combo3 patch set + 0182–0189 + two head-only bench fixups
(`0184z` rd.hot init, `0186z` forward declaration). 0190 excluded (crashes at default flags).
Runner: `scripts/bench_overnight.py`, blocks in `blocks/`, log in `status.log`
(first run part in `part1/`; the runner stopped after s07+s08 both failed restart, resumed from s09).

"best" = TF_DS_ENGRAM_LEAD=1 ENGRAM_SCALES=map ENGRAM_STRICT=1 TOKEN_CACHE_MB=32 DEVICE_ACCEPT=1
COMPRESS_SIDE=1 PDL=1 FUSED_COMPRESS2=1.

## Verdict

**Best config unchanged.** No single 0182–0189 switch rises above noise, and none of the eight
combinations tested (stack, stack-triage, best/stack + 4 odds, best + topk-fast + t15 (bt), bt + gather-overlap
(bto), and the two stack halves; phase D in `runs/combo5/`) beats best. The stack is slightly *worse* on decode
(the engram half carries it); gather-overlap costs ~0.8 % prefill at 32K; finer crashes the engine.

## Gate (a01, default flags)

validate passed; exact IDENTICAL (compare_rc=0); sampled shas 9100 f3b07d1aedd4, 9101 b2f0544ee7fc,
9102 8b41a5237de2, 9103 68310aa1585b (unchanged); clean 17/17. With every new switch off the image is
output-identical to combo3.

## Screen (one block per arm, vs mean of 4 best blocks s01/s05/s09/s13)

Best-block spread: decode 0.1–0.8 %, pd.32K/100K 0.3–0.7 %, pd.8K 4.8 %, pd.2K 11.5 % (2K is noise).

| arm | flag | pd.8K | pd.32K | pd.100K | dec1 | dec8 | t1 | t4 | exact | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| s02 topk-fast | TF_DS_TOPK_FAST=1 | +1.9 | −0.1 | −0.2 | +0.2 | +0.7 | +0.6 | +0.5 | IDENTICAL | neutral |
| s03 finer | TF_DS_IDX_BUCKET_FINER=1 | +1.7 | +0.3 | −0.1 | +0.1 | −0.8 | −0.9 | −2.8 | **FAIL** (D: d14) | **BROKEN** under exact (see D) |
| s04 finer+topk | both | +0.6 | +0.3 | +0.4 | +0.4 | +0.1 | +0.2 | +0.5 | **FAIL** (D: d15) | **BROKEN** under exact |
| s06 gather-overlap | TF_DS_PREFILL_GATHER_OVERLAP=1 | +1.0 | −0.7 | −0.8 | +0.3 | +0.5 | +0.7 | +0.7 | IDENTICAL | small prefill loss (confirmed in D: bto −0.8 % pd.32K) |
| s12 t15-place | TF_DS_T15_PLACE_BOUND=1 | +2.7 | +0.8 | – | −0.2 | +0.5 | −3.5 | +0.5 | IDENTICAL | inactive; cap absent |
| s07 bf16-partials | TF_DS_BF16_PARTIALS=1 | – | – | – | – | – | – | – | – | **BROKEN**: smoke reply garbage (2/2 restarts) |
| s08 argmax | TF_DS_GREEDY_ARGMAX=1 | – | – | – | – | – | – | – | – | **BROKEN**: smoke reply garbage (2/2); not the 0186z decl (matches the definition) |
| s10 head-last | TF_DS_HEAD_LAST_CHUNK=1 | – | – | – | – | – | – | – | – | **BROKEN**: rank pair hangs on the 8K warm prompt ("no progress for 120 s"), server restarts |

The s12 and phase D rank inspections show applied T13 prune but no `TF_DS_T13_EXPERT_CAP`. Patch 0053 returns no place bound in that case. The T15 flag in bt, bto and the odds stacks was inert. These results still compare top-k and gather overlap under that common setting. A later off/on/on/off run with cap 12,1 active on both ranks measured 4.18/4.18/4.17/4.19 s median decode at eight fixed rows. [Capped retest](../review-round/RESULTS.md#applied-cap-t15) [s12](blocks/s12.json) · [d02](../combo5/blocks/d02.json)

## Phase B + C (4 alternating passes each; MID cells; % vs mean of 8 best blocks b01/b06/b08/b10/c01/c06/c07/c12)

Best-block spread: dec1 0.8, dec4 1.0, dec8 1.1, pd.32K 2.6, t1 3.4, t4 2.9 %. pd.8K/ttft.8K have
sporadic first-sample outliers in every arm (e.g. b01, c05, c07); use pp.8000 (steady) instead.

| arm | dec1 | dec4 | dec8 | pd.32K | pp.8K (blocks) | t1 | t4 | exact | sampled |
|---|---|---|---|---|---|---|---|---|---|
| best (8) | 0 | 0 | 0 | 0 | 2168–2267 | 0 | 0 | gate | gate |
| stack (4) | −0.4 [−0.4,−0.3] | −0.5 [−0.7,−0.3] | −0.7 [−1.0,−0.4] | +0.3 | 2168–2273 | −1.2 | −0.6 | IDENTICAL | – |
| stack-triage (4) | −0.5 [−0.6,−0.3] | −0.5 [−0.7,−0.3] | −0.8 [−1.2,−0.3] | −0.1 | 2199–2264 | −0.2 | −1.3 | IDENTICAL | – |
| best-odds (4) | +0.0 | +0.2 | −0.0 | −0.1 | 2229–2263 (≈ −0.7 %) | +0.2 | +0.2 | **FAIL** | match |
| stack-odds (4) | −0.5 | −0.4 | −0.8 | +0.1 | 2241–2253 | −0.1 | −0.9 | **FAIL** | match |

- stack = best + ENGRAM_SCALES=map-warm, ENGRAM_FAST, CPUSET=5-9,15-19, ENGRAM_SUBMIT_EARLY,
  REPLY_SCAN_SUFFIX, TOKEN_ALIGN_MB=16, LOAD_READERS=4. stack-triage = stack − DEVICE_ACCEPT +
  DRAFT_TRIAGE=conf:0.2.
- odds = TOPK_FAST + PREFILL_GATHER_OVERLAP + IDX_BUCKET_FINER + T15_PLACE_BOUND.
- **Stack decode loss is consistent** (all 4 stack blocks below every best block's mean on dec4/dec8):
  the individually-neutral host-side switches do not add up; together they cost ~0.5–0.8 % decode.
  Not bisected (no upside to recover — the stack never beats best).
- **Odds exact FAIL is a sticky crash, not drift**: in c02/c03 the first error is `len_2048` ->
  HTTP 500 MissingTritonVariant (rank0 "no captured Triton variant of _index_score for this launch"), and
  every later case (repeat, resume, conc, solo_4+, conv) then gets CudaFailed: the engine is poisoned.
  No case returned different bytes. Phase D confirms the cause: finer alone (d14) and finer+topk (d15) fail
  the same way (CudaFailed on conc_*), while bt/bto without finer are IDENTICAL.

## Phase D (`runs/combo5/`, 15 blocks; % vs mean of 3 best blocks d01/d08/d11)

Best-block spread: dec1 0.4, dec4 0.1, dec8 0.8, pd.32K 0.1, pp.8K 0.3, t1 0.2, t4 2.2 %.

| arm | n | dec1 | dec4 | dec8 | pd.32K | pp.8K | t4 | exact / sampled | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bt = best + TOPK_FAST + T15_PLACE_BOUND | 3 | −0.1 | −0.1 | −0.3 | −0.2 | ≈0 (one −22.7 outlier sample) | +0.3 | IDENTICAL / match | neutral, no gain |
| bto = bt + PREFILL_GATHER_OVERLAP | 3 | +0.2 | +0.0 | −0.4 | **−0.8 [−1.0,−0.5]** | −1.1 | −0.7 | IDENTICAL / match | prefill loss |
| half-engram = best + map-warm + ENGRAM_FAST + SUBMIT_EARLY | 2 | −0.3 | −0.4 | **−1.1 [−1.4,−0.8]** | −0.1 | 0 | +0.3 | – | decode loss: the stack's loss lives here |
| half-host = best + CPUSET + SCAN_SUFFIX + TOKEN_ALIGN_MB=16 + LOAD_READERS=4 | 2 | −0.1 | +0.3 | −0.3 | −0.2 | 0 | −1.1 | – | neutral |
| finer | 1 | −0.3 | +0.1 | −0.1 | +0.5 | +0.7 | +1.1 | **FAIL** (CudaFailed) | broken |
| finer + topk-fast | 1 | +0.2 | +0.2 | −0.2 | +0.9 | +0.9 | +0.5 | **FAIL** (CudaFailed) | broken; the only arm with a prefill hint (+0.9 %, 1 block) |

## Production

Restored to image `234f9080b326` (combo3 set: b40ea899e367 + default-off 0160–0171 + head 0171z) with the head
`.env` flags; both ranks; `validate.py --compare runs/matrix/base.validate.json` all passed (8/8 equal to saved);
container env equals `production.rank0.env` except the image hash. Head `zig/patches` has no 018*/019* files.

## For the patch agent

1. Fold `0184z` into 0169 (Round.init never sets `rd.hot`; `var rd: Round = undefined`) and `0186z`
   (forward declaration of `tf_ds_block_argmax_v`) into 0186 — 0186 does not compile without it.
2. 0190 snap-trim: illegal address on the first request at default flags (0189 alone passes 3/3).
3. 0185 TF_DS_BF16_PARTIALS=1 and 0186 TF_DS_GREEDY_ARGMAX=1: garbage smoke reply at startup.
4. 0188 TF_DS_HEAD_LAST_CHUNK=1: rank-pair hang on an ~8K prompt.
5. TF_DS_IDX_BUCKET_FINER=1 (confirmed alone, d14): `_index_score` MissingTritonVariant on the first new
   launch shape, then CudaFailed on every request (engine poisoned). Worth fixing: finer(+topk) is the only arm
   with a prefill hint (+0.5–0.9 % in one block).
7. ENGRAM half of the stack (map-warm / ENGRAM_FAST / SUBMIT_EARLY together) costs ~1 % dec8.
6. prompt_reuse "resumed only 88%" also fails on best (pre-existing; see combo3).
