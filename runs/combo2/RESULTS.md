# Combo2: patches 0129–0145 on image exl3-tp2-bd9020f403bd (2026-11-16)

Completed campaign. Failed restart and validation gates are listed in phase D. Per-block medians come from `python3 runs/combo/stack_report.py runs/combo2 <prefix>`.

## Phase A: new-image gate (a01–a03, production flags)
- **Identity:**
  - Exact match against the old `runs/combo/blocks/p05.exact.json`: IDENTICAL cold in a01 (47,104 cached) and in b04 (map-warm); IDENTICAL resumed in a02 (344,064).
  - validate passed on every block that ran it.
  - The sampled_ident shas equal the old ones: 9100 f3b07d1aedd4, 9101 b2f0544ee7fc, 9102 8b41a5237de2, 9103 68310aa1585b.
- **Speed:** the same as the old q-base blocks.
  - pd.100000: 1915–1944 here, against 1911–1934 before.
  - dec1 80.4–80.5, dec8 134.8–136.5, t4 138–139.
  - The always-on 0134, 0137 and 0138 cost nothing measurable.

## Phase B: TF_DS_ENGRAM_SCALES io / map / map-warm (0129), all on the best flags
- **Steady state:** 3 full blocks per arm (io b01/b07/b13, map b02/b08/b12, map-warm b04/b06/b10). Values are the median of block medians, with the change vs io in brackets.

| arm | pd.32000 | pd.100000 | pp.8000 | pc.8000 | dec8 | t4 |
|---|---|---|---|---|---|---|
| io | 2054 | 2010 | 2203 | 1931 | 135.3 | 136.4 |
| map | 2061 (+0.3%) | 2013 (+0.1%) | 2252 (+2.2%) | 1940 (+0.5%) | 136.6 (+1.0%) | 138.7 (+1.7%) |
| map-warm | 2065 (+0.5%) | 2020 (+0.5%) | 2255 (+2.4%) | 1961 (+1.6%) | 137.0 (+1.3%) | 139.7 (+2.4%) |

  - pd.2000 and pd.8000 are too noisy right after a restart to rank the arms (941–1545 across blocks).
- **First request after a restart** (warm cell, TTFT in seconds):

| arm | n | 2K | 8K | 32K | 100K |
|---|---|---|---|---|---|
| io | 7 | 1.38–2.08 | 4.03–4.19 | 15.3–15.5 | 48.7–49.5 |
| map | 3 | 7.38–8.24 | 6.72–6.77 | 15.3–15.6 | 48.6–48.8 |
| map-warm | 3 | 7.10–7.21 | 6.13–6.86 | 15.6–16.2 | 49.0–49.1 |

- **Verdict:**
  - Map-warm does not do its job: it removes only about 0.3–1 s of the 6 s first-request penalty that map adds over io.
  - Steady state, map-warm is the same as map, within noise.
  - Best stays on `map`, which is what phases C and D use. The user ranks warm performance above first-request latency after a restart (2026-11-16), so `io` is not preferred despite its faster cold start.

## Phase C: row-cache soak (0136 fix, 0137 rehash), best flags
Each block runs dec1/4/8 and code 8K, then a soak of 15 × 100K uncached prefills (1.5M tokens), then dec1/dec8 again. The arms are control (c01, c04), `ROWCACHE_MB=256` (c02) and `64` (c03). Every block has clean 36/36. c02 output is IDENTICAL to p05 cold.

| block | dec1 before→after | dec8 per stream before→after | soak first3→last3 (tok/s) | pc.8000 | mem low H GiB |
|---|---|---|---|---|---|
| c01 control | 80.7→80.7 | 19.5→19.5 | 2014→1976 | 1986 | 6.9 |
| c02 rc256 | 79.7→79.9 | 19.3→19.3 | 1951→1882 | 1889 | 5.4 |
| c03 rc64 | 80.2→80.5 | 19.7→19.7 | 1963→1936 | 1953 | 6.5 |
| c04 control | 80.6→80.6 | 19.4→19.9 | 2023→1988 | 2006 | 6.3 |

- **The old progressive decode regression (-4/-9/-17% in the overnight run) is gone at both budgets.** Decode after 1.5M tokens equals decode before. So the kill test answers "fixed", not "rule the cache out".
- **The cache still has no upside.**
  - rc256 is slower everywhere: soak -3 to -4%, pc.8000 -5%. It also costs about 1.5 GiB of head MemAvailable.
  - rc64 is within about 1.5% of control.
  - **Verdict:** keep `ROWCACHE_MB` unset.
- **The soak drifts down about 2–4% in every arm, controls included.** It is not cache-specific and recovers by the next block. Temperatures reach 87–88 °C, so it is likely thermal.

## Phase D: switch screens
### Bugs found (patch fixes needed; no speed result possible)
- **`0133` `TF_DS_ENGRAM_AIO_BATCH`: values below the ring's default batch break long prompts.**
  - At 64 (d03) and 256 (d05), validate fails on the long prompt: `ended … reason=error (TooManyRows) prompt=2094`. Both blocks were aborted by the gate.
  - A smaller bound should split the batch, not refuse it.
  - 1024 (d06) passes, but it is probably a no-op: `per = min(slots/4, n)` only bounds downward.
  - So no value can lower the batch today, and the read-depth question is unanswered.
- **`0132` `TF_DS_ENGRAM_CPUS`: an in-range CPU list is refused.**
  - d09 used `CPUSET=5-9,15-19` with `ENGRAM_CPUS=5-9`. Both restarts failed with `TF_DS_ENGRAM_CPUS=5-9: expected … CPUs below 5` / `InvalidEngramCpus`.
  - `parseCpus` checks `hi >= std.Thread.getCpuCount()`. That function returns the affinity *count*, not the highest CPU id. So any absolute CPU id at or above the number of allowed CPUs is refused, which covers every CPU in a cpuset that doesn't start at 0.
  - The to-benchmark row (`ENGRAM_CPUS=10-14` with `CPUSET=10-19`) can't run either, so phase E's e10 was dropped from the schedule.

### Results so far (block medians; brackets are best d01/d04/d07)
- **`0143` `ALLOCATE_LEARNED=1` (d02):**
  - Output stays identical: sampled_ident shas equal, validate passed.
  - Prefill is unchanged: pd.32000 2079 vs 2070–2072, pd.100000 2029 vs 2020–2029.
  - Decode is slower: dec4 114.8 vs 117.6–118.6 (-2.5%), t1 93.2 vs 95.3–96.8 (-3%), t4 136.5 vs 134.2–139.2. One block only.
  - **Verdict:** no gain, a likely small decode loss; leave it off.
- **CPUSET `5-9,15-19` (d08, all fast cores):** within noise of best everywhere (pd.100000 2023, dec8 136.9, t4 138.0). The 5 extra fast cores buy nothing. The runner passes `CPUSET` to `start.sh --cpuset-cpus`; the live container on the default shows 15-19 on both nodes, but d08's own cpuset was not inspected.
- **`AIO_BATCH=1024` (d06):** equals best (pd.100000 2025, dec8 136.7). That is consistent with it being a no-op.
- **d13 (best) was low:** pp.8000 2009 against 2254–2266, and t4 133.3 against 134.2–139.2. Its peak temperature was 85/79 °C, cooler than the c-phase, so heat does not explain it. Unexplained. Check e06/e09 (best): if they also sit near 2010, the box shifted and the D/E brackets need re-reading.

### Phase D, rest (d10-d13)
- **FAST (d11):** within noise of best. pd.100000 2018 against 2003-2029; dec8 136.3; t4 139.4.
- **FAST with DECODE_THREADS=4 (d12):** also within noise. pd.100000 2020, t4 137.4.

## Phase E: audit gap-fill (e01-e12)
- **Best blocks (e06, e09, e12):** pd.100000 2013-2027, pp.8000 2193-2269, dec8 136.9-137.5, t4 134.2-136.8.
  - These are in line with d01-d10. d13's low pp.8000 did not repeat, so the D/E brackets stand.
- **`ENGRAM_AIO=0` (e01, e04):** about equal to best with LEAD, even though LEAD is off.
  - e04: pd.32000 2066, pd.100000 2022, pp.8000 2214, dec8 135.6, t4 136.5.
  - Nothing to gain by switching.
- **`0131` POOL (with AIO=0):**
  - 32 (e03): pd.100000 1925 (-5% vs e04), pp.8000 1602.
  - 128 (e05): pd.100000 1972 (-2.5%), t4 133.4.
  - 16 (e02, soak cell set): decode equal to e01.
  - No size beats the default 64. Keep it unset.
- **`0130` DECODE_THREADS on FAST:**
  - 2 (e07) and 5 (e08) are both within noise of best. e07's dec4 is 116.7, about -1.5%.
  - Together with d11/d12, FAST and its thread count are a wash.
- **`0117` ENERGY=1 (e11): output is identical, but decode slows.**
  - Exact is IDENTICAL cold, the sampled_ident shas are equal, and validate passes.
  - `energy_j=` appears on all 132 done lines.
  - Prefill is unchanged (pd.100000 2023).
  - Decode costs a lot:

    | metric | e11 | best | change |
    |---|---|---|---|
    | dec1 | 67.2 | 80.4-80.7 | -17% |
    | dec4 | 108.5 | 116-119 | -8% |
    | dec8 | 127.3 | about 137 | -7% |
    | t1 | 81.6 | 95-96.5 | -15% |
    | t4 | 129.3 | 134-137 | -5% |

  - Likely cause: the per-frame NVML read on rank 1. Use it for measurement runs only, never in production.
  - `idle_j` is `-` throughout.
