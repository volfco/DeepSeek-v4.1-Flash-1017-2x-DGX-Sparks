# ENGRAM_LEAD pre-soak A/B: +2.0% at 32K, +2.8% at 96K — misses its 3% keep bar at 32K

Window 2026-11-25 17:40–18:12 UTC. Image `tensorfold-dsv41:bench-fused-next-2bb052f`, ABBA (lead1a, lead0a, lead0b,
lead1b), production restored from `.env` alone at the end.

**This is the pre-soak half of the owed test only.** `to-benchmark.md`'s evaluation order also requires a soak of
≥1.5M uncached prefill tokens *and* an hour of process age before and after which the cells are re-measured. Neither is
covered here; the age-related degradation (the Nov 13 slowdown) is the specific thing this run cannot speak to.

## Gates

| Gate | Result |
|---|---|
| Exact panel IDENTICAL, every arm | **yes**, 4/4 (`39 cases`) |
| Decode unchanged | **yes** — see the table; all cells within 0.5% |
| Production restored | **yes**, `2bb052f`, `.env` alone |

## Cold prefill (fresh, salted prompts, `cached=0`)

| arm | lead | 32K tok/s | 96K tok/s |
|---|---|---:|---:|
| lead1a | on | 2038.3 | 1987.5 |
| lead0a | off | 2007.4 | 1940.9 |
| lead0b | off | 1999.5 | 1931.4 |
| lead1b | on | 2049.6 | 1993.7 |

Paired against the control bracketing each arm:

| depth | lead1a vs lead0a | lead1b vs lead0b |
|---|---:|---:|
| 32K | **+1.54%** | **+2.51%** |
| 96K (≈ the 100K cell) | **+2.40%** | **+3.23%** |

## Decode (no cost, as expected)

| arm | 4K×1 | 4K×4 | 32K×1 | 32K×4 | 128K×1 | 128K×4 |
|---|---:|---:|---:|---:|---:|---:|
| lead1a | 5.05 | 15.04 | 5.61 | 14.63 | 6.18 | 15.88 |
| lead0a | 5.04 | 15.06 | 5.62 | 14.60 | 6.17 | 15.92 |
| lead0b | 5.05 | 15.16 | 5.63 | 14.57 | 6.19 | 15.90 |
| lead1b | 5.03 | 15.07 | 5.61 | 14.59 | 6.18 | 15.78 |

The spread is within 0.5% everywhere and does not track the switch. Leaving the reader on costs decode nothing.

## Verdict against the keep rule

The rule: *keep if natural-text prefill gains at least 3% at 32K and 100K in every pass.* **It is not met.**

- 32K: +1.54% / +2.51% — **below 3% in both passes**.
- 96K: +2.40% / +3.23% — one pass under 3%.

The gain is real and reproducible (both lead arms beat both controls at both depths), but it is ~2–3%, not the
**+4.9% / +5.0% / +4.5%** the November 19 narrow run recorded at 8K/32K/100K on natural text. That run was one block
per arm on fresh processes; this one is ABBA with the controls bracketing. The earlier figure should be read as an
overestimate.

## What that means for production

`TF_DS_ENGRAM_LEAD=1` is a **provisional production default since November 19** that has never passed its full test.
On this half it delivers a genuine but sub-threshold prefill gain, with no decode cost and identical bits. Two things
keep it from being settled:

1. The soak and the hour of process age, which are the whole point of the owed test — the reader was on during the
   November 13 age-related slowdown.
2. 32K misses the 3% bar, so the case for keeping it is now weaker than the record said.

**Recommendation:** keep it for now (it is exact, free on decode, and positive everywhere measured), but correct the
recorded gain to ~2–3% and treat the soak/age half as still owed and still the deciding evidence.

## Files

`runs/engram-lead-20261125/main/` — `<arm>.{json,log,probe.txt,cmp.txt,exact.json,armed.txt,banner.txt}`, `run.log`.
Replay: `bash runs/engram-lead-20261125/run.sh` from the recipe root with production idle.
