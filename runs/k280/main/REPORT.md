# k280 main: 0281 Markov cache and 0282 draft prefetch both pay; 4096 rows clears its gate

Window 2026-11-25 06:49:20Z → 08:33:40Z. Image `tensorfold-dsv41:bench-fused-next-1421187`, 10 arms, production
restored from `.env` alone at the end (`restored: validate rc=0`). Launched from the recipe root on the head after
`rsync`ing `runs/k280/` and the newer `runs/overnight-20261122/arms.sh` (the head's copy lacked the `pass_counts`
grep).

## Gates — all four pass

| Gate | Result |
|---|---|
| Exact + sampled panels IDENTICAL every arm | **yes**, 10/10 arms |
| Drafts unchanged (`gate_drafts.py`) | **yes**, 19/19 comparisons, exit 0 |
| Banner counts as designed | **yes** (ctl arms 0, mc/dpf arms 1) |
| Production restored | **yes**, `restored: validate rc=0` |

`python3 runs/k280/gate_drafts.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b,mc4096a,mc4096b,dpf1a,dpf1b,dpf2a,dpf2b`
→ `drafts: IDENTICAL` on all 19 gated comparisons. The cache and the prefetch change no bit and no draft.

## Timings — `pass_ms` from the Rank 0 lifetime profile

Controls `ctl1` / `ctl2` = 4.20 / 4.22 ms; control spread 0.48%. That spread is the noise floor for every
comparison below.

| Arm | `pass_ms` | vs control | as % of pass | as % of round¹ | `forward_ms` | miss rate | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| mc2048 | 4.07 / 4.08 | −0.13 / −0.14 | −3.2% | −0.31% | +0.3% | 52.2 → 32.7% | **just under** the 0.15 ms gate |
| mc4096 | 4.00 / 4.00 | −0.20 / −0.22 | −5.0% | −0.49% | +0.3% | 52.2 → 24.2% | **CONFIRMED** |
| dpf1 | 4.02 / 4.05 | −0.18 / −0.17 | −4.2% | −0.42% | +0.2% | 52.2% | **CONFIRMED** |
| dpf2 | 4.12 / 4.11 | −0.08 / −0.11 | −2.3% | −0.21% | +0.2% | 52.3% | **neutral** on forward+pass |

¹ round ≈ `forward_ms` + `pass_ms` at one stream, so 42.9 ms; the share grows with streams per pass.

### H1 — 0281 Markov cache: confirmed at 4096, marginal at 2048

The gate was `pass_ms` ≥0.15 ms against both controls. **mc4096 delivers −0.20/−0.22 ms (−5.0%)**, 10× the control
spread, with the miss rate falling exactly as predicted: 52.2% → 24.2% against the predicted ~24%. **mc2048 delivers
−0.13/−0.14 ms**, which is 6.7× the control spread and therefore statistically clear, but 0.01–0.02 ms under the
pre-registered 0.15 ms threshold. Its miss rate (52.2% → 32.7%) also matches the predicted ~32%.

The pre-registered numbers were 0.16–0.18 ms at 2,048 rows and 0.23–0.26 ms at 4,096. Measured is ~80% of predicted
at both sizes, and the two sizes scale as predicted (0.14 → 0.21 ms, +50% for +2× rows). So the cache behaves as
modelled and only the absolute constant was ~20% optimistic. **Adopt 4096 rows, not 2048**: the extra 0.07 ms is the
whole reason to spend the memory, and 4096 costs 473 MiB/rank against 2048's 221 MiB.

`forward_ms` is unchanged (+0.2–0.3%, inside noise) — the cache touches only the pass, as designed.

### H2 — 0282 draft prefetch: dpf1 confirmed, dpf2 neutral

**dpf1** (the three in-pass prefetch sites) delivers **−0.18/−0.17 ms on the pass (−4.2%)** with no forward cost,
clearing its ≥0.08 ms gate by more than 2×. Its round-level saving is ~0.42% at one stream and grows with streams
per pass.

**dpf2** (dpf1 plus the layer hand-off) gives only −0.08/−0.11 ms on the pass, and its hand-off join shows up as
+0.19% on `forward_ms` — exactly the mechanism the README predicted. Judged on forward + pass, as instructed, dpf2 is
a **wash** (42.88 vs 42.87 ms). The hand-off costs as much as the prefetch saves. **Do not adopt dpf2; dpf1 is the
form to keep.**

### H3 — no equal-work regression

The two controls agree to 0.48% on `pass_ms`, and every test arm's exact and sampled panels are IDENTICAL. No
regression on the off path. The `rows_a_round` column reads +0.3–0.6% in the test arms, but that is downstream of a
0.8% spread in pass count (7,606 vs 7,542 passes) for identical draft output — workload timing noise, not a change.

## Memory

The arm artefacts carry no memory gauge, so the 473 MiB/rank cost of mc4096 is not instrumented here. `free -m` on
the head held ~7 GB available throughout the run and the memguard floor is 2.5 GiB, so no arm came near the guard.
Quantify the resident cost before adopting: production's headroom per the README is 6.9 GiB, and 473 MiB is ~7% of
it.

## The caveat that gates adoption

Every comparison here is **within the image**. Production's binary is a GCC 13.3 from-source build; this image's
executables are cross-built. A patch that wins against a cross-built host binary may not win against production's.
`todo.md` already carries this as "add a from-source control to the k280 suite before adopting anything". Either
build the from-source control or rebuild production from source and re-measure before `.env` changes.

## Artefacts

`runs/k280/main/` — 10 arms with `.json`, `.probe.json`, `.metrics.txt`, `.profile.txt`, `.armed.txt`, `.cmp.txt`,
`.banner.txt`. Replay: `bash runs/k280/run.sh` from the recipe root with production idle.
