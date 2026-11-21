# overnight-fold — the W2 deliverable: `runs/overnight/RESULTS.md` folded into one dated section

**Source of every number below: `runs/overnight/RESULTS.md`** — the overnight Engram / engine A/B campaign, status
`DONE 2026-11-13 21:39Z` in that file (folded here under the date the W2 hand-over names, **2026-11-15**).
**Folding this into `to-benchmark.md` is a PENDING EDIT, not done:** that repo is the operator's to commit, and
`to-benchmark.md` is where the two result sets have to stop disagreeing (`docs/benchmark-gaps-and-moe-review.md`,
finding 2 and gap A2). Nothing here was re-measured; nothing here was re-opened. The only edits are selection and
the bold markers the source already carried.

Not quoted, by rule: the campaign's saved container environments (`production.rank*.env`), `*.inspect.json`,
`*.cmd`. Those files hold whole `TF_*`/`HF_HUB_*` environments; the task forbids quoting env contents, and
`docs/benchmark-gaps-and-moe-review.md` (W2) says to check them for keys and tokens before anything there is shared.
Switch **names** are named below; values of the saved env files are not reproduced.

---

## Section for `to-benchmark.md` (paste after "Combinations", before "Dedupe same-prompt probe")

### Overnight pass (2026-11-13): the verdicts, and the three signs they move

Single pass, one block per candidate with a base block on each side; no 3x alternation, so **nothing here is
labelled POSITIVE**. Deltas are the candidate's median against the left and right base blocks; the band is the
larger of the per-rep spread of all three blocks and the base-to-base gap; decode figures are paired per rep index.
Bold = outside the band. Evidence: block ids, `runs/overnight/RESULTS.md`.

| Cand. | Block (bases) | Prefill 8K / 32K / 100K | Decode 1 / 4 / 8 streams (paired) | Other | Verdict |
| --- | --- | --- | --- | --- | --- |
| A `ENGRAM_LEAD=1` | b03 (b02/b04) | +2.9/+1.8% (band 2.8) / **+4.3/+3.4%** (1.5) / **+3.6/+2.4%** (1.2) | -0.4/-0.0, -0.4/-0.3, -0.0/-0.4% (band <= 0.8) | p125 8K +6.2/+5.4% | **SIGNAL +** at >= 32K prefill; decode neutral |
| B `ENGRAM_SCALES=map` | b05 (b04/b06) | **+6.1/+8.8%** (3.6) / **+4.7/+5.6%** (1.9) / **+3.5/+4.2%** (0.7) | -0.1/-0.3, -2.4/-1.9, +0.1/+0.6% (within) | code 8K **+4.6/+4.6%**; **cold cost: first request after restart (2K warm-up) 257 tok/s vs base 1399/1438 (-82%); second (8K) 1119 vs 1907/1891 (-41%)**; first 32K/100K cold 2074/1982 (no cost) | **SIGNAL +** in steady state (after one warm-up per size); large cold cost on the first two requests after a restart |
| C `ROWCACHE_MB=256` | b07 (b06/b08) | +0.9/-1.2% / -1.2/-2.0% / -16.3/-16.6% (band 26: one slow rep) | **-4.4/-3.8, -9.1/-9.3, -17.5/-17.1%** (band 1.1 / 3.6 / 7.2) | **code 8K -39%, p125 8K -61%** | **SIGNAL - / RULED OUT**: a progressive regression that starts after ~370-470K prefill tokens since the restart |
| C `ROWCACHE_MB=64` | b23 (b22/b24) | not measured | not measured | **rank pair broke 9 s after the cold 100K warm-up** (rank 0: `spill tier's agreement: rank 1: ReplyTimeout`, exit 75; rank 1 logged nothing after load) | **UNMEASURED** (pair broke); not re-run |
| D production as deployed (`LEAD=1 ROWCACHE_MB=256 STRICT=1`) vs clean base | b09 (b08/b10) | **+3.0/+2.9%** (0.8) / **+2.5/+2.9%** (0.9) / +1.2/+1.0% (0.8) | **-4.3/-4.2, -10.5/-10.4, -17.6/-17.6%** (band 1.5 / 3.3 / 6.2) | code 8K -22%, p125 8K -55%; TTFT 8K 4.3/2.9% better | **SIGNAL -** once past the onset: same progressive regression as C256; before it, LEAD's prefill gain with no decode cost (i01) |
| D `LEAD=1 + SCALES=map` | c01 (b24/c02) | **+6.9/+7.5%** (3.9) / **+6.0/+6.1%** (0.3) / **+4.4/+5.1%** (0.7) | -0.1/-0.2, +0.8/+0.7, **+1.0/+0.8%** (band 0.4) | code 8K +2.5/+8.1% (within); cold: first request (2K) 262 tok/s vs base 1403/~1400, second (8K) 1341 vs 1904 | **SIGNAL +**, largest prefill gain of the pass (A and B stack); decode neutral to +1% |
| E `GRAPH_CAP=512` / `GRAPH_CAP=0` | b11 (b10/b12) / b13 (b12/b14) | -0.3/-0.5 / -0.1/-0.3 / -0.4/-0.1% (and the same shape at 0) | +0.2/+0.2, +0.1/+0.1, +0.2/+0.5%; 100K-ctx lc1 -7.2/-3.7% (band 14.7), lc4 -0.1/-1.4% (cap 512) | **0 graphs destroyed in every block (64 kept, under the 128 cap)** | **INCONCLUSIVE**: no cost, but the bench never reached the thrash regime |
| F `ENGRAM_AIO=0` | b15 (b14/b16) | **+4.0/+3.6%** (0.6) / **+1.8/+2.1%** (1.4) / **+3.4/+2.8%** (0.9) | -1.3/-1.4, -1.0/-0.7, +0.3/+0.2% (within) | TTFT 8K 2.3% better | **SIGNAL +** on prefill (the prior said neutral) |
| G `ENGRAM_FAST=1` | b17 (b16/b18) | **-2.3/-2.1%** (0.9) / +0.3/+0.1 / -0.4/-0.8% | within | TTFT 8K 6% worse | **RULED OUT** (slightly negative at 8K, neutral elsewhere) |
| G `ENGRAM_DEDUPE=1` (LEAD=0, RC=0) | b19 (b18/b20) | +0.5/+2.9 / +0.3/+0.8 / -0.1/+0.1% | within | | **RULED OUT** (neutral) |
| H `T25_DRAFT_CAL=0.05` | b21 (b20/b22; thinking-on code cells in all three) | +1.8/+1.4 / -0.1/-1.0 / -0.6/-0.7% (within) | -2.1/-2.3, -7.5/-7.4, **-11.9/-11.9%** (band 3.6) | thinking-on code decode: t1 -0.0/-0.1%, t4 -0.3/-0.0% (within) | **SIGNAL - / RULED OUT**: -12% at 8-stream thinking-off decode, neutral with thinking on |
| I profile | i01 | - | - | production flags; decode at base speed (dec1 80.4, dec8 136.1), because i01 stayed under the row-cache onset | measured (attribution only) |
| J host overhead | b04 | - | - | gap below the 0.1 s done-line resolution | measured; not separable |
| K `SPILL_MIN_TOKENS` | b02 -> b03 | - | - | 2048: resumed 4096 of 6103 after a restart | measured at 2048; **8192 not run** (spill fingerprint) |
| L KV shard decode | - | - | - | - | **NOT RUN** |

**Row-cache regression, the per-request timeline** (`cum` = uncached prefill tokens since the restart):

| Block | flags | at cum ~370K | first slow request | after |
| --- | --- | --- | --- | --- |
| b07 | RC256 | 100K rep 1: 1846 tok/s (base 1959) | 100K rep 2 (cum 369-469K): 1421 | code 1168/1163/1125; p125 909/814/746; dec1 68/77/85 (base 71/81/89); dec8 116/112/111 (base 135-138) |
| b09 | LEAD+RC256+STRICT | 100K reps: 1989/1973 (LEAD-level) | code rep 1 (cum 470-479K): 1516 | code 1474/1405; p125 1040/930/852; dec8 115/112/111 |
| i01 | LEAD+RC256+STRICT | (only 143K total) | none | dec1 71/80/88, dec8 135/136/138 (base level) |
| b08 | base | 100K: 1959/1957 | none | code 1843-1918, p125 2079-2088, dec8 135-138 |

The degraded cells ran last in each block, so the cell order (code, then p125, then decode) is confounded with time;
the size of the loss on each corpus says nothing about the corpus itself (the "p125 is hurt most" reading is
withdrawn in the source).

**The three signs this pass moves** (the source's own inference, not re-measured here):

1. **Row cache (`0114`):** "the strongest finding, and conditional" — base speed up to ~370-470K cumulative uncached
   prefill tokens, then a monotone regression (8-stream decode -17%). "So this is a state the cache reaches, not a
   constant cost." Likely, not measured: production processes far more than 500K uncached tokens between restarts.
   `docs/benchmark-gaps-and-moe-review.md` W1 reads the cause off the code (`engram_io.zig:281-310`: tombstones in a
   map that never rehashes) and the onset's movement with the budget as its prediction; that link is **inference,
   not proof** — the kill test is in `env-vars.md`'s no-env rows.
2. **`SCALES=map`:** +3.5 to +8.8% prefill in steady state, and its cost is only the cold fault-in (first request
   -82%, second -41%, from the third on: no cost). The matrix's -5% / -11% "hold" was the cold requests. The plan's
   `map-warm` (W6) is the row that was meant to decide `map`; this pass says it would also remove the only cost.
3. **`AIO=0` and `T25`:** `AIO=0` is **positive** on prefill (+1.8 to +4.0%) where the matrix read neutral, and
   `T25_DRAFT_CAL=0.05` is **-12% at 8-stream decode** where the matrix read neutral at 4 (this pass: -7.5% at 4).
   Both are load-dependent: the matrix decodes at 1 and 4 streams only.

Single-pass caveat, restated in the source: each of these needs two more ABAB alternations plus
`tools/validate.py --compare` before it can be called POSITIVE; earlier matrix records already read `ident` for
lead, map and aio0. `AIO=0` must not be combined with `LEAD` or `map` untested — the earlier matrix measured
`AIO=0`+`map` at -44%.

---

## The W2 production-switches proposal — **PROPOSAL for the operator, not action**

Restated from `docs/benchmark-gaps-and-moe-review.md`'s W2, unchanged, with the overnight evidence attached. This
is a proposal to *write down* what production should be; it is not a restart instruction, and this file changes no
pair, no `.env`, no `scripts/local.sh`. The operator's call, in the operator's window.

| Proposed | Value | Evidence it rests on | Why not today |
| --- | --- | --- | --- |
| `TF_DS_ENGRAM_LEAD` | `1` | A: **+4.3/+3.4% at 32K, +3.6/+2.4% at 100K** prefill, decode neutral; c01 with `map`: +6-7% at 8K/32K | single pass; the row-cache regression contaminated D (b09) and not i01 (b03/b05/b06 rows are clean of it) |
| `TF_DS_ENGRAM_STRICT` | `1` | matrix arm read `ident`; no cost in any block that carried it | never measured alone against a base inside the band; no cost seen, no gain claimed |
| `TF_DS_ENGRAM_ROWCACHE_MB` | **unset** (no row cache) until the W1 retest passes | C256: **-4.4/-9.1/-17.5%** decode at 1/4/8 streams past the onset, code 8K -39%; b09 the same; RC64 broke the pair | `to-benchmark.md` still signs `0114` **keep** on `engram_lookup_seconds` -57%, a metric that is not speed — the sign-off is wrong as written (gap A1). A1's evidence is "withdrawn pending W1" |
| `TF_DS_ENGRAM_SCALES` | `io` until `map-warm` (W6) has data | B: `map` is +3.5-8.8% steady state but -82%/-41% on the first two requests after a restart; production restarts are what starve the first request | `map-warm` is unmeasured; the cold cost lands exactly on the first request after the restart the operator controls |

Written down where: the proposal is `scripts/local.sh` on the head (it does not exist there) or `.env` — again the
operator's call, and not this file's. Note for whoever edits it: the three switches the overnight pass found in the
live containers were **not in any file** (exported in the shell that restarted the pair), so a plain
`./start.sh restart` drops them; `to-benchmark.md`'s "production" arm and the matrix's base are therefore two
different configurations (gap A4). Naming the configuration on every result table is part of W2.

## Pending edits this hand-over does not perform

1. Paste the section above into `to-benchmark.md` (after "Combinations", before "Dedupe same-prompt probe"), as a
   diff the operator applies — including the two contradictions it settles: `0114`'s keep sign-off (gap A1) and
   `map`'s / `AIO=0`'s / `T25`'s signs (gap A2).
2. `tools/bench_div.py`'s diverse corpus becomes the matrix's prefill corpus, with `p125` kept as a labelled
   comparison (W2, and the source's "-10% optimistic" note on the 125-word corpus).
3. `runs/overnight/` stays untracked until its env/inspect artifacts are checked for keys (W2's own instruction).
