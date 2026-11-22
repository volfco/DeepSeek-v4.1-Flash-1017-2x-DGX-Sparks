# Forensic Attribution of the `dec8` Throughput Dip (`suite-fused-next-20261121`, `todo.md` Item 4)

## Executive Verdict

**Not a kernel or serving regression.** Across all five blocks (`s01..s05`), `bench-fused-next-d5ecba0` (`next` and `next+PDL`) produces **24/24 byte-identical greedy `dec8` replies** (`144/144` cross-block pairings) and **12/12 round-for-round, draft-for-draft identical solo generations** (`dec1` 3/3 + `probe` solo 9/9). True post-prefill-barrier active decode speed across all 3 waves is **identical in Pair A** (`80.651 -> 80.654 ms/stream-round`, `+0.003%`) and **faster in Pair B and PDL** (`80.951 -> 80.539 ms/stream-round`, `-0.51%`; `s05`: `80.565 ms`, `-0.30%` vs base mean), with both Sparks locked at **2379 MHz** median GPU clock throughout `dec8` and zero CUDA graph evictions, spill restores, or foreign requests.

The reported `-0.59%` (Pair A: `s01 138.97 -> s02 138.14 tok/s`, `+175.87 ms`) and `-0.69%` (Pair B: `s04 138.73 -> s03 137.77 tok/s`, `+204.65 ms`) dips come from **three measurement and multi-stream scheduling effects acting on Wave 0** (which is the median repetition in all five blocks):

1. **Critical-path round-count jitter (`+4` and `+5` physical rounds on Wave 0 straggler `s=7`) from random HTTP admission order under `TF_DS_ROUND_ROWS=16` (`+146.99 ms` = `83.6%` of Pair A dip; `+138.84 ms` = `67.8%` of Pair B dip):**
   With 8 active streams and `round_rows = 16`, 8 mandatory pending rows leave at most **8 speculative draft rows total** (~1.0 draft/stream/round). Marginal draft allocation across `h.admitted` favors earlier-admitted slots: across all 120 `dec8` stream-runs, `ord=0..2` averages **`-0.77` rounds** while `ord=6..7` averages **`+0.95` rounds** (`ord=7`: **`+1.27` rounds**). Under `threading.Barrier(8)` loopback socket races, Wave 0's slowest-accepting critical-path stream (`s=7`, 298 prompt tokens, ~70% accept) randomly landed at **`ord=1` in both `base` blocks (`s01`, `s04`)** (`166`/`167` accepted drafts -> **`345`/`344` physical rounds**), versus **`ord=3` (`s02`), `ord=7` (`s03`), and `ord=4` (`s05`)** (`162`, `162`, `164` accepted drafts -> **`349`, `349`, `347` physical rounds**).
2. **`bd.aggregate`'s `min(t_first)` first-prefill exclusion penalty (`+33.24 ms` in Pair A, `+20.90 ms` in Pair B):**
   `LaneHost` (`while (h.admitOne()) {}`) prefills all 8 requests sequentially (`~3.95–4.04 s`) before physical decode round 1 begins. `next` actually prefilled all 8 Wave-0 prompts **faster** than `base` (`-20.68 ms` in Pair A, `-16.24 ms` in Pair B, `-53.33 ms` in `s05` vs `s01` via `0216`'s short-prompt indexer bypass). However, `window = max(t_last) - min(t_first)` starts after `ord=0`'s first token, **excluding `ord=0`'s prefill from `window`**. Random socket arrival placed longer prompts at `ord=0` in `base` (`s01`: `s=2` 408t = `537.87 ms`; `s04`: `s=1` 366t = `512.63 ms`) than in `next` (`s02`: `s=0` 340t = `504.63 ms`; `s03`: `s=4` 322t = `491.73 ms`), turning `next`'s real prefill speedup into a `+12.47 ms` / `+4.63 ms` Phase-1 window loss.
3. **Why `ms/round` moved `+0.06%` / `-0.43%` while `tok/s` dipped `-0.59%` / `-0.69%`:**
   - **Statistic mismatch:** `dec8 tok/s` takes the **3-wave median** (selecting Wave 0 in every block: `Wave 1 < Wave 0 < Wave 2`), whereas `ms/round` pools `sum(decode_s) / sum(rounds)` across **all 3 waves** — including Wave 1 where `next` beat `base` in both pairs (`+0.06%` and `+0.09%`, with `s03` running `-240.66 ms` faster matched-width step speed than `s04`).
   - **Ratio blindness to round count:** `ms/round` divides out round count, and extra `<8`-stream tail rounds (`26–51 ms`) are cheaper than 8-stream rounds (`~84 ms`), so +4/+5 critical-path rounds lengthen `window` without increasing `ms/round`.
   - **Prefill stall inside `decode_s` (not `0.1 s` log rounding):** `done` line `0.1 s` rounding adds only `-0.016` to `+0.034 ms/round` error vs exact `server.seconds - server.time_to_first_token`, whereas `decode_s = finished_ns - prefilled_ns` includes the time streams `ord=0..6` sit waiting for `ord=1..7` to finish prefill (`41.09–41.82 s` per block = **`+5.93 to +6.04 ms/stream-round`**, ~7.0% of `decode_s`).

---

## 1. Exact `dec8` Metric Formulas & Reproduction (`analyze_dec8.py`)

From `run_suite.py`, `scripts/bench_overnight.py`, `tools/bench_div.py`, `compare_next.py`, `table.py`, and `zig/patches/{0070,0138}`:

- **Workload (`Block.decode("d8", 8, 3, 512)`):** 3 waves (`rep = 0, 1, 2`), 8 concurrent streams per wave gated on `threading.Barrier(8)`, `max_tokens = 512`, `temperature = 0`, `enable_thinking = False`, fixed seeds `dec-d8-{rep}-{s}` (`265–408` prompt tokens).
- **`dec8 tok/s` (`table.py` -> `st.median(decode([d], "dec8"))`):** Median across the 3 clean waves of `bd.aggregate(results)`:
  $$\text{aggregate\_tok\_s} = \frac{\sum_{i=0}^{7} \text{completion\_tokens}_i - 8}{\max_i t_{\text{last},i} - \min_i t_{\text{first},i}} = \frac{4088}{\text{window\_s}}$$
- **`dec8 ms/round` (`table.py` -> `round_ms([d], "dec8")[0]`):** Pooled over all 24 requests (`3 waves × 8 streams`) from rank-0 `done` lines:
  $$\text{ms/round} = \frac{1000 \times \sum_{r=1}^{24} \text{float}(\text{dl}[\text{decode\_s}])}{\sum_{r=1}^{24} \text{int}(\text{dl}[\text{rounds}])}$$

| Block | Arm | Wave 0 tok/s (**Median**) | Wave 1 tok/s | Wave 2 tok/s | Reported `dec8 tok/s` | Reported `ms/round` | Exact Server `ms/round` | `0.1s` Log Round Bias | Prefill Stall in `decode_s` | True Active `ms/stream-round` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s01` | `base` | **138.9694** | 137.3445 | 139.4804 | **138.97** | **86.62** (`86.6155`) | `86.6314` | `-0.0158 ms` | `41.481 s` (`+6.002 ms/r`) | **80.6514** |
| `s02` | `next` | **138.1435** | 137.4209 | 139.2147 | **138.14** (`-0.59%`) | **86.66** (`86.6647`) | `86.6750` | `-0.0102 ms` | `41.820 s` (`+6.035 ms/r`) | **80.6540** (`+0.003%`) |
| `s04` | `base` | **138.7286** | 137.4097 | 139.6601 | **138.73** | **86.92** (`86.9213`) | `86.8876` | `+0.0337 ms` | `41.211 s` (`+5.962 ms/r`) | **80.9512** |
| `s03` | `next` | **137.7718** | 137.5275 | 139.3370 | **137.77** (`-0.69%`) | **86.54** (`86.5443`) | `86.5427` | `+0.0016 ms` | `41.685 s` (`+6.025 ms/r`) | **80.5389** (`-0.509%`) |
| `s05` | `next_pdl` | **138.5912** | 137.8321 | 139.5145 | **138.59** (`-0.19%` vs bm) | **86.48** (`86.4818`) | `86.4810` | `+0.0009 ms` | `41.090 s` (`+5.935 ms/r`) | **80.5654** (`-0.295%` vs bm) |

---

## 2. Exact Millisecond Decomposition of the Batch Wall Window

For every wave, `window = max(t_last) - min(t_first)` satisfies the exact telescoping identity (verified to `< 10^-9 ms`):

$$\text{window} = \underbrace{(t_{\text{first}}[\text{ord}_7] - t_{\text{first}}[\text{ord}_0])}_{\text{Phase 1: Prefills of }\text{ord}_{1..7}\text{ after }\text{ord}_0\text{ token 1}} + \underbrace{(t_{\text{last}}[\text{fin}] - t_{\text{first}}[\text{ord}_7])}_{\text{Phase 2: Lockstep multi-stream decode (8 streams }\to\text{ 1 stream)}}$$

where:
- $\text{Phase 1} = \sum_{i=0}^{7} \text{srv\_prefill}_i - \text{srv\_prefill}_{\text{ord}_0} + \text{handoff\_sse\_gap}$ (handoff + SSE delta is only `+0.14` to `+0.37 ms` across all 15 waves because `LaneHost.admitOne()` runs all 8 prefills back-to-back before `core.step()`).
- $\text{Phase 2} = \sum_{k=8}^{1} \Delta t_k$, decomposed via exact symmetric Shapley attribution at each active-stream width $k \in \{8,\dots,1\}$ into **round-count / width-mix effect** ($\Delta r_k \cdot \bar{w}_k$) plus **matched-width step-speed effect** ($\bar{r}_k \cdot \Delta w_k$).

### 2.1 Wave 0 (Median Wave in All 5 Blocks — Explains 100% of Reported `-0.59%` & `-0.69%`)

| Component | `s01` (base) | `s02` (next) | **Pair A (`s02 - s01`)** | `s04` (base) | `s03` (next) | **Pair B (`s03 - s04`)** | `s05` (`next_pdl`) | **`s05 - s01` / `s05 - s04`** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aggregate_tok_s` | `138.9694` | `138.1435` | **`-0.8259` (`-0.59%`)** | `138.7286` | `137.7718` | **`-0.9568` (`-0.69%`)** | `138.5912` | `-0.27%` / `-0.10%` |
| **Total `window` (ms)** | **`29,416.55`** | **`29,592.42`** | **`+175.87 ms`** | **`29,467.61`** | **`29,672.26`** | **`+204.65 ms`** | **`29,496.82`** | **`+80.26` / `+29.21 ms`** |
| **Phase 1: `ord=1..7` Prefill (ms)** | **`3,463.09`** | **`3,475.56`** | **`+12.47 ms`** | **`3,470.46`** | **`3,475.09`** | **`+4.63 ms`** | **`3,436.13`** | **`-26.96` / `-34.33 ms`** |
| ↳ Sum of all 8 `srv_prefill` (ms) | `4,000.60` | `3,979.92` | **`-20.68 ms` (faster)** | `3,982.81` | `3,966.57` | **`-16.24 ms` (faster)** | `3,947.27` | `-53.33` / `-35.54 ms` |
| ↳ Minus excluded `ord=0` prefill | `-537.87` (`s2`,408t) | `-504.63` (`s0`,340t) | **`+33.24 ms` (penalty)** | `-512.63` (`s1`,366t) | `-491.73` (`s4`,322t) | **`+20.90 ms` (penalty)** | `-511.47` (`s1`,366t) | `+26.40` / `+1.16 ms` |
| ↳ Inter-prefill handoff + SSE | `+0.36` | `+0.27` | `-0.09 ms` | `+0.28` | `+0.25` | `-0.03 ms` | `+0.33` | `-0.03` / `+0.05 ms` |
| **Phase 2: Decode Span (ms)** | **`25,953.46`** | **`26,116.86`** | **`+163.40 ms`** | **`25,997.14`** | **`26,197.16`** | **`+200.02 ms`** | **`26,060.69`** | **`+107.23` / `+63.54 ms`** |
| ↳ Full 8-stream stage (`k=8`) | `17,114.64` (`204r`) | `16,438.51` (`196r`) | `-676.14 ms` (`-8r`) | `16,666.13` (`198r`) | `16,954.07` (`201r`) | `+287.95 ms` (`+3r`) | `16,478.10` (`196r`) | `-636.55` / `-188.03 ms` |
| ↳ Tail-drain stage (`k=7..1`) | `8,838.82` (`141r`) | `9,678.35` (`153r`) | `+839.53 ms` (`+12r`) | `9,331.02` (`146r`) | `9,243.09` (`148r`) | `-87.93 ms` (`+2r`) | `9,582.59` (`151r`) | `+743.77` / `+251.57 ms` |
| ↳ Straggler `s=7` (`adm_ord`, `rounds`) | `ord=1`, **`345r`** | `ord=3`, **`349r`** | **`+4 phys rounds`** | `ord=1`, **`344r`** | `ord=7`, **`349r`** | **`+5 phys rounds`** | `ord=4`, **`347r`** | **`+2r` / `+3r`** |
| ↳ **Shapley: Round-count & width mix** | — | — | **`+146.99 ms` (`83.6%`)** | — | — | **`+138.84 ms` (`67.8%`)** | — | **`+93.63` / `+138.00 ms`** |
| ↳ **Shapley: Matched-width step speed** | `83.90 ms/r` @8s | `83.87 ms/r` @8s | **`+16.40 ms` (`+0.06%`)** | `84.17 ms/r` @8s | `84.35 ms/r` @8s | **`+61.18 ms` (`+0.24%`)** | `84.07 ms/r` @8s | **`+13.60` / `-74.45 ms`** |

### 2.2 Waves 1 & 2 Decomposition (Non-Median Waves)

| Comparison | Wave | Legacy `Δtok/s` | Pure Decode `Δtok/s` | Full Batch `Δtok/s` | `Δwindow` (ms) | `ΔPhase 1` (`8pref` / `ord0_excl`) | `ΔPhase 2` (ms) | `Δphys_rnd` | Round-Mix Effect | Matched Step-Speed Effect |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pair A (`s02 - s01`)** | W1 | **`+0.06%`** | `+0.11%` | `+0.18%` | `-16.53` | `+13.52` (`-24.40` / `+38.03`) | `-30.05` | `0` (`341->341`) | `-24.49 ms` | **`-5.56 ms`** |
| **Pair A (`s02 - s01`)** | W2 | **`-0.19%`** | `-0.20%` | `-0.16%` | `+55.94` | `+2.94` (`-5.53` / `+8.43`) | `+53.01` | `-2` (`328->326`) | `+36.89 ms` | **`+16.12 ms`** |
| **Pair B (`s03 - s04`)** | W1 | **`+0.09%`** | `+0.29%` | `+0.21%` | `-25.47` | `+51.07` (`+12.40` / `+38.63`) | `-76.53` | `+4` (`338->342`) | `+164.13 ms` | **`-240.66 ms` (faster)** |
| **Pair B (`s03 - s04`)** | W2 | **`-0.23%`** | `-0.24%` | `-0.23%` | `+67.87` | `+5.20` (`+3.36` / `+1.82`) | `+62.67` | `+4` (`327->331`) | `+139.28 ms` | **`-76.60 ms` (faster)** |
| **PDL (`s05 - s01`)** | W1 | **`+0.36%`** | `+0.40%` | `+0.51%` | `-105.29` | `-1.71` (`-49.19` / `+47.41`) | `-103.58` | `+1` (`341->342`) | `-36.42 ms` | **`-67.16 ms` (faster)** |
| **PDL (`s05 - s01`)** | W2 | **`+0.02%`** | `-0.23%` | `+0.04%` | `-7.17` | `-66.44` (`-62.42` / `-4.09`) | `+59.27` | `0` (`328->328`) | `+29.93 ms` | **`+29.34 ms`** |

In Wave 2 Pair A (`s02 - s01`), even though `s02`'s final stream finished 2 rounds earlier (`326` vs `328`), `s02` ran `+7` more total stream-rounds (`2352` vs `2345`), keeping streams alive in expensive `6s`/`5s` stages (`+4r` at `~79 ms/r`, `+4r` at `~63 ms/r`) instead of cheap `4s`/`3s`/`1s` stages (`-4r`, `-5r`, `-4r` at `31–57 ms/r`), costing `+36.89 ms` in width mix.

---

## 3. Admission Order, Client HTTP/SSE Overhead & Hardware Telemetry

- **Admission-order (`ord = 0..7`) round-count gradient across all 120 `dec8` stream-runs:**
  Mean stream rounds minus per-(wave, stream) 5-block mean:
  `ord=0: -0.79r`, `ord=1: -0.52r`, `ord=2: -0.99r`, `ord=3: -0.07r`, `ord=4: +0.32r`, `ord=5: +0.15r`, `ord=6: +0.63r`, `ord=7: +1.27r`.
  Because `rounds + accepted == 511` (`512 - 1` prefill token), every draft lost to `TF_DS_ROUND_ROWS=16` contention adds exactly 1 round to that stream.
- **Client HTTP / SSE scheduling overhead:**
  - Client `Barrier(8)` `t_req` spread: `0.69–3.31 ms` across all 15 waves; rank-0 `POST /v1/chat/completions` completion log span: `4.8–8.5 ms`.
  - Mean `cli_ttft - srv_ttft`: `1.6–2.9 ms` across waves; `cli_decode - srv_decode` on the critical-path finisher stream: `2.1–8.1 ms` (`< 0.03%` of window).
  - Zero `restore_s` (`0.000 s` on all 120 requests), zero prompt-cache hits (`cached=0` on all 120 requests), zero CUDA graph evictions during `dec8` (`CUDA graphs: 64 kept` occurs *after* `dec8` ends in all 5 blocks).
- **GPU clocks, temperatures, power & memory during `dec8` (`32–33` samples/rank/block):**
  - `s01` (`base`): Head `2366–2379 MHz` (**median `2379`**), `78–81°C`, `65.5 W`, `5.97 GiB` low; Worker `2366–2379 MHz` (**median `2379`**), `70–76°C`, `66.5 W`.
  - `s02` (`next`): Head `2379–2392 MHz` (**median `2379`**), `70–79°C`, `62.3 W`, `6.14 GiB` low; Worker `2379–2385 MHz` (**median `2379`**), `62–75°C`, `62.9 W`.
  - `s04` (`base`): Head `2372–2379 MHz` (**median `2379`**), `77–80°C`, `65.8 W`, `6.07 GiB` low; Worker `2379–2379 MHz` (**median `2379`**), `71–77°C`, `67.3 W`.
  - `s03` (`next`): Head `2379–2379 MHz` (**median `2379`**), `75–79°C`, `64.2 W`, `6.08 GiB` low; Worker `2379–2385 MHz` (**median `2379`**), `63–77°C`, `65.2 W`.
  - `s05` (`next_pdl`): Head `2379–2392 MHz` (**median `2379`**), `71–80°C`, `62.9 W`, `6.05 GiB` low; Worker `2379–2385 MHz` (**median `2379`**), `63–76°C`, `64.2 W`.

---

## 4. Post-Restart Replay Tool (`dec8_replay_audit.py`)

`/tmp/pre-bench-20261121/dec8-only/dec8_replay_audit.py` is offline-verified (`--offline-verify` passed against `s01..s05` + synthetic SSE per-round trace) and ready for a future restart window:

- **Eliminates random admission-permutation noise:** `--admission-mode staggered --stagger-ms 2.5` spaces stream submissions by `2.5 ms` (`s=0..7`, total `17.5 ms` << `~500 ms` `ord=0` prefill), deterministically locking `adm_ord == [0, 1, 2, 3, 4, 5, 6, 7]` on every arm while keeping all 7 followers queued before `ord=0` finishes prefill.
- **Captures per-round timestamps (`events_mono`):** Records `t_req`, `t_first`, every SSE round event timestamp, and `t_last` per request, reconstructing per-round latency by active-stream occupancy `8..1` without `--profile` stream sync.
- **Reports unbiased metrics alongside legacy `aggregate_tok_s`:** `pure_decode_tok_s = (sum(tokens) - 8) / (max(t_last) - max(t_first))` (immune to `ord=0` prefill length) and `full_batch_tok_s = sum(tokens) / (max(t_last) - min(t_req))` (all 8 prefills included), plus prompt-token and `token_sha` assertions against `s01.json`.

```bash
# Offline verification (workstation-only, zero network/GPU):
python3 /tmp/pre-bench-20261121/dec8-only/dec8_replay_audit.py --offline-verify

# Future post-restart paired run on head Spark (idle-guarded):
python3 dec8_replay_audit.py record --corpus-dir runs/suite-fused-next-20261121/corpus \
  --expected-suite runs/suite-fused-next-20261121 --admission-mode staggered --reps 5 --label base --out base_dec8.json
python3 dec8_replay_audit.py record --corpus-dir runs/suite-fused-next-20261121/corpus \
  --expected-suite runs/suite-fused-next-20261121 --admission-mode staggered --reps 5 --label next --out next_dec8.json
python3 dec8_replay_audit.py compare base_dec8.json next_dec8.json
```
