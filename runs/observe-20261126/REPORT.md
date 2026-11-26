# Live vs expected on image `1421187` (2026-11-25 19:29 → 2026-11-26 04:22 UTC)

Read-only review of OpenObserve (the engine `done` lines shipped since `runs/observe-20261125`, the `tensorfold:*`
series, host metrics and DCGM) plus `docker inspect` on both ranks. Nothing was restarted or changed. The expected
figures come from:

- `runs/observe-review-20261123/RESULTS.md`: the one-stream line `ms/round = 24.73 + 3.524 × drafted/round`, `dec8`,
  and live E1, E2 and E7.
- `runs/deploy-1421187-20261125/REPORT.md`.
- `runs/mixed-20261125/REPORT.md`.

**Config verified live.** Both ranks run `bench-fused-next-1421187`, started 19:27:45, with `MARKOV_CACHE=4096`,
`DRAFT_PREFETCH=1`, `KV_SHARD=1`, `PREFILL_INTERLEAVE_MS=250`, `PREFILL_SLICE=5` and the rest of `best-runtime.md`'s set.
The model is 105.71 GiB: 105.21 plus the 473 MiB Markov rows.

**The serving lifetime starts at 19:29**, not at the 19:10 deploy: four restarts followed it. Its first 23 requests
are the loopback validate and are excluded. After them come 2,852 requests from one non-loopback address. The `done`
lines carry no client identity, so "one client" below means one source address.

## Summary

| Area | Expected | Live | Verdict |
|---|---|---|---|
| 1-stream round, no other request in flight during decode (exact, from `done` intervals) | 2bb052f line −~1.2% (the adopted pass gain). E2 excess +7.5 / +9.5 / +12.3% | +3.0 to +6.3% (<8K, n=4) · **+6.1%** (8–64K, n=24) · **+8.0%** (64–200K, n=57) · +9.7 to +11.0% (≥200K, n=5) | **As expected.** Same depth gradient as before, 1.5–2 points under E2. The −1.2% gain is below what live traffic can resolve: no regression, and the gain is not confirmed |
| 2–3 streams, decode round vs the line at the total drafted rows | E2 +30% at 2 streams | **+18%** (n=104 and 99, 100–180K deep, no prefill overlap) | As expected, better than E2 |
| 8-stream round | `dec8` ~87 ms; live E7 91.7 ms | **212.6 ms** median (p10 152, p90 271), n=1,647 | **2.3× off.** Source: D1 |
| 8-stream throughput | 136–138 tok/s `dec8`; 164 tok/s live E7 | **68–70 tok/s** by wall (10.2 tok/s a stream) | **−50%.** Source: D1 |
| 5–8-stream round with prompt-pass time removed | ~87 ms (`dec8`) | **86–93 ms**: 86.2 if decode stops completely during a pass, 93 from the two-rate fit | **As expected.** The decode path is fine |
| Short-tail prefill (one-chunk passes, ~1.7K uncached) | ~1,680 tok/s (cache replay, a 1,796-token tail) | **1,491** median. 1,528 in the lowest Engram tercile, 1,369 in the highest | **−9 to −11%.** Part of it is D2 |
| Cold prefill | 1,886–1,910 tok/s at 300K; E7 2,051 at 15.4K | 1,873 at 271K; 1,958 (×2) at 14.5K | 271K as expected; 14.5K −4.5%, see D4 |
| Resumed tails of 4–8K | E2 1,619 tok/s | 1,580 (n=10) | As expected |
| TTFT for 256–2,047-token tails, alone | E2 1.28 s | 1.20 s (n=41) | As expected |
| GPU and host | | SM 2,171–2,190 MHz, ≤76 °C. Swap 18 MiB, major faults ~0 | No throttling, no paging |
| Head memory | 7.4 GiB available at deploy | `free` shows 6 GiB available (7.4 GiB free + cached + buffered) | Down ~1.4 GiB in 9 h. Memguard floor is 2.5 GiB; worth watching |

## Divergences and their sources

**D1. During the saturated closed-loop burst (03:00–04:22), decoders run at about 5% of their rate while any prompt
pass runs, because short passes never yield. This is the whole 8-stream divergence.**

*Workload*
- One source address keeps about 21 requests outstanding: 8 running and 12–14 waiting.
- Prompts are about 5K tokens and 65–82% cached, over 1,310 distinct prompt lengths.
- Median 1,904 uncached tokens and a median 104-token reply, with `effort=low`. About 29 new requests a minute.
- That is about 880 uncached tok/s of prefill demand, so prompt passes fill **60–64% of wall time**.
- The 23 s median queue follows arithmetically from a closed loop of 21 against 8 slots; it is not an engine fault.

*Measurement*
- Each 8-stream request's decode window is intersected with the union of every request's prompt-pass interval, taken
  from the `done` lines.
- The two-rate fit is `1/ms = 0.01072 − 0.01013 × overlap`. That gives a decode-only round of 93 ms, with **decoders
  keeping f = 0.05 of their rate during a pass**.
- `mixed-20261125`'s 128K cold prefill gave f = 0.38 (95.5 → 251 ms). At f = 0.38, the live 60% overlap would predict
  about 137 ms, not 213. This burst is worse than the interference benchmark because of the mechanism below.

*Mechanism (source `tfref-kstack5` at `1421187`)*
1. `TF_DS_PREFILL_INTERLEAVE_MS=250` runs other streams' rounds only *between* 2,048-row chunks. In `lanes.zig:1138`,
   `if (f.start < len …) if (s.between) …` follows each chunk. `lane_host.zig:769` (`interleaving()`) gates it.
2. Production has `TF_DS_KV_SHARD=1`. `lanes.zig:1118–1119` reads "TF_DS_KV_SHARD pins the twins for the whole chunk:
   it runs whole", then `sliced = m.slice_layers > 0 and self.kv == null …`. So **`TF_DS_PREFILL_SLICE=5` is disabled
   under KV shard.** This also explains `mixed-20261125`'s "SLICE is inert" finding.
3. Chunks are aligned to 2,048 rows (`model.zig:45`). Of the 2,291 saturated passes:
   - **1,427 (62%) are one chunk and never yield.** They take 49% of prefill seconds.
   - 851 are two chunks and yield once, for up to 250 ms.
   - 13 are longer.

   A 128K cold pass has 64 boundaries to interleave at.

*The same mechanism at lower counts*

| Streams | Requests | Prompt-pass overlap | Round with pass time removed |
|---:|---:|---:|---:|
| 5 | 40 | 0.50 | 86.6 ms |
| 6 | 214 | 0.52 | 87.1 ms |
| 7 | 37 | 0.59 | 86.7 ms |
| 8 | 1,613 | 0.59 | 86.2 ms |

At 4 streams (n=34, overlap 0.60) the figure is −6% against the line.

*Restores add latency of their own, but not through disk and not by waiting for a pass*
- 302 disk hits, 291 restore waits totalling 328 s: **1.13 s a wait**. The `done` median is 1.04 s.
- `restores_room_waits` 0, `refused_budget` 0, and `spill_read_bytes` under 1 MB/s.
- Restore windows overlap no other request's prompt pass (median 0.00).
- Before 03:00, two restores of 12K tokens took 0.12 s.
- So restores slow down under load somewhere in the restore service's own pacing. **Where is open.**

*Why there are restores at all*
- `prompt_cache_held_bytes` is 0.49–0.53 GiB with 77–79 entries. That is the 512 MiB memory tier full (about 80
  untrimmed entries, per `best-runtime.md`), not an entry cap.
- `evicted_lru` adds 410 in 70 minutes.

**D2. Engram host time per request rose about 50× with the prefill rate. The deploy did not cause it, and it costs
short passes about 10% at the high end.**

*The rise*
- `engram_host_ms` per uncached token: 0.019 (01–02 UTC) → 0.30 (02–03) → 1.00 (03+).
- Lookup latency rose from 3.5 to 18 ms.
- Host disk reads are about 390 MB/s on both nodes, with page cache flat at 6.5 GiB: Engram page-ins.

*Not the deploy*
- Disk MB per uncached prefill token is 0.46 now, against 0.50 (E1), 0.55 (E2) and 0.62 (E7).
- Lookup latency rises with prefill rate here as it did on E1 (13 ms at 615 tok/s).
- So the Markov cache's 473 MiB taken from the page cache did not raise disk reads per token.

*Cost*
- `ENGRAM_LEAD` hides most of the latency, but not all of it.
- One-chunk passes in the highest tercile of Engram ms per token (median 2.8) run at 1,369 tok/s, against 1,528 in the
  lowest (0.5). Two-chunk passes: 1,394 against 1,516.
- The remaining −9% against the replay's 1,680 is not attributed. The replay ran on 8K contexts with no other load.

*Not swap*
- The `system_paging_operations` series is about 2× the disk-read bytes, with near-zero major faults and 18 MiB of swap.
- It is file page-in.

**D3. One-stream rounds remain 6–11% above the line, rising with depth.** This was already present on E1, E2 and E7
(D1 of the 11-23 review) and is not new. The earlier ≥200K outliers of +49% to +136% were requests the 30 s gauge
called "alone" while 1–3 others were in flight; the exact filter removes them.

**D4. Minor, unconfirmed: a cold 14.5K prefill ran at 1,958 tok/s.** Two different client prompts both gave 1,958,
against one E7 request of 15.4K at 2,051 (−4.5%). Neither adopted switch touches the prompt pass, and the samples are
2 against 1. A 16K cold cell on `2bb052f` vs `1421187` would settle it.

## What would move D1 (not done: decisions for the operator)

**Correction (added after review): items 1 and 2 move latency, not throughput.** The GPU is serialized.
- In `mixed-20261125`, every interleaved arm's wall time (120 s) equals solo prefill (~63 s) plus solo decode (53.9 s)
  plus about 3%, at any interleave setting.
- Live, prompt passes (~372 s) plus decode rounds (~2,350 × 86 ms ≈ 202 s) fill about 96% of each 10 minutes.
- So this closed loop is bound by prefill work. Interleaving more makes decode rounds faster and prompt passes, and
  therefore TTFT, slower by about the same GPU time.
- End-to-end time per request (≈ 21 outstanding ÷ throughput) does not improve.
- Item 2 at N above ~8 rounds per completion idles slots and lowers throughput.
- What would raise throughput on this mix is less prefill work or faster short-tail prefill: the −9% residual in D2,
  and cache hits.

1. **Make short passes yield.** Let layer slicing work under `KV_SHARD`, which today pins the twins for the whole
   chunk, or give single-chunk passes a sub-chunk boundary. This is an engine change. The bound is f 0.05 → ~0.4,
   which is about 213 → ~140 ms a round for decoders, paid for by correspondingly longer prompt passes.
2. **Cap prefill share while a queue exists.** Admit at most one pass per N rounds while k or more streams decode. That
   trades TTFT for decode: the same priority dial `mixed-20261125` measured.
3. **Memory-tier budget or snapshot trim.** The 512 MiB tier is full. `best-runtime.md` records that trim takes it from
   80 to 143 ordinary entries, which would turn more of the ~1 s restores into memory hits.
4. **Add a suite cell with this shape.** 21 closed-loop clients, 5K prompts with ~2K-token tails, 100-token replies.
   No current benchmark has many short passes; `mixed-20261125` has one long one.

## Open

- Where the ~1.1 s restore wait goes under load (D1). Small reads, no room or budget waits, no pass overlap.
- The −9% short-tail prefill residual left after Engram (D2).
- Head MemAvailable fell from 7.4 to 6 GiB over 9 h.

## Files

| File | What |
|---|---|
| `scripts/pulls.py` | Every OpenObserve query: logs, gauges, host and counter tables, the epoch comparison, restore counters |
| `scripts/parse.py` | `engine_logs.json` → `data/reqs.json` |
| `scripts/score.py` | First-pass benchmark comparisons (gauge-based "alone"; superseded by `fixes.py` for one stream) |
| `scripts/churn.py` | The 10-minute time budget |
| `scripts/overlap.py` | Prefill-overlap decomposition |
| `scripts/deeper.py` | Two-rate fit, Engram, restores |
| `scripts/fixes.py` | Exact-concurrency one-stream bins, restore overlap excluding the request's own pass, short-tail prefill by chunk count and Engram tercile, 2–8 streams |
| `data/gauges.json` | The 30 s running and waiting gauges |
| `data/restore_wait.txt` | Counter deltas for 03:10–04:20 |

The raw log pull (`engine_logs.json`, which holds the client address) is not committed. To rerun, from the repo root:

1. `. runs/observe-review-20261123/scripts/o2env.sh`
2. `python3 -I runs/observe-20261126/scripts/pulls.py OUTDIR`
3. Run the other scripts in `OUTDIR` in the order of the table.
