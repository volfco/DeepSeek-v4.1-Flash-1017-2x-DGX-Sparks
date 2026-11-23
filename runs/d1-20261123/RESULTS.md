# Drafter D1 overlay: rejected (exact, but 2.4% slower decode)

Suite 02:32–03:26 UTC 2026-11-23 on image 10c9866 + production `.env` (incl. `NCCL_MAX_CTAS=4`). Arms ctl1, d1a, ctl2, d1b;
D1 = ursuciprian/DeepSeek-V4.1-Flash-1017-DSpark-D1 @ ca1a99a8 (165 `mtp.*` tensors) laid over our checkpoint in a
sibling data dir (`build.sh`, `DATA_SUFFIX=-d1`).

- **Exact:** all four arms IDENTICAL 39/39 and sampled IDENTICAL (drafts are verified, as expected).
- **Engaged:** every probe row's round count differs from control (e.g. json s700 360 → 371, prose s701 433 → 442,
  sql s700 275 → 282), the same in both D1 arms. D1 accepts *fewer* tokens a round on every row.
- **Decode probe tok/s: −2.42% mean over 20 rows**; every row negative (−1.0% ts … −4.1% sql; streamed x1 −5.15%,
  x4 bursts −4.6%). Chunk-gap p95 −1.02%.
- Round-time workload: mean +0.41% (within the ±2–3% cell noise); rank-0 forward_ms 38.98/38.92 vs 38.80/38.94 and
  rounds 7,638/7,604 vs 7,503/7,490 (rows a round 6.31/6.34 vs 6.34/6.36).
- Likely cause: D1 was refit against LibertAIDAI's EXL3 2.9 bpw checkpoint (`afc7aa0a`); our served model is Mia EAQ v1
  2.9 bpw with the Keys ablit overlay, whose hidden states the stock `mtp.*` tracks better.
- The first D1 arm took 24 min (vs 9) — first load of a new 106 GB weight file from cold page cache.

Cleanup: both ranks' D1 weight files deleted (worker back to 236 GB free). The `-d1` checkpoint links dir (hard links +
the 0.5 GB overlay) is kept on the head; `DATA_SUFFIX` stays as a general hook.
