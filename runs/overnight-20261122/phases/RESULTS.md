# Both ranks' phases on production's config, November 22: the ranks are balanced; prefill is experts, gathers and dense GEMMs

**Setup.** Image `bench-fused-next-73bbdfd`: production (84806bb + 0259) plus a diagnostic that makes the follower print its decode `phases_ms` at exit. `TF_DS_ROUND_PROFILE=2 GRAPHS=0 RANK1_PROFILE=2 TF_DS_PREFILL_PROFILE=1` with production's `.env` (reuse on), 08:33–08:42 UTC.
- Arm d32x4: four cold 32K histories, then 4-stream greedy decode of 320 tokens.
- Arm p128: one cold 128K prompt.

Rank 0's lines come from `/metrics`. Rank 1's are its exit print after `docker stop` of rank 0. Production was restored and validated (all passed). The profiles synchronize every round and chunk: use them for attribution only. [run](../phases.sh) · raw: `d32x4.rank{0,1}.txt`, `p128.rank{0,1}.txt`

## Decode: the ranks are balanced (item 1 closed)

d32x4, about 10.6 rows a round, eager, ms a round:

| Phase | rank 0 | rank 1 |
|---|---:|---:|
| moe | 39.04 | 39.27 |
| a_q | 2.61 | 2.65 |
| attn | 2.31 | 2.30 |
| a_core | 2.15 | 2.14 |
| gather_m | 1.85 | 1.57 |
| a_woa | 1.69 | 1.69 |
| a_in | 1.67 | 1.75 |
| gather_a | 1.65 | 1.59 |
| a_idx | 1.64 | 1.65 |
| head | 1.26 | 1.28 |
| m_rt | 1.08 | 1.08 |
| sh_rows | 0.55 | 0.55 |
| sh_fetch | 0.11 | 0.11 |

The two ranks' phases agree within about 1–2%. The November 18 stage windows, where the worker waited on the head (`gather_m` 9.3% vs 2.9%), came from an older image under stage instrumentation and no longer hold.
- The remaining collective cost is the collectives themselves: `gather_a` + `gather_m` ≈ 3.2–3.5 ms a round over 80 hidden-state all-gathers (two per layer), about 40 µs each, about 5.5% of a round.
- With 0259 on, `sh_rows` is 0.55 ms for 8 KV exchanges, about 69 µs each.
- Both are latency-bound. The next decode lever outside the kit is collective latency (a custom ring for the hidden all-gathers, as the KV path already has), not rank balance.

`forward_ms` 60.0 / 59.9. Drafter `pass_ms` was 6.86 at 3.19 streams a pass. In p128, `pass_ms` was 18.70 at 1 stream over only 14 passes (deep context; small sample, worth a look).

## Prefill: where a chunk's time goes (rank 1, `prefill_profile_ms`, ms a chunk)

| Phase | 4 × 32K cold (77 chunks) | 1 × 128K cold (66 chunks) |
|---|---:|---:|
| moe | 454.8 | 481.5 |
| ↳ e_gu (experts gate/up through the epilogue) | 281.3 | 302.3 |
| ↳ e_down | 144.2 | 156.7 |
| attn | 203.8 | 263.4 |
| ↳ a_core | 59.7 | 59.8 |
| ↳ a_q | 34.0 | 36.2 |
| ↳ a_woa | 34.9 | 38.5 |
| ↳ **a_idx (keyed indexer)** | **23.7** | **72.8** |
| ↳ a_in | 21.9 | 23.2 |
| engram | 52.9 | 43.6 |
| gather_a + gather_m | 99.8 | 101.2 |
| mix_attn + mix_ffn | 78.8 | 85.5 |
| dense (d_gemm / d_rot / d_unpack) | 90.3 / 19.6 / 18.8 | 101.2 / 21.1 / 20.7 |
| **chunk total (phases)** | **≈ 890** | **≈ 975** |

- **The keyed prefill indexer (item 2) is 2.7% of a chunk at 32K and 7.5% at 128K.** It grows linearly with depth, so it will be larger on the 300K–500K prompts the November 13 traffic carried. It is the only prefill phase that grows with depth here (`attn` 204 → 263 is almost all `a_idx`). A pool-style restriction does not apply (the keyed path has no pool). The lever is the keyed scoring and pruned top-k themselves.
- The experts' gate/up and down GEMMs are about 47% of a prefill chunk. The hidden-state all-gathers are 11%, the dense projections about 10–14% with unpack and rotation, and engram about 5%.
