# k278: 0278 TAPS_PACKED and 0279 IDX_PICK_WIDE, both exact, adopted (production 2bb052f since 05:20 UTC 2026-11-23)

Suite 04:31–05:11 UTC on `bench-fused-next-2bb052f` (`kernel-stack-0278` = 10c9866 + 0278 + 0279), production `.env`
(incl. `NCCL_MAX_CTAS=4`). Arms ctl1, t278a, w279a, ctl2, w279b, t278b. Workload `work.py`: cold 32K ×2 / 96K prefill
(salted), then greedy 256-token decode twice at ~300K (indexer nbk 32,768) and ~600K (nbk 65,536), unsalted.

- **Exact:** all six arms IDENTICAL 39/39 vs `deploy-10c9866/exact.json` and sampled IDENTICAL. Long-context decode text
  sha equal in every arm at both depths (`955a78bc29e9` 300K, `33354e45dc0a` 600K): 0279 keeps bits where it engages.
- **Engaged:** banners 1 on both ranks in every test arm, 0 in controls.

| Cold prefill tok/s | ctl1 | ctl2 | t278a | t278b | w279a | w279b |
|---|---:|---:|---:|---:|---:|---:|
| 32K (rep 1 / 2) | 2108.8 / 2104.6 | 2108.7 / 2108.6 | 2115.9 / 2117.8 | 2115.6 / 2115.8 | 2112.1 / 2108.8 | 2107.5 / 2105.9 |
| 96K | 2041.2 | 2040.0 | 2044.9 | 2059.1 | 2039.8 | 2041.2 |

| Decode tok/s (rep 0 / 1) | ctl1 | ctl2 | w279a | w279b | t278a | t278b |
|---|---:|---:|---:|---:|---:|---:|
| 300K | 73.62* / 74.25 | 74.14 / 74.38 | 74.12 / 74.66 | 73.85 / 74.27 | 74.03 / 74.34 | 74.12 / 74.14 |
| 600K | 71.56* / 73.10 | 73.11 / 73.37 | 73.02 / 73.31 | 72.90 / 73.12 | 72.86 / 73.10 | 72.93 / 73.24 |

\* ctl1 rep 0 decoded right after a cold 152 s / 332 s prompt pass; every later arm restored the prompt from the spill tier.

- **0278:** cold prefill +0.38% at 32K and +0.55% at 96K; every test cell is above both controls. Decode unaffected.
- **0279:** neutral at depth (300K 74.23 vs 74.26 control mean, 600K 73.09 vs 73.19), i.e. within run noise; exact; adopted
  under the exact-and-neutral rule (it removes one launch a decode round past 262K tokens).
- Rank-0 forward_ms (whole arm): 34.78–34.99 everywhere.
- **Spill tier (first real exercise):** after each restart the long prompts restored from disk: 300K TTFT 1.50 s (299,008
  cached) vs 152 s cold; 600K 2.71 s (598,016 cached) vs 332 s cold. The test text repeats one 145K corpus, so its
  decode rate (~73 tok/s at 600K) overstates real long-context decode (drafts accept easily).

Deploy (`runs/deploy-2bb052f-20261123/`): image 2bb052f + `TF_DS_TAPS_PACKED=1` + `TF_DS_IDX_PICK_WIDE=1`; both ranks armed,
exact IDENTICAL, sampled IDENTICAL, validate rc=0. Undo: `cp ~/dsv41-env-backups/.env.bak-20261123-pre-2bb052f .env &&
PREPARE=0 ./start.sh restart`.
