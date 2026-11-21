# Depth quality and the Engram lead reader, November 20

**Superseded in part by the [sampled rerun](#sampled-rerun-2026-11-20-1740-1756-utc-production-no-restart).** At the clients' sampling (T=1.0, top_p 0.95, top_k 20) production recalls all eight codes and the sum at 530K on 5 of 5 seeds; only the order of mid-context entries slips. The greedy 530K failure below is a greedy-decoding artifact (upstream MiaAI-Lab#54). Runaway reasoning to the token cap happens at any depth (2 of 5 seeds at 64K). The lead reader's bit-identity and the cache-resume result below still stand. The repo baseline was tested greedy only.

**Result.** The production configuration fails a planted-fact recall task at 530K tokens, with or without
`TF_DS_ENGRAM_LEAD`. The lead reader changes no output bit at any depth tested. It recalls all eight facts at 4K, 64K
and 256K. At 530K it reasons until the 4,096-token cap and never answers (16K characters of reasoning, 96% draft
acceptance, a repetitive loop). Resuming the 530K prompt from the prompt cache gives the same tokens as the cold run.
The reported degradation (a live agent session at ~533K tokens, 69.6% of a 768K client window) is a depth effect, not
a configuration change.

| Depth | Lead on (`on1`) | Lead off | token_sha (both arms) |
| --- | --- | --- | --- |
| 4K | 8/8, sum ok, 530 tokens | same | 88547d8fd281 |
| 64K | 8/8, sum ok, 667 tokens | same | b53a41c4cb74 |
| 256K | 8/8, sum ok, 1,901 tokens | same | 81ebb9e2aec5 |
| 530K cold | 0/8, cap reached | same | e0c1358b9f0c |
| 530K resumed (528,384 cached) | 0/8, cap reached | same | e0c1358b9f0c |

Setup: production image `exl3-tp2-ea522d353626` (digest 38bf190d) and `.env` on both arms, `TF_DS_ENGRAM_LEAD` set
per arm (banners checked), a fresh process with a private spill directory for each. Greedy (temperature 0, seed 1234),
thinking on. The probe is eight "vault code" entries at 2–98% of a natural-text prompt; the task asks for them in
order plus their sum ([probe.py](probe.py)). The second lead-on arm (`on2`) was dropped when a live client blocked it.
Production was restored at 06:18 UTC and passed `validate.py` (8 of 8 saved replies). One prompt per depth: the
failure point between 256K and 530K is not located.

Supporting trace: in the production log, draft acceptance fell from 82–83% (prompts under 185K) to 69–74% (260K to
533K) on November 19–20.

## Repo baseline, November 20 16:00 UTC

Same image and probe with every tuning key removed from `.env` for the start (`PARALLEL`, `KV_SHARD`,
`TENSORFOLD_TOKEN_CACHE_MB`, `TF_DS_T13_EXPERT_PRUNE`, `TF_DS_T26_KEYED_DRAFT`, `TF_DS_T27_DEVICE_DRAW`,
`TF_DS_PREFILL_SLICE`, `TF_DS_DRAFT_BLOCK`, `TF_DS_ENGRAM_LEAD`, `TF_DS_RDMA_CPUS`). `ABLIT=1` kept, at colum's choice.
Recipe defaults only on both ranks (4 streams, replicated KV). [Script](baseline.sh) · [Log](baseline.log)

| Depth | Production | Repo baseline |
| --- | --- | --- |
| 4K | 8/8, sum ok | 8/8, sum ok (726 tokens) |
| 64K | 8/8, sum ok | 8/8, sum ok (660 tokens) |
| 256K | 8/8, sum ok (1,901 tokens) | **0/8, cap reached** |
| 530K cold and resumed | 0/8, cap reached | 0/8, cap reached |

The baseline is no better: it fails at 530K too, and at 256K where production passed. Its tokens differ from
production's at every depth (expert pruning off changes the arithmetic). Both configurations sit near a cliff
around 256K, and which side a given prompt lands on is configuration-sensitive. One prompt per depth: neither the
256K difference nor the cliff's position is a measured rate. Production restored at 16:15 UTC, `validate.py` all passed.

## Caveat from upstream: greedy decoding loops on this model

Every run above is greedy (temperature 0), and every failure is a reasoning loop to the cap. Upstream reports that
greedy decoding itself makes this model loop: "At T=0 both builds sometimes loop in reasoning until `max_tokens`. The
same items pass at T=1.0, so greedy decoding is not a good default for evaluations of this model"
([MiaAI-Lab#54](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/54), stebarit,
2026-10-30). The same reporter found no task-level difference from the FP8 weights at T=1.0 up to ~300K-token
archives, and nothing beyond 300K was tested. The recipe's own needle test found a single passphrase 3 of 3 times in a
479,012-token prompt (docs/measurements.md). So these results show where *greedy* reasoning starts to loop, not a
sampled failure rate. The lead-on/off and production/baseline comparisons still hold as bit comparisons. The open PR
[#138](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/138) backports TensorFold's opt-in
long reasoning-loop guard (`LOOP_GUARD=1`) for this failure mode. Owed: the same probe at T=1.0 / top_p 0.95 /
top_k 20 with several explicit seeds at 256K and 530K.

## Sampled rerun (2026-11-20 17:40–17:56 UTC, production, no restart)

`sampled.py`/`sampled.sh`: the same prompts, at the clients' sampling (T=1.0, top_p 0.95, top_k 20), explicit seeds
11/22/33/44/55, max_tokens 8192, against the live production config (lead on, ABLIT, all tuning keys as deployed).

| depth | seeds 8/8 codes + sum | order exact | ran to cap | notes |
|---|---|---|---|---|
| 64K  | 3/5 | 3/5 | 2/5 (seeds 44, 55) | greedy passed here |
| 256K | 4/5 | 4/5 | 0/5 | seed 22 dropped Garnet (the entry at 98%, just before the question) |
| 530K | 5/5 | 0/5 | 0/5 | greedy 0/8; every seed swaps Ember/Alder (entries at 45%/60%), seed 44 also Garnet/Brisk |

The 530K "cliff" was greedy decoding: at T=1.0 every seed recalls all eight codes and the right sum. What does
degrade at 530K is ordering of mid-context entries (consistent Ember/Alder swap), and the model is no worse at
530K than at 64K on finishing.

Runaway reasoning is depth-independent: `repro-64k-s44.json` reproduces seed 44 at 64K bit-exactly (sha
185248de0c1e). The reasoning had all eight codes, the order and the sum within ~2,000 chars, then started
"verifying order" by quoting the surrounding corpus and drifted into transcribing it until the 8192 cap. That is
over-verification, not a recall failure and not a short-period loop (no periodic tail), so PR #138's LOOP_GUARD
probably would not catch it; a thinking budget / lower reasoning effort is the relevant knob.
`reasoning_repeat` in sampled.json uses aligned 64-char chunks and misses such cases; ignore it.
