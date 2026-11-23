# Prompt-cache replay on production (2026-11-22/23): 0276 and 0271 are inert in production

Production image `10c9866`, production `.env` (`SPILL_GIB=180`, `SPILL_MIN_TOKENS=2048`, no `TF_DS_EXACT_RESUME`).
Workload `replay.py` (greedy, thinking off, C=4 concurrent, 48 output tokens):

- `repeat`: K distinct word-list documents, each asked R times with a different question;
- `turns`: K conversations of T turns, each resending the full history plus the greedy answer.

Counter deltas come from `/metrics`; `usage.prompt_tokens_details.cached_tokens` gives each request's resume point.

| Run | Requests | Prompt tokens | Memory hits / misses | Stale misses | Kept | Wall |
|---|---:|---:|---:|---:|---:|---:|
| `prod-repeat` K=6 R=3 | 18 | ~7.9K | 12 / 6 (no prefix) | 0 | 6 | 39.7 s |
| `prod-turns` K=6 T=4 | 24 | 7.9–8.1K | 24 / 0 | 0 | 0 | 39.0 s |
| `prod-big` K=16 R=2 | 32 | ~22.6K | 22 / 10 (no prefix) | 0 | 16 | 175.8 s |

Docs 0–5 were already cached by the earlier runs (same text), so the later runs show hits on their first rep.

## 0276 `TF_DS_CACHE_REPLACE_STALE`: no hardware arm, held

- No kept state went stale (0 `misses_stale`, no "goes stale" lines). That's expected rather than tested: ~480K of the
  pool's 2,101,248 positions were in use, and staleness only comes from a placement needing pool room
  (`lanes.zig` `placeKept` / `staleOver`). A pool-pressure replay (~100 distinct ~22K documents) is still owed. It
  would also exercise the spill-restore path, which has had 0 `disk_hits` all night.
- Even when a state does go stale, production's spill tier writes a durable point for every kept state
  (`prompt_cache_writes_enqueued` = `prompt_cache_kept`, 23/23 since restart). The point is assigned synchronously at keep time (`prompt_cache.zig:1250`, `t.vtable.keep` returns the id), so there is no window before the write publishes. 0276 deliberately skips entries with
  `e.point != null` (the tier's `adopt` already replaces their stale twin on restore). Only full-end keeps
  (`TF_DS_PROMPT_ENDSTATE`, off in production) carry no point.
- So the arm cannot fire under production's configuration. It stays committed off-by-default on `kernel-0276`
  (`7b7e4e1`). Revisit only for an off-tier deployment with `TF_DS_PROMPT_ENDSTATE` or `TF_DS_EXACT_RESUME`.

## 0271 `TF_DS_RESUME_ASSEMBLE`: inert in production, held

- It only runs on an exact-resume prepend (`replayCut`, `pre > 0`). `pre` comes from `tailPre` on the chunk's `Tail`, which only exists under `TF_DS_EXACT_RESUME=1` (`prompt.zig:188`, verified in `tfref-k0271`).
- `decisions.md` §0121 rules exact resume **lossy** (any off-grid chunk shape moves bits) and opt-in only.
  Production resumes on the 2,048-row grid, so 0271 can never engage there and needs no further hardware time.

## What the replay does show: multi-turn cost is the grid remainder

- Every resume lands on the 2,048-row grid: 7.9–8.1K-token turns resume at 6,144 and 22.6K-token prompts at
  20,480 or 22,528. Each later turn recomputes up to 2,047 rows of history plus its new text.
- At the head of the queue that costs ≈1.05–1.19 s TTFT per ~1.9K-row turn (≈1.7K tok/s). Under C=4 it queues to
  4–6.7 s, because prompt passes run one at a time: cold 7.9K prompts finish at 3.5 / 7.8 / 12.1 / 16.3 s
  (≈4.2 s each, ≈1.9K tok/s).
- A lossless cut needs the cold pass to chunk on the same grid as the resume, so a finer resume grid would also
  shrink cold-pass chunks and slow cold prefill. That makes exact resume a quality/latency trade for the operator,
  not an automatic adoption.
