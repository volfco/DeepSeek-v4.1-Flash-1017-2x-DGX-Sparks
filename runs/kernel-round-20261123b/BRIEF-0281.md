# BRIEF-0281 — TF_DS_MARKOV_CACHE: more cached Markov bias rows

**Switch:** `TF_DS_MARKOV_CACHE=<rows>` (unset: 256, today's). **Module:** extend `markov_tokens.zig` (+ a small
parse module if cleaner).

**Why.** Each Markov step looks up a bias row for the previous token. 256 rows are precomputed at load
(`draft.zig:424–436`, `tri_markov.zig` `fill`, `cache_rows = 256`; the list is `markov_tokens.cached`). Served miss
rate is 52–53% (k267, kall). On a miss the kit `_bias` recomputes the row against the Markov head, 64,640 × 256 fp16 =
33.1 MB a rank. `markov_tokens.zig` says it: "Which tokens have cached rows changes the time a step takes, never a
draft." The kit calls this knob `TF_DS_MARKOV_CACHE`; the port hard-coded its default.

**Build.**
- `TF_DS_MARKOV_CACHE=N`: the first N ids of an extended ranking are cached. **The first 256 must be today's list in
  today's order** (unset and `256` are then byte-identical to `2bb052f`, slot numbers included).
- The extended ranking's source matters. Preference order:
  1. the kit's own longer ranking, if the served kit's `markov_tokens` list holds more than 256 ids (the kit is on the
     head at the path the armed env names, `TF_DS_KIT=/hf/tensorfold-dsv41/kit-abe07c7d/kit`; a read-only
     `ssh colum@100.95.49.21 cat <file>` is allowed for this, nothing else on the head is);
  2. otherwise a frequency ranking you build from general text.
  **Do not rank from `runs/overnight-20261122/corpus.json`**: that is the benchmark's own prompt corpus, and a list fitted
  to it would overstate the gain. You may use that corpus only as a held-out check of the hit rate (the served 256
  hit 49.2% of its tokens). Record the source and the held-out hit rates at 512 / 1,024 / 2,048 / 4,096 rows in the report.
- Bounds: 256 ≤ N ≤ a cap you justify. A row is 64,640 × 2 bytes a rank (4,096 rows = 530 MB a rank); the arena take
  in `Drafter.init` must follow N; and `rows × cols` must stay well under 2^31, because the kit `_score` indexes the
  cache with arithmetic we cannot read (it is a cubin). Refuse startup beyond the cap.
- The slot table, the host-side hit/miss counter from 0280, and the load-time `fill` (16 rows a launch) all follow N.
- Banner at load: rows cached and bytes a rank.

**Parity and the gate that actually checks it.** Output tokens cannot move whatever the cache holds, so the exact
panel is **not** a test of this patch. A wrong cache row would change drafts, which shows only as different rounds or
accepted/drafted counts. State in the report that the hardware gate must compare **per-request rounds and
accepted/drafted** against control (seeded requests), not just token hashes.

**Out of scope (note as follow-up only):** a per-reply LRU filled on a miss (the saved-corpus estimate is 66–73% with
256 extra rows). It needs device-side slot updates inside the pass graph.

**What only hardware can settle.** The cost of a miss step. `runs/negative-kernels-20261122/markov.py` times `_bias`
at 0..N missing rows; it has never been run. Two unknowns it resolves: whether the 33 MB head stays in L2 between
steps, and whether `_bias` skips the head read when every row hits.
