# BRIEF-0276 — the prompt cache replaces a stale duplicate instead of refreshing it (upstream PR #134, keep side)

**Switch:** `TF_DS_CACHE_REPLACE_STALE` (off by default). **Module:** `zig/src/families/dsv41/cache_replace_stale.zig` (pure), or a
`core` module if the change lives in `zig/src/core/prompt_cache.zig`. Follow where the code is.
**Commit subject:** `0276 core: TF_DS_CACHE_REPLACE_STALE (off by default): keeping a state the cache already holds replaces a stale duplicate`
**Source:** `src/upstream-pr134-0069.patch` (upstream MiaAI recipe PR #134, "cache usable without spill").

In our tree `zig/src/core/prompt_cache.zig` `keepWith` (~line 1134), a keep that matches an existing entry removes it only
when `full_end and !e.full_end`. Otherwise it refreshes `used` and keeps the old state. Our earlier review
(`docs/prefix-cache-hitrate-review.md:147-155` in the recipe repo) notes the `usable` half is already wired because production
runs with a spill tier (`SPILL_GIB=180`, `native.zig:443-449`). Port **only the half we lack**: the keep side.

- Understand and document what makes an entry "stale" in the source patch and in our tree (`usable`, `endstate`, tier state).
- Off: identical. On: show from the code which keeps change behaviour, and that a replaced entry can't be in use (lifetime/refs).
- The decode path must not change. This is a cache-policy change: it can change *which* prefix is restored, never a value.
- Harness: host test of `keepWith` sequences (keep, mark stale, keep again; with and without a tier; full_end vs not)
  against an independent model of the intended policy. Banner on first replacement.
