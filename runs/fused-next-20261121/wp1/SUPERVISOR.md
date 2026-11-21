# WP1 supervisor check (2026-11-21)

- Full-harness parity, base `3c54a9d` (OLD) against the WP1 tip `d01975d` (NEW), all kernels, parity only:
  every committed kernel family passes (0195 idx 29, idx-small 7, block-max 4, idx-hist 12, 0197 decode 25 /
  prompt 6, 0198 key 5 / latent 6 / query 5 / prompt-key 3 / prompt-latent 3, 0199 prune 8, kit score-keys 24).
  [supervisor-parity.log](supervisor-parity.log)
- Harness fixes needed first (pi had only run filtered subsets): the k == nb indexer cases launched the unfused
  `topk_indices` reference outside its contract and faulted the context (the engine pools only when
  `nbk > candidate_blocks`, so k < nb at every call site); those cases now compare new with old only. The key-fold
  cases (the closed 0206 experiment) now run only with `--kernel fold`, since the committed tree has no fold variant.
- Host tests, fresh cache: 794/796, the two pre-existing failures only.
- Open for the pair gate: 0216's prompt bypass changes a host path the harness cannot byte-check.
