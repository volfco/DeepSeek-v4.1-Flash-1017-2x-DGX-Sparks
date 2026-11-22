# BRIEF-0267 — DSpark Markov bias and scoring fused on greedy draft steps

**Switch:** `TF_DS_MARKOV_FUSED` (off by default). **Module:** `zig/src/families/dsv41/markov_fused.zig` (pure).
**Commit subject:** `0267 dsv41: TF_DS_MARKOV_FUSED (off by default): a greedy draft step's Markov bias miss and tile scoring in one kernel`

## The item (todo.md, "Drafting and text prefill")

On a cache miss, round the computed bias tile to FP16, add the logits and emit the tile winner with the same tie rule.
Writing and rereading 64,640 staged FP16 values costs 252.5 KiB per missed stream, step and rank. Measure hits and
misses separately. The sampled path consumes an intermediate between these launches (`tri_markov.zig:124–141`). Count
bias-cache misses under KEPT=2 to size the opportunity. Compare tile winners on ties and final greedy draft IDs, and
keep the staged bias available to the sampled hook.

## Where

`zig/src/families/dsv41/tri_markov.zig` (`Markov.bias`, `fill`, `localBest`, `localBestWith`, lines ~64–170) and its
callers in `draft.zig` (the Markov loop after the head). The bias cache: find how a hit/miss is decided and where.

## Contract

- Off: launch- and byte-identical.
- On, greedy steps only: the draft IDs and every buffer a later reader uses are bit-identical. The sampled hook still
  gets its staged bias (if a step may be sampled, keep the old path for it).
- **Two deliverables:** (1) a hit/miss counter, emitted only under the existing profile switch
  (`TF_DS_ROUND_PROFILE`), so the operator can size it on hardware without the fused kernel; it must add nothing when
  the profile is off. Prefer a host-side count if hit/miss is decided on the host. (2) the fused kernel.
- If the bias computation is a kit kernel (a cubin with no source here), you cannot fuse into it: in that case the
  fused kernel must reproduce its arithmetic from the kit's source semantics bit-exactly, and the report must say
  exactly which rounding points are assumed. If that cannot be established, deliver (1) only and report.

## Harness (`h0267/`)

Emulate the fused kernel against an independent reference of bias → FP16 round → add → tile argmax with the existing
tie rule (lowest index? find it), including ties, NaN/inf logits, and misses for 1–4 streams.
