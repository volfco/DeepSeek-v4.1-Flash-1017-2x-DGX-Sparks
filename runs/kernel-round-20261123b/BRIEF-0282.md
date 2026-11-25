# BRIEF-0282 — TF_DS_DRAFT_PREFETCH: the paced L2 prefetch inside the drafter pass

**Switch:** `TF_DS_DRAFT_PREFETCH` (off by default; `1` on). **Module:** `zig/src/families/dsv41/draft_prefetch.zig`.

**Why.** The forward forks a paced L2 prefetch three times a layer (`round.zig` `usePrefetch` / `prefetch` /
`joinPrefetch`; call sites near `:1216`, `:1428`, `:1763`; kernel `ops.l2Prefetch`). Turning it off costs 5.3% at one
stream (attention producer 240 → 198 µs a layer, MoE 559 → 537 µs). The drafter pass runs three more EXL3 stages and a
full vocabulary-head read with no prefetch at all: `draft.zig:10` says the served build's forks there "move no bytes
and are left out". Separately, k267 showed a change inside the pass moving the next forward's GPU time by about
0.25 ms in 12 of 12 cells, so cache state across the pass-to-forward boundary is worth something.

**Build.** With the switch on, `Drafter.body` forks the same paced prefetch on a stream of the drafter's own (created
at `Drafter.init` only when the switch is on), mirroring the forward's three sites per stage:
1. behind the attention input linears (`wq_a`, `wkv` issued): this stage's `wo_a` groups and `wo_b`;
2. behind `wo_b`: this stage's ffn mix weights, the router gate, the shared expert;
3. behind the experts: the next stage's attention mix weights and input linears (`wq_a`, `wkv`, `wq_b`); after the
   last stage: the head (`e.w.head`), as the forward does after its last layer.
4. **Hand-off, value `2`:** additionally, behind the Markov loop's last step, the target's layer-0 input linears
   (what the next forward reads first). `1` = sites 1–3 only, `2` = 1–4, so the two effects can be separated.
Use the forward's byte budgets (16 MiB, half behind `wo_b`) and the same `l2Prefetch` call; if 0283 has landed in
your tree, take its values. Join the prefetch stream before the capture ends and before `Drafter.pass` synchronizes,
exactly as a stretch does. Ranges come from `weights.DSpark` blocks; a plain (non-trellis) linear has no range and is
skipped, as `linearBytes` does in `round.zig`.

**Parity.** The kernel writes nothing; drafts and outputs cannot move. The pass graph gains launches on a second
stream: give the count per pass in the report.

**Report.** Bytes prefetched per site at the served shape, and anything about the drafter's weights that makes a site
pointless (for example a linear small enough to sit in L2 anyway).
