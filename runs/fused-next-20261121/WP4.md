# WP4: PDL overlap and the cross-kernel boundaries

Evidence folder: `runs/fused-next-20261121/wp4/`. Engine commits numbered from **0250**. Use the WP0 harness, extended
with a chain mode that replays several kernels back to back in one graph (the serving shape) instead of one kernel.

Items (todo.md text is the spec):

1. **Measure the decode chain's launch floor** (research item "Measure the decode launch floor"): the fused kernels of
   one decode layer in their serving order in one graph, with and without PDL (`TF_DS_PDL`), against the same number of
   empty nodes. This says how much of a layer is launch/dependency overhead the kernel bodies cannot reduce.
2. **Split PDL entry around proven independent work** (research item): launch dependents earlier and prefetch only
   fixed weights / provably initialized storage before `griddepcontrol.wait`; every dependent read stays after the
   wait. Delayed-producer canaries, graph and non-graph parity, SASS placement of the wait, measured chain overlap.
3. **Prove engagement** (research item): an opt-in counter or a trace that shows which fused path ran at each row
   count / width / PDL setting on the real call sites (host-side, via the engine's own test harness where possible).
4. **Durable parity target** (research item "Make fused-vs-unfused parity durable"): make sure `tools/fused_bench`
   covers every kernel WP1-WP3 changed, with the fallbacks, as one `just fused-bench` command that prints pass/fail.

Finish with `wp4/RESULTS.md`, including the per-layer budget: launch floor, kernel bodies, and what is left to fuse.
