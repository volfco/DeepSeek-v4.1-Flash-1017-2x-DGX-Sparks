# WP2: the 0197 ratio-2 compressor (`tf_ds_compress2_norm_kernel`, `tf_ds_compress2x_norm_kernel`, `compress2_norm.zig`)

Evidence folder: `runs/fused-next-20261121/wp2/`. Engine commits numbered from **0230**. Use the WP0 harness.
WP0's baseline says how close the decode form already is to the launch floor (1.6-2.3 µs at R<=48): at decode shapes
only a change that lowers the launch count or the node's latency can show up; judge prompt shapes separately.

Items (todo.md text is the spec):

1. **128 threads with four channels a thread** (todo item 6): re-test 128/256/512 on the *current* arithmetic at R =
   1/7/32/48, widths 128/512; byte-identical output, rings, stride padding and the invalid flag.
2. **Duplicate exponentials** (item 2) and **0207's one-exp form** (parked: bit-identical but 13-17% slower at the served
   decode shape because of its `isfinite` branch): read the SASS; try a branchless form (select, or a finite-pair
   predicate hoisted per warp) and keep it only if it beats the current kernel at the served decode shape.
3. **Duplicated partner resolution across warps** (research item): measure executed ballots/loads; try resolving once
   per block into shared memory or registers broadcast by shuffle.
4. **Compute a repeated ratio-2 pair once per speculative window** (Compression item): rows with the same resolved
   source pair share the softmax and normalized latent; every input row still writes its own raw-ring slot. Check
   separate requests, ring wrap and partners from an earlier round.
5. **Fold prompt raw-ring writes and position setup into the fused path** (Compression item): remove the metadata
   uploads and scatter launches; preserve ring wrap, an odd trailing row and the one-row case.
6. **Occupancy and shared sizing** (research item): only if ptxas/occupancy shows shared memory limits residency.

Finish with `wp2/RESULTS.md` and the kernel's final position against the launch floor and roofline.
