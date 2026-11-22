# BRIEF-0275 — TF_DS_COPY_MIN: the copy-draft minimum length as a switch

**Switch:** `TF_DS_COPY_MIN=<n>` (unset = today's hard-coded minimum, i.e. off-path identical).
**Module:** `zig/src/families/dsv41/copy_min.zig` (pure: parse, bounds).
**Commit subject:** `0275 dsv41: TF_DS_COPY_MIN (unset: today's minimum): the copy draft's minimum match length`
**Source:** `src/yushell-0006-copy-min.patch`. It claims 78.4→89.8 tok/s geomean on repetitive text at n=3, on a different
quant. Our gate is hard-coded at `zig/src/families/dsv41/windows.zig` ~line 141 (`copied.len < 2`); also `lanes.zig`.

- Unset must reproduce today's behaviour byte for byte. Read the source patch's semantics carefully: is its n the same
  quantity as our `copied.len < 2` threshold? Map it explicitly in the report, including what n=2 and n=3 mean in our tree.
- Lossless by construction (it only changes which drafts are proposed; verification is unchanged). State why.
- Parse: integer within a sane range, else refuse startup. Banner on first engagement when set.
- Harness: host emulation of the copy-draft proposal over token sequences (repetitive and non-repetitive), comparing
  proposals under n vs an independent reference, and showing unset == today.
- Also report whether `TF_DS_NO_COPIES` / `TF_DS_SOLO_OVERHEAD` exist in our tree and how they interact.
