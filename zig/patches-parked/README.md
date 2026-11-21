# zig/patches-parked

Patches that are written and kept, but not built: `zig/Dockerfile` applies only `zig/patches/*.patch`, and
`scripts/config.sh`'s recipe hash reads only that directory, so nothing here changes an image.

| Patch | Why it is parked |
| --- | --- |
| `0206-dsv41-tf-ds-0195-key-fold.patch` | Changes the reply on four of 39 exact cases (`len_20000`-`len_64000`) under `TF_DS_IDX_FUSED=1`. Likely cause: the fold writes `scoreKey`'s sign-flipped key where the kit's `topk_select_pruned` expects `_score_keys`' encoding. [Gate](../../runs/fused-followup-20261121/RESULTS.md) |
| `0207-dsv41-tf-ds-pair-softmax-one-exp.patch` | Bit-identical but 13-17% slower on the served decode shape under `TF_DS_FUSED_COMPRESS2_NORM=1`. |
| `0208-dsv41-tf-ds-0195-high-word-cache.patch` | Caches 0206's key high words; unproven until 0206 is fixed. |

To build with them, copy them into `zig/patches/` and apply them in this order after the whole series: the result is
engine tip `ffdce65` (`~/Workspace/tfref-recipe`, branch `recipe`) byte for byte. Each needs its own pass through the
39-case exact suite before it moves back.
