# zstd recompression check

The one-shot check that closes or opens "a lossless recompression of the trellis" (`docs/benchmark-gaps-and-moe-review.md`,
section 5; `docs/baseten-ultrafast-plan.md`, second pass §4.5 "entropy coding"). A round reads ~3.5-7 GB of trellis
bytes at ~200-220 GB/s; if the stored bytes can be losslessly compressed and decompressed on the fly, the bus carries
less. The review's own test: **one `zstd -19` of a 1 GB slice of the rank cache settles it; under 3% saving: close.**

## Run (not run by us; the pair is the operator's)

```
just zstd-check                     # default: the rank cache under ~/.cache/dsv41-tensorfold/rank-cache-*, 1 GiB
just zstd-check file=/path/to/rank-cache size=2048
```

Or on the head directly: `bash runs/zstd-recompress/check.sh RANK_CACHE [SIZE_MIB] [OFFSET_MIB]`. The slice defaults
to the file's middle (past the header region); pass `OFFSET_MIB` to sample elsewhere. Read-only against the cache;
scratch is a `mktemp -d` removed on exit (the restore step is nothing: no state changes).

## What it prints, how to read it

One row per level (-1, -6, -19): compressed MiB, saving vs the slice, encode/decode MiB/s, and `lossless` — every
level is decompressed and compared byte for byte before its row prints (a failure stops the script). Then a verdict
line with the best saving against the 3% bar.

- **Under 3%: close the idea** (the review's stop rule). The trellis is already near-entropy-coded.
- **At or above 3%:** a recompression design opens (a decompression kernel beside `grouped_cp`, or a decompress-on-
  load pack format); the encode/decode columns say whether on-the-fly decode is plausible at bus speed (it will not
  be at `-19`; the design would need a fast codec or a load-time decode to a wider format — that is M5's ground).
- **The sample matters:** one slice is the review's bar. If a verdict is close to 3%, rerun with `OFFSET_MIB` at a
  few offsets before deciding.

## Provenance

Script added 2026-11-16 with the second-pass patch series (`zig/patches/0160`-`0181`); no engine patch is involved —
this is a check, not a change. Nothing here has been run on the pair.
