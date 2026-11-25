# M3 draft-triage probe study (started 2026-11-25)

The killing experiment from `docs/benchmark-gaps-and-moe-review.md` M3: **can a linear probe on a mid-layer hidden
state predict "this draft row will be rejected" at ≥95% precision with ≥50% recall?** If not, drop M3.

M3's premise: a draft row whose predecessor picks another token is wasted work, and marking it dead mid-forward
cannot change a reply (a dropped row is an unverified draft). The mechanism already exists (`0177`,
`TF_DS_DRAFT_TRIAGE`) and its `conf:` stand-in (the drafter's chain-survival confidence) measured **−0.5%** — the
mask overhead plus lost speculation ate the gain. `file:<path>` is the trained probe's hook and refuses with
`TriageProbeUnsupported` because the probe was never built.

## Why this needed a build, not a benchmark arm

Nothing the study needs existed on the box: no hidden-state capture (`--dump` in the layer gate only dumps Engram
data), no recordings (`REC_DIR`), no layer-gate binary, no probe format. So the chain is:

| Stage | What | State |
|---|---|---|
| A | `probe-dump` engine patch: env-gated tap at one layer + ids/picks/drafts per verify round | in flight (worker) |
| B | build an image from the branch; capture a few thousand verify rounds | todo |
| C | `train.py`: logistic probe, precision/recall on a held-out split | written |
| D | report; the probe-load path (`file:`) is a separate follow-up | todo |

## Record format (what stage A writes)

Little-endian, one record per verify round:

    u32 magic = 0x4d503331
    u32 R                      rows in the round (pending token + drafts)
    u32 D                      hidden dim
    u32 layer                  the tapped layer
    u32 ids[R]
    u32 picks[R]               row r's sampled/greedy token (o.sampled[r])
    u32 drafts[R]              o.drafts[r]; the last slot unused
    u16 h[R * D]               bf16 bit patterns, row-major

Rows are `[pending, draft1 .. draftR-1]`. Draft j (row j+1) is accepted iff every earlier draft was accepted and
`picks[j] == drafts[j]`, so `alive[r] = AND_{i<r}(picks[i] == drafts[i])` and the target is `dead = !alive`.
Row 0 is not a draft and is excluded.

## Switches (stage A)

- `TF_DS_PROBE_LAYER=<n>` — the layer to tap (0..39); unset = off.
- `TF_DS_PROBE_DUMP=<path>` — the output file; required with `PROBE_LAYER`.
- Off path unchanged; no CUDA kernel added (reuses `tri_basic.tap`).

## Running

    # B: capture (production idle; instrumented image)
    TF_DS_PROBE_LAYER=13 TF_DS_PROBE_DUMP=/tmp/m3-13.bin ... serve ...
    # run a few thousand decode rounds, then pull the dump

    # C: score
    python3 runs/m3-20261125/train.py /tmp/m3-13.bin

## What the numbers bound

- Dead rows are 834 of 3,085 drafted (27%); MoE is ~24 ms of a 91 ms round.
- A *perfect* probe at a mid-depth cut saves ~6% of a round (the doc's ceiling).
- The `conf:` stand-in nets −0.5%, so the mask overhead + lost speculation ≈ the naive gain.
- Realistic upside if the probe is near-perfect: ~1–2% of a decode round.

## Two traps found while first running the capture (2026-11-25)

1. **`start.sh restart` runs a smoke test, so the engine's first verify round happens before the workload.** The
   probe writer opens `TF_DS_PROBE_DUMP` on that first round and holds the fd. A capture script that then
   `rm -f`s the dump unlinks the file out from under the open fd: the engine keeps writing to a deleted inode, the
   banner still prints, and nothing ever reaches the disk. Do not delete the dump after the restart — the writer
   truncates on open, so a stale file from an earlier run clears itself.
2. **`find / -xdev` skips the bind-mounted `/spill`** (and any other mount on its own device), so a locate step
   must not use `-xdev`. `std.Io.Dir.cwd().createFile(io, "/abs/path", .{})` does resolve absolute paths correctly
   (checked with a standalone Zig program against the same Threaded Io), so the absolute path is fine once (1) is
   respected.
