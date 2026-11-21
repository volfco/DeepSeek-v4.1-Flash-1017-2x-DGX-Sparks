# Prune stats short benchmark, November 20, 2026

Stats-off produced no measurable service throughput improvement in this short bracket. The synthetic CUDA comparison shows substantial kernel savings from stats removal and six-slot specialization; those are component timings, not measured service gains. Nothing deployed.

## Service bracket

Head gx10-eb0e and worker 10.50.0.11 were idle (0 connections/waiting, no running benchmark campaign, zero non-loopback POSTs in current service log). Used existing image `tensorfold-dsv41:bench-fused-rework-20261120` (`dae2a97e…`) and existing `arm.sh`/`probe.py`. Production environment plus `TF_DS_DRAFT_TRIAGE=conf:0.2 TF_DS_FUSED_ROUTE=1` throughout; only `TF_DS_T13_PRUNE_STATS=1/0/1` varied. T13 prune enabled on both ranks. Private spill per arm. Each arm ran greedy Tetris 1024, greedy code 1024, sampled Tetris x4 768, twice.

| Arm | Tetris tok/s | Code tok/s | Sampled x4 wall tok/s |
| --- | --- | --- | --- |
| control1 | 126.3 / 126.5 | 118.7 / 118.7 | 208.37 / 211.59 |
| candidate | 126.0 / 126.7 | 118.6 / 118.4 | 210.17 / 210.18 |
| control2 | 126.1 / 126.6 | 118.6 / 118.4 | 210.12 / 208.59 |

Candidate versus pooled-control medians: Tetris −0.04%, code −0.13%, sampled x4 +0.39%. Control ranges: 126.1–126.6, 118.4–118.7, 208.37–211.59 respectively. Sampled rounds/acceptance varied despite identical reply hashes. All 18 request-group reply hashes matched across arms (six distinct output hashes including four sampled streams); greedy round counts matched. This is reply-hash equality on these probes, not a full exact-suite claim.

Production restored at 19:35:51 UTC, `validate.py` passed 8/8 saved replies. Both ranks subsequently verified on `exl3-tp2-ea522d353626` with no triage/fused/stats override; post-micro health remained healthy with no clients. See `restore-validate.log`, `post-micro-restoration.txt`.

## Standalone CUDA microbenchmark

Source supplied by parent review: `micro/prune4.cu`, extracted actual generic prune kernel and experimental six-slot and eight-lane-per-row variants. Compiled in the same image with `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false -Xptxas=-v`. Run only after service timing and restoration, with idle client check. One warm pass and 12 rotated measured passes per mode/shape; each CUDA graph has 256 kernel nodes over distinct synthetic rows. Raw pick and weight buffers equal to generic stats-on on warm pass for every shape/mode, both null and synthetic 25% dead masks. No failure stderr; both exited zero. These variants are standalone, not implemented service patches. Synthetic rows test ties and missing experts; no representative routed-traffic distribution or full edge-case coverage claim.

Median microseconds per graph node (kernel) and observed min/max in CSV.

| Mask | Rows | Generic stats | Generic no stats | Six-slot | Eight-lane tile |
| --- | --- | --- | --- | --- | --- |
| null-mask | 1 | 8.453 | 7.388 | 1.796 | 1.667 |
| null-mask | 7 | 9.765 | 8.615 | 2.120 | 2.581 |
| null-mask | 28 | 10.686 | 9.413 | 2.400 | 2.591 |
| null-mask | 64 | 10.933 | 9.734 | 2.480 | 2.591 |
| dead25 | 1 | 8.621 | 7.558 | 1.669 | 1.589 |
| dead25 | 7 | 9.998 | 8.847 | 2.247 | 2.550 |
| dead25 | 28 | 10.942 | 9.668 | 2.545 | 2.737 |
| dead25 | 64 | 11.218 | 9.984 | 2.662 | 2.743 |

Generic stats removal saves approximately 11–13% of kernel time here. The six-slot variant saves approximately 72–76% versus generic no-stats at 7–64 rows; tile is faster at one row but slower than six-slot at the other tested shapes. Service benefit is unmeasured for either specialization.

Artifacts include exact launched commands in `campaign.sh`, filtered rank switch evidence, timing JSON/logs, microbenchmark source/CSV/compiler log, and restoration evidence. The initial runner recorded complete environments briefly; retained environment files were filtered on the remote before copying. The saved runner now filters directly.
