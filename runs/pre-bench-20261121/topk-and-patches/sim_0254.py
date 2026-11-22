#!/usr/bin/env python3
"""Exact simulation of patch 0254 (tf_ds_topk_indices_cut_kernel vs tf_ds_topk_indices_kernel).

Verifies:
1. Exact radix-pass counts, row global-read counts, total shared-histogram atomicAdd counts,
   single-bin peak atomicAdd counts, and warp-32 serialized atomic wavefront counts per row
   at 128 x 32,768 (k=512) and across the full served shape matrix:
   (a) random scores (exact bench.py seed=254 + general N(0,1)),
   (b) applyPool-masked rows in layers 24/28/32/36 (16,384 finite + 16,384 -inf when vis >= 512,
       plus 100K and 300K prefill widths),
   (c) causal early prompt rows (vis < 512 where n - vis columns are -inf),
   (d) deep fallback decode (nbk > 16,384, k=2048: homogeneous deep rows vs mixed-depth batches),
   (e) all -inf ("flat", exact bench.py case).
2. Bit-exact output parity of simulated EarlyCut=True and EarlyCut=False against NumPy sort.
3. Quantitative timing decomposition of index-timing.json's 100.717 us (random) vs 138.788 us (all -inf).
"""

import json
import math
import numpy as np


def score_key_u64(scores_f32: np.ndarray, cols_u32: np.ndarray) -> np.ndarray:
    """Exact bit-level replica of dsv41_ops.cu:tf_ds_score_key."""
    bits = scores_f32.view(np.uint32).copy()
    bits[bits == np.uint32(0x80000000)] = np.uint32(0)  # -0.0 -> +0.0
    neg = (bits & np.uint32(0x80000000)) != 0
    ordered = np.where(neg, bits ^ np.uint32(0x7FFFFFFF), bits).astype(np.int32)
    # Sign-extend ordered (int32 -> int64 -> uint64) << 32, OR inverted column
    hi = (ordered.astype(np.int64).view(np.uint64)) << np.uint64(32)
    lo = (np.uint32(0xFFFFFFFF) - cols_u32).astype(np.uint64)
    return (hi | lo) ^ np.uint64(0x8000000000000000)


def clz32(x: int) -> int:
    if x == 0:
        return 32
    return 32 - int(x).bit_length()


def simulate_row(scores_row: np.ndarray, k: int, vis: int, early_cut: bool = True, verify_parity: bool = True):
    """Simulate one CTA (1024 threads) of tf_ds_topk_indices<early_cut> on one row."""
    n = scores_row.shape[0]
    cols = np.arange(n, dtype=np.uint32)

    if early_cut and k == 0:
        return {
            "radix_passes": 0,
            "global_row_reads": 0,
            "bytes_read": 0,
            "hist_atomics_total": 0,
            "hist_atomics_per_pass": [],
            "max_bin_atomics_per_pass": [],
            "warp_serial_steps": 0,
            "collect_atomics": 0,
            "digits": [],
        }

    if early_cut and k == n:
        out = np.where(cols.astype(np.int64) < vis, cols.astype(np.int64), -1)
        if verify_parity:
            u = score_key_u64(scores_row, cols)
            ref_cols = np.sort(np.argsort(u, kind="stable")[-k:])
            ref_out = np.where(ref_cols < vis, ref_cols, -1)
            assert np.array_equal(out, ref_out)
        return {
            "radix_passes": 0,
            "global_row_reads": 0,
            "bytes_read": 0,
            "hist_atomics_total": 0,
            "hist_atomics_per_pass": [],
            "max_bin_atomics_per_pass": [],
            "warp_serial_steps": 0,
            "collect_atomics": 0,
            "digits": [],
        }

    u = score_key_u64(scores_row, cols)
    column_bytes = (32 - clz32(n - 1) + 7) // 8 if early_cut else 4
    fixed = ((0xFFFFFFFF << (8 * column_bytes)) & 0xFFFFFFFF)
    prefix = np.uint64(fixed) if early_cut else np.uint64(0)
    mask = np.uint64(fixed) if early_cut else np.uint64(0)
    want = k

    digits = []
    hist_atomics_per_pass = []
    max_bin_per_pass = []
    winning_bins = []
    warp_serial_steps = 0  # sum over warps & loop tiles of max duplicate bin count inside a 32-lane warp

    d = 7
    while d >= 0:
        digits.append(d)
        active = (u & mask) == prefix
        bins = ((u >> np.uint64(8 * d)) & np.uint64(0xFF)).astype(np.int32)
        active_bins = bins[active]
        hist = np.bincount(active_bins, minlength=256)

        total_act = int(active_bins.size)
        hist_atomics_per_pass.append(total_act)
        max_bin_per_pass.append(int(hist.max()) if total_act > 0 else 0)

        # Exact warp-level shared-atomic serialization count:
        # Threads iterate i = tid, tid + 1024, ...; warp w in tile t processes i = 1024*t + 32*w + lane (0..31).
        # On NVIDIA sm_90/sm_120 shared memory, K active lanes in the same warp targeting the SAME address
        # serialize into K atomic cycles (or hardware warp-aggregated/serialized steps depending on SASS).
        # Let's record both: (1) total active lanes colliding on identical address within a warp, and
        # (2) sum of max per-address multiplicity per warp-step (the serialization depth across warps).
        n_pad = ((n + 31) // 32) * 32
        act_pad = np.zeros(n_pad, dtype=bool)
        bin_pad = np.full(n_pad, -1, dtype=np.int32)
        act_pad[:n] = active
        bin_pad[:n] = np.where(active, bins, -1 - np.arange(n, dtype=np.int32))
        # Sort each 32-lane warp; inactive lanes have unique negative sentinels so multiplicity=1 only if active
        act_warps = act_pad.reshape(-1, 32)
        has_act = act_warps.any(axis=1)
        if has_act.any():
            sb = np.sort(bin_pad.reshape(-1, 32)[has_act], axis=1)
            # Count max equal-run length in each 32-element sorted row (only positive bins >= 0 can repeat!)
            # Using equality matrix (M, 32, 32) restricted to >=0: (M <= 4096 -> tiny 4096*32*32 = 4M bools!)
            valid = sb >= 0
            eq = (sb[:, :, None] == sb[:, None, :]) & valid[:, :, None]
            warp_serial_steps += int(eq.sum(axis=2).max(axis=1).sum())

        if early_cut:
            # Descending cumulative scan
            rem = want
            chosen_b = 0
            for b in range(255, -1, -1):
                if hist[b] >= rem:
                    chosen_b = b
                    break
                rem -= int(hist[b])
            winning_bins.append(chosen_b)
            prefix = prefix | (np.uint64(chosen_b) << np.uint64(8 * d))
            mask = mask | (np.uint64(0xFF) << np.uint64(8 * d))
            want = 0 if rem == int(hist[chosen_b]) else rem
            if want == 0:
                break
        else:
            rem = want
            chosen_b = 0
            for b in range(255, 0, -1):
                if hist[b] >= rem:
                    chosen_b = b
                    break
                rem -= int(hist[b])
            winning_bins.append(chosen_b)
            want = rem
            prefix = prefix | (np.uint64(chosen_b) << np.uint64(8 * d))
            mask = mask | (np.uint64(0xFF) << np.uint64(8 * d))

        d = (column_bytes if d == 4 else d) - 1

    kth = prefix
    picked_mask = u >= kth
    collect_atomics = int(np.count_nonzero(picked_mask))
    assert collect_atomics == k, f"Expected exactly k={k} keys >= kth, got {collect_atomics} (want={want})"

    if verify_parity:
        picked_cols = np.sort(cols[picked_mask].astype(np.int64))
        out = np.where(picked_cols < vis, picked_cols, -1)
        ref_cols = np.sort(np.argsort(u, kind="stable")[-k:].astype(np.int64))
        ref_out = np.where(ref_cols < vis, ref_cols, -1)
        assert np.array_equal(out, ref_out), "Parity mismatch against reference sort!"

    radix_passes = len(digits)
    global_reads = radix_passes + 1  # radix_passes + 1 collection scan
    return {
        "radix_passes": radix_passes,
        "global_row_reads": global_reads,
        "bytes_read": global_reads * n * 4,
        "hist_atomics_total": int(sum(hist_atomics_per_pass)),
        "hist_atomics_per_pass": hist_atomics_per_pass,
        "max_bin_atomics_per_pass": max_bin_per_pass,
        "winning_bins": winning_bins,
        "warp_serial_steps": warp_serial_steps,
        "collect_atomics": collect_atomics,
        "digits": digits,
    }


def summarize_matrix(scores: np.ndarray, k: int, vis_arr: np.ndarray, early_cut: bool = True, verify: bool = True):
    rows, n = scores.shape
    stats = [simulate_row(scores[r], k, int(vis_arr[r]), early_cut=early_cut, verify_parity=verify) for r in range(rows)]
    passes = [s["radix_passes"] for s in stats]
    reads = [s["global_row_reads"] for s in stats]
    bytes_r = [s["bytes_read"] for s in stats]
    atoms = [s["hist_atomics_total"] for s in stats]
    serial = [s["warp_serial_steps"] for s in stats]
    p0_max_bin = [s["max_bin_atomics_per_pass"][0] if s["max_bin_atomics_per_pass"] else 0 for s in stats]

    # Wave model on GB10: 48 SMs, 1 CTA/SM (1024 threads, 66.7% occupancy).
    # Each wave runs up to 48 CTAs concurrently; wave duration is bounded by max passes in that wave (approx).
    sm_count = 48
    wave_max_passes = [max(passes[i:i + sm_count]) for i in range(0, rows, sm_count)]

    return {
        "rows": rows,
        "n": n,
        "k": k,
        "early_cut": early_cut,
        "passes_min": int(min(passes)),
        "passes_mean": round(float(np.mean(passes)), 4),
        "passes_median": float(np.median(passes)),
        "passes_max": int(max(passes)),
        "passes_hist": {int(v): int(passes.count(v)) for v in sorted(set(passes))},
        "wave_max_passes_sum": int(sum(wave_max_passes)),
        "global_reads_mean": round(float(np.mean(reads)), 4),
        "bytes_per_row_mean": round(float(np.mean(bytes_r)), 1),
        "total_bytes_mib": round(float(sum(bytes_r)) / (1024 * 1024), 3),
        "hist_atomics_per_row_mean": round(float(np.mean(atoms)), 1),
        "hist_atomics_per_row_max": int(max(atoms)),
        "pass0_max_bin_mean": round(float(np.mean(p0_max_bin)), 1),
        "warp_serial_steps_per_row_mean": round(float(np.mean(serial)), 1),
        "sample_row0": stats[0],
    }


def make_bench_case(rows: int, width: int, k: int, dist: str):
    """Exact replica of tools/fused_bench/bench.py:case_topk_indices."""
    rng = np.random.default_rng(254)
    stride = width + 3
    scores = rng.standard_normal((rows, stride), dtype=np.float32)
    if dist == "ties":
        scores[:] = rng.integers(-3, 4, scores.shape)
    elif dist == "flat":
        scores.fill(-np.inf)
    elif dist == "special":
        patterns = np.array([
            0, 0x80000000, 0x7f800000, 0xff800000, 1, 0x80000001,
            0x7fc00001, 0x7fc00002, 0xffc00001
        ], dtype=np.uint32)
        scores.view(np.uint32)[:] = rng.choice(patterns, scores.shape)
    visible = np.resize(np.array([width + 1, width // 2, 0, -1, width - 1], dtype=np.int64), rows)
    return scores[:, :width].copy(), visible


def main():
    results = {}

    # 1. Exact bench.py cases at 128 x 32,768 (k=512): random vs flat (old and new)
    sc_rand, vis_bench = make_bench_case(128, 32768, 512, "random")
    sc_flat, _ = make_bench_case(128, 32768, 512, "flat")

    results["a_random_128x32768_new"] = summarize_matrix(sc_rand, 512, vis_bench, early_cut=True)
    results["a_random_128x32768_old"] = summarize_matrix(sc_rand, 512, vis_bench, early_cut=False)
    results["e_all_ninf_128x32768_new"] = summarize_matrix(sc_flat, 512, vis_bench, early_cut=True)
    results["e_all_ninf_128x32768_old"] = summarize_matrix(sc_flat, 512, vis_bench, early_cut=False)

    # 2. (b) applyPool-masked rows in layers 24/28/32/36 at 128 x 32,768 (k=512):
    # At n_comp_end = 32,768 (nbk = 4,096 > 2,048), replay = 32,640 -> vis = 32,641..32,768.
    # Layer 20 picks 2,048 blocks of 8 (= 16,384 columns); applyPool sets the other 2,048 blocks (16,384 cols) to -inf.
    rng = np.random.default_rng(20261121)
    sc_pool = rng.standard_normal((128, 32768), dtype=np.float32)
    vis_32k = np.arange(32641, 32769, dtype=np.int64)
    for r in range(128):
        # Causal mask first (columns >= vis[r] are -inf from _index_score_rows)
        sc_pool[r, vis_32k[r]:] = -np.inf
        # Simulate layer-20 candidate pool: pick top 2,048 blocks of 8 (with newest visible block pinned)
        blocks = sc_pool[r].reshape(4096, 8)
        bmax = blocks.max(axis=1)
        last_blk = (vis_32k[r] - 1) // 8
        bmax[last_blk] = np.inf
        bkeys = score_key_u64(bmax, np.arange(4096, dtype=np.uint32))
        top_blks = np.argsort(bkeys, kind="stable")[-2048:]
        mask = np.zeros(4096, dtype=bool)
        mask[top_blks] = bmax[top_blks] > -np.inf
        # New layers 24/28/32/36 compute fresh scores (causal masked), then applyPool sets !mask blocks to -inf
        fresh = rng.standard_normal(32768, dtype=np.float32)
        fresh[vis_32k[r]:] = -np.inf
        fresh.reshape(4096, 8)[~mask] = -np.inf
        sc_pool[r] = fresh

    results["b_applyPool_128x32768_new"] = summarize_matrix(sc_pool, 512, vis_32k, early_cut=True)

    # Also test applyPool at 100K (n=100,000, 16,384 finite + 83,616 -inf) for 16 rows (one indexRowBlock/sample)
    sc_pool_100k = rng.standard_normal((16, 100000), dtype=np.float32)
    vis_100k = np.arange(100000 - 15, 100001, dtype=np.int64)
    for r in range(16):
        sc_pool_100k[r, vis_100k[r]:] = -np.inf
        keep_blks = rng.choice(12500, size=2048, replace=False)
        mask = np.zeros(12500, dtype=bool)
        mask[keep_blks] = True
        sc_pool_100k[r].reshape(12500, 8)[~mask] = -np.inf
    results["b_applyPool_16x100000_new"] = summarize_matrix(sc_pool_100k, 512, vis_100k, early_cut=True)

    # 3. (c) Causal early prompt rows (vis < 512 where n - vis columns are -inf, or post-vis):
    # Subcase c1: 128 x 32,768 (k=512) hypothetical with vis < 512 (e.g., vis = 0..511 sampled across 128 rows)
    vis_early_hyp = np.linspace(0, 511, 128, dtype=np.int64)
    sc_early_32k = rng.standard_normal((128, 32768), dtype=np.float32)
    for r in range(128):
        sc_early_32k[r, max(0, vis_early_hyp[r]):] = -np.inf
    results["c1_causal_vis_lt_512_at_128x32768_causal_ninf"] = summarize_matrix(
        sc_early_32k, 512, vis_early_hyp, early_cut=True
    )

    # Subcase c2: Exact ONLY served prompt shape where vis < 512 actually reaches topk_indices_cut under TF_DS_IDX_FUSED=1!
    # Prompt length L in [513, 639]: replay = L - 128 in [385, 511], ch.n = 128, n_comp_end = L (e.g. L = 513 -> vis = 386..513)
    L = 513
    vis_513 = np.arange(L - 128 + 1, L + 1, dtype=np.int64)  # 386..513 (126 rows have vis < 512!)
    sc_513 = rng.standard_normal((128, L), dtype=np.float32)
    for r in range(128):
        sc_513[r, vis_513[r]:] = -np.inf
    results["c2_served_L513_128x513_vis386_513"] = summarize_matrix(sc_513, 512, vis_513, early_cut=True)

    # Subcase c3: Verify exhaustively ALL vis in 0..511 at n = 2048, k = 512 (e.g. replay=0 chunk 0 if replay disabled)
    sc_all_vis = rng.standard_normal((512, 2048), dtype=np.float32)
    vis_0_511 = np.arange(512, dtype=np.int64)
    for r in range(512):
        sc_all_vis[r, r:] = -np.inf
    results["c3_exhaustive_vis0_511_at_n2048_k512"] = summarize_matrix(sc_all_vis, 512, vis_0_511, early_cut=True)

    # Subcase c4: What if post-vis columns were finite (unmasked) while vis < 512? (for completeness of prompt wording)
    sc_unmasked = rng.standard_normal((128, 32768), dtype=np.float32)
    results["c4_unmasked_post_vis_128x32768"] = summarize_matrix(sc_unmasked, 512, vis_early_hyp, early_cut=True)

    # 4. (d) Deep fallback decode (nbk > 16,384, k = 2,048):
    # Subcase d1: Served 262,144-token bucket (nbk = 32,768, k = 2,048), R = 7 homogeneous deep rows (pos ~ 150,000, vis_nbk = 18,751)
    nbk_262k = 32768
    every_262k = np.full(7, 1 << 40, dtype=np.int64)
    bmax_deep = rng.standard_normal((7, nbk_262k), dtype=np.float32) + 1.2  # max of 8 normals
    for r in range(7):
        vis_nbk = 18751 + r
        bmax_deep[r, vis_nbk - 1] = np.inf
        bmax_deep[r, vis_nbk:] = -np.inf
    results["d1_deep_decode_homogeneous_7x32768_k2048"] = summarize_matrix(bmax_deep, 2048, every_262k, early_cut=True)

    # Subcase d2: Mixed-depth batch in 262,144-token bucket (R = 14: 7 rows at 150K + 7 rows of a short stream at 3K -> vis_nbk = 375 < 2048!)
    every_mix = np.full(14, 1 << 40, dtype=np.int64)
    bmax_mix = rng.standard_normal((14, nbk_262k), dtype=np.float32) + 1.2
    for r in range(7):
        bmax_mix[r, 18750] = np.inf
        bmax_mix[r, 18751:] = -np.inf
    for r in range(7, 14):
        bmax_mix[r, 374] = np.inf
        bmax_mix[r, 375:] = -np.inf
    results["d2_deep_decode_mixed_short_stream_14x32768_k2048"] = summarize_matrix(bmax_mix, 2048, every_mix, early_cut=True)

    # Subcase d3: 1M-token bucket (nbk = 131,072 > 65,536 -> column_bytes = 3!), homogeneous deep vs mixed short row
    nbk_1m = 131072
    every_1m = np.full(8, 1 << 40, dtype=np.int64)
    bmax_1m = rng.standard_normal((8, nbk_1m), dtype=np.float32) + 1.2
    for r in range(7):
        bmax_1m[r, 75000] = np.inf
        bmax_1m[r, 75001:] = -np.inf
    bmax_1m[7, 374] = np.inf
    bmax_1m[7, 375:] = -np.inf  # 1 short stream row in a 1M bucket
    results["d3_deep_decode_1M_bucket_mixed_8x131072_k2048"] = summarize_matrix(bmax_1m, 2048, every_1m, early_cut=True)

    # 5. Also simulate all other timed topk_indices shapes from index-timing.json to fit exact GB10 timing model!
    bench_timed = [
        (128, 4096, 512, "random", 77.955, 30.861, 0.808),
        (128, 32768, 512, "random", 237.022, 100.717, 0.806),
        (128, 32768, 512, "flat", 227.199, 138.788, 0.807),
        (32, 2049, 2048, "flat", 37.661, 23.445, 0.509),
        (7, 8192, 2048, "random", 48.893, 24.888, 0.504),
        (32, 16385, 2048, "ties", 60.855, 36.581, 0.500),
        (7, 65535, 512, "special", 138.326, 101.349, 0.502),
        (7, 65536, 512, "ties", 144.174, 101.617, 0.501),
        (7, 65537, 513, "special", 142.804, 119.970, 0.502),
        (32, 131072, 512, "random", 241.737, 120.256, 0.499),
    ]
    calibration = []
    for rows, w, k, dist, old_us, new_us, floor_us in bench_timed:
        sc, vis = make_bench_case(rows, w, k, dist)
        s_new = summarize_matrix(sc, k, vis, early_cut=True, verify=True)
        s_old = summarize_matrix(sc, k, vis, early_cut=False, verify=False)
        calibration.append({
            "label": f"R={rows} n={w} k={k} {dist}",
            "old_us": old_us,
            "new_us": new_us,
            "floor_us": floor_us,
            "new_passes_mean": s_new["passes_mean"],
            "new_passes_hist": s_new["passes_hist"],
            "new_reads_mean": s_new["global_reads_mean"],
            "new_mib": s_new["total_bytes_mib"],
            "new_atomics_mean": s_new["hist_atomics_per_row_mean"],
            "new_serial_mean": s_new["warp_serial_steps_per_row_mean"],
            "old_reads_mean": s_old["global_reads_mean"],
            "old_mib": s_old["total_bytes_mib"],
            "old_atomics_mean": s_old["hist_atomics_per_row_mean"],
            "old_serial_mean": s_old["warp_serial_steps_per_row_mean"],
        })
    results["gb10_calibration"] = calibration

    with open("/tmp/pre-bench-20261121/topk-patches/sim_0254_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== SUMMARY OF 128 x 32,768 (k=512) & SERVED CASES ===")
    for key in [
        "a_random_128x32768_new",
        "a_random_128x32768_old",
        "e_all_ninf_128x32768_new",
        "e_all_ninf_128x32768_old",
        "b_applyPool_128x32768_new",
        "b_applyPool_16x100000_new",
        "c1_causal_vis_lt_512_at_128x32768_causal_ninf",
        "c2_served_L513_128x513_vis386_513",
        "c3_exhaustive_vis0_511_at_n2048_k512",
        "c4_unmasked_post_vis_128x32768",
        "d1_deep_decode_homogeneous_7x32768_k2048",
        "d2_deep_decode_mixed_short_stream_14x32768_k2048",
        "d3_deep_decode_1M_bucket_mixed_8x131072_k2048",
    ]:
        r = results[key]
        print(
            f"{key}: passes={r['passes_hist']} (mean={r['passes_mean']}), "
            f"reads={r['global_reads_mean']}, MiB={r['total_bytes_mib']}, "
            f"atomics/row={r['hist_atomics_per_row_mean']}, "
            f"pass0_max_bin={r['pass0_max_bin_mean']}, "
            f"warp_serial/row={r['warp_serial_steps_per_row_mean']}"
        )
    print("\n=== GB10 CALIBRATION TABLE ===")
    for c in calibration:
        print(
            f"{c['label']:30s} | old={c['old_us']:7.3f}us (reads={c['old_reads_mean']:.1f}, atom={c['old_atomics_mean']:8.1f}, ser={c['old_serial_mean']:7.1f}) "
            f"| new={c['new_us']:7.3f}us (passes={str(c['new_passes_hist']):14s}, reads={c['new_reads_mean']:.2f}, MiB={c['new_mib']:6.2f}, atom={c['new_atomics_mean']:8.1f}, ser={c['new_serial_mean']:7.1f})"
        )


if __name__ == "__main__":
    main()
