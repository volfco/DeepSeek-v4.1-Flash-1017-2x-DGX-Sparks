#!/usr/bin/env python3
"""Unit tests for measure.py: health guards, host NumPy references, distributions, and stream-ordering invariants."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import measure


class TestHealthGuard(unittest.TestCase):
    def test_valid_idle_window(self) -> None:
        h0 = {"live": {"connections": 0}, "prompt_tokens_total": 100, "generation_tokens_total": 50}
        h1 = {"live": {"connections": 0}, "prompt_tokens_total": 100, "generation_tokens_total": 50}
        self.assertTrue(measure.check_idle_health(h0))
        self.assertTrue(measure.is_window_clean(h0, h1))

    def test_rejects_missing_or_bool_or_negative_counters(self) -> None:
        for bad in (
            {"live": {"connections": 0}},
            {"live": {"connections": 0}, "prompt_tokens_total": None, "generation_tokens_total": None},
            {"live": {"connections": 0}, "prompt_tokens_total": True, "generation_tokens_total": 10},
            {"live": {"connections": 0}, "prompt_tokens_total": -1, "generation_tokens_total": 10},
            {"live": {"connections": False}, "prompt_tokens_total": 10, "generation_tokens_total": 10},
            {"live": {"connections": 1}, "prompt_tokens_total": 10, "generation_tokens_total": 10},
        ):
            self.assertFalse(measure.check_idle_health(bad))
            self.assertFalse(measure.is_window_clean(bad, bad))

    def test_rejects_moved_counters_or_post_connections(self) -> None:
        h0 = {"live": {"connections": 0}, "prompt_tokens_total": 100, "generation_tokens_total": 50}
        h_prompt_moved = {"live": {"connections": 0}, "prompt_tokens_total": 101, "generation_tokens_total": 50}
        h_gen_moved = {"live": {"connections": 0}, "prompt_tokens_total": 100, "generation_tokens_total": 51}
        h_busy = {"live": {"connections": 1}, "prompt_tokens_total": 100, "generation_tokens_total": 50}
        self.assertFalse(measure.is_window_clean(h0, h_prompt_moved))
        self.assertFalse(measure.is_window_clean(h0, h_gen_moved))
        self.assertFalse(measure.is_window_clean(h0, h_busy))


class TestHostReferences(unittest.TestCase):
    def test_score_key_matches_idx_fused_zig(self) -> None:
        # +5.0 > -5.0
        k_pos = measure.score_key_host(np.array([5.0], dtype=np.float32), np.array([0], dtype=np.uint64))[0]
        k_neg = measure.score_key_host(np.array([-5.0], dtype=np.float32), np.array([0], dtype=np.uint64))[0]
        self.assertGreater(int(k_pos), int(k_neg))

        # -0.0 and +0.0 produce identical keys at the same column
        k_nz = measure.score_key_host(np.array([-0.0], dtype=np.float32), np.array([3], dtype=np.uint64))[0]
        k_pz = measure.score_key_host(np.array([0.0], dtype=np.float32), np.array([3], dtype=np.uint64))[0]
        self.assertEqual(int(k_nz), int(k_pz))

        # Tie breaks to lower column
        t0 = measure.score_key_host(np.array([1.0], dtype=np.float32), np.array([0], dtype=np.uint64))[0]
        t1 = measure.score_key_host(np.array([1.0], dtype=np.float32), np.array([1], dtype=np.uint64))[0]
        self.assertGreater(int(t0), int(t1))

        # NaN > +inf > 0.0 > -inf
        qnan = np.uint32(0x7FC00000).view(np.float32)
        vals = np.array([qnan, np.inf, 0.0, -np.inf], dtype=np.float32)
        keys = measure.score_key_host(vals, np.array([1, 1, 1, 1], dtype=np.uint64))
        self.assertGreater(int(keys[0]), int(keys[1]))
        self.assertGreater(int(keys[1]), int(keys[2]))
        self.assertGreater(int(keys[2]), int(keys[3]))

    def test_block_maxima_and_pool_pick_match_idx_fused_zig(self) -> None:
        qnan = np.uint32(0x7FC00000).view(np.float32)
        row = np.array([[1, 2, 3, qnan, -np.inf, 4, 5, 6, 7]], dtype=np.float32)
        bmax = measure.block_maxima_ref(row, width=9, bsize=4, vis_1d=np.array([100], dtype=np.int64))
        self.assertEqual(int(bmax[0, 0].view(np.uint32)), 0x7FC00000)
        self.assertEqual(float(bmax[0, 1]), 6.0)
        self.assertEqual(float(bmax[0, 2]), 7.0)

        # vis = 7 pins block 1 ((7 - 1) // 4 == 1) to +inf
        bmax_pin = measure.block_maxima_ref(row, width=9, bsize=4, vis_1d=np.array([7], dtype=np.int64))
        self.assertTrue(np.isposinf(bmax_pin[0, 1]))

        # vis = 0 pins nothing
        bmax_zero = measure.block_maxima_ref(row, width=9, bsize=4, vis_1d=np.array([0], dtype=np.int64))
        self.assertEqual(float(bmax_zero[0, 1]), 6.0)

        # poolPick excludes NaN and -inf maxima
        bmax_p = np.array([[5.0, -np.inf, qnan]], dtype=np.float32)
        idx_p = np.array([[0, 1, 2]], dtype=np.int64)
        cblk = measure.pool_pick_ref(bmax_p, idx_p)
        np.testing.assert_array_equal(cblk[0], np.array([0, -1, -1], dtype=np.int32))


class TestDistributionsAndDriverSafety(unittest.TestCase):
    def test_topk_distributions(self) -> None:
        # 1. random: 0 -inf
        s_rand, v_rand, _ = measure.make_topk_case_inputs("random", rows=4, width=32768)
        self.assertEqual(int(np.isneginf(s_rand).sum()), 0)
        np.testing.assert_array_equal(v_rand, np.full(4, 32768, dtype=np.int64))

        # 2. applyPool_50pct_ninf: exact 16384 finite and 16384 -inf per row
        s_pool, v_pool, mask = measure.make_topk_case_inputs("applyPool_50pct_ninf", rows=4, width=32768)
        self.assertIsNotNone(mask)
        for r in range(4):
            self.assertEqual(int(np.isneginf(s_pool[r]).sum()), 16384)
            self.assertEqual(int(np.isfinite(s_pool[r]).sum()), 16384)
        np.testing.assert_array_equal(v_pool, np.full(4, 32768, dtype=np.int64))

        # 3. vis128_lt_k: 128 finite in [0, 128) and 32640 -inf in [128, 32768); output is 0..127 then 384 -1s
        s_vis, v_vis, _ = measure.make_topk_case_inputs("vis128_lt_k", rows=2, width=32768)
        for r in range(2):
            self.assertTrue(np.all(np.isfinite(s_vis[r, :128])))
            self.assertTrue(np.all(np.isneginf(s_vis[r, 128:])))
        out_vis = measure.topk_indices_ref(s_vis, 32768, 512, v_vis)
        for r in range(2):
            np.testing.assert_array_equal(out_vis[r, :128], np.arange(128, dtype=np.int64))
            np.testing.assert_array_equal(out_vis[r, 128:], np.full(384, -1, dtype=np.int64))

        # 4. all_ninf: all 32768 -inf; output is 0..511 ascending
        s_flat, v_flat, _ = measure.make_topk_case_inputs("all_ninf", rows=2, width=32768)
        self.assertTrue(np.all(np.isneginf(s_flat)))
        out_flat = measure.topk_indices_ref(s_flat, 32768, 512, v_flat)
        for r in range(2):
            np.testing.assert_array_equal(out_flat[r], np.arange(512, dtype=np.int64))

    def test_no_stream0_memcpy_in_source(self) -> None:
        src = Path(measure.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        called_apis: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cu":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    called_apis.add(node.args[0].value)
        self.assertIn("cuMemcpyHtoDAsync_v2", called_apis)
        self.assertIn("cuMemcpyDtoHAsync_v2", called_apis)
        self.assertNotIn("cuMemcpyHtoD_v2", called_apis)
        self.assertNotIn("cuMemcpyDtoH_v2", called_apis)

    def test_mem_available_parser(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("MemTotal:       134217728 kB\nMemAvailable:    4194304 kB\n")
            path = f.name
        try:
            self.assertAlmostEqual(measure.mem_available_gib(path), 4.0)
        finally:
            Path(path).unlink(missing_ok=True)


class FakeBurstDriver:
    vram_free_before = 100 * 1024 * 1024
    min_vram_free_seen = 90 * 1024 * 1024
    peak_alloc_bytes = 10 * 1024 * 1024


class TestBurstStateTransitions(unittest.TestCase):
    def _initial_state(self) -> dict:
        return {
            "all_parity_passed": True,
            "total_timing_retries": 0,
            "cases": [
                {"case_id": "c1", "group": "0215_crossover", "timing": {}, "timing_valid": False, "timing_retries": 0},
                {"case_id": "c2", "group": "0215_crossover", "timing": {}, "timing_valid": False, "timing_retries": 0},
            ],
        }

    def test_continuous_traffic_never_publishes_contaminated_medians(self) -> None:
        state = self._initial_state()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            jpath = Path(f.name)
        try:
            h0 = {"live": {"connections": 0}, "prompt_tokens_total": 10, "generation_tokens_total": 20}
            h1 = {"live": {"connections": 0}, "prompt_tokens_total": 10, "generation_tokens_total": 25}
            with mock.patch.object(measure, "time_one_case", return_value=({}, False, h0, h1)):
                captured, rem = measure.run_timing_burst(FakeBurstDriver(), state, jpath, 100, 20, "http://x")
            self.assertEqual(captured, 0)
            self.assertEqual(rem, 2)
            measure.finalize_state(jpath)
            saved = json.loads(jpath.read_text(encoding="utf-8"))
            self.assertFalse(saved["timing_complete"])
            self.assertEqual(saved["cases"][0]["timing"], {})
            self.assertFalse(saved["cases"][0]["timing_valid"])
            self.assertEqual(saved["cases"][0]["timing_retries"], 1)
        finally:
            jpath.unlink(missing_ok=True)

    def test_clean_retry_keeps_retry_count_separate_from_validity(self) -> None:
        state = self._initial_state()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            jpath = Path(f.name)
        try:
            h_idle = {"live": {"connections": 0}, "prompt_tokens_total": 42, "generation_tokens_total": 99}
            h_moved = {"live": {"connections": 0}, "prompt_tokens_total": 42, "generation_tokens_total": 100}
            stats_ok = {"idx_fused": {"median": 8.5, "iqr": 0.01, "n": 20}}

            # Burst 1: c1 contaminated mid-window -> yields lock
            with mock.patch.object(measure, "time_one_case", return_value=({}, False, h_idle, h_moved)):
                c, rem = measure.run_timing_burst(FakeBurstDriver(), state, jpath, 100, 20, "http://x")
            self.assertEqual((c, rem), (0, 2))
            self.assertEqual(state["cases"][0]["timing_retries"], 1)
            self.assertFalse(state["cases"][0]["timing_valid"])

            # Burst 2: idle gap captures both c1 and c2 cleanly
            with mock.patch.object(measure, "time_one_case", return_value=(stats_ok, True, h_idle, h_idle)):
                c, rem = measure.run_timing_burst(FakeBurstDriver(), state, jpath, 100, 20, "http://x")
            self.assertEqual((c, rem), (2, 0))
            self.assertTrue(state["cases"][0]["timing_valid"])
            self.assertEqual(state["cases"][0]["timing_retries"], 1)
            self.assertEqual(state["cases"][0]["timing"], stats_ok)
            self.assertTrue(state["cases"][1]["timing_valid"])
            self.assertEqual(state["cases"][1]["timing_retries"], 0)
        finally:
            jpath.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
