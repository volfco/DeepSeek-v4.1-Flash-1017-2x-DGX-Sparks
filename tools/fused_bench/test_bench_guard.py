#!/usr/bin/env python3
"""CPU unit tests for tools/fused_bench/bench.py timing-label, idle-health, and stream-sync guards."""

import ctypes as ct
import gc
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import chain  # noqa: E402


def idle_health(prompt=100, gen=50, connections=0):
    return {
        "live": {"connections": connections},
        "prompt_tokens_total": prompt,
        "generation_tokens_total": gen,
    }


class FakeCuda:
    """Minimal ctypes-compatible CUDA driver stub for CPU-only harness unit tests."""

    def __init__(self, elapsed_ms_fn=None):
        self.next_addr = 0x10000
        self.next_handle = 1
        self.calls = []
        self.call_args = []
        self.device_mem = {}
        self.pending_stream_ops = {}
        self.pending_stream0_ops = []
        self.on_stream_sync = None
        self.elapsed_ms_fn = elapsed_ms_fn or (lambda: 0.12)

    def _new_handle(self):
        h = self.next_handle
        self.next_handle += 1
        return h

    @staticmethod
    def _val(x):
        v = getattr(x, "value", x)
        return 0 if v is None else int(v)

    def _write_mem(self, dst_addr, payload):
        for base, buf in self.device_mem.items():
            if base <= dst_addr and dst_addr + len(payload) <= base + len(buf):
                off = dst_addr - base
                buf[off:off + len(payload)] = payload
                return

    def _read_mem(self, src_addr, size):
        for base, buf in self.device_mem.items():
            if base <= src_addr and src_addr + size <= base + len(buf):
                off = src_addr - base
                return bytes(buf[off:off + size])
        return b"\x00" * size

    def cuMemAlloc_v2(self, addr_ref, nbytes):
        self.calls.append("cuMemAlloc_v2")
        size = self._val(nbytes)
        addr = self.next_addr
        addr_ref._obj.value = addr
        self.device_mem[addr] = bytearray(b"\xaa" * max(1, size))
        self.next_addr += max(256, (size + 255) & ~255)
        return 0

    def cuStreamCreate(self, stream_ref, flags):
        self.calls.append("cuStreamCreate")
        h = self._new_handle()
        stream_ref._obj.value = h
        self.call_args.append(("cuStreamCreate", h, self._val(flags)))
        return 0

    def cuMemcpyHtoDAsync_v2(self, dst, src, nbytes, st, *_extra):
        self.calls.append("cuMemcpyHtoDAsync_v2")
        dst_addr, src_addr, size, st_val = self._val(dst), self._val(src), self._val(nbytes), self._val(st)
        self.call_args.append(("cuMemcpyHtoDAsync_v2", dst_addr, src_addr, size, st_val))
        payload = bytes((ct.c_ubyte * size).from_address(src_addr)) if size > 0 and src_addr else b""
        # Deferred until cuStreamSynchronize(st_val): simulates <=64 KiB async DMA ordered only on `st`.
        self.pending_stream_ops.setdefault(st_val, []).append(lambda: self._write_mem(dst_addr, payload))
        return 0

    def cuMemcpyHtoD_v2(self, dst, src, nbytes):
        self.calls.append("cuMemcpyHtoD_v2")
        dst_addr, src_addr, size = self._val(dst), self._val(src), self._val(nbytes)
        self.call_args.append(("cuMemcpyHtoD_v2", dst_addr, src_addr, size))
        payload = bytes((ct.c_ubyte * size).from_address(src_addr)) if size > 0 and src_addr else b""
        # Deferred on legacy stream 0 until cuCtxSynchronize (does NOT synchronize with CU_STREAM_NON_BLOCKING).
        self.pending_stream0_ops.append(lambda: self._write_mem(dst_addr, payload))
        return 0

    def cuStreamSynchronize(self, st):
        self.calls.append("cuStreamSynchronize")
        st_val = self._val(st)
        self.call_args.append(("cuStreamSynchronize", st_val))
        if self.on_stream_sync is not None:
            self.on_stream_sync(st_val)
        for op in self.pending_stream_ops.pop(st_val, []):
            op()
        return 0

    def cuCtxSynchronize(self):
        self.calls.append("cuCtxSynchronize")
        self.call_args.append(("cuCtxSynchronize",))
        ops = self.pending_stream0_ops[:]
        self.pending_stream0_ops.clear()
        for op in ops:
            op()
        return 0

    def cuMemcpyDtoH_v2(self, dst, src, nbytes):
        self.calls.append("cuMemcpyDtoH_v2")
        dst_addr, src_addr, size = self._val(dst), self._val(src), self._val(nbytes)
        self.call_args.append(("cuMemcpyDtoH_v2", dst_addr, src_addr, size))
        # Legacy stream 0 DtoH flushes stream-0 work only, NOT CU_STREAM_NON_BLOCKING streams!
        ops = self.pending_stream0_ops[:]
        self.pending_stream0_ops.clear()
        for op in ops:
            op()
        if size > 0 and dst_addr:
            ct.memmove(dst_addr, self._read_mem(src_addr, size), size)
        return 0

    def cuModuleLoad(self, mod_ref, _path):
        self.calls.append("cuModuleLoad")
        mod_ref._obj.value = self._new_handle()
        return 0

    def cuModuleGetFunction(self, fn_ref, _mod, _name):
        self.calls.append("cuModuleGetFunction")
        fn_ref._obj.value = self._new_handle()
        return 0

    def cuModuleGetFunctionCount(self, count_ref, _mod):
        self.calls.append("cuModuleGetFunctionCount")
        count_ref._obj.value = 0
        return 0

    def cuFuncGetParamInfo(self, _fn, _idx, _off_ref, _size_ref):
        self.calls.append("cuFuncGetParamInfo")
        return 1  # non-zero terminates param_count loop cleanly

    def cuStreamEndCapture(self, _stream, graph_ref):
        self.calls.append("cuStreamEndCapture")
        graph_ref._obj.value = self._new_handle()
        return 0

    def cuGraphInstantiateWithFlags(self, exec_ref, _graph, _flags):
        self.calls.append("cuGraphInstantiateWithFlags")
        exec_ref._obj.value = self._new_handle()
        return 0

    def cuEventCreate(self, ev_ref, _flags):
        self.calls.append("cuEventCreate")
        ev_ref._obj.value = self._new_handle()
        return 0

    def cuEventElapsedTime(self, ms_ref, _start, _stop):
        self.calls.append("cuEventElapsedTime")
        ms_ref._obj.value = float(self.elapsed_ms_fn())
        return 0

    def __getattr__(self, name):
        if name.startswith("cu"):
            def _ok(*_args):
                self.calls.append(name)
                return 0
            return _ok
        raise AttributeError(name)


class BenchGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        kit_dir = root / "kit"
        kit_dir.mkdir()
        for prefix, _, _ in bench.kit.values():
            (kit_dir / f"{prefix}_stub.cubin").write_bytes(b"")
        for f in ("old-ops.cubin", "old-torch.cubin", "new-ops.cubin", "new-torch.cubin", "floor.cubin"):
            (root / f).write_bytes(b"")

        self.fake_cuda = FakeCuda()
        self.orig_cuda = bench.cuda
        self.orig_stream = bench.P(bench.stream.value)
        self.orig_opts = bench.opts
        self.orig_do_timing = bench.DO_TIMING
        self.orig_do_parity = bench.DO_PARITY

        bench.cuda = self.fake_cuda
        bench.stream = bench.P()
        bench.modules.clear()
        bench.kernels.clear()
        bench.module_symbols.cache_clear()
        bench.allocations.clear()
        bench.allocated = 0
        bench.results.clear()
        bench.failures.clear()
        bench.timing_retries = 0
        bench.DO_TIMING = True
        bench.DO_PARITY = False
        bench.opts = SimpleNamespace(
            old_ops=root / "old-ops.cubin",
            old_torch=root / "old-torch.cubin",
            new_ops=root / "new-ops.cubin",
            new_torch=root / "new-torch.cubin",
            floor=root / "floor.cubin",
            kit=kit_dir,
            kernel="all",
            shape=None,
            batch=10,
            reps=3,
            no_guard=False,
            wait=5,
            tries=3,
        )

    def tearDown(self):
        bench.cuda = self.orig_cuda
        bench.stream = self.orig_stream
        bench.opts = self.orig_opts
        bench.DO_TIMING = self.orig_do_timing
        bench.DO_PARITY = self.orig_do_parity
        bench.modules.clear()
        bench.kernels.clear()
        bench.module_symbols.cache_clear()
        bench.allocations.clear()
        bench.results.clear()
        bench.failures.clear()
        self.tmpdir.cleanup()

    def test_lazy_libcuda_not_loaded_at_import(self):
        self.assertIsNone(self.orig_cuda)
        bench.cuda = None
        with mock.patch.object(bench.ct, "CDLL", side_effect=OSError("no libcuda")) as cdll_mock:
            self.assertTrue(bench.counters_valid((1, 2)))
            cdll_mock.assert_not_called()
            with self.assertRaises(OSError):
                bench.cu("cuInit", 0)
            cdll_mock.assert_called_once_with("libcuda.so.1")

    def test_a_chain_name_mismatch_raises_before_capture(self):
        bench.cuda = None
        chain_calls = 0

        def arm():
            nonlocal chain_calls
            chain_calls += 1

        with mock.patch.object(bench.ct, "CDLL", side_effect=AssertionError("CUDA should not be touched")):
            # 3 chains vs 4 names (the historical 'newnf' bug)
            with self.assertRaises(ValueError):
                bench.time_graphs("bad-count", [arm, arm, arm], ["ref", "old", "new", "newnf"], arm, 10, 2)
            # 2 chains vs 1 name
            with self.assertRaises(ValueError):
                bench.time_graphs("too-few-names", [arm, arm], ["old"], None, 10, 2)
            # duplicate arm names
            with self.assertRaises(ValueError):
                bench.time_graphs("dup-names", [arm, arm], ["new", "new"], None, 10, 2)
            # reserved 'floor' in caller names
            with self.assertRaises(ValueError):
                bench.time_graphs("reserved-floor", [arm, arm], ["new", "floor"], None, 10, 2)

        self.assertEqual(chain_calls, 0)

    def test_b_missing_counters_and_failed_health_rejected(self):
        # Direct predicate checks
        self.assertTrue(bench.counters_valid((0, 0)))
        self.assertTrue(bench.counters_valid((12345, 678)))
        for bad in ((None, None), (10, None), (None, 5), (True, 0), (0, False), [1, 2], (1,), (1, 2, 3), (-1, 0), (1.5, 2)):
            self.assertFalse(bench.counters_valid(bad), f"expected invalid for {bad!r}")

        self.assertTrue(bench.check_idle_health(idle_health(10, 20, connections=0)))
        self.assertFalse(bench.check_idle_health({}))
        self.assertFalse(bench.check_idle_health({"live": {"connections": 0}}))
        self.assertFalse(bench.check_idle_health({"live": {"connections": 1}, "prompt_tokens_total": 1, "generation_tokens_total": 1}))
        self.assertFalse(bench.check_idle_health({"live": {"connections": False}, "prompt_tokens_total": 1, "generation_tokens_total": 1}))
        self.assertFalse(bench.check_idle_health({"live": None, "prompt_tokens_total": 1, "generation_tokens_total": 1}))

        # wait_health rejects connections==0 when counters are missing, and raises TimeoutError cleanly on expiry
        with (
            mock.patch.object(bench, "health_now", return_value={"live": {"connections": 0}}),
            mock.patch.object(bench.time, "time", side_effect=[100.0, 101.0, 106.0]),
            mock.patch.object(bench.time, "sleep"),
        ):
            with self.assertRaises(TimeoutError):
                bench.wait_health(timeout=5)

        # Guarded timing when /health omits counters on BOTH before and after: (None, None) == (None, None) must fail
        row_missing = {}
        with (
            mock.patch.object(bench, "wait_health", return_value={"live": {"connections": 0}}),
            mock.patch.object(bench, "health_now", return_value={"live": {"connections": 0}}),
        ):
            bench.attach_timing(row_missing, "missing-counters", [lambda: None], ["new"], None, 10, 2)
        self.assertEqual(row_missing["timing"], {})
        self.assertFalse(row_missing["timing_valid"])
        self.assertEqual(row_missing["timing_retries"], bench.opts.tries)
        self.assertTrue(row_missing["timing_guarded"])

        # Guarded timing when /health is unreachable (TimeoutError from wait_health): returns {} and timing_valid=False
        row_down = {}
        with (
            mock.patch.object(bench, "health_now", return_value={}),
            mock.patch.object(bench.time, "time", side_effect=[200.0, 206.0]),
            mock.patch.object(bench.time, "sleep"),
        ):
            bench.attach_timing(row_down, "health-down", [lambda: None], ["new"], None, 10, 2)
        self.assertEqual(row_down["timing"], {})
        self.assertFalse(row_down["timing_valid"])
        self.assertTrue(row_down["timing_guarded"])

        # Post-block active connection with unchanged finished-token counters is also rejected
        row_inflight = {}
        with (
            mock.patch.object(bench, "wait_health", return_value=idle_health(50, 25, connections=0)),
            mock.patch.object(bench, "health_now", return_value=idle_health(50, 25, connections=1)),
        ):
            bench.attach_timing(row_inflight, "inflight-conn", [lambda: None], ["new"], None, 10, 2)
        self.assertEqual(row_inflight["timing"], {})
        self.assertFalse(row_inflight["timing_valid"])
        self.assertEqual(row_inflight["timing_retries"], bench.opts.tries)

    def test_c_continuous_traffic_across_all_tries_returns_empty_timing(self):
        bench.opts.tries = 4
        step = 0

        def next_pre(_timeout):
            nonlocal step
            step += 1
            return idle_health(100 + step * 10, 50 + step)

        def next_post():
            return idle_health(100 + step * 10 + 5, 50 + step + 1)

        row = {"kernel": "0195-idx", "shape": "idx-busy"}
        with (
            mock.patch.object(bench, "wait_health", side_effect=next_pre),
            mock.patch.object(bench, "health_now", side_effect=next_post),
        ):
            out = bench.attach_timing(row, "idx-busy", [lambda: None, lambda: None], ["old", "new"], lambda: None, 10, 2)

        self.assertEqual(out, {})
        self.assertEqual(row["timing"], {})
        self.assertFalse(row["timing_valid"])
        self.assertEqual(row["timing_retries"], 4)
        self.assertTrue(row["timing_guarded"])

        summary = bench.summarize_health([row], True, idle_health(100, 50), idle_health(145, 55), bench.timing_retries)
        self.assertFalse(summary["timing_clean"])
        self.assertEqual(summary["timing_retries"], 4)

    def test_d_traffic_on_try0_then_clean_try1_publishes_clean_medians(self):
        bench.opts.tries = 3
        bench.opts.batch = 10
        bench.opts.reps = 3
        attempt_idx = -1

        def pre_health(_timeout):
            nonlocal attempt_idx
            attempt_idx += 1
            return idle_health(100, 50) if attempt_idx == 0 else idle_health(120, 60)

        def post_health():
            # try 0 sees served tokens move (100 -> 120); try 1 stays idle at (120, 60)
            return idle_health(120, 60)

        # Contaminate try 0 with 9.9 ms (990 us/launch) vs clean try 1 with 0.02 ms (2.0 us/launch)
        self.fake_cuda.elapsed_ms_fn = lambda: 9.9 if attempt_idx == 0 else 0.02

        row = {"kernel": "0198-key", "shape": "rope key R=7 D=128"}
        with (
            mock.patch.object(bench, "wait_health", side_effect=pre_health),
            mock.patch.object(bench, "health_now", side_effect=post_health),
        ):
            bench.attach_timing(row, row["shape"], [lambda: None, lambda: None, lambda: None],
                                ["ref", "old", "new"], lambda: None, bench.opts.batch, bench.opts.reps)

        self.assertTrue(row["timing_valid"])
        self.assertEqual(row["timing_retries"], 1)
        self.assertTrue(row["timing_guarded"])
        self.assertEqual(list(row["timing"].keys()), ["ref", "old", "new", "floor"])
        for arm in ("ref", "old", "new", "floor"):
            self.assertAlmostEqual(row["timing"][arm]["median"], 2.0, places=3)
            self.assertEqual(row["timing"][arm]["n"], 3)

        # Top-level timing_clean is True when guarded, overall counters valid, and every timed row has timing_valid=True,
        # even though timing_retries == 1.
        summary = bench.summarize_health([row], True, idle_health(100, 50), idle_health(120, 60), bench.timing_retries)
        self.assertTrue(summary["timing_clean"])
        self.assertEqual(summary["timing_retries"], 1)

        # Unguarded or missing overall counters must set timing_clean=False
        self.assertFalse(bench.summarize_health([row], False, None, None, 0)["timing_clean"])
        self.assertFalse(bench.summarize_health([row], True, idle_health(100, 50), {}, 1)["timing_clean"])

    def test_e_all_case_callers_pass_matching_chains_and_names(self):
        bench.opts.batch = 2
        bench.opts.reps = 2
        bench.opts.tries = 2

        invocations = [
            ("case_latent_owner", lambda: bench.case_latent_owner(4, 2, timing=True), ["ref", "old", "new", "floor"]),
            ("case_topk_indices", lambda: bench.case_topk_indices(2, 16, 4, "random"), ["old", "new", "floor"]),
            ("case_candidate_mask", lambda: bench.case_candidate_mask(2, 8), ["old", "new", "floor"]),
            ("case_idx", lambda: bench.case_idx(2, 8, 4, False, "ties", False, True), ["ref", "old", "new", "floor"]),
            ("case_idx_small", lambda: bench.case_idx_small(2, 16, 8, "random"), ["ref", "old", "fused", "small", "floor"]),
            ("case_blockmax", lambda: bench.case_blockmax(2, 8), ["kit", "floor"]),
            ("case_idx_hist", lambda: bench.case_idx_hist(2, 8, 4, "random", 256), ["new", "floor"]),
            ("case_idx_fold", lambda: bench.case_idx_fold(1, 16388, "mixed", False, True), ["ref", "old", "new", "newnf", "floor"]),
            ("case_score_keys_only", lambda: bench.case_score_keys_only(1, 16388), ["kit"]),
            ("case_compress_decode", lambda: bench.case_compress_decode(2, 128, "mixed", True, False), ["ref", "old", "new", "newdup", "floor"]),
            ("case_compress_prompt", lambda: bench.case_compress_prompt(2, 128, True, False), ["ref", "old", "new", "floor"]),
            ("case_rope:key", lambda: bench.case_rope("key", 2, False, True), ["ref", "old", "new", "floor"]),
            ("case_rope:latent", lambda: bench.case_rope("latent", 2, False, True), ["ref", "old", "new", "floor"]),
            ("case_rope:query", lambda: bench.case_rope("query", 2, False, True), ["ref", "old", "new", "floor"]),
            ("case_rope:prompt-key", lambda: bench.case_rope("prompt key", 2, False, True), ["ref", "old", "new", "floor"]),
            ("case_rope:prompt-latent", lambda: bench.case_rope("prompt latent", 2, False, True), ["ref", "old", "new", "floor"]),
            ("case_latent_values", lambda: bench.case_latent_values(timing=False), None),
            ("case_prune", lambda: bench.case_prune(2, False, False, True), ["ref", "old", "new", "newnf", "floor"]),
            ("case_fallback_idx", lambda: bench.case_fallback_idx(2, 16, 8), ["old", "new", "floor"]),
            ("case_fallback_compress", lambda: bench.case_fallback_compress(2, 256), ["old", "new", "floor"]),
            ("case_prompt_source", lambda: bench.case_prompt_source(2, 128, False, True, False), ["ref", "old", "new", "floor"]),
            ("case_score_keys", lambda: bench.case_score_keys(1, 16384), ["kit"]),
        ]

        # Ensure every case_* function defined in bench.py is exercised here
        all_case_fns = {
            name for name, fn in inspect.getmembers(bench, inspect.isfunction)
            if name.startswith("case_") and fn.__module__ == bench.__name__
        }
        tested_case_fns = {tag.split(":")[0] for tag, _, _ in invocations}
        self.assertEqual(tested_case_fns, all_case_fns)

        with (
            mock.patch.object(bench, "wait_health", return_value=idle_health(200, 80)),
            mock.patch.object(bench, "health_now", return_value=idle_health(200, 80)),
        ):
            for tag, call, expected_keys in invocations:
                before_len = len(bench.results)
                call()
                self.assertEqual(len(bench.results), before_len + 1, f"{tag} did not append a result row")
                row = bench.results[-1]
                if expected_keys is None:
                    self.assertNotIn("timing", row, f"{tag} should be untimed")
                else:
                    self.assertEqual(list(row["timing"].keys()), expected_keys, f"{tag} arm keys mismatch")
                    self.assertTrue(row["timing_valid"], f"{tag} timing_valid should be True")
                    self.assertEqual(row["timing_retries"], 0, f"{tag} timing_retries should be 0")
                    self.assertTrue(row["timing_guarded"], f"{tag} timing_guarded should be True")

            # Also verify chain.py's mode_chain caller against bench.time_graphs
            with mock.patch.object(bench.Buffer, "read", lambda self_buf: bench.np.zeros(self_buf.shape, self_buf.dtype).view("u1").ravel()):
                chain_row = chain.mode_chain(1, 8, 8)
            self.assertEqual(
                list(chain_row["timing"].keys()),
                ["old", "oldpdl", "new", "newpdl", "empty", "emptypdl"],
            )

    def test_report_compatible_with_new_timing_schema(self):
        root = Path(self.tmpdir.name)
        bench_json = root / "bench.json"
        bw_json = root / "bw.json"
        sass_json = root / "sass.json"
        rows = [
            {
                "kernel": "0195-idx",
                "shape": "idx R=7 nb=8 width=61 k=8 list ties",
                "bytes": 2044,
                "parity": {"new_vs_ref": 0},
                "timing": {
                    "ref": {"median": 26.6, "iqr": 0.01, "n": 20},
                    "old": {"median": 2.02, "iqr": 0.01, "n": 20},
                    "new": {"median": 1.98, "iqr": 0.01, "n": 20},
                    "floor": {"median": 0.495, "iqr": 0.002, "n": 20},
                },
                "timing_valid": True,
                "timing_retries": 1,
                "timing_guarded": True,
            },
            {
                "kernel": "0198-key",
                "shape": "rope key R=7 D=128",
                "bytes": 2048,
                "parity": {"new_vs_ref": 0},
                "timing": {},
                "timing_valid": False,
                "timing_retries": 3,
                "timing_guarded": True,
            },
        ]
        health = bench.summarize_health(rows, True, idle_health(10, 5), idle_health(20, 8), 4)
        self.assertFalse(health["timing_clean"])
        bench_json.write_text(json.dumps({"results": rows, "failures": [], "health": health}))
        bw_json.write_text(json.dumps({"peak_GBps": 950.0, "cache_GBps": 950.0, "dram_GBps": 224.0}))
        sass_json.write_text(json.dumps({"kernels": {}, "limits": {}}))

        report_py = Path(__file__).resolve().parent / "report.py"
        proc = subprocess.run(
            [sys.executable, str(report_py), str(bench_json), str(bw_json), str(sass_json)],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        self.assertIn("0195-idx", proc.stdout)
        self.assertNotIn("0198-key", proc.stdout)

    def test_f_buffer_stream_ordered_upload_and_read_sync_when_stream_non_null(self):
        # Initialize bench.stream as CU_STREAM_NON_BLOCKING (flags == 1), matching bench.main().
        bench.cu("cuStreamCreate", bench.ref(bench.stream), 1)
        st_handle = bench.stream.value
        self.assertIsNotNone(st_handle)
        self.assertNotEqual(st_handle, 0)

        # Verify temporary non-contiguous input stays alive across cuMemcpyHtoDAsync_v2 AND cuStreamSynchronize.
        contig_refs = []
        real_ascontig = bench.np.ascontiguousarray

        def spy_ascontig(a, *args, **kwargs):
            out = real_ascontig(a, *args, **kwargs)
            contig_refs.append(weakref.ref(out))
            return out

        alive_during_sync = []

        def check_alive_on_sync(sync_stream):
            gc.collect()
            if contig_refs:
                alive_during_sync.append((sync_stream, contig_refs[-1]() is not None))

        self.fake_cuda.on_stream_sync = check_alive_on_sync
        self.fake_cuda.calls.clear()
        self.fake_cuda.call_args.clear()

        expected = bench.np.arange(32, dtype="i4")[::2]  # non-contiguous view -> fresh contiguous temporary
        self.assertFalse(expected.flags["C_CONTIGUOUS"])
        with mock.patch.object(bench.np, "ascontiguousarray", side_effect=spy_ascontig):
            buf = bench.Buffer(bench.np.arange(32, dtype="i4")[::2])

        self.assertEqual(
            self.fake_cuda.calls,
            ["cuMemAlloc_v2", "cuMemcpyHtoDAsync_v2", "cuStreamSynchronize"],
        )
        self.assertNotIn("cuMemcpyHtoD_v2", self.fake_cuda.calls)
        self.assertEqual(self.fake_cuda.call_args[0][0], "cuMemcpyHtoDAsync_v2")
        self.assertEqual(self.fake_cuda.call_args[0][1], buf.addr.value)
        self.assertEqual(self.fake_cuda.call_args[0][3], expected.nbytes)
        self.assertEqual(self.fake_cuda.call_args[0][4], st_handle)
        self.assertEqual(self.fake_cuda.call_args[1], ("cuStreamSynchronize", st_handle))
        self.assertEqual(alive_during_sync, [(st_handle, True)])

        # Verify uploaded bytes are present in device memory immediately after Buffer.__init__.
        self.assertEqual(
            self.fake_cuda._read_mem(buf.addr.value, expected.nbytes),
            bench.np.ascontiguousarray(expected).tobytes(),
        )

        # Simulate an async kernel write queued on `bench.stream` that has NOT yet completed when read() / readarr() starts.
        self.fake_cuda.on_stream_sync = None
        updated1 = (bench.np.arange(16, dtype="i4") + 1000)
        self.fake_cuda.pending_stream_ops.setdefault(st_handle, []).append(
            lambda: self.fake_cuda._write_mem(buf.addr.value, updated1.tobytes())
        )
        self.fake_cuda.calls.clear()
        self.fake_cuda.call_args.clear()
        raw_bytes = buf.read()
        self.assertEqual(self.fake_cuda.calls, ["cuStreamSynchronize", "cuMemcpyDtoH_v2"])
        self.assertEqual(self.fake_cuda.call_args[0], ("cuStreamSynchronize", st_handle))
        self.assertTrue(bench.np.array_equal(raw_bytes, updated1.view("u1").ravel()))

        updated2 = (bench.np.arange(16, dtype="i4") + 2000)
        self.fake_cuda.pending_stream_ops.setdefault(st_handle, []).append(
            lambda: self.fake_cuda._write_mem(buf.addr.value, updated2.tobytes())
        )
        self.fake_cuda.calls.clear()
        self.fake_cuda.call_args.clear()
        arr_out = buf.readarr()
        self.assertEqual(self.fake_cuda.calls, ["cuStreamSynchronize", "cuMemcpyDtoH_v2"])
        self.assertEqual(self.fake_cuda.call_args[0], ("cuStreamSynchronize", st_handle))
        self.assertTrue(bench.np.array_equal(arr_out, updated2))

        # End-to-end parity path with non-null stream exercises stream-ordered Buffer.__init__ + compare() -> Buffer.read().
        bench.DO_TIMING = False
        bench.DO_PARITY = True
        row = bench.case_idx_small(2, 16, 8, "random")
        self.assertEqual(row["parity"], {"new_vs_ref": 0, "fused_vs_ref": 0, "old_vs_ref": 0})

    def test_g_buffer_null_stream_fallback_uses_ctx_sync(self):
        self.assertIsNone(bench.stream.value)
        src = bench.np.array([7, 11, 13, 17], dtype="i4")
        self.fake_cuda.calls.clear()
        buf = bench.Buffer(src)
        self.assertEqual(
            self.fake_cuda.calls,
            ["cuMemAlloc_v2", "cuMemcpyHtoD_v2", "cuCtxSynchronize"],
        )
        self.fake_cuda.calls.clear()
        self.assertTrue(bench.np.array_equal(buf.readarr(), src))
        self.assertEqual(self.fake_cuda.calls, ["cuMemcpyDtoH_v2"])
        self.fake_cuda.calls.clear()
        self.assertTrue(bench.np.array_equal(buf.read(), src.view("u1").ravel()))
        self.assertEqual(self.fake_cuda.calls, ["cuMemcpyDtoH_v2"])


if __name__ == "__main__":
    unittest.main()
