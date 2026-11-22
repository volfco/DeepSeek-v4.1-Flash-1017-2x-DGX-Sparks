#!/usr/bin/env python3
"""GB10 measurement harness for:
  1. 0215 small-row crossover: single-launch `idx_fused` vs two-launch `block_max + idx_pick`
     (plus parity against unfused `block_max + topk_indices_cut + pool_pick` and NumPy reference)
     at R in {1, 7}, nbk in {2049, 4096, 5120, 6144, 7168, 8192, 12288, 16384}, dist in {random, ties}.
  2. 0254 `topk_indices_cut` vs `topk_indices` distribution timings & NumPy parity at 128 x 32768, k=512:
     `random` vs `applyPool_50pct_ninf` vs `vis128_lt_k` vs `all_ninf`.

Hard invariants enforced:
  - MemAvailable >= 3.5 GiB before touching GPU
  - Live device allocations strictly < 64 MiB (hard limit < 192 MiB)
  - Stream-ordered copies/sync ONLY (`cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` + `cuStreamSynchronize`
    on `CU_STREAM_NON_BLOCKING`); legacy stream-0 `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` are never called
  - Strict per-case live-engine health guard on timings: `live.connections == 0` and non-null non-bool
    non-negative integer `(prompt_tokens_total, generation_tokens_total)` identical before and after each case.
    Contaminated samples are discarded immediately and never published as valid guarded medians.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

I = ct.c_int
L = ct.c_int64
P = ct.c_void_p
F = ct.c_float
U64 = ct.c_uint64
SZ = ct.c_size_t
ref = ct.byref

MIN_MEM_AVAILABLE_GIB = 3.5
MAX_VRAM_BYTES = 64 * 1024 * 1024  # Strictly < 192 MiB hard rule
CU_STREAM_NON_BLOCKING = 1
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
MAX_DYNAMIC_SHARED_BYTES = 16384 * 4  # 65,536 bytes for nbk <= 16384

CROSSOVER_ROWS = (1, 7)
CROSSOVER_NBKS = (2049, 4096, 5120, 6144, 7168, 8192, 12288, 16384)
CROSSOVER_DISTS = ("random", "ties")
CROSSOVER_K = 2048
CROSSOVER_BSIZE = 8

TOPK_ROWS = 128
TOPK_WIDTH = 32768
TOPK_K = 512
TOPK_DISTS = ("random", "applyPool_50pct_ninf", "vis128_lt_k", "all_ninf")


# ---------------------------------------------------------------------------------------------------------------------
# System & live-engine health guard helpers


def mem_available_gib(meminfo_path: str = "/proc/meminfo") -> float:
    with open(meminfo_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


def health_counters(h: Any) -> tuple[Any, Any]:
    if not isinstance(h, dict):
        return (None, None)
    return (h.get("prompt_tokens_total"), h.get("generation_tokens_total"))


def counters_valid(c: Any) -> bool:
    return (
        isinstance(c, tuple)
        and len(c) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in c)
    )


def check_idle_health(h: Any) -> bool:
    if not isinstance(h, dict):
        return False
    live = h.get("live")
    if not isinstance(live, dict):
        return False
    conns = live.get("connections")
    return (
        isinstance(conns, int)
        and not isinstance(conns, bool)
        and conns == 0
        and counters_valid(health_counters(h))
    )


def is_window_clean(h_before: Any, h_after: Any) -> bool:
    if not (check_idle_health(h_before) and check_idle_health(h_after)):
        return False
    c0 = health_counters(h_before)
    c1 = health_counters(h_after)
    return counters_valid(c0) and counters_valid(c1) and c0 == c1


def health_now(url: str = "http://127.0.0.1:8000/health", timeout: float = 2.5) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------------------------------------------------
# Exact CPU / NumPy references (matching idx_fused.zig and dsv41_ops.cu)


def score_key_host(scores: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorized IEEE-754 float32 + column total-order key matching tf_ds_score_key."""
    bits = np.asarray(scores, dtype=np.float32).view(np.uint32).astype(np.int64)
    bits = np.where(bits == 0x80000000, 0, bits)  # -0.0 -> +0.0
    sb = np.where(bits >= (1 << 31), bits - (1 << 32), bits)
    ordered = np.where(sb < 0, sb ^ 0x7FFFFFFF, sb).astype(np.uint64)
    inv_col = (np.uint64(0xFFFFFFFF) - np.asarray(cols, dtype=np.uint64)) & np.uint64(0xFFFFFFFF)
    key = (ordered << np.uint64(32)) | inv_col
    return key ^ np.uint64(0x8000000000000000)


def block_maxima_ref(scores_2d: np.ndarray, width: int, bsize: int, vis_1d: np.ndarray) -> np.ndarray:
    """Exact CPU reference for tf_ds_block_max_kernel: [rows, nbk] float32."""
    rows = scores_2d.shape[0]
    nbk = (width + bsize - 1) // bsize
    out = np.empty((rows, nbk), dtype=np.float32)
    qnan = np.uint32(0x7FC00000).view(np.float32)
    pinf = np.float32(np.inf)
    ninf = np.float32(-np.inf)

    # Pad partial tail with -inf if width < nbk * bsize
    padded = np.full((rows, nbk * bsize), ninf, dtype=np.float32)
    padded[:, :width] = scores_2d[:, :width]
    blocks = padded.reshape(rows, nbk, bsize)

    has_nan = np.isnan(blocks).any(axis=2)
    clean = np.where(np.isnan(blocks), ninf, blocks)
    amax = clean.max(axis=2)
    out[:] = np.where(has_nan, qnan, amax)

    for r in range(rows):
        v = int(vis_1d[r])
        last = (v - 1) // bsize if v > 0 else -1
        if 0 <= last < nbk:
            out[r, last] = pinf
    return out


def topk_indices_ref(scores_2d: np.ndarray, width: int, k: int, vis_1d: np.ndarray) -> np.ndarray:
    """Exact CPU reference for tf_ds_topk_indices[_cut]_kernel: [rows, k] int64."""
    rows = scores_2d.shape[0]
    out = np.empty((rows, k), dtype=np.int64)
    if k == 0:
        return out
    cols = np.arange(width, dtype=np.uint64)
    for r in range(rows):
        keys = score_key_host(scores_2d[r, :width], cols)
        # Keys are strictly unique due to inverted column bits in [0:32]
        picks = np.sort(np.argpartition(keys, width - k)[-k:]).astype(np.int64)
        out[r] = np.where(picks < int(vis_1d[r]), picks, -1)
    return out


def pool_pick_ref(bmax_2d: np.ndarray, idx_2d: np.ndarray) -> np.ndarray:
    """Exact CPU reference for tf_ds_pool_pick_kernel decode list arm: [rows, k] int32."""
    rows, k = idx_2d.shape
    cblk = np.full((rows, k), -1, dtype=np.int32)
    ninf = np.float32(-np.inf)
    for r in range(rows):
        for j in range(k):
            c = int(idx_2d[r, j])
            if c >= 0 and bool(bmax_2d[r, c] > ninf):
                cblk[r, j] = c
    return cblk


def crossover_ref(scores_2d: np.ndarray, width: int, bsize: int, k: int,
                  vis_1d: np.ndarray, every_1d: np.ndarray) -> np.ndarray:
    bmax = block_maxima_ref(scores_2d, width, bsize, vis_1d)
    idx = topk_indices_ref(bmax, bmax.shape[1], k, every_1d)
    return pool_pick_ref(bmax, idx)


# ---------------------------------------------------------------------------------------------------------------------
# Deterministic input builders


def make_crossover_inputs(rows: int, nbk: int, dist: str, partial_tail: bool = False) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Build deterministic inputs for 0215 crossover cases."""
    seed = 215000 + rows * 10000 + nbk + (0 if dist == "random" else 500000) + (900000 if partial_tail else 0)
    rng = np.random.default_rng(seed)
    ss = CROSSOVER_BSIZE * nbk
    width = ss - 3 if partial_tail else ss
    if dist == "random":
        scores = rng.standard_normal((rows, ss), dtype=np.float32)
    elif dist == "ties":
        scores = rng.integers(-10, 10, size=(rows, ss)).astype(np.float32)
    else:
        raise ValueError(f"unknown crossover dist: {dist}")
    if partial_tail:
        scores[:, width:] = np.float32(np.inf)  # Catch any read past width
    if rows == 1:
        vis = np.array([width], dtype=np.int64)
    else:
        vis = np.linspace(1, width, rows, dtype=np.int64)
    every = np.full(rows, 1 << 40, dtype=np.int64)
    return scores, width, vis, every


def make_topk_case_inputs(dist: str, rows: int = TOPK_ROWS, width: int = TOPK_WIDTH,
                          pad: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build deterministic inputs for the 4 0254 topk_indices_cut distributions at 128 x 32768, k=512.

    Returns (scores [rows, width + pad] float32, vis [rows] int64, pool_mask [rows, ms] uint8 | None).
    """
    seed_map = {
        "random": 254001,
        "applyPool_50pct_ninf": 254002,
        "vis128_lt_k": 254003,
        "all_ninf": 254004,
    }
    if dist not in seed_map:
        raise ValueError(f"unknown topk dist: {dist}")
    rng = np.random.default_rng(seed_map[dist] + pad * 100)
    stride = width + pad
    scores = rng.standard_normal((rows, stride), dtype=np.float32)
    pool_mask: np.ndarray | None = None

    if dist == "random":
        vis = np.full(rows, width, dtype=np.int64)
    elif dist == "applyPool_50pct_ninf":
        # Candidate-source pool at width=32768, cb=8 -> nbk=4096 blocks, candidate_blocks=2048 retained (exact 50%).
        bsize = 8
        nbk = width // bsize
        keep_blocks = nbk // 2  # 2048 blocks = 16,384 finite columns; 2048 blocks = 16,384 -inf columns
        ms = nbk + 8
        pool_mask = np.zeros((rows, ms), dtype=np.uint8)
        for r in range(rows):
            # Pin newest block (nbk - 1) and choose keep_blocks - 1 additional blocks uniformly
            chosen = rng.choice(nbk - 1, size=keep_blocks - 1, replace=False)
            pool_mask[r, chosen] = 1
            pool_mask[r, nbk - 1] = 1
        expanded = np.repeat(pool_mask[:, :nbk].astype(bool), bsize, axis=1)
        scores[:, :width] = np.where(expanded, scores[:, :width], np.float32(-np.inf))
        vis = np.full(rows, width, dtype=np.int64)
    elif dist == "vis128_lt_k":
        # Causal early prompt rows where vis=128 < k=512: _index_score writes finite scores in [0, 128) and -inf in [128, width).
        scores[:, 128:width] = np.float32(-np.inf)
        vis = np.full(rows, 128, dtype=np.int64)
    elif dist == "all_ninf":
        scores[:, :width] = np.float32(-np.inf)
        vis = np.full(rows, width, dtype=np.int64)

    if pad > 0:
        scores[:, width:] = np.float32(np.inf)  # Sentinel to expose any out-of-bounds row read
    return scores, vis, pool_mask


# ---------------------------------------------------------------------------------------------------------------------
# Stream-ordered CUDA Driver API wrapper (strictly zero stream-0 copies)


class StreamOrderedDriver:
    def __init__(self, cubin_path: Path, libcuda: Any = None):
        avail = mem_available_gib()
        if avail < MIN_MEM_AVAILABLE_GIB:
            raise RuntimeError(f"MemAvailable {avail:.2f} GiB < {MIN_MEM_AVAILABLE_GIB:.2f} GiB; refusing GPU run")
        self.mem_avail_gib_at_init = round(avail, 3)
        self.cuda = ct.CDLL("libcuda.so.1") if libcuda is None else libcuda
        self.cu("cuInit", I(0))
        self.ctx = P()
        self.cu("cuDevicePrimaryCtxRetain", ref(self.ctx), I(0))
        self.cu("cuCtxSetCurrent", self.ctx)
        self.stream = P()
        self.cu("cuStreamCreate", ref(self.stream), ct.c_uint(CU_STREAM_NON_BLOCKING))

        self.live_allocs: dict[int, int] = {}
        self.cur_bytes = 0
        self.peak_alloc_bytes = 0
        self.vram_free_before, self.vram_total = self.mem_info()
        self.min_vram_free_seen = self.vram_free_before

        self.mod = P()
        self.cu("cuModuleLoad", ref(self.mod), str(cubin_path).encode("utf-8"))
        self.fns: dict[str, tuple[P, int]] = {}

        for sym in (
            "tf_ds_block_max_kernel",
            "tf_ds_idx_pick_kernel",
            "tf_ds_idx_fused_kernel",
            "tf_ds_topk_indices_kernel",
            "tf_ds_topk_indices_cut_kernel",
            "tf_ds_pool_pick_kernel",
            "tf_ds_apply_pool_kernel",
            "empty_kernel",
        ):
            fn = P()
            self.cu("cuModuleGetFunction", ref(fn), self.mod, sym.encode("utf-8"))
            nparams = self._query_param_count(fn)
            if sym in ("tf_ds_idx_fused_kernel", "tf_ds_idx_pick_kernel", "empty_kernel"):
                self.cu(
                    "cuFuncSetAttribute",
                    fn,
                    I(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                    I(MAX_DYNAMIC_SHARED_BYTES),
                )
            self.fns[sym] = (fn, nparams)

    def cu(self, name: str, *args: Any) -> None:
        if name in ("cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"):
            raise AssertionError(f"Forbidden legacy stream-0 call: {name}")
        err = getattr(self.cuda, name)(*args)
        if err != 0:
            msg = ct.c_char_p()
            self.cuda.cuGetErrorString(err, ref(msg))
            text = msg.value.decode("utf-8", errors="replace") if msg.value else "?"
            raise RuntimeError(f"{name} failed with error {err}: {text}")

    def _query_param_count(self, fn: P) -> int:
        n = 0
        off, size = SZ(), SZ()
        while self.cuda.cuFuncGetParamInfo(fn, SZ(n), ref(off), ref(size)) == 0:
            n += 1
        return n

    def mem_info(self) -> tuple[int, int]:
        free_b, total_b = SZ(), SZ()
        self.cu("cuMemGetInfo_v2", ref(free_b), ref(total_b))
        return int(free_b.value), int(total_b.value)

    def sync(self) -> None:
        self.cu("cuStreamSynchronize", self.stream)

    def upload(self, host_arr: np.ndarray) -> tuple[int, tuple[int, ...], np.dtype]:
        arr = np.ascontiguousarray(host_arr)
        nbytes = max(16, int(arr.nbytes))
        if self.cur_bytes + nbytes > MAX_VRAM_BYTES:
            raise RuntimeError(
                f"Device allocation cap exceeded: {self.cur_bytes + nbytes} > {MAX_VRAM_BYTES} bytes"
            )
        addr = U64()
        self.cu("cuMemAlloc_v2", ref(addr), SZ(nbytes))
        ptr = int(addr.value)
        self.live_allocs[ptr] = nbytes
        self.cur_bytes += nbytes
        self.peak_alloc_bytes = max(self.peak_alloc_bytes, self.cur_bytes)
        free_now, _ = self.mem_info()
        self.min_vram_free_seen = min(self.min_vram_free_seen, free_now)

        if arr.nbytes > 0:
            # Keep `arr` alive until stream-ordered HtoD copy completes on self.stream
            self.cu("cuMemcpyHtoDAsync_v2", U64(ptr), P(arr.ctypes.data), SZ(arr.nbytes), self.stream)
            self.sync()
        return ptr, arr.shape, arr.dtype

    def alloc_sentinel(self, shape: tuple[int, ...], dtype: str | np.dtype, fill_byte: int = 0x55) -> tuple[int, tuple[int, ...], np.dtype]:
        dt = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dt.itemsize
        raw = np.full(nbytes, fill_byte, dtype=np.uint8)
        arr = raw.view(dt).reshape(shape)
        return self.upload(arr)

    def download(self, buf: tuple[int, tuple[int, ...], np.dtype]) -> np.ndarray:
        ptr, shape, dtype = buf
        host = np.empty(shape, dtype=dtype)
        if host.nbytes > 0:
            self.cu("cuMemcpyDtoHAsync_v2", P(host.ctypes.data), U64(ptr), SZ(host.nbytes), self.stream)
            self.sync()
        return host

    def free_all(self) -> None:
        self.sync()
        for ptr in list(self.live_allocs.keys()):
            nbytes = self.live_allocs.pop(ptr)
            self.cu("cuMemFree_v2", U64(ptr))
            self.cur_bytes -= nbytes

    def close(self) -> None:
        try:
            self.free_all()
        finally:
            if self.mod.value:
                self.cu("cuModuleUnload", self.mod)
                self.mod = P()
            if self.stream.value:
                self.cu("cuStreamDestroy_v2", self.stream)
                self.stream = P()
            if self.ctx.value:
                self.cu("cuDevicePrimaryCtxRelease", I(0))
                self.ctx = P()

    def launch(
        self,
        sym: str,
        grid: int | tuple[int, int],
        block: int | tuple[int, int],
        shared: int,
        *args: Any,
    ) -> None:
        fn, expected_n = self.fns[sym]
        if len(args) != expected_n:
            raise ValueError(f"{sym}: expected {expected_n} args, got {len(args)}")
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        bx, by = block if isinstance(block, tuple) else (block, 1)
        if args:
            argv = (P * len(args))(*(ct.addressof(a) for a in args))
        else:
            argv = None
        self.cu(
            "cuLaunchKernel",
            fn,
            ct.c_uint(gx),
            ct.c_uint(gy),
            ct.c_uint(1),
            ct.c_uint(bx),
            ct.c_uint(by),
            ct.c_uint(1),
            ct.c_uint(shared),
            self.stream,
            argv,
            None,
        )

    def capture_graph(self, chain: Callable[[], None], batch: int) -> P:
        chain()
        self.sync()
        self.cu("cuStreamBeginCapture_v2", self.stream, I(0))
        for _ in range(batch):
            chain()
        graph = P()
        exec_graph = P()
        self.cu("cuStreamEndCapture", self.stream, ref(graph))
        try:
            self.cu("cuGraphInstantiateWithFlags", ref(exec_graph), graph, U64(0))
        finally:
            if graph.value:
                self.cu("cuGraphDestroy", graph)
        free_now, _ = self.mem_info()
        self.min_vram_free_seen = min(self.min_vram_free_seen, free_now)
        return exec_graph

    def time_interleaved_single_window(
        self,
        chains: list[Callable[[], None]],
        names: list[str],
        batch: int,
        reps: int,
        health_url: str,
    ) -> tuple[dict[str, Any], bool, dict[str, Any], dict[str, Any]]:
        """Capture CUDA graphs, warm once, and run a single ~0.2-0.7s guarded window.

        Returns (stats, valid, h_before, h_after). Never publishes stats when valid is False.
        """
        if len(chains) != len(names) or len(set(names)) != len(names):
            raise ValueError(f"Invalid arm names: {names!r}")

        h_pre = health_now(health_url)
        if not check_idle_health(h_pre):
            return {}, False, h_pre, {}

        exes: list[P] = []
        start_ev, stop_ev = P(), P()
        try:
            for ch in chains:
                exes.append(self.capture_graph(ch, batch))
            for exe in exes:
                self.cu("cuGraphLaunch", exe, self.stream)
            self.sync()

            self.cu("cuEventCreate", ref(start_ev), ct.c_uint(0))
            self.cu("cuEventCreate", ref(stop_ev), ct.c_uint(0))

            h0 = health_now(health_url)
            if not check_idle_health(h0):
                return {}, False, h0, {}

            samples: list[list[float]] = [[] for _ in exes]
            n_arms = len(exes)
            for rep in range(reps):
                for off in range(n_arms):
                    idx = (rep + off) % n_arms
                    self.cu("cuEventRecord", start_ev, self.stream)
                    self.cu("cuGraphLaunch", exes[idx], self.stream)
                    self.cu("cuEventRecord", stop_ev, self.stream)
                    self.cu("cuEventSynchronize", stop_ev)
                    ms = F()
                    self.cu("cuEventElapsedTime", ref(ms), start_ev, stop_ev)
                    samples[idx].append(float(ms.value) * 1000.0 / float(batch))

            h1 = health_now(health_url)
            if not is_window_clean(h0, h1):
                return {}, False, h0, h1

            stats: dict[str, Any] = {}
            for name, s in zip(names, samples, strict=True):
                arr = np.asarray(s, dtype=np.float64)
                stats[name] = {
                    "median": round(float(median(s)), 3),
                    "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 3),
                    "min": round(float(arr.min()), 3),
                    "max": round(float(arr.max()), 3),
                    "n": len(s),
                }
            return stats, True, h0, h1
        finally:
            self.sync()
            for ev in (start_ev, stop_ev):
                if ev.value:
                    self.cu("cuEventDestroy_v2", ev)
            for exe in exes:
                if exe.value:
                    self.cu("cuGraphExecDestroy", exe)


# ---------------------------------------------------------------------------------------------------------------------
# Parity execution (Step 2: load-independent, runs before any idle wait)


def byte_diff_count(a: np.ndarray, b: np.ndarray) -> int:
    ba = np.ascontiguousarray(a).view(np.uint8).ravel()
    bb = np.ascontiguousarray(b).view(np.uint8).ravel()
    if ba.shape != bb.shape:
        raise ValueError(f"shape mismatch: {ba.shape} vs {bb.shape}")
    return int(np.count_nonzero(ba != bb))


def run_crossover_parity_one(
    drv: StreamOrderedDriver, rows: int, nbk: int, dist: str, partial_tail: bool = False
) -> dict[str, int]:
    scores, width, vis, every = make_crossover_inputs(rows, nbk, dist, partial_tail=partial_tail)
    ss = scores.shape[1]
    k = CROSSOVER_K
    bsize = CROSSOVER_BSIZE
    guard_elems = 16

    expected_cblk = crossover_ref(scores, width, bsize, k, vis, every)
    expected_full = np.full(rows * k + guard_elems, 0x55555555, dtype=np.uint32).view(np.int32)
    expected_full[: rows * k] = expected_cblk.ravel()

    try:
        d_score = drv.upload(scores)
        d_vis = drv.upload(vis)
        d_every = drv.upload(every)

        d_bmax_unfused = drv.alloc_sentinel((rows, nbk), np.float32)
        d_pidx_unfused = drv.alloc_sentinel((rows, k), np.int64)
        d_cblk_unfused = drv.alloc_sentinel((rows * k + guard_elems,), np.int32)

        d_bmax_two = drv.alloc_sentinel((rows, nbk), np.float32)
        d_cblk_two = drv.alloc_sentinel((rows * k + guard_elems,), np.int32)

        d_cblk_fused = drv.alloc_sentinel((rows * k + guard_elems,), np.int32)

        # Arm 1: unfused block_max + topk_indices_cut + pool_pick
        drv.launch(
            "tf_ds_block_max_kernel",
            ((nbk + 255) // 256, rows),
            256,
            0,
            U64(d_score[0]), L(ss), I(width), I(bsize), U64(d_vis[0]), U64(d_bmax_unfused[0]), L(nbk), I(nbk), I(0),
        )
        drv.launch(
            "tf_ds_topk_indices_cut_kernel",
            rows,
            1024,
            0,
            U64(d_bmax_unfused[0]), L(nbk), I(nbk), I(k), U64(d_every[0]), U64(d_pidx_unfused[0]), I(0),
        )
        drv.launch(
            "tf_ds_pool_pick_kernel",
            rows,
            1024,
            0,
            U64(d_bmax_unfused[0]), L(nbk), I(nbk), U64(d_pidx_unfused[0]), I(k), U64(0), L(0), U64(d_cblk_unfused[0]), I(0),
        )

        # Arm 2: two-launch block_max + idx_pick
        drv.launch(
            "tf_ds_block_max_kernel",
            ((nbk + 255) // 256, rows),
            256,
            0,
            U64(d_score[0]), L(ss), I(width), I(bsize), U64(d_vis[0]), U64(d_bmax_two[0]), L(nbk), I(nbk), I(0),
        )
        drv.launch(
            "tf_ds_idx_pick_kernel",
            rows,
            1024,
            nbk * 4,
            U64(d_bmax_two[0]), L(nbk), I(nbk), I(k), U64(d_every[0]), U64(d_cblk_two[0]), I(0),
        )

        # Arm 3: single-launch idx_fused
        drv.launch(
            "tf_ds_idx_fused_kernel",
            rows,
            1024,
            nbk * 4,
            U64(d_score[0]), L(ss), I(width), I(bsize), U64(d_vis[0]), U64(d_every[0]), I(k), U64(0), L(0), U64(d_cblk_fused[0]), I(0),
        )

        out_unfused = drv.download(d_cblk_unfused)
        out_two = drv.download(d_cblk_two)
        out_fused = drv.download(d_cblk_fused)

        return {
            "fused_vs_two_phase": byte_diff_count(out_fused, out_two),
            "fused_vs_unfused": byte_diff_count(out_fused, out_unfused),
            "two_phase_vs_unfused": byte_diff_count(out_two, out_unfused),
            "fused_vs_numpy": byte_diff_count(out_fused, expected_full),
            "two_phase_vs_numpy": byte_diff_count(out_two, expected_full),
            "unfused_vs_numpy": byte_diff_count(out_unfused, expected_full),
        }
    finally:
        drv.free_all()


def run_topk_parity_one(drv: StreamOrderedDriver, dist: str, pad: int = 0) -> dict[str, int]:
    rows, width, k = TOPK_ROWS, TOPK_WIDTH, TOPK_K
    scores, vis, pool_mask = make_topk_case_inputs(dist, rows=rows, width=width, pad=pad)
    ss = scores.shape[1]
    guard_elems = 16

    expected = topk_indices_ref(scores, width, k, vis)
    expected_full = np.full(rows * k + guard_elems, 0x5555555555555555, dtype=np.uint64).view(np.int64)
    expected_full[: rows * k] = expected.ravel()

    try:
        apply_pool_diff = 0
        if dist == "applyPool_50pct_ninf" and pool_mask is not None:
            # Verify GPU tf_ds_apply_pool_kernel produces identical 50%-inf scores before topk
            raw_rng = np.random.default_rng(254002 + pad * 100)
            raw_scores = raw_rng.standard_normal((rows, ss), dtype=np.float32)
            if pad > 0:
                raw_scores[:, width:] = np.float32(np.inf)
            d_raw = drv.upload(raw_scores)
            d_mask = drv.upload(pool_mask)
            drv.launch(
                "tf_ds_apply_pool_kernel",
                ((width + 255) // 256, rows),
                256,
                0,
                U64(d_raw[0]), L(ss), I(width), I(8), U64(d_mask[0]), L(pool_mask.shape[1]), I(0),
            )
            gpu_masked = drv.download(d_raw)
            apply_pool_diff = byte_diff_count(gpu_masked, scores)
            d_score = d_raw
        else:
            d_score = drv.upload(scores)

        d_vis = drv.upload(vis)
        d_out_old = drv.alloc_sentinel((rows * k + guard_elems,), np.int64)
        d_out_cut = drv.alloc_sentinel((rows * k + guard_elems,), np.int64)
        d_out_pdl = drv.alloc_sentinel((rows * k + guard_elems,), np.int64)

        drv.launch(
            "tf_ds_topk_indices_kernel",
            rows, 1024, 0,
            U64(d_score[0]), L(ss), I(width), I(k), U64(d_vis[0]), U64(d_out_old[0]), I(0),
        )
        drv.launch(
            "tf_ds_topk_indices_cut_kernel",
            rows, 1024, 0,
            U64(d_score[0]), L(ss), I(width), I(k), U64(d_vis[0]), U64(d_out_cut[0]), I(0),
        )
        drv.launch(
            "tf_ds_topk_indices_cut_kernel",
            rows, 1024, 0,
            U64(d_score[0]), L(ss), I(width), I(k), U64(d_vis[0]), U64(d_out_pdl[0]), I(1),
        )

        out_old = drv.download(d_out_old)
        out_cut = drv.download(d_out_cut)
        out_pdl = drv.download(d_out_pdl)

        res = {
            "cut_vs_numpy": byte_diff_count(out_cut, expected_full),
            "old_vs_numpy": byte_diff_count(out_old, expected_full),
            "cut_pdl_vs_numpy": byte_diff_count(out_pdl, expected_full),
            "cut_vs_old": byte_diff_count(out_cut, out_old),
        }
        if dist == "applyPool_50pct_ninf":
            res["apply_pool_vs_numpy"] = apply_pool_diff
        return res
    finally:
        drv.free_all()


def run_all_parity(drv: StreamOrderedDriver, cubin_path: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    crossover_cases: list[dict[str, Any]] = []
    failures: list[str] = []

    for rows in CROSSOVER_ROWS:
        for nbk in CROSSOVER_NBKS:
            for dist in CROSSOVER_DISTS:
                case_id = f"crossover_R{rows}_nbk{nbk}_{dist}"
                p_exact = run_crossover_parity_one(drv, rows, nbk, dist, partial_tail=False)
                p_tail = run_crossover_parity_one(drv, rows, nbk, dist, partial_tail=True)
                ok = all(v == 0 for v in p_exact.values()) and all(v == 0 for v in p_tail.values())
                if not ok:
                    failures.append(f"{case_id}: exact={p_exact}, partial_tail={p_tail}")
                print(
                    f"[PARITY] {'PASS' if ok else 'FAIL'} {case_id:<34} exact={p_exact} tail_diff={sum(p_tail.values())}",
                    flush=True,
                )
                crossover_cases.append(
                    {
                        "case_id": case_id,
                        "group": "0215_crossover",
                        "rows": rows,
                        "nbk": nbk,
                        "width": CROSSOVER_BSIZE * nbk,
                        "k": CROSSOVER_K,
                        "dist": dist,
                        "currently_dispatched_two_phase": bool(rows <= 7 and nbk >= 6144),
                        "parity": p_exact,
                        "parity_partial_tail": p_tail,
                        "parity_pass": ok,
                    }
                )

    topk_cases: list[dict[str, Any]] = []
    for dist in TOPK_DISTS:
        case_id = f"topk_R{TOPK_ROWS}_n{TOPK_WIDTH}_k{TOPK_K}_{dist}"
        p_aligned = run_topk_parity_one(drv, dist, pad=0)
        p_padded = run_topk_parity_one(drv, dist, pad=3)
        ok = all(v == 0 for v in p_aligned.values()) and all(v == 0 for v in p_padded.values())
        if not ok:
            failures.append(f"{case_id}: aligned={p_aligned}, padded={p_padded}")
        print(
            f"[PARITY] {'PASS' if ok else 'FAIL'} {case_id:<44} aligned={p_aligned} pad3_diff={sum(p_padded.values())}",
            flush=True,
        )
        topk_cases.append(
            {
                "case_id": case_id,
                "group": "0254_topk_indices_cut",
                "rows": TOPK_ROWS,
                "width": TOPK_WIDTH,
                "k": TOPK_K,
                "dist": dist,
                "parity": p_aligned,
                "parity_pad3": p_padded,
                "parity_pass": ok,
            }
        )

    elapsed_s = round(time.monotonic() - t0, 3)
    vram_spike_bytes = max(0, drv.vram_free_before - drv.min_vram_free_seen)
    return {
        "cubin_path": str(cubin_path),
        "cubin_sha256": hashlib.sha256(cubin_path.read_bytes()).hexdigest(),
        "mem_available_gib": drv.mem_avail_gib_at_init,
        "peak_alloc_mib": round(drv.peak_alloc_bytes / (1024.0 * 1024.0), 3),
        "observed_vram_delta_mib": round(vram_spike_bytes / (1024.0 * 1024.0), 3),
        "parity_elapsed_s": elapsed_s,
        "all_parity_passed": len(failures) == 0,
        "failures": failures,
        "cases": crossover_cases + topk_cases,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Guarded per-case CUDA-graph timing execution (Step 3)


def time_one_case(
    drv: StreamOrderedDriver,
    case: dict[str, Any],
    batch: int,
    reps: int,
    health_url: str,
) -> tuple[dict[str, Any], bool, dict[str, Any], dict[str, Any]]:
    try:
        if case["group"] == "0215_crossover":
            rows = int(case["rows"])
            nbk = int(case["nbk"])
            dist = str(case["dist"])
            scores, width, vis, every = make_crossover_inputs(rows, nbk, dist, partial_tail=False)
            ss = scores.shape[1]
            k = CROSSOVER_K
            bsize = CROSSOVER_BSIZE

            d_score = drv.upload(scores)
            d_vis = drv.upload(vis)
            d_every = drv.upload(every)
            d_bmax = drv.alloc_sentinel((rows, nbk), np.float32)
            d_cblk_fused = drv.alloc_sentinel((rows, k), np.int32)
            d_cblk_two = drv.alloc_sentinel((rows, k), np.int32)

            def chain_fused() -> None:
                drv.launch(
                    "tf_ds_idx_fused_kernel",
                    rows, 1024, nbk * 4,
                    U64(d_score[0]), L(ss), I(width), I(bsize), U64(d_vis[0]), U64(d_every[0]),
                    I(k), U64(0), L(0), U64(d_cblk_fused[0]), I(0),
                )

            def chain_two_phase() -> None:
                drv.launch(
                    "tf_ds_block_max_kernel",
                    ((nbk + 255) // 256, rows), 256, 0,
                    U64(d_score[0]), L(ss), I(width), I(bsize), U64(d_vis[0]), U64(d_bmax[0]),
                    L(nbk), I(nbk), I(0),
                )
                drv.launch(
                    "tf_ds_idx_pick_kernel",
                    rows, 1024, nbk * 4,
                    U64(d_bmax[0]), L(nbk), I(nbk), I(k), U64(d_every[0]), U64(d_cblk_two[0]), I(0),
                )

            def chain_floor() -> None:
                drv.launch("empty_kernel", rows, 1024, nbk * 4)

            return drv.time_interleaved_single_window(
                [chain_fused, chain_two_phase, chain_floor],
                ["idx_fused", "two_phase", "floor"],
                batch=batch,
                reps=reps,
                health_url=health_url,
            )

        if case["group"] == "0254_topk_indices_cut":
            rows = int(case["rows"])
            width = int(case["width"])
            k = int(case["k"])
            dist = str(case["dist"])
            scores, vis, _ = make_topk_case_inputs(dist, rows=rows, width=width, pad=0)
            ss = scores.shape[1]

            d_score = drv.upload(scores)
            d_vis = drv.upload(vis)
            d_out_old = drv.alloc_sentinel((rows, k), np.int64)
            d_out_cut = drv.alloc_sentinel((rows, k), np.int64)

            def chain_old() -> None:
                drv.launch(
                    "tf_ds_topk_indices_kernel",
                    rows, 1024, 0,
                    U64(d_score[0]), L(ss), I(width), I(k), U64(d_vis[0]), U64(d_out_old[0]), I(0),
                )

            def chain_cut() -> None:
                drv.launch(
                    "tf_ds_topk_indices_cut_kernel",
                    rows, 1024, 0,
                    U64(d_score[0]), L(ss), I(width), I(k), U64(d_vis[0]), U64(d_out_cut[0]), I(0),
                )

            def chain_floor() -> None:
                drv.launch("empty_kernel", rows, 1024, 0)

            return drv.time_interleaved_single_window(
                [chain_old, chain_cut, chain_floor],
                ["topk_indices", "topk_indices_cut", "floor"],
                batch=batch,
                reps=reps,
                health_url=health_url,
            )

        raise ValueError(f"Unknown case group: {case['group']}")
    finally:
        drv.free_all()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_timing_burst(
    drv: StreamOrderedDriver,
    state: dict[str, Any],
    json_path: Path,
    batch: int,
    reps: int,
    health_url: str,
) -> tuple[int, int]:
    """Run consecutive ~0.2-0.7s guarded cases while the live engine remains idle.

    Stops immediately as soon as all cases are valid OR a health check sees activity,
    so the caller can release gpu.lock during 3s polling sleeps.
    Returns (newly_captured_count, remaining_count).
    """
    newly_captured = 0
    for case in state["cases"]:
        if case.get("timing_valid") is True:
            continue
        t_case0 = time.monotonic()
        stats, valid, h0, h1 = time_one_case(drv, case, batch=batch, reps=reps, health_url=health_url)
        dt = round(time.monotonic() - t_case0, 3)
        if valid:
            c0 = health_counters(h0)
            c1 = health_counters(h1)
            case["timing"] = stats
            case["timing_valid"] = True
            case["timing_guarded"] = True
            case["timing_retries"] = int(case.get("timing_retries", 0))
            case["window_s"] = dt
            case["health_before"] = {
                "connections": h0["live"]["connections"],
                "prompt_tokens_total": c0[0],
                "generation_tokens_total": c0[1],
            }
            case["health_after"] = {
                "connections": h1["live"]["connections"],
                "prompt_tokens_total": c1[0],
                "generation_tokens_total": c1[1],
            }
            newly_captured += 1
            summary = " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in stats.items())
            print(
                f"[TIMING] VALID {case['case_id']:<44} ({dt:.3f}s) counters={c0} {summary}",
                flush=True,
            )
            atomic_write_json(json_path, state)
        else:
            case["timing"] = {}
            case["timing_valid"] = False
            case["timing_guarded"] = True
            case["timing_retries"] = int(case.get("timing_retries", 0)) + 1
            state["total_timing_retries"] = int(state.get("total_timing_retries", 0)) + 1
            c0 = health_counters(h0)
            c1 = health_counters(h1)
            conns0 = h0.get("live", {}).get("connections") if isinstance(h0, dict) and isinstance(h0.get("live"), dict) else None
            conns1 = h1.get("live", {}).get("connections") if isinstance(h1, dict) and isinstance(h1.get("live"), dict) else None
            print(
                f"[TIMING] BUSY/CONTAMINATED {case['case_id']} ({dt:.3f}s) "
                f"before=(conns={conns0}, c={c0}) after=(conns={conns1}, c={c1}); discarding & yielding lock",
                flush=True,
            )
            atomic_write_json(json_path, state)
            break

    vram_spike_bytes = max(0, drv.vram_free_before - drv.min_vram_free_seen)
    state["timing_peak_alloc_mib"] = max(
        float(state.get("timing_peak_alloc_mib", 0.0)),
        round(drv.peak_alloc_bytes / (1024.0 * 1024.0), 3),
    )
    state["timing_observed_vram_delta_mib"] = max(
        float(state.get("timing_observed_vram_delta_mib", 0.0)),
        round(vram_spike_bytes / (1024.0 * 1024.0), 3),
    )
    remaining = sum(1 for c in state["cases"] if c.get("timing_valid") is not True)
    state["timing_complete"] = remaining == 0
    state["timing_valid_count"] = len(state["cases"]) - remaining
    state["timing_invalid_count"] = remaining
    atomic_write_json(json_path, state)
    return newly_captured, remaining


def finalize_state(json_path: Path) -> int:
    state = json.loads(json_path.read_text(encoding="utf-8"))
    for case in state["cases"]:
        if case.get("timing_valid") is not True:
            case["timing"] = {}
            case["timing_valid"] = False
            case["timing_guarded"] = True
            case["timing_retries"] = int(case.get("timing_retries", 0))
    remaining = sum(1 for c in state["cases"] if c.get("timing_valid") is not True)
    state["timing_complete"] = remaining == 0
    state["timing_valid_count"] = len(state["cases"]) - remaining
    state["timing_invalid_count"] = remaining
    atomic_write_json(json_path, state)
    print(
        f"[FINALIZE] valid={state['timing_valid_count']}/{len(state['cases'])} "
        f"invalid={remaining} total_retries={state.get('total_timing_retries', 0)}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("parity", "timing-burst", "finalize"), required=True)
    ap.add_argument("--cubin", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    args = ap.parse_args()

    if args.mode == "finalize":
        return finalize_state(args.json)

    drv = StreamOrderedDriver(args.cubin)
    try:
        if args.mode == "parity":
            payload = run_all_parity(drv, args.cubin)
            for case in payload["cases"]:
                case["timing"] = {}
                case["timing_valid"] = False
                case["timing_guarded"] = True
                case["timing_retries"] = 0
            payload["total_timing_retries"] = 0
            payload["timing_complete"] = False
            payload["timing_valid_count"] = 0
            payload["timing_invalid_count"] = len(payload["cases"])
            atomic_write_json(args.json, payload)
            print(
                f"[PARITY DONE] all_passed={payload['all_parity_passed']} "
                f"cases={len(payload['cases'])} peak_alloc_mib={payload['peak_alloc_mib']} "
                f"elapsed_s={payload['parity_elapsed_s']}",
                flush=True,
            )
            return 0 if payload["all_parity_passed"] else 1

        if args.mode == "timing-burst":
            state = json.loads(args.json.read_text(encoding="utf-8"))
            if not state.get("all_parity_passed"):
                print("Refusing timing-burst because all_parity_passed is False", file=sys.stderr)
                return 1
            captured, remaining = run_timing_burst(
                drv, state, args.json, batch=args.batch, reps=args.reps, health_url=args.health_url
            )
            print(f"[BURST DONE] newly_captured={captured} remaining={remaining}", flush=True)
            # Exit 0 when all done, exit 2 when more cases remain (for outer idle-poll loop)
            return 0 if remaining == 0 else 2
    finally:
        drv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
