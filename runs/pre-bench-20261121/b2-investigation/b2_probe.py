#!/usr/bin/env python3
"""B2 Instrumented GPU Diagnostic Probe (per-process PID-tagged artifacts).

Tests WP1 tip (d01975d, old), WP2 tip (b28310e, wp2), and current tip (d0c9fb1, new):
  1. Pre-poisons freed device memory and output/scratch buffers with adversarial patterns
     (0xFF, 0x7FC00000 NaN, 0xDEADBEEF, random bytes) across varied pointer alignments/offsets.
  2. Computes SHA-256 of all device input buffers BEFORE and AFTER launch, logs exact hex
     device addresses and resolved kernel symbol names + cubin SHA-256.
  3. Runs 100 iterations each (eager + CUDA-graph replay, pdl=0 and pdl=1) of:
       - idx-small R=1 nb=8192 k=2048 (wp2_seed197, random, ties, flat, adversarial) + R=7 nb=8192
       - compress-decode D=512 mixed (wp2_seed197 R=32, plus R=1, 7, 32, 48, and R=32 adversarial/unaligned)
     recording first differing byte offset + actual/expected byte values if any mismatch occurs.
  4. With --repro-stream0-race, demonstrates the exact bench.py legacy-stream-0 cuMemcpyHtoD_v2
     (<= 64 KiB pageable staging) vs CU_STREAM_NON_BLOCKING race that produced 3,984 (first 4)
     and 3,707 (first 29952) in WP2, and verifies cuMemcpyHtoDAsync_v2(..., stream) eliminates it.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
U32, U64 = ct.c_uint32, ct.c_uint64
ref = ct.byref

MAX_VRAM_BYTES = 256 * 1024 * 1024
MIN_MEM_AVAIL_GIB = 3.5
CU_STREAM_NON_BLOCKING = 1
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6


class LaunchAttr(ct.Structure):
    _fields_ = [("id", ct.c_uint), ("pad", ct.c_ubyte * 4), ("value", ct.c_ubyte * 64)]


class LaunchConfig(ct.Structure):
    _fields_ = [
        ("grid_x", ct.c_uint), ("grid_y", ct.c_uint), ("grid_z", ct.c_uint),
        ("block_x", ct.c_uint), ("block_y", ct.c_uint), ("block_z", ct.c_uint),
        ("shared_bytes", ct.c_uint), ("stream", P), ("attrs", P), ("num_attrs", ct.c_uint),
    ]


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_poison_bytes(nbytes: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    n4 = (nbytes + 3) // 4
    if pattern == "0xFF":
        arr = np.full(n4, 0xFFFFFFFF, dtype=np.uint32)
    elif pattern == "0x7FC00000":
        arr = np.full(n4, 0x7FC00000, dtype=np.uint32)
    elif pattern == "0xDEADBEEF":
        arr = np.full(n4, 0xDEADBEEF, dtype=np.uint32)
    elif pattern == "random":
        arr = rng.integers(1, 0xFFFFFFFF, size=n4, dtype=np.uint32)
    elif pattern == "0x55":
        arr = np.full(n4, 0x55555555, dtype=np.uint32)
    else:
        raise ValueError(f"unknown poison pattern: {pattern}")
    return np.ascontiguousarray(arr.view(np.uint8)[:nbytes])


class Driver:
    def __init__(self):
        avail = mem_available_gib()
        if avail < MIN_MEM_AVAIL_GIB:
            raise SystemExit(f"ABORT: MemAvailable {avail:.2f} GiB < {MIN_MEM_AVAIL_GIB:.2f} GiB")
        self.cuda = ct.CDLL("libcuda.so.1")
        self.cu("cuInit", 0)
        self.ctx = P()
        self.cu("cuDevicePrimaryCtxRetain", ref(self.ctx), 0)
        self.cu("cuCtxSetCurrent", self.ctx)
        self.stream = P()
        self.cu("cuStreamCreate", ref(self.stream), CU_STREAM_NON_BLOCKING)
        self.modules = {}
        self.functions = {}
        self.live_allocs = {}
        self.cur_bytes = 0
        self.peak_bytes = 0

    def cu(self, name: str, *args):
        err = getattr(self.cuda, name)(*args)
        if err:
            msg = ct.c_char_p()
            self.cuda.cuGetErrorString(err, ref(msg))
            raise RuntimeError(f"{name}: {err} ({msg.value.decode() if msg.value else '?'})")

    def alloc(self, nbytes: int) -> int:
        nbytes = max(16, int(nbytes))
        if self.cur_bytes + nbytes >= MAX_VRAM_BYTES:
            raise RuntimeError(f"VRAM cap exceeded: {self.cur_bytes + nbytes} >= {MAX_VRAM_BYTES}")
        addr = U64()
        self.cu("cuMemAlloc_v2", ref(addr), ct.c_size_t(nbytes))
        self.live_allocs[addr.value] = nbytes
        self.cur_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.cur_bytes)
        return addr.value

    def free(self, base_addr: int):
        nbytes = self.live_allocs.pop(base_addr)
        self.cu("cuMemFree_v2", U64(base_addr))
        self.cur_bytes -= nbytes

    def free_all(self):
        self.sync()
        for addr in list(self.live_allocs.keys()):
            self.free(addr)

    def htod_async(self, dst_addr: int, arr: np.ndarray, on_stream=None):
        arr = np.ascontiguousarray(arr)
        st = self.stream if on_stream is None else on_stream
        self.cu("cuMemcpyHtoDAsync_v2", U64(dst_addr), P(arr.ctypes.data), ct.c_size_t(arr.nbytes), st)

    def dtod_async(self, dst_addr: int, src_addr: int, nbytes: int, on_stream=None):
        st = self.stream if on_stream is None else on_stream
        self.cu("cuMemcpyDtoDAsync_v2", U64(dst_addr), U64(src_addr), ct.c_size_t(nbytes), st)

    def dtoh_async(self, src_addr: int, nbytes: int, on_stream=None) -> np.ndarray:
        host = np.empty(nbytes, dtype=np.uint8)
        st = self.stream if on_stream is None else on_stream
        self.cu("cuMemcpyDtoHAsync_v2", P(host.ctypes.data), U64(src_addr), ct.c_size_t(nbytes), st)
        return host

    def read_sync(self, src_addr: int, nbytes: int) -> np.ndarray:
        host = self.dtoh_async(src_addr, nbytes)
        self.sync()
        return host

    def sync(self):
        self.cu("cuStreamSynchronize", self.stream)

    def load_module(self, path: Path):
        key = str(path.resolve())
        if key not in self.modules:
            mod = P()
            self.cu("cuModuleLoad", ref(mod), key.encode())
            self.modules[key] = (mod, sha256_bytes(path.read_bytes()))
        return self.modules[key]

    def get_fn(self, path: Path, symbol: str):
        key = (str(path.resolve()), symbol)
        if key not in self.functions:
            mod, mod_sha = self.load_module(path)
            fn = P()
            self.cu("cuModuleGetFunction", ref(fn), mod, symbol.encode())
            resolved = ct.c_char_p()
            self.cu("cuFuncGetName", ref(resolved), fn)
            self.functions[key] = (fn, resolved.value.decode(), mod_sha)
        return self.functions[key]

    def launch(self, path: Path, symbol: str, grid, block, shared: int, *args, pdl: bool = False, on_stream=None):
        fn, resolved_name, mod_sha = self.get_fn(path, symbol)
        if shared > 49152:
            self.cu("cuFuncSetAttribute", fn, I(8), I(shared))
        pack = []
        for a in args:
            if isinstance(a, U64):
                pack.append(a)
            elif isinstance(a, (int, np.integer)):
                pack.append(I(int(a)))
            else:
                pack.append(a)
        argv = (P * len(pack))(*(ct.addressof(x) for x in pack))
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        bx, by = block if isinstance(block, tuple) else (block, 1)
        st = self.stream if on_stream is None else on_stream
        if not pdl:
            self.cu("cuLaunchKernel", fn, gx, gy, 1, bx, by, 1, shared, st, argv, None)
        else:
            attr = LaunchAttr(
                CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                (ct.c_ubyte * 4)(0, 0, 0, 0),
                (ct.c_ubyte * 64)(1, *([0] * 63)),
            )
            cfg = LaunchConfig(gx, gy, 1, bx, by, 1, shared, st, ct.cast(ct.pointer(attr), P), 1)
            self.cu("cuLaunchKernelEx", ref(cfg), fn, argv, None)
        return resolved_name, mod_sha

    def poison_freelist(self, pattern: str, rng: np.random.Generator):
        sizes = [8192, 16384, 32768, 33792, 65536, 262144, 524288]
        addrs = []
        for sz in sizes:
            a = self.alloc(sz)
            self.htod_async(a, make_poison_bytes(sz, pattern, rng))
            addrs.append(a)
        self.sync()
        for a in addrs:
            self.free(a)

    def close(self):
        self.free_all()
        self.cu("cuStreamDestroy_v2", self.stream)
        self.cu("cuDevicePrimaryCtxRelease", 0)


class ArenaSlice:
    """Oversized device allocation supporting byte-offset pointer views + head/tail guard verification."""

    def __init__(self, drv: Driver, nbytes: int, max_offset: int = 256, guard: int = 64):
        self.drv = drv
        self.nbytes = int(nbytes)
        self.max_offset = int(max_offset)
        self.guard = int(guard)
        self.total = self.guard + self.max_offset + self.nbytes + self.guard
        self.base = drv.alloc(self.total)
        self._last_host = None

    def ptr_at(self, offset: int = 0) -> U64:
        assert 0 <= offset <= self.max_offset
        return U64(self.base + self.guard + offset)

    def stage_input(self, data: np.ndarray, offset: int, poison_buf: np.ndarray):
        raw = np.ascontiguousarray(data).view(np.uint8).ravel()
        assert raw.nbytes == self.nbytes
        full = poison_buf[: self.total].copy()
        start = self.guard + offset
        full[start : start + self.nbytes] = raw
        self._last_host = full
        self.drv.htod_async(self.base, full)

    def stage_poison(self, poison_buf: np.ndarray):
        self._last_host = poison_buf
        self.drv.htod_async(self.base, poison_buf[: self.total])

    def stage_from_device(self, d_src: int):
        self.drv.dtod_async(self.base, d_src, self.total)

    def read_payload(self, offset: int = 0) -> np.ndarray:
        return self.drv.read_sync(self.base + self.guard + offset, self.nbytes)

    def check_guards(self, offset: int, poison_buf: np.ndarray) -> int:
        full = self.drv.read_sync(self.base, self.total)
        exp = poison_buf[: self.total]
        head_end = self.guard + offset
        tail_start = head_end + self.nbytes
        return int(np.count_nonzero(full[:head_end] != exp[:head_end])) + int(
            np.count_nonzero(full[tail_start:] != exp[tail_start:])
        )


def reconstruct_wp2_seed197():
    """Reconstruct exact RNG state of tools/fused_bench/bench.py (seed 197) at WP2 cases #31 and #57."""
    rng = np.random.default_rng(197)

    def bf(shape):
        return (rng.normal(size=shape).astype("f4").view("u4") >> 16).astype("u2")

    def sim_idx(rows, nb, dist="ties", pad=0):
        ss = 8 * nb + pad
        rng.integers(-10, 10, (rows, ss))
        if dist == "random":
            rng.normal(size=(rows, ss))

    idx_shapes = [
        (1, 8, "ties"), (7, 8, "ties"), (32, 8, "ties"), (48, 8, "ties"),
        (7, 4096, "random"), (32, 8192, "ties"),
        (128, 4096, "ties"), (1024, 4096, "ties"), (2048, 4096, "ties"),
        (7, 2049, "random"), (7, 8192, "flat"),
    ]
    for r, nb, d in idx_shapes:
        sim_idx(r, nb, d)
    for args in [
        (7, 8, "random"), (7, 8, "flat"), (128, 4096, "random"), (7, 4096, "random"),
        (7, 8193, "random"), (32, 8193, "ties"), (7, 16384, "random"), (32, 16384, "ties"),
        (7, 16385, "random"), (7, 4096, "random"), (7, 4096, "random"), (7, 8192, "random"),
        (32, 8192, "flat"), (128, 4096, "flat"), (32, 8192, "same"), (128, 4096, "same"),
    ]:
        sim_idx(*args)
    sim_idx(7, 4096, "random", pad=1)
    sim_idx(32, 8192, "ties", pad=3)

    wp2_idx_scores = None
    for rows in (1, 7):
        for nb in (2049, 4096, 8192):
            sc = rng.normal(size=(rows, 8 * nb)).astype("f4")
            if rows == 1 and nb == 8192:
                wp2_idx_scores = sc.copy()
    rng.integers(-10, 10, (7, 8 * 4096))

    for rows, nb in ((1, 4096), (7, 4096), (7, 8192), (32, 8192)):
        rng.normal(size=(rows, 8 * nb))
    for dist in ("random", "ties", "flat"):
        for _block in (256, 512, 1024):
            rng.integers(-10, 10, (7, 8 * 4096))
            if dist == "random":
                rng.normal(size=(7, 8 * 4096))
    for _block in (256, 512, 1024):
        rng.integers(-10, 10, (32, 8 * 8192))

    wp2_comp = None
    for rows in (1, 7, 32, 48):
        for d in (128, 512):
            n = rows * 2
            slots = rng.permutation(n)[:rows].astype("i8")
            pair = np.column_stack((slots, np.setdiff1d(np.arange(n), slots)))
            pair[1::2] = pair[1::2, ::-1]
            inp0 = rng.normal(size=(rows, d)).astype("f4")
            inp1 = rng.normal(size=(rows, d)).astype("f4")
            rk = rng.normal(size=(n, d)).astype("f4")
            rg = rng.normal(size=(n, d)).astype("f4")
            w = bf(d)
            if rows == 32 and d == 512:
                wp2_comp = (slots.copy(), np.ascontiguousarray(pair), inp0.copy(), inp1.copy(), rk.copy(), rg.copy(), w.copy())
    return wp2_idx_scores, wp2_comp


def diff_detail(actual: np.ndarray, expected: np.ndarray):
    mm = np.flatnonzero(actual != expected)
    if mm.size == 0:
        return None
    first = int(mm[0])
    lo = max(0, first - 4)
    hi = min(len(actual), first + 20)
    return {
        "diff_bytes": int(mm.size),
        "first_byte_offset": first,
        "actual_at_first": int(actual[first]),
        "expected_at_first": int(expected[first]),
        "actual_slice_hex": actual[lo:hi].tobytes().hex(),
        "expected_slice_hex": expected[lo:hi].tobytes().hex(),
        "slice_range": [lo, hi],
    }


def probe_idx_small(drv: Driver, bin_dir: Path, rows: int, nb: int, k: int, dist: str,
                    fixed_scores=None, iters: int = 100, logf=None):
    width, ss = 8 * nb - 3, 8 * nb
    label = f"idx-small R={rows} nb={nb} k={k} {dist}"
    rng = np.random.default_rng(1000 + rows * 37 + nb + sum( map(ord, dist) ))
    poison_rng = np.random.default_rng(9001)
    poisons = ["0xFF", "0x7FC00000", "0xDEADBEEF", "random"]
    offsets4 = [0, 4, 8, 12, 16, 20, 36, 68, 132, 260]
    offsets8 = [0, 8, 16, 24, 32, 40, 64, 128, 200, 264]

    if fixed_scores is not None:
        scores = fixed_scores.copy()
    elif dist == "random":
        scores = rng.normal(size=(rows, ss)).astype("f4")
    elif dist == "ties":
        scores = rng.integers(-6, 7, size=(rows, ss)).astype("f4")
    elif dist == "flat":
        scores = np.full((rows, ss), -np.inf, dtype="f4")
    elif dist == "adversarial":
        scores = rng.integers(-2, 3, size=(rows, ss)).astype("f4")
        special = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC01234, 0xFFC00001, 1], dtype="u4")
        scores.view("u4")[:, ::5] = rng.choice(special, size=scores[:, ::5].shape)
    else:
        raise ValueError(dist)

    vis = np.linspace(0, width, rows, dtype="i8")
    if dist == "adversarial" and rows == 1:
        vis = np.array([width // 2], dtype="i8")
    every = np.full(rows, 1 << 40, dtype="i8")

    # Pre-poison freed memory pool
    drv.poison_freelist("0xFF", poison_rng)

    poison_templates = {p: make_poison_bytes(rows * ss * 4 + 1024, p, poison_rng) for p in poisons}

    # Reference arm: old-ops block_max + topk_indices + pool_pick (with 0x55 initial output to match bench.py ref)
    old_ops = bin_dir / "old-ops.cubin"
    wp2_ops = bin_dir / "wp2-ops.cubin"
    new_ops = bin_dir / "new-ops.cubin"

    s_score = ArenaSlice(drv, scores.nbytes, max_offset=264)
    s_vis = ArenaSlice(drv, vis.nbytes, max_offset=264)
    s_every = ArenaSlice(drv, every.nbytes, max_offset=264)
    s_mx = ArenaSlice(drv, rows * nb * 4, max_offset=264)
    s_ix = ArenaSlice(drv, rows * k * 8, max_offset=264)
    s_bmax = ArenaSlice(drv, rows * nb * 4, max_offset=264)
    s_blocks = ArenaSlice(drv, rows * k * 4, max_offset=264)

    p55 = make_poison_bytes(scores.nbytes + 1024, "0x55", poison_rng)
    s_score.stage_input(scores, 0, p55)
    s_vis.stage_input(vis, 0, p55)
    s_every.stage_input(every, 0, p55)
    s_mx.stage_poison(p55)
    s_ix.stage_poison(p55)
    s_blocks.stage_poison(p55)

    drv.launch(old_ops, "tf_ds_block_max_kernel", ((nb + 255) // 256, rows), 256, 0,
               s_score.ptr_at(0), L(ss), width, 8, s_vis.ptr_at(0), s_mx.ptr_at(0), L(nb), nb, 0)
    drv.launch(old_ops, "tf_ds_topk_indices_kernel", rows, 1024, 0,
               s_mx.ptr_at(0), L(nb), nb, k, s_every.ptr_at(0), s_ix.ptr_at(0), 0)
    drv.launch(old_ops, "tf_ds_pool_pick_kernel", rows, 1024, 0,
               s_mx.ptr_at(0), L(nb), nb, s_ix.ptr_at(0), k, P(), L(0), s_blocks.ptr_at(0), 0)
    gold = s_blocks.read_payload(0).copy()

    host_sha = {
        "score": sha256_bytes(np.ascontiguousarray(scores).tobytes()),
        "vis": sha256_bytes(np.ascontiguousarray(vis).tobytes()),
        "every": sha256_bytes(np.ascontiguousarray(every).tobytes()),
    }

    case_report = {
        "case": label,
        "rows": rows,
        "nb": nb,
        "k": k,
        "dist": dist,
        "output_bytes": int(gold.nbytes),
        "input_sha256_host": host_sha,
        "arms": {},
    }

    for arm_name, cubin in (("old_wp1", old_ops), ("wp2", wp2_ops), ("new_tip", new_ops)):
        _, sym_bmax, sha_cubin = drv.get_fn(cubin, "tf_ds_block_max_kernel")
        _, sym_pick, _ = drv.get_fn(cubin, "tf_ds_idx_pick_kernel")
        arm_mismatches = []
        guard_corruptions = 0
        input_hash_mismatches = 0
        mode_counts = {}
        addr_samples = []

        # Stage inputs at rotated alignment for this arm and verify SHA-256 BEFORE
        in_off4 = 12 if arm_name == "new_tip" else (4 if arm_name == "wp2" else 0)
        in_off8 = 24 if arm_name == "new_tip" else (8 if arm_name == "wp2" else 0)
        p_init = poison_templates["0xDEADBEEF"]
        s_score.stage_input(scores, in_off4, p_init)
        s_vis.stage_input(vis, in_off8, p_init)
        s_every.stage_input(every, in_off8, p_init)
        sha_before = {
            "score": sha256_bytes(s_score.read_payload(in_off4).tobytes()),
            "vis": sha256_bytes(s_vis.read_payload(in_off8).tobytes()),
            "every": sha256_bytes(s_every.read_payload(in_off8).tobytes()),
        }
        if sha_before != host_sha:
            input_hash_mismatches += 1

        # 1) Eager pdl=0 and pdl=1: 100 iterations cycling poison patterns and output/scratch alignments
        for pdl in (0, 1):
            mode_key = f"eager_pdl{pdl}"
            fails = 0
            for it in range(iters):
                pat = poisons[it % len(poisons)]
                pbuf = poison_templates[pat]
                off_bmax = offsets4[it % len(offsets4)]
                off_out = offsets4[(it * 3 + 1) % len(offsets4)]
                s_bmax.stage_poison(pbuf)
                s_blocks.stage_poison(pbuf)
                drv.launch(cubin, "tf_ds_block_max_kernel", ((nb + 255) // 256, rows), 256, 0,
                           s_score.ptr_at(in_off4), L(ss), width, 8, s_vis.ptr_at(in_off8),
                           s_bmax.ptr_at(off_bmax), L(nb), nb, pdl, pdl=bool(pdl))
                drv.launch(cubin, "tf_ds_idx_pick_kernel", rows, 1024, nb * 4,
                           s_bmax.ptr_at(off_bmax), L(nb), nb, k, s_every.ptr_at(in_off8),
                           s_blocks.ptr_at(off_out), pdl, pdl=bool(pdl))
                actual = s_blocks.read_payload(off_out)
                if it < 2 and pdl == 0:
                    addr_samples.append({
                        "iter": it, "poison": pat,
                        "score_ptr": hex(s_score.ptr_at(in_off4).value),
                        "vis_ptr": hex(s_vis.ptr_at(in_off8).value),
                        "every_ptr": hex(s_every.ptr_at(in_off8).value),
                        "bmax_ptr": hex(s_bmax.ptr_at(off_bmax).value),
                        "cblk_ptr": hex(s_blocks.ptr_at(off_out).value),
                    })
                d = diff_detail(actual, gold)
                if d is not None:
                    fails += 1
                    if len(arm_mismatches) < 5:
                        d.update({"mode": mode_key, "iter": it, "poison": pat,
                                  "bmax_ptr": hex(s_bmax.ptr_at(off_bmax).value),
                                  "cblk_ptr": hex(s_blocks.ptr_at(off_out).value)})
                        arm_mismatches.append(d)
                if it % 25 == 0:
                    guard_corruptions += s_blocks.check_guards(off_out, pbuf)
            mode_counts[mode_key] = {"iters": iters, "mismatches": fails}

        # 2) CUDA-graph replay pdl=0 and pdl=1: 100 replays each (graph includes DtoD poison + both kernels)
        d_poison_src = drv.alloc(max(s_bmax.total, s_blocks.total))
        for pdl in (0, 1):
            mode_key = f"graph_pdl{pdl}"
            pat = poisons[pdl]
            pbuf = poison_templates[pat]
            off_bmax = offsets4[2 + pdl]
            off_out = offsets4[5 + pdl]
            drv.htod_async(d_poison_src, pbuf[: max(s_bmax.total, s_blocks.total)])
            drv.sync()

            drv.cu("cuStreamBeginCapture_v2", drv.stream, 0)
            s_bmax.stage_from_device(d_poison_src)
            s_blocks.stage_from_device(d_poison_src)
            drv.launch(cubin, "tf_ds_block_max_kernel", ((nb + 255) // 256, rows), 256, 0,
                       s_score.ptr_at(in_off4), L(ss), width, 8, s_vis.ptr_at(in_off8),
                       s_bmax.ptr_at(off_bmax), L(nb), nb, pdl, pdl=bool(pdl))
            drv.launch(cubin, "tf_ds_idx_pick_kernel", rows, 1024, nb * 4,
                       s_bmax.ptr_at(off_bmax), L(nb), nb, k, s_every.ptr_at(in_off8),
                       s_blocks.ptr_at(off_out), pdl, pdl=bool(pdl))
            g, exe = P(), P()
            drv.cu("cuStreamEndCapture", drv.stream, ref(g))
            drv.cu("cuGraphInstantiateWithFlags", ref(exe), g, U64(0))
            drv.cu("cuGraphDestroy", g)

            fails = 0
            for it in range(iters):
                drv.cu("cuGraphLaunch", exe, drv.stream)
                actual = s_blocks.read_payload(off_out)
                d = diff_detail(actual, gold)
                if d is not None:
                    fails += 1
                    if len(arm_mismatches) < 5:
                        d.update({"mode": mode_key, "iter": it, "poison": pat})
                        arm_mismatches.append(d)
            drv.cu("cuGraphExecDestroy", exe)
            mode_counts[mode_key] = {"iters": iters, "mismatches": fails}
        drv.free(d_poison_src)

        sha_after = {
            "score": sha256_bytes(s_score.read_payload(in_off4).tobytes()),
            "vis": sha256_bytes(s_vis.read_payload(in_off8).tobytes()),
            "every": sha256_bytes(s_every.read_payload(in_off8).tobytes()),
        }
        if sha_after != sha_before:
            input_hash_mismatches += 1

        case_report["arms"][arm_name] = {
            "cubin": cubin.name,
            "cubin_sha256": sha_cubin,
            "resolved_symbols": [sym_bmax, sym_pick],
            "input_sha256_before": sha_before,
            "input_sha256_after": sha_after,
            "input_sha256_unchanged": (sha_before == sha_after == host_sha),
            "guard_byte_corruptions": guard_corruptions,
            "address_samples": addr_samples,
            "modes": mode_counts,
            "mismatches": arm_mismatches,
        }
        msg = f"[idx] {label:<42} arm={arm_name:<8} modes={mode_counts} in_sha_ok={sha_before == sha_after == host_sha}"
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")

    drv.free_all()
    return case_report


def probe_compress_decode(drv: Driver, bin_dir: Path, kit_dir: Path, rows: int, d: int, layout: str,
                          fixed_comp=None, iters: int = 100, logf=None):
    n = rows * 2
    stride = d + (1 if layout == "adversarial" else 0)
    ldo = d + 16
    label = f"compress-decode R={rows} D={d} {layout}"
    rng = np.random.default_rng(2000 + rows * 53 + d + sum(map(ord, layout)))
    poison_rng = np.random.default_rng(9002)
    poisons = ["0xFF", "0x7FC00000", "0xDEADBEEF", "random"]
    offsets2 = [0, 2, 6, 8, 14, 16, 24, 32, 66, 130]
    offsets4 = [0, 4, 8, 12, 16, 20, 36, 68, 132, 260]
    offsets8 = [0, 8, 16, 24, 32, 40, 64, 128, 200, 264]

    if fixed_comp is not None:
        slots, pair, ckv, cgate, ring_kv0, ring_sc0, w = [a.copy() for a in fixed_comp]
    else:
        slots = rng.permutation(n)[:rows].astype("i8")
        pair = np.ascontiguousarray(np.column_stack((slots, np.setdiff1d(np.arange(n), slots))))
        pair[1::2] = pair[1::2, ::-1]
        if layout == "adversarial":
            pair[::5, 1] = -1
            pair[::7, 0] = n
        ckv = rng.normal(size=(rows, stride)).astype("f4")
        cgate = rng.normal(size=(rows, stride)).astype("f4")
        ring_kv0 = rng.normal(size=(n, d)).astype("f4")
        ring_sc0 = rng.normal(size=(n, d)).astype("f4")
        if layout == "adversarial":
            special = np.array([0, 0x80000000, 1, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F7FFFFF, 0x00800000], dtype="u4")
            cgate.view("u4")[:, ::7] = np.resize(special, cgate[:, ::7].shape)
            ring_sc0.view("u4")[:, ::7] = np.resize(special, ring_sc0[:, ::7].shape)
            ckv.view("u4")[:, ::5] = np.resize(special, ckv[:, ::5].shape)
        w = (rng.normal(size=d).astype("f4").view("u4") >> 16).astype("u2")

    drv.poison_freelist("0x7FC00000", poison_rng)

    max_bytes = max(ring_kv0.nbytes, ckv.nbytes, rows * ldo * 2) + 1024
    poison_templates = {p: make_poison_bytes(max_bytes, p, poison_rng) for p in poisons}
    p55 = make_poison_bytes(max_bytes, "0x55", poison_rng)

    old_torch = bin_dir / "old-torch.cubin"
    wp2_torch = bin_dir / "wp2-torch.cubin"
    new_torch = bin_dir / "new-torch.cubin"
    (kit_norm,) = list(kit_dir.glob(("0cca97" if d == 128 else "ee9e02") + "*.cubin"))

    s_ckv = ArenaSlice(drv, ckv.nbytes, max_offset=264)
    s_cgate = ArenaSlice(drv, cgate.nbytes, max_offset=264)
    s_slot = ArenaSlice(drv, slots.nbytes, max_offset=264)
    s_gi = ArenaSlice(drv, pair.nbytes, max_offset=264)
    s_w = ArenaSlice(drv, w.nbytes, max_offset=264)
    s_rk = ArenaSlice(drv, ring_kv0.nbytes, max_offset=264)
    s_rg = ArenaSlice(drv, ring_sc0.nbytes, max_offset=264)
    s_mid = ArenaSlice(drv, rows * d * 2, max_offset=264)
    s_dst = ArenaSlice(drv, rows * ldo * 2, max_offset=264)
    s_flag = ArenaSlice(drv, 4, max_offset=264)

    # Compute reference output once per poison pattern for active + pad bytes?
    # Note: the 16 stride-pad uint16s per row in dst are NEVER written by kit _rmsnorm or compress2x_norm512,
    # so they preserve whatever pre-poison pattern dst started with! Thus for exact whole-buffer parity under
    # poison pattern `pat`, expected dst has active [:, :d] from kit _rmsnorm and pad [:, d:] from `pat`!
    s_ckv.stage_input(ckv, 0, p55)
    s_cgate.stage_input(cgate, 0, p55)
    s_slot.stage_input(slots, 0, p55)
    s_gi.stage_input(pair, 0, p55)
    s_w.stage_input(w, 0, p55)
    s_rk.stage_input(ring_kv0, 0, p55)
    s_rg.stage_input(ring_sc0, 0, p55)
    s_mid.stage_poison(p55)
    s_dst.stage_poison(p55)
    s_flag.stage_input(np.zeros(1, dtype=np.uint32), 0, p55)

    eps = F(1e-20)
    drv.launch(old_torch, "tf_ds_compress2x_kernel", rows, 256, 0,
               s_ckv.ptr_at(0), L(stride), s_cgate.ptr_at(0), L(stride), s_slot.ptr_at(0),
               s_rk.ptr_at(0), s_rg.ptr_at(0), s_gi.ptr_at(0), L(n), s_mid.ptr_at(0), L(d), rows, d, s_flag.ptr_at(0), 0)
    drv.launch(kit_norm, "_rmsnorm", rows, 128, 16,
               s_mid.ptr_at(0), d, s_w.ptr_at(0), s_dst.ptr_at(0), ldo, eps, P(), P())
    gold_dst_2d = s_dst.read_payload(0).view(np.uint16).reshape(rows, ldo).copy()
    gold_rk = s_rk.read_payload(0).copy()
    gold_rg = s_rg.read_payload(0).copy()
    gold_flag = s_flag.read_payload(0).copy()

    def expected_dst_for_poison(pbuf: np.ndarray) -> np.ndarray:
        base = pbuf[: rows * ldo * 2].copy().view(np.uint16).reshape(rows, ldo)
        base[:, :d] = gold_dst_2d[:, :d]
        return base.view(np.uint8).ravel()

    host_sha = {
        "ckv": sha256_bytes(np.ascontiguousarray(ckv).tobytes()),
        "cgate": sha256_bytes(np.ascontiguousarray(cgate).tobytes()),
        "slot": sha256_bytes(np.ascontiguousarray(slots).tobytes()),
        "gi": sha256_bytes(np.ascontiguousarray(pair).tobytes()),
        "w": sha256_bytes(np.ascontiguousarray(w).tobytes()),
    }

    case_report = {
        "case": label,
        "rows": rows,
        "d": d,
        "layout": layout,
        "output_bytes": int(rows * ldo * 2),
        "input_sha256_host": host_sha,
        "arms": {},
    }

    arms = [
        ("old_wp1", old_torch, "tf_ds_compress2x_norm_kernel"),
        ("wp2", wp2_torch, "tf_ds_compress2x_norm128_kernel" if d == 128 else "tf_ds_compress2x_norm512_kernel"),
        ("new_tip", new_torch, "tf_ds_compress2x_norm128_kernel" if d == 128 else "tf_ds_compress2x_norm512_kernel"),
    ]

    for arm_name, cubin, sym_req in arms:
        _, sym_resolved, sha_cubin = drv.get_fn(cubin, sym_req)
        arm_mismatches = []
        guard_corruptions = 0
        mode_counts = {}
        addr_samples = []

        in_off2 = 6 if arm_name == "new_tip" else (2 if arm_name == "wp2" else 0)
        in_off4 = 12 if arm_name == "new_tip" else (4 if arm_name == "wp2" else 0)
        in_off8 = 24 if arm_name == "new_tip" else (8 if arm_name == "wp2" else 0)
        p_init = poison_templates["0xDEADBEEF"]

        s_ckv.stage_input(ckv, in_off4, p_init)
        s_cgate.stage_input(cgate, in_off4, p_init)
        s_slot.stage_input(slots, in_off8, p_init)
        s_gi.stage_input(pair, in_off8, p_init)
        s_w.stage_input(w, in_off2, p_init)

        sha_before = {
            "ckv": sha256_bytes(s_ckv.read_payload(in_off4).tobytes()),
            "cgate": sha256_bytes(s_cgate.read_payload(in_off4).tobytes()),
            "slot": sha256_bytes(s_slot.read_payload(in_off8).tobytes()),
            "gi": sha256_bytes(s_gi.read_payload(in_off8).tobytes()),
            "w": sha256_bytes(s_w.read_payload(in_off2).tobytes()),
        }

        # 1) Eager pdl=0 and pdl=1: 100 iterations each
        for pdl in (0, 1):
            mode_key = f"eager_pdl{pdl}"
            fails = 0
            for it in range(iters):
                pat = poisons[it % len(poisons)]
                pbuf = poison_templates[pat]
                off_dst = offsets2[it % len(offsets2)]
                off_ring = offsets4[(it * 3) % len(offsets4)]
                off_flag = offsets4[(it * 5) % len(offsets4)]

                s_rk.stage_input(ring_kv0, off_ring, pbuf)
                s_rg.stage_input(ring_sc0, off_ring, pbuf)
                s_dst.stage_poison(pbuf)
                s_flag.stage_input(np.zeros(1, dtype=np.uint32), off_flag, pbuf)

                drv.launch(cubin, sym_req, rows, d, 0,
                           s_ckv.ptr_at(in_off4), L(stride), s_cgate.ptr_at(in_off4), L(stride),
                           s_slot.ptr_at(in_off8), s_rk.ptr_at(off_ring), s_rg.ptr_at(off_ring),
                           s_gi.ptr_at(in_off8), L(n), s_w.ptr_at(in_off2), s_dst.ptr_at(off_dst),
                           L(ldo), rows, d, eps, s_flag.ptr_at(off_flag), pdl, pdl=bool(pdl))

                exp_dst = expected_dst_for_poison(pbuf[s_dst.guard + off_dst :])
                act_dst = s_dst.read_payload(off_dst)
                act_rk = s_rk.read_payload(off_ring)
                act_rg = s_rg.read_payload(off_ring)
                act_flag = s_flag.read_payload(off_flag)

                if it < 2 and pdl == 0:
                    addr_samples.append({
                        "iter": it, "poison": pat,
                        "ckv_ptr": hex(s_ckv.ptr_at(in_off4).value),
                        "w_ptr": hex(s_w.ptr_at(in_off2).value),
                        "rk_ptr": hex(s_rk.ptr_at(off_ring).value),
                        "dst_ptr": hex(s_dst.ptr_at(off_dst).value),
                    })

                for bidx, (act, exp) in enumerate([(act_dst, exp_dst), (act_rk, gold_rk), (act_rg, gold_rg), (act_flag, gold_flag)]):
                    dd = diff_detail(act, exp)
                    if dd is not None:
                        fails += 1
                        if len(arm_mismatches) < 5:
                            dd.update({"mode": mode_key, "iter": it, "buffer": bidx, "poison": pat,
                                       "dst_ptr": hex(s_dst.ptr_at(off_dst).value)})
                            arm_mismatches.append(dd)
                        break
                if it % 25 == 0:
                    guard_corruptions += s_dst.check_guards(off_dst, pbuf)
            mode_counts[mode_key] = {"iters": iters, "mismatches": fails}

        # 2) CUDA-graph replay pdl=0 and pdl=1: 100 replays each (graph includes DtoD poison + kernel)
        d_poison_src = drv.alloc(s_dst.total)
        d_zero_flag = drv.alloc(s_flag.total)
        for pdl in (0, 1):
            mode_key = f"graph_pdl{pdl}"
            pat = poisons[pdl]
            pbuf = poison_templates[pat]
            off_dst = offsets2[3 + pdl]
            off_ring = offsets4[2 + pdl]
            off_flag = offsets4[4 + pdl]
            exp_dst = expected_dst_for_poison(pbuf[s_dst.guard + off_dst :])

            s_rk.stage_input(ring_kv0, off_ring, pbuf)
            s_rg.stage_input(ring_sc0, off_ring, pbuf)
            drv.htod_async(d_poison_src, pbuf[: s_dst.total])
            flag_full = pbuf[: s_flag.total].copy()
            flag_full[s_flag.guard + off_flag : s_flag.guard + off_flag + 4] = 0
            drv.htod_async(d_zero_flag, flag_full)
            drv.sync()

            drv.cu("cuStreamBeginCapture_v2", drv.stream, 0)
            s_dst.stage_from_device(d_poison_src)
            s_flag.stage_from_device(d_zero_flag)
            drv.launch(cubin, sym_req, rows, d, 0,
                       s_ckv.ptr_at(in_off4), L(stride), s_cgate.ptr_at(in_off4), L(stride),
                       s_slot.ptr_at(in_off8), s_rk.ptr_at(off_ring), s_rg.ptr_at(off_ring),
                       s_gi.ptr_at(in_off8), L(n), s_w.ptr_at(in_off2), s_dst.ptr_at(off_dst),
                       L(ldo), rows, d, eps, s_flag.ptr_at(off_flag), pdl, pdl=bool(pdl))
            g, exe = P(), P()
            drv.cu("cuStreamEndCapture", drv.stream, ref(g))
            drv.cu("cuGraphInstantiateWithFlags", ref(exe), g, U64(0))
            drv.cu("cuGraphDestroy", g)

            fails = 0
            for it in range(iters):
                drv.cu("cuGraphLaunch", exe, drv.stream)
                act_dst = s_dst.read_payload(off_dst)
                dd = diff_detail(act_dst, exp_dst)
                if dd is not None:
                    fails += 1
                    if len(arm_mismatches) < 5:
                        dd.update({"mode": mode_key, "iter": it, "buffer": 0, "poison": pat})
                        arm_mismatches.append(dd)
            # Also check rings & flag after 100 graph replays
            for bidx, (act, exp) in enumerate([(s_rk.read_payload(off_ring), gold_rk),
                                               (s_rg.read_payload(off_ring), gold_rg),
                                               (s_flag.read_payload(off_flag), gold_flag)], start=1):
                dd = diff_detail(act, exp)
                if dd is not None:
                    fails += 1
                    if len(arm_mismatches) < 5:
                        dd.update({"mode": mode_key, "iter": iters - 1, "buffer": bidx})
                        arm_mismatches.append(dd)
            drv.cu("cuGraphExecDestroy", exe)
            mode_counts[mode_key] = {"iters": iters, "mismatches": fails}
        drv.free(d_poison_src)
        drv.free(d_zero_flag)

        sha_after = {
            "ckv": sha256_bytes(s_ckv.read_payload(in_off4).tobytes()),
            "cgate": sha256_bytes(s_cgate.read_payload(in_off4).tobytes()),
            "slot": sha256_bytes(s_slot.read_payload(in_off8).tobytes()),
            "gi": sha256_bytes(s_gi.read_payload(in_off8).tobytes()),
            "w": sha256_bytes(s_w.read_payload(in_off2).tobytes()),
        }

        case_report["arms"][arm_name] = {
            "cubin": cubin.name,
            "cubin_sha256": sha_cubin,
            "resolved_symbols": [sym_resolved],
            "input_sha256_before": sha_before,
            "input_sha256_after": sha_after,
            "input_sha256_unchanged": (sha_before == sha_after == host_sha),
            "guard_byte_corruptions": guard_corruptions,
            "address_samples": addr_samples,
            "modes": mode_counts,
            "mismatches": arm_mismatches,
        }
        msg = f"[cmp] {label:<42} arm={arm_name:<8} modes={mode_counts} in_sha_ok={sha_before == sha_after == host_sha}"
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")

    drv.free_all()
    return case_report


def run_stream0_race_repro(drv: Driver, bin_dir: Path, kit_dir: Path, wp2_idx_scores, wp2_comp, logf=None):
    """Controlled experimental proof of the bench.py legacy-stream-0 cuMemcpyHtoD_v2 (<= 64 KiB pageable)
    vs CU_STREAM_NON_BLOCKING race on GB10, and verification that cuMemcpyHtoDAsync_v2(..., stream) fixes it."""
    floor_cubin = bin_dir / "floor.cubin"
    old_ops = bin_dir / "old-ops.cubin"
    new_ops = bin_dir / "new-ops.cubin"
    old_torch = bin_dir / "old-torch.cubin"
    new_torch = bin_dir / "new-torch.cubin"
    (kit_norm,) = list(kit_dir.glob("ee9e02*.cubin"))

    d_sink = drv.alloc(16)
    repro = {"idx_small_R1_nb8192": {}, "compress_decode_R32_D512": {}}

    # 1) idx-small R=1 nb=8192 (exact WP2 seed 197 input)
    rows, nb, k = 1, 8192, 2048
    width, ss = 8 * nb - 3, 8 * nb
    vis = np.linspace(0, width, rows, dtype="i8")
    every = np.full(rows, 1 << 40, dtype="i8")
    d_x = drv.alloc(wp2_idx_scores.nbytes)
    d_vis = drv.alloc(vis.nbytes)
    d_every = drv.alloc(every.nbytes)
    d_mx = drv.alloc(rows * nb * 4)
    d_ix = drv.alloc(rows * k * 8)
    d_gold = drv.alloc(rows * k * 4)
    drv.htod_async(d_x, wp2_idx_scores)
    drv.htod_async(d_vis, vis)
    drv.htod_async(d_every, every)
    drv.launch(old_ops, "tf_ds_block_max_kernel", ((nb + 255) // 256, rows), 256, 0,
               U64(d_x), L(ss), width, 8, U64(d_vis), U64(d_mx), L(nb), nb, 0)
    drv.launch(old_ops, "tf_ds_topk_indices_kernel", rows, 1024, 0,
               U64(d_mx), L(nb), nb, k, U64(d_every), U64(d_ix), 0)
    drv.launch(old_ops, "tf_ds_pool_pick_kernel", rows, 1024, 0,
               U64(d_mx), L(nb), nb, U64(d_ix), k, P(), L(0), U64(d_gold), 0)
    gold_idx = drv.read_sync(d_gold, rows * k * 4).copy()

    host_mx_55 = np.full((rows, nb), 0x55555555, dtype=np.uint32)
    host_ix_55 = np.full((rows, k), 0x5555555555555555, dtype=np.uint64)
    host_bmax_55 = np.full((rows, nb), 0x55555555, dtype=np.uint32)
    host_blk_55 = np.full((rows, k), 0x55555555, dtype=np.uint32)
    d_bmax = drv.alloc(host_bmax_55.nbytes)   # 32,768 B <= 64 KiB!
    d_blk = drv.alloc(host_blk_55.nbytes)     #  8,192 B <= 64 KiB!
    n_trials = 200

    for upload_mode in ("bench_py_stream0_cuMemcpyHtoD_v2", "fixed_stream_cuMemcpyHtoDAsync_v2"):
        diffs = []
        for trial in range(n_trials):
            if upload_mode == "bench_py_stream0_cuMemcpyHtoD_v2":
                # Exact unmodified bench.py case_idx_small sequence: 4 pageable <=64KiB HtoDs on stream 0,
                # followed immediately by kernel launch on CU_STREAM_NON_BLOCKING stream!
                drv.cu("cuMemcpyHtoD_v2", U64(d_mx), P(host_mx_55.ctypes.data), ct.c_size_t(host_mx_55.nbytes))
                drv.cu("cuMemcpyHtoD_v2", U64(d_ix), P(host_ix_55.ctypes.data), ct.c_size_t(host_ix_55.nbytes))
                drv.cu("cuMemcpyHtoD_v2", U64(d_blk), P(host_blk_55.ctypes.data), ct.c_size_t(host_blk_55.nbytes))
                drv.cu("cuMemcpyHtoD_v2", U64(d_bmax), P(host_bmax_55.ctypes.data), ct.c_size_t(host_bmax_55.nbytes))
            else:
                drv.htod_async(d_mx, host_mx_55)
                drv.htod_async(d_ix, host_ix_55)
                drv.htod_async(d_blk, host_blk_55)
                drv.htod_async(d_bmax, host_bmax_55)
            drv.launch(new_ops, "tf_ds_block_max_kernel", ((nb + 255) // 256, rows), 256, 0,
                       U64(d_x), L(ss), width, 8, U64(d_vis), U64(d_bmax), L(nb), nb, 0)
            drv.launch(new_ops, "tf_ds_idx_pick_kernel", rows, 1024, nb * 4,
                       U64(d_bmax), L(nb), nb, k, U64(d_every), U64(d_blk), 0)
            drv.sync()
            drv.cu("cuStreamSynchronize", P(0))
            act = drv.read_sync(d_blk, host_blk_55.nbytes)
            dd = diff_detail(act, gold_idx)
            if dd is not None:
                diffs.append({"trial": trial, "diff_bytes": dd["diff_bytes"], "first_offset": dd["first_byte_offset"]})
        repro["idx_small_R1_nb8192"][upload_mode] = {
            "trials": n_trials, "failures": len(diffs), "samples": diffs[:10]
        }

    # 2) compress-decode R=32 D=512 mixed (exact WP2 seed 197 input)
    slots, pair, ckv, cgate, rk0, rg0, w = wp2_comp
    rows, d, n, ldo = 32, 512, 64, 528
    eps = F(1e-20)
    d_ckv = drv.alloc(ckv.nbytes)
    d_cgate = drv.alloc(cgate.nbytes)
    d_slot = drv.alloc(slots.nbytes)
    d_gi = drv.alloc(pair.nbytes)
    d_w = drv.alloc(w.nbytes)
    d_rk = drv.alloc(rk0.nbytes)
    d_rg = drv.alloc(rg0.nbytes)
    d_mid = drv.alloc(rows * d * 2)
    d_dst = drv.alloc(rows * ldo * 2)  # 33,792 B <= 64 KiB!
    d_flag = drv.alloc(4)
    host_flag_0 = np.zeros(1, dtype=np.uint32)
    host_mid_55 = np.full((rows, d), 0x5555, dtype=np.uint16)
    host_dst_55 = np.full((rows, ldo), 0x5555, dtype=np.uint16)

    drv.htod_async(d_ckv, ckv)
    drv.htod_async(d_cgate, cgate)
    drv.htod_async(d_slot, slots)
    drv.htod_async(d_gi, pair)
    drv.htod_async(d_w, w)
    drv.htod_async(d_rk, rk0)
    drv.htod_async(d_rg, rg0)
    drv.htod_async(d_dst, host_dst_55)
    drv.htod_async(d_flag, host_flag_0)
    drv.launch(old_torch, "tf_ds_compress2x_kernel", rows, 256, 0,
               U64(d_ckv), L(d), U64(d_cgate), L(d), U64(d_slot), U64(d_rk), U64(d_rg),
               U64(d_gi), L(n), U64(d_mid), L(d), rows, d, U64(d_flag), 0)
    drv.launch(kit_norm, "_rmsnorm", rows, 128, 16,
               U64(d_mid), d, U64(d_w), U64(d_dst), ldo, eps, P(), P())
    gold_cmp = drv.read_sync(d_dst, host_dst_55.nbytes).copy()

    for upload_mode in ("bench_py_stream0_cuMemcpyHtoD_v2", "fixed_stream_cuMemcpyHtoDAsync_v2"):
        diffs = []
        for trial in range(n_trials):
            drv.htod_async(d_rk, rk0)
            drv.htod_async(d_rg, rg0)
            drv.sync()
            if upload_mode == "bench_py_stream0_cuMemcpyHtoD_v2":
                # Exact unmodified bench.py case_compress_decode sequence: 3 pageable <=64KiB HtoDs on stream 0,
                # followed immediately by compress2x_norm512 launch on CU_STREAM_NON_BLOCKING stream!
                drv.cu("cuMemcpyHtoD_v2", U64(d_flag), P(host_flag_0.ctypes.data), ct.c_size_t(host_flag_0.nbytes))
                drv.cu("cuMemcpyHtoD_v2", U64(d_mid), P(host_mid_55.ctypes.data), ct.c_size_t(host_mid_55.nbytes))
                drv.cu("cuMemcpyHtoD_v2", U64(d_dst), P(host_dst_55.ctypes.data), ct.c_size_t(host_dst_55.nbytes))
            else:
                drv.htod_async(d_flag, host_flag_0)
                drv.htod_async(d_mid, host_mid_55)
                drv.htod_async(d_dst, host_dst_55)
            drv.launch(new_torch, "tf_ds_compress2x_norm512_kernel", rows, d, 0,
                       U64(d_ckv), L(d), U64(d_cgate), L(d), U64(d_slot), U64(d_rk), U64(d_rg),
                       U64(d_gi), L(n), U64(d_w), U64(d_dst), L(ldo), rows, d, eps, U64(d_flag), 0)
            drv.sync()
            drv.cu("cuStreamSynchronize", P(0))
            act = drv.read_sync(d_dst, host_dst_55.nbytes)
            dd = diff_detail(act, gold_cmp)
            if dd is not None:
                diffs.append({"trial": trial, "diff_bytes": dd["diff_bytes"], "first_offset": dd["first_byte_offset"]})
        repro["compress_decode_R32_D512"][upload_mode] = {
            "trials": n_trials, "failures": len(diffs), "samples": diffs[:10]
        }

    drv.free_all()
    msg = f"[stream0-repro] {repro}"
    print(msg.replace("{", "(").replace("}", ")"), flush=True)
    if logf:
        logf.write(json.dumps(repro) + "\n")
    return repro


def load_or_build_wp2_seed197(cache_path: Path):
    if cache_path.exists():
        z = np.load(cache_path)
        wp2_comp = (z["slots"], z["pair"], z["inp0"], z["inp1"], z["rk"], z["rg"], z["w"])
        return z["idx_scores"], wp2_comp
    idx_scores, wp2_comp = reconstruct_wp2_seed197()
    slots, pair, inp0, inp1, rk, rg, w = wp2_comp
    np.savez(cache_path, idx_scores=idx_scores, slots=slots, pair=pair, inp0=inp0, inp1=inp1, rk=rk, rg=rg, w=w)
    return idx_scores, wp2_comp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path("bin"))
    parser.add_argument("--kit-dir", type=Path, default=Path("/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--suite", choices=("all", "idx", "compress", "stream0-repro", "precompute-cpu"), default="all")
    parser.add_argument("--repro-stream0-race", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "wp2_seed197_inputs.npz"
    wp2_idx_scores, wp2_comp = load_or_build_wp2_seed197(cache_path)
    if args.suite == "precompute-cpu":
        print(f"Precomputed CPU-only {cache_path}")
        return

    pid = os.getpid()
    tag = args.suite.replace("-", "_")
    log_path = args.out_dir / f"b2_probe_{tag}_pid{pid}.log"
    json_path = args.out_dir / f"b2_probe_{tag}_pid{pid}.json"

    t0 = time.time()
    avail_start = mem_available_gib()
    drv = Driver()

    cases = []
    stream0_repro = None
    with open(log_path, "w") as logf:
        hdr = f"PID={pid} suite={args.suite} start_MemAvailable={avail_start:.2f}GiB iters={args.iters}"
        print(hdr, flush=True)
        logf.write(hdr + "\n")

        if args.suite in ("all", "idx"):
            # idx-small R=1 nb=8192 (wp2_seed197, random, ties, flat, adversarial) + R=7 nb=8192
            cases.append(probe_idx_small(drv, args.bin_dir, 1, 8192, 2048, "wp2_seed197_random",
                                         fixed_scores=wp2_idx_scores, iters=args.iters, logf=logf))
            for dist in ("random", "ties", "flat", "adversarial"):
                cases.append(probe_idx_small(drv, args.bin_dir, 1, 8192, 2048, dist, iters=args.iters, logf=logf))
            cases.append(probe_idx_small(drv, args.bin_dir, 7, 8192, 2048, "random", iters=args.iters, logf=logf))

        if args.suite in ("all", "compress"):
            # compress-decode D=512 mixed (wp2_seed197 R=32 + R=1,7,32,48 + R=32 adversarial)
            cases.append(probe_compress_decode(drv, args.bin_dir, args.kit_dir, 32, 512, "wp2_seed197_mixed",
                                               fixed_comp=wp2_comp, iters=args.iters, logf=logf))
            for r in (1, 7, 32, 48):
                cases.append(probe_compress_decode(drv, args.bin_dir, args.kit_dir, r, 512, "mixed",
                                                   iters=args.iters, logf=logf))
            cases.append(probe_compress_decode(drv, args.bin_dir, args.kit_dir, 32, 512, "adversarial",
                                               iters=args.iters, logf=logf))

        if args.repro_stream0_race or args.suite == "stream0-repro":
            stream0_repro = run_stream0_race_repro(drv, args.bin_dir, args.kit_dir, wp2_idx_scores, wp2_comp, logf=logf)

        elapsed = time.time() - t0
        total_mismatches = sum(
            m["mismatches"]
            for c in cases
            for arm in c["arms"].values()
            for m in arm["modes"].values()
        )
        all_sha_ok = all(
            arm["input_sha256_unchanged"] and arm["guard_byte_corruptions"] == 0
            for c in cases
            for arm in c["arms"].values()
        )
        summary = {
            "pid": pid,
            "elapsed_sec": round(elapsed, 3),
            "mem_available_gib_start": round(avail_start, 3),
            "mem_available_gib_end": round(mem_available_gib(), 3),
            "peak_cuda_alloc_bytes": drv.peak_bytes,
            "peak_cuda_alloc_mib": round(drv.peak_bytes / (1024 * 1024), 3),
            "total_cases": len(cases),
            "iters_per_mode": args.iters,
            "total_kernel_parity_mismatches": total_mismatches,
            "all_input_sha256_and_guards_clean": all_sha_ok,
        }
        out_doc = {"summary": summary, "cases": cases}
        if stream0_repro is not None:
            out_doc["stream0_race_repro"] = stream0_repro
        json_path.write_text(json.dumps(out_doc, indent=2))
        ftr = f"DONE pid={pid} elapsed={elapsed:.2f}s peak_vram={summary['peak_cuda_alloc_mib']:.2f}MiB mismatches={total_mismatches} sha_ok={all_sha_ok} -> {json_path}"
        print(ftr, flush=True)
        logf.write(ftr + "\n")

    drv.close()
    if total_mismatches != 0 or not all_sha_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
