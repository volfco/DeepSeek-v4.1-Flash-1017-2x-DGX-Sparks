#!/usr/bin/env python3
"""0231 Multi-Chunk Stateful GPU Byte-Parity Verification.

Compares the unfused vs fused ratio-2 prompt KV-source step (0231) across a
stateful 4-chunk progression on GB10 (sm_121):
  1. start=0,  n=50  (even, 25 pairs, raw ring 0..49)
  2. start=50, n=1   (no pair: full=0, scatter fallback on both arms, slot 50)
  3. start=51, n=19  (odd + wraps 64: full=9, grid_rows=10, slots 51..63 & 0..5,
                      rmsLanes(512,9)=64 multi-warp norm, trailing block g=9)
  4. start=70, n=131 (odd + raw_first=67>0 + wraps 64: full=65, grid_rows=66,
                      pair g=33 split at raw_first=67, slots 9..63 & 0..8,
                      trailing block g=65)

Checks byte-for-byte equality after EVERY chunk across:
  - raw_kv[64, 512] (fp32)      + CPU modular-ring ground truth
  - raw_score[64, 512] (fp32)   + CPU modular-ring ground truth
  - compressed lat ([full, 512] active + full [128, 512] bf16 buffer)
  - idx_codes[128, 64] (u8) & idx_scales[128, 4] (u8)
  - comp_codes[128, 256] (u8) & comp_scales[128, 32] (u8)

Safety invariants:
  - Checks MemAvailable >= 3.5 GiB before initializing CUDA.
  - Enforces total VRAM < 256 MiB.
  - Uses a single CU_STREAM_NON_BLOCKING stream for all async HtoD/DtoH/memset/kernel
    calls and stream synchronization (never mixes stream 0 cuMemcpyHtoD_v2).
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
U64 = ct.c_uint64
ref = ct.byref

MAX_VRAM_BYTES = 256 * 1024 * 1024
MIN_MEM_AVAIL_GIB = 3.5
CU_STREAM_NON_BLOCKING = 1
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6

RAW_ROWS = 64
HD = 512
ID_DIM = 128
RD = 64
RATIO = 2
EPS = 1e-6
CAP_GROUPS = 128
MAX_POS = 256


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


def last_pow2(n: int) -> int:
    return 0 if n == 0 else (1 << (n.bit_length() - 1))


def rms_lanes(d: int, rows: int) -> int:
    dim0 = d // 4
    d0 = last_pow2(dim0) if dim0 < 512 else 512
    d1 = last_pow2(rows) if rows < 512 else 512
    bw = min(d0, 32)
    bh = min(d1, 512 // bw)
    return min(d0, 512 // bh)


def pack_lanes(lanes: int, half: int) -> int:
    if lanes == 0 or half == 0 or half % lanes != 0:
        return 1
    most = max(1, 512 // lanes)
    y = min(half // lanes, most)
    while y > 1 and half % (lanes * y) != 0:
        y -= 1
    return y


def launch_blocks(count: int, threads: int) -> int:
    return min((count + threads - 1) // threads, 65535)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


@dataclass
class KvSource2Plan:
    full: int
    keep: int
    raw_first: int
    raw_pos0: int
    do_scatter_upload: bool
    raw_idx: np.ndarray
    do_compress: bool
    compress_fused: bool
    grid_rows: int
    base_group: int
    idx_key_fused: bool
    lat_fused: bool
    stage_group_pos: bool


def plan_kv_source2(
    start: int,
    n: int,
    hd: int,
    id_dim: int,
    rd: int,
    compress2_norm_on: bool,
    rope_store_on: bool,
) -> KvSource2Plan:
    full = n // 2
    keep = min(n, RAW_ROWS)
    raw_first = n - keep
    raw_pos0 = (start + raw_first) % RAW_ROWS
    is_pow2_hd = hd > 0 and (hd & (hd - 1)) == 0
    compress_fused = bool(compress2_norm_on and full > 0 and is_pow2_hd and 128 <= hd <= 2048)
    do_scatter_upload = (not compress_fused) or (full == 0)
    raw_idx = np.zeros(RAW_ROWS, dtype=np.int64)
    for j in range(keep):
        raw_idx[j] = (start + raw_first + j) % RAW_ROWS
    do_compress = full > 0
    grid_rows = (n + 1) // 2
    base_group = start // 2
    idx_key_fused = bool(rope_store_on and full > 0 and id_dim in (128, 256, 512) and rd == 64)
    lat_fused = bool(rope_store_on and full > 0 and hd in (128, 256, 512) and rd == 64)
    stage_group_pos = do_compress and ((not idx_key_fused) or (not lat_fused))
    return KvSource2Plan(
        full=full,
        keep=keep,
        raw_first=raw_first,
        raw_pos0=raw_pos0,
        do_scatter_upload=do_scatter_upload,
        raw_idx=raw_idx,
        do_compress=do_compress,
        compress_fused=compress_fused,
        grid_rows=grid_rows,
        base_group=base_group,
        idx_key_fused=idx_key_fused,
        lat_fused=lat_fused,
        stage_group_pos=stage_group_pos,
    )


class Driver:
    def __init__(self):
        avail = mem_available_gib()
        if avail < MIN_MEM_AVAIL_GIB:
            raise SystemExit(f"ABORT: MemAvailable {avail:.2f} GiB < {MIN_MEM_AVAIL_GIB:.2f} GiB")
        self.mem_avail_start_gib = avail
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
        self._staged_refs = []
        self.cur_bytes = 0
        self.peak_bytes = 0

    def cu(self, name: str, *args):
        err = getattr(self.cuda, name)(*args)
        if err:
            msg = ct.c_char_p()
            self.cuda.cuGetErrorString(err, ref(msg))
            raise RuntimeError(f"{name}: {err} ({msg.value.decode() if msg.value else '?'})")

    def alloc(self, nbytes: int) -> int:
        nbytes = max(16, ((int(nbytes) + 15) // 16) * 16)
        if self.cur_bytes + nbytes >= MAX_VRAM_BYTES:
            raise RuntimeError(f"VRAM cap exceeded: {self.cur_bytes + nbytes} >= {MAX_VRAM_BYTES}")
        addr = U64()
        self.cu("cuMemAlloc_v2", ref(addr), ct.c_size_t(nbytes))
        assert addr.value % 16 == 0, f"unaligned device allocation: 0x{addr.value:x}"
        self.live_allocs[addr.value] = nbytes
        self.cur_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.cur_bytes)
        return addr.value

    def free_all(self):
        self.sync()
        for addr in list(self.live_allocs.keys()):
            nbytes = self.live_allocs.pop(addr)
            self.cu("cuMemFree_v2", U64(addr))
            self.cur_bytes -= nbytes

    def htod_async(self, dst_addr: int, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        self._staged_refs.append(arr)
        self.cu("cuMemcpyHtoDAsync_v2", U64(dst_addr), P(arr.ctypes.data), ct.c_size_t(arr.nbytes), self.stream)

    def dtoh_sync(self, src_addr: int, nbytes: int) -> np.ndarray:
        host = np.empty(nbytes, dtype=np.uint8)
        self.cu("cuMemcpyDtoHAsync_v2", P(host.ctypes.data), U64(src_addr), ct.c_size_t(nbytes), self.stream)
        self.sync()
        return host

    def sync(self):
        self.cu("cuStreamSynchronize", self.stream)
        self._staged_refs.clear()

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

    def launch(self, path: Path, symbol: str, grid, block, shared: int, *args, pdl: bool = False):
        fn, resolved_name, mod_sha = self.get_fn(path, symbol)
        if shared > 49152:
            self.cu("cuFuncSetAttribute", fn, I(8), I(shared))
        pack = []
        for a in args:
            if isinstance(a, (U64, L, I, F)):
                pack.append(a)
            elif isinstance(a, (int, np.integer)):
                pack.append(I(int(a)))
            else:
                raise TypeError(f"unsupported kernel arg type: {type(a)}")
        argv = (P * len(pack))(*(ct.addressof(x) for x in pack))
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        bx, by = block if isinstance(block, tuple) else (block, 1)
        if not pdl:
            self.cu("cuLaunchKernel", fn, gx, gy, 1, bx, by, 1, shared, self.stream, argv, None)
        else:
            attr = LaunchAttr(
                CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                (ct.c_ubyte * 4)(0, 0, 0, 0),
                (ct.c_ubyte * 64)(1, *([0] * 63)),
            )
            cfg = LaunchConfig(gx, gy, 1, bx, by, 1, shared, self.stream, ct.cast(ct.pointer(attr), P), 1)
            self.cu("cuLaunchKernelEx", ref(cfg), fn, argv, None)
        return resolved_name, mod_sha

    def close(self):
        self.free_all()
        self.cu("cuStreamDestroy_v2", self.stream)
        self.cu("cuDevicePrimaryCtxRelease", 0)


def find_fp4_store_spec(aot_dir: Path, d: int, block: int, e4m3: int) -> dict:
    aot_json = json.loads((aot_dir / "aot.json").read_text())
    for k in aot_json["kernels"]:
        if k["fn"] != "_fp4_store":
            continue
        c = k["consts"]
        if (
            c.get("D", {}).get("int") == d
            and c.get("BLOCK", {}).get("int") == block
            and c.get("E4M3", {}).get("int") == e4m3
            and c.get("PDL", {}).get("int") == 1
        ):
            return k
    raise RuntimeError(f"missing _fp4_store spec for D={d} BLOCK={block} E4M3={e4m3}")


class ArmBuffers:
    """Device buffers for one stateful KV-source execution arm."""

    def __init__(self, drv: Driver, max_chunk_rows: int):
        self.drv = drv
        self.max_chunk_rows = max_chunk_rows
        self.d_raw_kv = drv.alloc(RAW_ROWS * HD * 4)
        self.d_raw_score = drv.alloc(RAW_ROWS * HD * 4)
        self.d_lat = drv.alloc(CAP_GROUPS * HD * 2)
        self.d_lat2 = drv.alloc(CAP_GROUPS * HD * 2)
        self.d_ik = drv.alloc(CAP_GROUPS * ID_DIM * 2)
        self.d_idx_codes = drv.alloc(CAP_GROUPS * (ID_DIM // 2))
        self.d_idx_scales = drv.alloc(CAP_GROUPS * (ID_DIM // 32))
        self.d_comp_codes = drv.alloc(CAP_GROUPS * (HD // 2))
        self.d_comp_scales = drv.alloc(CAP_GROUPS * (HD // 16))
        self.d_kvc = drv.alloc(max_chunk_rows * HD * 4)
        self.d_scc = drv.alloc(max_chunk_rows * HD * 4)
        self.d_raw_idx = drv.alloc(RAW_ROWS * 8)
        self.d_groups = drv.alloc(CAP_GROUPS * 8)
        self.d_gpos = drv.alloc(CAP_GROUPS * 8)

    def reset_state(self, init_blobs: dict[str, np.ndarray], poison_staging: np.ndarray):
        self.drv.htod_async(self.d_raw_kv, init_blobs["raw_kv"])
        self.drv.htod_async(self.d_raw_score, init_blobs["raw_score"])
        self.drv.htod_async(self.d_lat, init_blobs["lat"])
        self.drv.htod_async(self.d_lat2, init_blobs["lat2"])
        self.drv.htod_async(self.d_ik, init_blobs["ik"])
        self.drv.htod_async(self.d_idx_codes, init_blobs["idx_codes"])
        self.drv.htod_async(self.d_idx_scales, init_blobs["idx_scales"])
        self.drv.htod_async(self.d_comp_codes, init_blobs["comp_codes"])
        self.drv.htod_async(self.d_comp_scales, init_blobs["comp_scales"])
        # Poison index staging arrays so any unintended read by fused kernels is caught immediately.
        self.drv.htod_async(self.d_raw_idx, poison_staging[:RAW_ROWS])
        self.drv.htod_async(self.d_groups, poison_staging[:CAP_GROUPS])
        self.drv.htod_async(self.d_gpos, poison_staging[:CAP_GROUPS])
        self.drv.sync()

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "raw_kv": self.drv.dtoh_sync(self.d_raw_kv, RAW_ROWS * HD * 4),
            "raw_score": self.drv.dtoh_sync(self.d_raw_score, RAW_ROWS * HD * 4),
            "lat": self.drv.dtoh_sync(self.d_lat, CAP_GROUPS * HD * 2),
            "idx_codes": self.drv.dtoh_sync(self.d_idx_codes, CAP_GROUPS * (ID_DIM // 2)),
            "idx_scales": self.drv.dtoh_sync(self.d_idx_scales, CAP_GROUPS * (ID_DIM // 32)),
            "comp_codes": self.drv.dtoh_sync(self.d_comp_codes, CAP_GROUPS * (HD // 2)),
            "comp_scales": self.drv.dtoh_sync(self.d_comp_scales, CAP_GROUPS * (HD // 16)),
        }


def run_step_gpu(
    drv: Driver,
    buf: ArmBuffers,
    ops_cubin: Path,
    torch_cubin: Path,
    fp4_idx_cubin: tuple[Path, dict],
    fp4_lat_cubin: tuple[Path, dict],
    d_comp_norm: int,
    d_idx_k_norm: int,
    d_cos: int,
    d_sin: int,
    start: int,
    n: int,
    kvc: np.ndarray,
    scc: np.ndarray,
    ik_proj: np.ndarray,
    compress2_norm_on: bool,
    rope_store_on: bool,
    pdl: bool,
) -> KvSource2Plan:
    plan = plan_kv_source2(start, n, HD, ID_DIM, RD, compress2_norm_on, rope_store_on)
    pdl_i = 1 if pdl else 0

    drv.htod_async(buf.d_kvc, kvc)
    drv.htod_async(buf.d_scc, scc)

    if plan.do_scatter_upload:
        drv.htod_async(buf.d_raw_idx, plan.raw_idx[: plan.keep])
        row_bytes = HD * 4
        drv.launch(
            ops_cubin,
            "tf_ds_scatter_rows_kernel",
            plan.keep,
            256,
            0,
            U64(buf.d_kvc + plan.raw_first * row_bytes),
            L(row_bytes),
            U64(buf.d_raw_idx),
            U64(buf.d_raw_kv),
            L(row_bytes),
            L(row_bytes),
            I(plan.keep),
            I(pdl_i),
            pdl=pdl,
        )
        drv.launch(
            ops_cubin,
            "tf_ds_scatter_rows_kernel",
            plan.keep,
            256,
            0,
            U64(buf.d_scc + plan.raw_first * row_bytes),
            L(row_bytes),
            U64(buf.d_raw_idx),
            U64(buf.d_raw_score),
            L(row_bytes),
            L(row_bytes),
            I(plan.keep),
            I(pdl_i),
            pdl=pdl,
        )

    if not plan.do_compress:
        drv.sync()
        return plan

    full = plan.full
    if plan.compress_fused:
        lanes = rms_lanes(HD, full)
        target = 512 if full <= 32 else (256 if full <= 128 else 128)
        groups = max(1, min(HD, target) // lanes)
        factor = float(np.float32(full) / np.float32(full * HD))
        drv.launch(
            torch_cubin,
            "tf_ds_compress2_norm_kernel",
            (plan.grid_rows, 1),
            (lanes, groups),
            0,
            U64(buf.d_kvc),
            U64(buf.d_scc),
            U64(d_comp_norm),
            U64(buf.d_lat),
            L(HD),
            I(full),
            I(HD),
            F(factor),
            F(EPS),
            I(pdl_i),
            U64(buf.d_raw_kv),
            U64(buf.d_raw_score),
            I(RAW_ROWS),
            L(plan.raw_pos0),
            I(plan.raw_first),
            I(n),
            pdl=pdl,
        )
    else:
        drv.launch(
            torch_cubin,
            "tf_ds_compress2_kernel",
            min(4096, launch_blocks(full * HD, 256)),
            256,
            0,
            U64(buf.d_kvc),
            U64(buf.d_scc),
            U64(buf.d_lat2),
            I(full),
            I(HD),
            I(pdl_i),
            pdl=pdl,
        )
        lanes = rms_lanes(HD, full)
        h = max(1, min(512 // lanes, 64))
        factor = float(np.float32(full) / np.float32(full * HD))
        smem = lanes * h * 4 if lanes > 32 else 0
        drv.launch(
            torch_cubin,
            "tf_ds_rms_norm_kernel",
            ((full + h - 1) // h, 1),
            (lanes, h),
            smem,
            U64(buf.d_lat2),
            L(HD),
            U64(d_comp_norm),
            U64(buf.d_lat),
            L(HD),
            I(full),
            I(HD),
            F(factor),
            F(EPS),
            I(pdl_i),
            pdl=pdl,
        )

    if plan.stage_group_pos:
        g_host = (np.arange(full, dtype=np.int64) + plan.base_group).astype(np.int64)
        p_host = (g_host * RATIO).astype(np.int64)
        drv.htod_async(buf.d_groups, g_host)
        drv.htod_async(buf.d_gpos, p_host)

    # Simulate mmRows(iwk, ch.lat -> ch.lat2) writing [full, ID_DIM] bf16 into d_lat2.
    drv.htod_async(buf.d_lat2, ik_proj)

    if plan.idx_key_fused:
        lanes = rms_lanes(ID_DIM, full)
        y = pack_lanes(lanes, ID_DIM // 2)
        factor = float(np.float32(full) / np.float32(full * ID_DIM))
        drv.launch(
            torch_cubin,
            "tf_ds_fused_prompt_idx_key_kernel",
            (full, 1),
            (lanes, y),
            0,
            U64(buf.d_lat2),
            L(ID_DIM),
            U64(d_idx_k_norm),
            F(EPS),
            F(factor),
            U64(d_cos),
            U64(d_sin),
            L(plan.base_group),
            I(RATIO),
            U64(buf.d_idx_codes),
            U64(buf.d_idx_scales),
            I(ID_DIM),
            I(RD),
            I(pdl_i),
            pdl=pdl,
        )
    else:
        lanes = rms_lanes(ID_DIM, full)
        h = max(1, min(512 // lanes, 64))
        factor = float(np.float32(full) / np.float32(full * ID_DIM))
        smem = lanes * h * 4 if lanes > 32 else 0
        drv.launch(
            torch_cubin,
            "tf_ds_rms_norm_kernel",
            ((full + h - 1) // h, 1),
            (lanes, h),
            smem,
            U64(buf.d_lat2),
            L(ID_DIM),
            U64(d_idx_k_norm),
            U64(buf.d_ik),
            L(ID_DIM),
            I(full),
            I(ID_DIM),
            F(factor),
            F(EPS),
            I(pdl_i),
            pdl=pdl,
        )
        half = RD // 2
        drv.launch(
            torch_cubin,
            "tf_ds_rope_kernel",
            min(4096, launch_blocks(full * half, 256)),
            256,
            0,
            U64(buf.d_ik),
            L(ID_DIM),
            I(ID_DIM - RD),
            U64(d_cos),
            U64(d_sin),
            U64(buf.d_gpos),
            I(full),
            I(half),
            I(0),
            I(pdl_i),
            pdl=pdl,
        )
        idx_cubin_path, idx_meta = fp4_idx_cubin
        drv.launch(
            idx_cubin_path,
            idx_meta["name"],
            (full, 1),
            (32 * idx_meta["num_warps"], 1),
            int(idx_meta["shared"]),
            U64(buf.d_ik),
            I(ID_DIM),
            U64(buf.d_idx_codes),
            U64(buf.d_idx_scales),
            U64(buf.d_groups),
            U64(0),
            U64(0),
            pdl=bool(idx_meta.get("pdl", True)),
        )

    if plan.lat_fused:
        drv.launch(
            torch_cubin,
            "tf_ds_fused_prompt_latent_kernel",
            (full, 1),
            (min(HD // 2, 256), 1),
            0,
            U64(buf.d_lat),
            U64(d_cos),
            U64(d_sin),
            L(plan.base_group),
            I(RATIO),
            U64(buf.d_comp_codes),
            U64(buf.d_comp_scales),
            I(HD),
            I(RD),
            I(pdl_i),
            pdl=pdl,
        )
    else:
        n16 = (HD * 2) // 16
        drv.launch(
            ops_cubin,
            "tf_ds_copy_rows16_kernel",
            (launch_blocks(n16, 256), full),
            (256, 1),
            0,
            U64(buf.d_lat),
            U64(n16),
            U64(buf.d_lat2),
            U64(n16),
            U64(n16),
            I(pdl_i),
            pdl=pdl,
        )
        half = RD // 2
        drv.launch(
            torch_cubin,
            "tf_ds_rope_kernel",
            min(4096, launch_blocks(full * half, 256)),
            256,
            0,
            U64(buf.d_lat2),
            L(HD),
            I(HD - RD),
            U64(d_cos),
            U64(d_sin),
            U64(buf.d_gpos),
            I(full),
            I(half),
            I(0),
            I(pdl_i),
            pdl=pdl,
        )
        lat_cubin_path, lat_meta = fp4_lat_cubin
        drv.launch(
            lat_cubin_path,
            lat_meta["name"],
            (full, 1),
            (32 * lat_meta["num_warps"], 1),
            int(lat_meta["shared"]),
            U64(buf.d_lat2),
            I(HD),
            U64(buf.d_comp_codes),
            U64(buf.d_comp_scales),
            U64(buf.d_groups),
            U64(0),
            U64(0),
            pdl=bool(lat_meta.get("pdl", True)),
        )

    drv.sync()
    return plan


def diff_bytes(a: np.ndarray, b: np.ndarray) -> dict:
    assert a.shape == b.shape and a.dtype == np.uint8 and b.dtype == np.uint8
    neq = np.flatnonzero(a != b)
    if neq.size == 0:
        return {"diff_bytes": 0, "first_offset": None}
    idx = int(neq[0])
    return {
        "diff_bytes": int(neq.size),
        "first_offset": idx,
        "expected_u8": int(a[idx]),
        "actual_u8": int(b[idx]),
    }


def make_chunk_inputs(rng: np.random.Generator, n: int, mode: str):
    if mode == "normal":
        kvc = rng.normal(loc=0.0, scale=1.5, size=(n, HD)).astype(np.float32)
        scc = rng.normal(loc=0.0, scale=2.0, size=(n, HD)).astype(np.float32)
        ik_f32 = rng.normal(loc=0.0, scale=1.25, size=(max(1, n // 2), ID_DIM)).astype(np.float32)
    elif mode == "wide_edge":
        scales = np.exp2(rng.integers(-20, 12, size=(n, 1))).astype(np.float32)
        kvc = (rng.normal(size=(n, HD)).astype(np.float32)) * scales
        scc = rng.uniform(-12.0, 12.0, size=(n, HD)).astype(np.float32)
        # Inject exact ties and zeros into scores/kv to exercise edge rounding and FP4 scale floors.
        if n >= 2:
            scc[0, :32] = scc[1, :32]
            kvc[0, 32:64] = 0.0
            kvc[1, 32:64] = 0.0
        ik_scales = np.exp2(rng.integers(-24, 10, size=(max(1, n // 2), 1))).astype(np.float32)
        ik_f32 = (rng.normal(size=(max(1, n // 2), ID_DIM)).astype(np.float32)) * ik_scales
        ik_f32[:, :32] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")
    ik_proj = f32_to_bf16_bits(ik_f32[: (n // 2)]) if (n // 2) > 0 else np.empty((0, ID_DIM), dtype=np.uint16)
    return kvc, scc, ik_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops-cubin", type=Path, required=True)
    ap.add_argument("--torch-cubin", type=Path, required=True)
    ap.add_argument("--kit-aot-dir", type=Path, default=Path("/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot"))
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    drv = Driver()
    try:
        idx_spec = find_fp4_store_spec(args.kit_aot_dir, d=ID_DIM, block=32, e4m3=0)
        lat_spec = find_fp4_store_spec(args.kit_aot_dir, d=HD, block=16, e4m3=1)
        fp4_idx = (args.kit_aot_dir / "cubins" / f"{idx_spec['hash']}.cubin", idx_spec)
        fp4_lat = (args.kit_aot_dir / "cubins" / f"{lat_spec['hash']}.cubin", lat_spec)

        # Primary 4-chunk stateful progression required by task spec:
        primary_chunks = [
            (0, 50, "even_initial"),
            (50, 1, "single_row_no_pair"),
            (51, 19, "odd_wraps_64_multiwarp_norm"),
            (70, 131, "odd_raw_first_67_gt0_wraps_64"),
        ]
        max_n = max(n for _, n, _ in primary_chunks)

        # Fixed weights & rotary tables
        w_rng = np.random.default_rng(20261121)
        comp_norm_w = f32_to_bf16_bits(w_rng.uniform(0.5, 1.75, size=(HD,)).astype(np.float32))
        idx_k_norm_w = f32_to_bf16_bits(w_rng.uniform(0.5, 1.75, size=(ID_DIM,)).astype(np.float32))
        inv_freq = (1.0 / (10000.0 ** (np.arange(0, RD, 2, dtype=np.float64) / RD))).astype(np.float64)
        angles = np.outer(np.arange(MAX_POS, dtype=np.float64), inv_freq)
        cos_tbl = np.cos(angles).astype(np.float32)
        sin_tbl = np.sin(angles).astype(np.float32)

        d_comp_norm = drv.alloc(comp_norm_w.nbytes)
        d_idx_k_norm = drv.alloc(idx_k_norm_w.nbytes)
        d_cos = drv.alloc(cos_tbl.nbytes)
        d_sin = drv.alloc(sin_tbl.nbytes)
        drv.htod_async(d_comp_norm, comp_norm_w)
        drv.htod_async(d_idx_k_norm, idx_k_norm_w)
        drv.htod_async(d_cos, cos_tbl)
        drv.htod_async(d_sin, sin_tbl)

        buf_unfused = ArmBuffers(drv, max_n)
        buf_fused = ArmBuffers(drv, max_n)
        buf_comp_only = ArmBuffers(drv, max_n)
        buf_rope_only = ArmBuffers(drv, max_n)

        poison_staging = np.full(max(RAW_ROWS, CAP_GROUPS), -999999, dtype=np.int64)

        suites = [
            {"name": "seed231_normal_pdl0", "seed": 231, "mode": "normal", "pdl": False},
            {"name": "seed231_normal_pdl1", "seed": 231, "mode": "normal", "pdl": True},
            {"name": "seed2026_wide_edge_pdl0", "seed": 2026, "mode": "wide_edge", "pdl": False},
            {"name": "seed2026_wide_edge_pdl1", "seed": 2026, "mode": "wide_edge", "pdl": True},
        ]

        all_suite_reports = []
        total_failures = 0

        for suite in suites:
            rng = np.random.default_rng(suite["seed"])
            init_blobs = {
                "raw_kv": rng.uniform(-3.0, 3.0, size=(RAW_ROWS, HD)).astype(np.float32),
                "raw_score": rng.uniform(-3.0, 3.0, size=(RAW_ROWS, HD)).astype(np.float32),
                "lat": rng.integers(0, 0x7BFF, size=(CAP_GROUPS, HD), dtype=np.uint16),
                "lat2": rng.integers(0, 0x7BFF, size=(CAP_GROUPS, HD), dtype=np.uint16),
                "ik": rng.integers(0, 0x7BFF, size=(CAP_GROUPS, ID_DIM), dtype=np.uint16),
                "idx_codes": rng.integers(0, 256, size=(CAP_GROUPS, ID_DIM // 2), dtype=np.uint8),
                "idx_scales": rng.integers(1, 250, size=(CAP_GROUPS, ID_DIM // 32), dtype=np.uint8),
                "comp_codes": rng.integers(0, 256, size=(CAP_GROUPS, HD // 2), dtype=np.uint8),
                "comp_scales": rng.integers(1, 250, size=(CAP_GROUPS, HD // 16), dtype=np.uint8),
            }
            cpu_raw_kv = init_blobs["raw_kv"].copy()
            cpu_raw_score = init_blobs["raw_score"].copy()

            # Deliberately give unfused vs fused DIFFERENT initial scratch in lat2 and ik to prove zero scratch leakage.
            init_fused = dict(init_blobs)
            init_fused["lat2"] = (init_blobs["lat2"] ^ np.uint16(0x5555)).copy()
            init_fused["ik"] = (init_blobs["ik"] ^ np.uint16(0x3333)).copy()

            buf_unfused.reset_state(init_blobs, poison_staging)
            buf_fused.reset_state(init_fused, poison_staging)
            buf_comp_only.reset_state(init_fused, poison_staging)
            buf_rope_only.reset_state(init_blobs, poison_staging)

            chunk_reports = []
            for step_idx, (start, n, label) in enumerate(primary_chunks):
                kvc, scc, ik_proj = make_chunk_inputs(rng, n, suite["mode"])

                # Update CPU modular-ring ground truth
                keep = min(n, RAW_ROWS)
                raw_first = n - keep
                for j in range(keep):
                    slot = (start + raw_first + j) % RAW_ROWS
                    cpu_raw_kv[slot] = kvc[raw_first + j]
                    cpu_raw_score[slot] = scc[raw_first + j]

                plan_u = run_step_gpu(
                    drv, buf_unfused, args.ops_cubin, args.torch_cubin, fp4_idx, fp4_lat,
                    d_comp_norm, d_idx_k_norm, d_cos, d_sin, start, n, kvc, scc, ik_proj,
                    compress2_norm_on=False, rope_store_on=False, pdl=suite["pdl"],
                )
                plan_f = run_step_gpu(
                    drv, buf_fused, args.ops_cubin, args.torch_cubin, fp4_idx, fp4_lat,
                    d_comp_norm, d_idx_k_norm, d_cos, d_sin, start, n, kvc, scc, ik_proj,
                    compress2_norm_on=True, rope_store_on=True, pdl=suite["pdl"],
                )
                run_step_gpu(
                    drv, buf_comp_only, args.ops_cubin, args.torch_cubin, fp4_idx, fp4_lat,
                    d_comp_norm, d_idx_k_norm, d_cos, d_sin, start, n, kvc, scc, ik_proj,
                    compress2_norm_on=True, rope_store_on=False, pdl=suite["pdl"],
                )
                run_step_gpu(
                    drv, buf_rope_only, args.ops_cubin, args.torch_cubin, fp4_idx, fp4_lat,
                    d_comp_norm, d_idx_k_norm, d_cos, d_sin, start, n, kvc, scc, ik_proj,
                    compress2_norm_on=False, rope_store_on=True, pdl=suite["pdl"],
                )

                snap_u = buf_unfused.snapshot()
                snap_f = buf_fused.snapshot()
                snap_c = buf_comp_only.snapshot()
                snap_r = buf_rope_only.snapshot()

                comparisons = {}
                for field in ("raw_kv", "raw_score", "lat", "idx_codes", "idx_scales", "comp_codes", "comp_scales"):
                    d_uf = diff_bytes(snap_u[field], snap_f[field])
                    d_uc = diff_bytes(snap_u[field], snap_c[field])
                    d_ur = diff_bytes(snap_u[field], snap_r[field])
                    comparisons[field] = {
                        "nbytes": int(snap_u[field].nbytes),
                        "sha256": sha256_bytes(snap_f[field].tobytes()),
                        "fused_vs_unfused": d_uf,
                        "comp_only_vs_unfused": d_uc,
                        "rope_only_vs_unfused": d_ur,
                    }
                    if d_uf["diff_bytes"] or d_uc["diff_bytes"] or d_ur["diff_bytes"]:
                        total_failures += 1

                # Active compressed lat slice check + CPU raw ring ground-truth check
                active_lat_bytes = plan_f.full * HD * 2
                lat_active_diff = diff_bytes(snap_u["lat"][:active_lat_bytes], snap_f["lat"][:active_lat_bytes])
                cpu_kv_diff = diff_bytes(cpu_raw_kv.view(np.uint8).ravel(), snap_f["raw_kv"])
                cpu_sc_diff = diff_bytes(cpu_raw_score.view(np.uint8).ravel(), snap_f["raw_score"])
                if lat_active_diff["diff_bytes"] or cpu_kv_diff["diff_bytes"] or cpu_sc_diff["diff_bytes"]:
                    total_failures += 1

                step_ok = all(
                    comparisons[k]["fused_vs_unfused"]["diff_bytes"] == 0
                    and comparisons[k]["comp_only_vs_unfused"]["diff_bytes"] == 0
                    and comparisons[k]["rope_only_vs_unfused"]["diff_bytes"] == 0
                    for k in comparisons
                ) and cpu_kv_diff["diff_bytes"] == 0 and cpu_sc_diff["diff_bytes"] == 0

                print(
                    f"[{suite['name']}] chunk#{step_idx} start={start:3d} n={n:3d} ({label:30s}) "
                    f"full={plan_f.full:2d} keep={plan_f.keep:2d} raw_first={plan_f.raw_first:2d} "
                    f"raw_pos0={plan_f.raw_pos0:2d} grid_rows={plan_f.grid_rows:2d} "
                    f"scatter={int(plan_f.do_scatter_upload)} -> {'PASS (0 byte diffs)' if step_ok else 'FAIL'}"
                )

                chunk_reports.append({
                    "chunk_index": step_idx,
                    "start": start,
                    "n": n,
                    "label": label,
                    "plan_unfused": {
                        "full": plan_u.full,
                        "keep": plan_u.keep,
                        "raw_first": plan_u.raw_first,
                        "raw_pos0": plan_u.raw_pos0,
                        "do_scatter_upload": plan_u.do_scatter_upload,
                        "stage_group_pos": plan_u.stage_group_pos,
                    },
                    "plan_fused": {
                        "full": plan_f.full,
                        "keep": plan_f.keep,
                        "raw_first": plan_f.raw_first,
                        "raw_pos0": plan_f.raw_pos0,
                        "grid_rows": plan_f.grid_rows,
                        "base_group": plan_f.base_group,
                        "do_scatter_upload": plan_f.do_scatter_upload,
                        "compress_fused": plan_f.compress_fused,
                        "idx_key_fused": plan_f.idx_key_fused,
                        "lat_fused": plan_f.lat_fused,
                        "stage_group_pos": plan_f.stage_group_pos,
                    },
                    "lat_active_bytes": active_lat_bytes,
                    "lat_active_diff": lat_active_diff,
                    "cpu_raw_kv_diff": cpu_kv_diff,
                    "cpu_raw_score_diff": cpu_sc_diff,
                    "buffers": comparisons,
                    "pass": step_ok,
                })

            all_suite_reports.append({
                "suite": suite["name"],
                "seed": suite["seed"],
                "mode": suite["mode"],
                "pdl": suite["pdl"],
                "chunks": chunk_reports,
            })

        resolved_kernels = {
            f"{Path(k[0]).name}:{k[1]}": {"resolved": v[1], "module_sha256": v[2]}
            for k, v in drv.functions.items()
        }
        summary = {
            "pid": os.getpid(),
            "elapsed_sec": round(time.time() - t0, 4),
            "mem_available_start_gib": round(drv.mem_avail_start_gib, 3),
            "mem_available_end_gib": round(mem_available_gib(), 3),
            "peak_vram_bytes": drv.peak_bytes,
            "peak_vram_mib": round(drv.peak_bytes / (1024 * 1024), 3),
            "vram_cap_mib": MAX_VRAM_BYTES // (1024 * 1024),
            "ops_cubin": str(args.ops_cubin.resolve()),
            "ops_cubin_sha256": sha256_bytes(args.ops_cubin.read_bytes()),
            "torch_cubin": str(args.torch_cubin.resolve()),
            "torch_cubin_sha256": sha256_bytes(args.torch_cubin.read_bytes()),
            "kit_idx_fp4_store_hash": idx_spec["hash"],
            "kit_lat_fp4_store_hash": lat_spec["hash"],
            "resolved_kernels": resolved_kernels,
            "total_failures": total_failures,
            "pass": total_failures == 0,
            "suites": all_suite_reports,
        }

        out_path = args.out_json or Path(f"verify_0231_chunks_pid{os.getpid()}.json")
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"\nSUMMARY: pass={summary['pass']} total_failures={total_failures} "
            f"peak_vram={summary['peak_vram_mib']:.3f} MiB "
            f"mem_avail={summary['mem_available_end_gib']:.2f} GiB "
            f"elapsed={summary['elapsed_sec']:.3f}s json={out_path}"
        )
        if total_failures != 0:
            raise SystemExit(1)
    finally:
        drv.close()


if __name__ == "__main__":
    main()
