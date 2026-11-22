#!/usr/bin/env python3
"""Parity + CUDA-graph timing harness for the dsv41 fused kernels (0195/0197/0198/0199).

Loads the old and new cubins built by build.sh plus the pinned kit cubins through the CUDA driver API, and for each
kernel runs
  (a) byte parity of every live output: new fused vs old fused vs the unfused/kit producer, over random, tied, flat,
      NaN, +-inf, signed-zero, partial-block and boundary inputs at the served shapes, and
  (b) interleaved CUDA-graph timings (A B A B ...) of >= --reps graphs of --batch replays, warmed, median + IQR, plus
      an empty-kernel launch floor at the fused launch shape.

It refuses to run below 3 GiB MemAvailable or while production is serving, and holds ~/tf-bench/gpu.lock while timing.

Usage (on the head):
  python3 bench.py OLD_BIN NEW_BIN KIT_CUBINS [--kernel ...] [--shape ...] [--json out.json] [--parity-only]
See README.md for the per-kernel cases and the exact commands.
"""

import argparse
import ctypes as ct
import fcntl
import json
import os
import sys
import time
import urllib.request
from functools import cache
from pathlib import Path
from statistics import median

import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
U64 = ct.c_ulonglong
ref = ct.byref
cuda = None
stream = P()
modules, kernels, allocations = {}, {}, []
allocated = 0
rng = np.random.default_rng(197)
eps = F(1e-20)
opts = None
results = []
failures = []
DO_TIMING = True
DO_PARITY = True
timing_retries = 0

# kit cubin prefixes (aot.json hashes) -> (name, shared bytes)
kit = {
    "norm128": ("0cca97", "_rmsnorm", 16),
    "norm512": ("ee9e02", "_rmsnorm", 16),
    "rope128": ("e8bd0a", "_rope_heads", 0),
    "rope512": ("a22128", "_rope_heads", 0),
    "rope32": ("5ed6b4", "_rope_heads", 0),
    "store128": ("345438", "_fp4_store", 512),
    "store512": ("9cd357", "_fp4_store", 2048),
    "qd": ("4cbef7", "_fp4qd_p2", 0),
    "scorekeys": ("df85d0", "_score_keys", 4096),
}

# ---------------------------------------------------------------------------------------------------------------------
# driver plumbing

def libcuda():
    global cuda
    if cuda is None:
        cuda = ct.CDLL("libcuda.so.1")
    return cuda


def cu(name, *args):
    if os.environ.get("FUSED_BENCH_TRACE"):
        print(f"    cu {name}", flush=True)
    lib = libcuda()
    err = getattr(lib, name)(*args)
    if err:
        msg = ct.c_char_p()
        lib.cuGetErrorString(err, ref(msg))
        raise RuntimeError(f"{name}: {err} {msg.value}")


def argptr(x):
    if isinstance(x, ct._Pointer):  # c_void_p
        return ct.addressof(x)
    if isinstance(x, ct.Structure):
        return ct.addressof(x)
    if isinstance(x, ct.Array):
        return ct.addressof(x)
    return ct.addressof(x)


class Buffer:
    def __init__(self, data):
        global allocated
        data = np.ascontiguousarray(data)
        self.shape, self.dtype = data.shape, data.dtype
        self.addr = P()
        cu("cuMemAlloc_v2", ref(self.addr), ct.c_size_t(data.nbytes))
        # stream uses CU_STREAM_NON_BLOCKING, which does not synchronize with legacy stream 0.
        if stream.value:
            cu("cuMemcpyHtoDAsync_v2", self.addr, P(data.ctypes.data), ct.c_size_t(data.nbytes), stream)
            cu("cuStreamSynchronize", stream)
        else:
            cu("cuMemcpyHtoD_v2", self.addr, P(data.ctypes.data), ct.c_size_t(data.nbytes))
            cu("cuCtxSynchronize")
        allocations.append(self.addr)
        allocated += data.nbytes

    def read(self):
        host = np.empty(self.shape, self.dtype)
        if stream.value:
            cu("cuStreamSynchronize", stream)
        cu("cuMemcpyDtoH_v2", P(host.ctypes.data), self.addr, ct.c_size_t(host.nbytes))
        return host.view("u1").ravel()

    def readarr(self):
        host = np.empty(self.shape, self.dtype)
        if stream.value:
            cu("cuStreamSynchronize", stream)
        cu("cuMemcpyDtoH_v2", P(host.ctypes.data), self.addr, ct.c_size_t(host.nbytes))
        return host


def out(shape, dtype):
    return Buffer(np.full(shape, 0x55, dtype))


def module(path):
    if path not in modules:
        mod = P()
        cu("cuModuleLoad", ref(mod), str(path).encode())
        modules[path] = mod
    return modules[path]


def function(path, name):
    key = (path, name)
    if key not in kernels:
        fn = P()
        cu("cuModuleGetFunction", ref(fn), module(path), name.encode())
        kernels[key] = fn
    return kernels[key]


class LaunchAttr(ct.Structure):  # CUlaunchAttribute: id, 4 pad bytes, a 64-byte value union
    _fields_ = [("id", ct.c_uint), ("pad", ct.c_ubyte * 4), ("value", ct.c_ubyte * 64)]


class LaunchConfig(ct.Structure):  # CUlaunchConfig
    _fields_ = [("grid_x", ct.c_uint), ("grid_y", ct.c_uint), ("grid_z", ct.c_uint),
                ("block_x", ct.c_uint), ("block_y", ct.c_uint), ("block_z", ct.c_uint),
                ("shared_bytes", ct.c_uint), ("stream", P), ("attrs", P), ("num_attrs", ct.c_uint)]


PROGRAMMATIC_STREAM_SERIALIZATION = 6  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION


def launch(path, name, grid, threads, shared, *args, pdl=False, on_stream=None):
    fn = function(path, name)
    if shared > 49152:  # past 48 KiB the kernel must opt into dynamic shared memory (CU_FUNC_ATTRIBUTE 8)
        cu("cuFuncSetAttribute", fn, I(8), I(shared))
    pack = []
    for a in args:
        if isinstance(a, Buffer):
            pack.append(a.addr)
        elif isinstance(a, int):
            pack.append(I(a))
        elif isinstance(a, np.integer):
            pack.append(I(int(a)))
        else:
            pack.append(a)
    argv = (P * len(pack))(*(argptr(x) for x in pack))
    gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
    bx, by = threads if isinstance(threads, tuple) else (threads, 1)
    st = stream if on_stream is None else on_stream
    if not pdl:  # the engine's plain launch path
        cu("cuLaunchKernel", fn, gx, gy, 1, bx, by, 1, shared, st, argv, None)
        return
    # TF_DS_PDL: the engine sets this attribute on every launch of a pdl-taking kernel (cuda/launch.zig)
    attr = LaunchAttr(PROGRAMMATIC_STREAM_SERIALIZATION, (ct.c_ubyte * 4)(0, 0, 0, 0), (ct.c_ubyte * 64)(1, *([0] * 63)))
    cfg = LaunchConfig(gx, gy, 1, bx, by, 1, shared, st, ct.cast(ct.pointer(attr), P), 1)
    cu("cuLaunchKernelEx", ref(cfg), fn, argv, None)
    if os.environ.get("FUSED_BENCH_DEBUG"):
        print(f"    launch {Path(path).name}:{name} grid={gx},{gy} block={bx},{by}", flush=True)
        cu("cuStreamSynchronize", stream)


def aot(key, grid, *args):
    prefix, name, smem = kit[key]
    (path,) = opts.kit.glob(prefix + "*.cubin")
    launch(path, name, grid, 128, smem, *args, P(), P())


def native(mod, name, grid, block, *args, shared=0, pdl=0):
    launch(mod, "tf_ds_" + name + "_kernel", grid, block, shared, *args, pdl, pdl=bool(pdl))


def param_count(path, name):
    """How many parameters the kernel declares: a tree that predates a signature change would otherwise read past the
    argv array in cuLaunchKernel (a driver-side fault, not a CUDA error)."""
    fn = function(path, name)
    lib = libcuda()
    n, off, size = 0, I(), I()
    while lib.cuFuncGetParamInfo(fn, ct.c_uint(n), ref(off), ref(size)) == 0:
        n += 1
    return n


def native_any(mod, names, grid, block, *args, shared=0):
    """The first of `names` the module exports: the decode compressor's per-width kernels exist on some trees only."""
    for name in names:
        try:
            launch(mod, "tf_ds_" + name + "_kernel", grid, block, shared, *args, 0)
            return name
        except RuntimeError as err:
            if "named symbol not found" not in str(err):
                raise


def free_all():
    global allocated
    cu("cuStreamSynchronize", stream)
    for addr in allocations:
        cu("cuMemFree_v2", addr)
    allocations.clear()
    allocated = 0


def read_byte_diff(actual, expected):
    return int(np.count_nonzero(actual.read() != expected.read()))


def compare(label, actual, expected):
    if not DO_PARITY:
        return 0
    cu("cuStreamSynchronize", stream)
    total = 0
    for i, (buf, gold) in enumerate(zip(actual, expected, strict=True)):
        mismatch = np.flatnonzero(buf.read() != gold.read())
        if mismatch.size:
            failures.append(f"{label} buffer {i}: {len(mismatch)} bytes differ (first {mismatch[0]})")
            total += len(mismatch)
    return total


def case_latent_owner(rows, own, timing=True):
    """TF_DS_KV_SHARD's decode latent: `own` rows target this rank's plane, the rest the trash area (one row a round
    row, past the part's rows). Both arms write every target, so the packed bytes are compared whole - this is the
    dest-indirection case the shard path really runs."""
    d = 512
    label = f"latent-owner R={rows} own={own} D={d}"
    if not shape_ok(label):
        return None
    src = bf((rows, 1, d))
    pos = Buffer(rng.permutation(rows).astype("i8"))
    angle = rng.uniform(-np.pi, np.pi, (rows, 32))
    cos, sin = Buffer(np.cos(angle).astype("f4")), Buffer(np.sin(angle).astype("f4"))
    owned = rng.permutation(own)
    dest = np.concatenate([owned, np.arange(own, rows)]).astype("i8")   # the rest: trash rows above the part
    destb = Buffer(dest)
    chains, states, names, floors = [], [], [], []
    for arm in ["ref", "old", "new"]:
        x = Buffer(src)
        rot = Buffer(src)
        codes, scales = out((rows, d // 2), "u1"), out((rows, d // 16), "u1")

        def chain(arm=arm, x=x, rot=rot, codes=codes, scales=scales):
            if arm == "ref":
                aot("rope512", rows, x, cos, sin, pos)
                aot("store512", rows, x, d, codes, scales, destb)
            elif arm == "new":
                native(opts.new_torch, "fused_rope_store", rows, min(d // 2, 256), x, cos, sin, pos, codes, scales,
                       destb, d, 64, own, 0)
            else:
                native(opts.old_torch, "fused_rope_store", rows, min(d // 2, 256), x, cos, sin, pos, codes, scales,
                       destb, d, 64, 0)

        chain()
        chains.append(chain)
        names.append(arm)
        states.append([codes, scales])
        floors.append(arm)
    diffs = {}
    for k, arm in enumerate(names):
        if arm == "ref":
            continue
        diffs[arm] = compare(f"{label} {arm}", states[k], states[0])
    row = {"kernel": "0198-latent", "shape": label, "rows": rows, "d": d, "parity": diffs}
    if timing and DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, min(d // 2, 256), 0)
        attach_timing(row, label, chains, names, floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# timing

def _time_graphs(label, chains, names, floor, batch, reps):
    """Interleaved CUDA-graph timings returning (stats, timing_valid, timing_retries, timing_guarded)."""
    global timing_retries
    chains = list(chains)
    names = list(names)
    if len(chains) != len(names):
        raise ValueError(f"{label}: chains/names length mismatch ({len(chains)} != {len(names)})")
    if len(set(names)) != len(names):
        raise ValueError(f"{label}: duplicate arm names in {names!r}")
    if "floor" in names:
        raise ValueError(f"{label}: 'floor' is reserved for the launch-floor graph")

    def build(chain):
        chain()
        cu("cuStreamSynchronize", stream)
        cu("cuStreamBeginCapture_v2", stream, 0)
        for _ in range(batch):
            chain()
        g, e = P(), P()
        cu("cuStreamEndCapture", stream, ref(g))
        cu("cuGraphInstantiateWithFlags", ref(e), g, ct.c_uint64(0))
        cu("cuGraphDestroy", g)
        return e

    exes = [build(c) for c in chains]
    if floor is not None:
        exes.append(build(floor))
        names = names + ["floor"]
    for e in exes:  # warm
        cu("cuGraphLaunch", e, stream)
        cu("cuStreamSynchronize", stream)
    start, stop = P(), P()
    for ev in (start, stop):
        cu("cuEventCreate", ref(ev), 0)
    # A guarded timing block is valid only when before-health and after-health both report connections == 0 and valid
    # unchanged served-token counters.
    guarded = not bool(getattr(opts, "no_guard", False))
    valid = False
    case_retries = 0
    samples = None
    try:
        for attempt in range(opts.tries):
            c0 = None
            if guarded:
                try:
                    h0 = wait_health(opts.wait)
                except TimeoutError as exc:
                    print(f"{label}: {exc}", flush=True)
                    break
                c0 = health_counters(h0)
                if not check_idle_health(h0):
                    case_retries += 1
                    timing_retries += 1
                    print(f"{label}: invalid pre-block health, retry {attempt + 1}/{opts.tries}", flush=True)
                    continue
            samples = [[] for _ in exes]
            for rep in range(reps):
                for off in range(len(exes)):
                    i = (rep + off) % len(exes)
                    cu("cuEventRecord", start, stream)
                    cu("cuGraphLaunch", exes[i], stream)
                    cu("cuEventRecord", stop, stream)
                    cu("cuEventSynchronize", stop)
                    ms = F()
                    cu("cuEventElapsedTime", ref(ms), start, stop)
                    samples[i].append(ms.value * 1000.0 / batch)
            if not guarded:
                valid = True
                break
            h1 = health_now()
            c1 = health_counters(h1)
            if check_idle_health(h1) and counters_valid(c0) and counters_valid(c1) and c1 == c0:
                valid = True
                break
            case_retries += 1
            timing_retries += 1
            print(f"{label}: production served or health invalid during the timing block, retry {attempt + 1}/{opts.tries}",
                  flush=True)
        else:
            if guarded:
                print(f"{label}: WARNING no clean timing window after {opts.tries} tries", flush=True)
    finally:
        for e in exes:
            cu("cuGraphExecDestroy", e)
        for ev in (start, stop):
            cu("cuEventDestroy_v2", ev)
    if not valid or samples is None:
        return {}, False, case_retries, guarded
    stats = {}
    for name, s in zip(names, samples, strict=True):
        a = np.asarray(s)
        stats[name] = {"median": round(float(median(s)), 3),
                       "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3),
                       "n": len(s)}
    print(f"{label}: " + " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in stats.items()), flush=True)
    return stats, True, case_retries, guarded


def time_graphs(label, chains, names, floor, batch, reps):
    """Interleaved CUDA-graph timings: one graph per arm of `batch` replays, plus the launch-floor graph."""
    stats, _, _, _ = _time_graphs(label, chains, names, floor, batch, reps)
    return stats


def attach_timing(row, label, chains, names, floor, batch, reps):
    stats, valid, case_retries, guarded = _time_graphs(label, chains, names, floor, batch, reps)
    row["timing"] = stats
    row["timing_valid"] = valid
    row["timing_retries"] = case_retries
    row["timing_guarded"] = guarded
    return stats


# ---------------------------------------------------------------------------------------------------------------------
# engine launch geometry (zig/src/families/dsv41/exact.zig)

def last_pow2(n):
    return 0 if n == 0 else 1 << (n.bit_length() - 1)


def rms_lanes(d, rows):
    dim0 = d // 4
    d0 = last_pow2(dim0) if dim0 < 512 else 512
    d1 = last_pow2(rows) if rows < 512 else 512
    bw = min(d0, 32)
    bh = min(d1, 512 // bw)
    return min(d0, 512 // bh)


def pack_lanes(lanes, half):
    if lanes == 0 or half == 0 or half % lanes != 0:
        return 1
    most = max(1, 512 // lanes)
    y = min(half // lanes, most)
    while y > 1 and half % (lanes * y) != 0:
        y -= 1
    return y


def prompt_target(rows):
    return 512 if rows <= 32 else 256 if rows <= 128 else 128


# ---------------------------------------------------------------------------------------------------------------------
# 0195 indexer

def score_key_host(x, j):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    bits = np.where(bits == 0x80000000, 0, bits)
    sb = np.where(bits >= 2**31, bits - 2**32, bits)
    ordered = np.where(sb < 0, sb ^ 0x7FFFFFFF, sb).astype(np.uint64)
    key = (ordered << np.uint64(32)) | (np.uint64(0xFFFFFFFF) - np.uint64(j))
    return key ^ np.uint64(0x8000000000000000)


@cache
def module_symbols(path):
    """Enumerate once so testing an older cubin never produces a CUDA API error under memcheck."""
    mod = module(path)
    count = ct.c_uint()
    cu("cuModuleGetFunctionCount", ref(count), mod)
    handles = (P * count.value)()
    cu("cuModuleEnumerateFunctions", handles, count, mod)
    names = set()
    for handle in handles:
        name = ct.c_char_p()
        cu("cuFuncGetName", ref(name), P(handle))
        names.add(name.value.decode())
    return names


def index_symbol(mod, original, optimized):
    return optimized if "tf_ds_" + optimized + "_kernel" in module_symbols(mod) else original


def check_index_case(label, original, optimized, expected, launch_arm, floor, eligible=True, bytes_moved=0):
    gold = Buffer(expected)
    chains, parity, symbols = [], {}, {}
    for arm, mod in (("old", opts.old_ops), ("new", opts.new_ops), ("new_off", opts.new_ops),
                     ("new_pdl", opts.new_ops)):
        symbol = index_symbol(mod, original, optimized) if eligible and arm != "new_off" else original
        symbols[arm] = symbol
        output = out(expected.shape, expected.dtype)

        def chain(mod=mod, symbol=symbol, output=output, pdl=int(arm == "new_pdl")):
            launch_arm(mod, symbol, output, pdl)

        chain()
        parity[arm] = compare(f"{label} {arm}", [output], [gold])
        if arm in ("old", "new"):
            chains.append(chain)
    row = {"kernel": original, "shape": label, "symbols": symbols, "parity": parity, "bytes": bytes_moved}
    if DO_TIMING:
        attach_timing(row, label, chains, ["old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    free_all()


def case_topk_indices(rows, width, k, dist):
    label = f"topk R={rows} n={width} k={k} {dist}"
    if not shape_ok(label):
        return
    rng = np.random.default_rng(254)
    # An odd stride exposes reads from the next row. The output tail catches excess writes.
    stride = width + 3
    scores = rng.standard_normal((rows, stride), dtype=np.float32)
    if dist == "ties":
        scores[:] = rng.integers(-3, 4, scores.shape)
    elif dist == "flat":
        scores.fill(-np.inf)
    elif dist == "special":
        patterns = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 1, 0x80000001,
                             0x7fc00001, 0x7fc00002, 0xffc00001], dtype="u4")
        scores.view("u4")[:] = rng.choice(patterns, scores.shape)
    scores[:, width:] = np.inf
    visible = np.resize(np.array([width + 1, width // 2, 0, -1, width - 1], dtype="i8"), rows)
    x, vis = Buffer(scores), Buffer(visible)
    expected = np.full(rows * k + 8, 0x55, "i8")
    columns = np.arange(width, dtype="u4")
    if k:
        for r in range(rows):
            keys = score_key_host(scores[r, :width], columns)
            picks = np.sort(np.argpartition(keys, width - k)[-k:])
            expected[r * k:(r + 1) * k] = np.where(picks < visible[r], picks, -1)

    def launch_arm(mod, symbol, output, pdl):
        native(mod, symbol, rows, 1024, x, L(stride), width, k, vis, output, pdl=pdl)

    floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, 0)
    bytes_moved = rows * width * 4 + rows * 8 + rows * k * 8
    check_index_case(label, "topk_indices", "topk_indices_cut", expected, launch_arm, floor, bytes_moved=bytes_moved)


def case_candidate_mask(rows, blocks, size=8, offset=0, pad=0):
    label = f"cand-mask R={rows} nb={blocks} bs={size} offset={offset} pad={pad}"
    if not shape_ok(label):
        return
    rng = np.random.default_rng(255)
    width, ms = blocks * size, blocks + 3
    storage = rng.standard_normal(offset + rows * (width + pad), dtype=np.float32)
    scores = storage[offset:].reshape(rows, width + pad)[:, :width].reshape(rows, blocks, size)
    scores[:, 0::4] = -np.inf
    scores[:, 1::4] = -0.0
    scores[:, 2::4, -1] = np.inf
    for j in range(size):
        scores[:, 3 + 4 * j::4 * size, j] = np.nan
    visible = np.resize(np.array([width, 0, -1, size, size + 1, width + 1], dtype="i8"), rows)
    expected = np.full((rows, ms), 0x55, "u1")
    expected[:, :blocks] = (~np.isnan(scores).any(axis=2) & (scores > -np.inf).any(axis=2)
                             | (np.arange(blocks) == (visible[:, None] - 1) // size))
    x, vis = Buffer(storage), Buffer(visible)
    ptr = P(x.addr.value + offset * storage.itemsize)
    grid = ((blocks + 255) // 256, rows)

    def launch_arm(mod, symbol, output, pdl):
        native(mod, symbol, grid, 256, ptr, L(width + pad), blocks, size, vis, output, L(ms), pdl=pdl)

    floor = lambda: launch(opts.floor, "empty_kernel", grid, 256, 0)
    aligned8 = size == 8 and ptr.value % 16 == 0 and (width + pad) % 4 == 0
    bytes_moved = rows * width * 4 + rows * 8 + rows * blocks
    check_index_case(label, "cand_fast", "cand_fast8", expected, launch_arm, floor, aligned8, bytes_moved=bytes_moved)


def case_idx(rows, nb, k, mask_on, dist="ties", special=False, timing=True, pad=0):
    width, ss, ms = 8 * nb - 3, 8 * nb + pad, nb + 7
    label = f"idx R={rows} nb={nb} width={width} k={k} {'mask' if mask_on else 'list'} {dist}{' special' if special else ''}" + (f" pad={pad}" if pad else "")
    if not shape_ok(label):
        return None
    scores = rng.integers(-10, 10, (rows, ss)).astype("f4")
    if dist == "random":
        scores = rng.normal(size=(rows, ss)).astype("f4")
    elif dist == "flat":
        scores.fill(-np.inf)
    elif dist == "same":
        scores.fill(1.0)
    if special:
        scores[:, 8:16], scores[:, 24:32], scores[:, 32:40], scores[:, 40:48] = np.nan, -np.inf, -0.0, np.inf
        scores[0, 0] = 0.0
        scores[-1, width - 1] = np.inf
    x = Buffer(scores)
    vis = Buffer(np.linspace(0, width, rows, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    chains, states = [], []
    old_mod, new_mod = opts.old_ops, opts.new_ops
    for i, mod in enumerate((old_mod, old_mod, new_mod)):
        mx, ix, mask, blocks = (out((rows, nb), "u4"), out((rows, k), "i8"),
                                out((rows, ms), "u1"), out((rows, k), "i4"))
        m, c = (mask, P()) if mask_on else (P(), blocks)

        def chain(i=i, mod=mod, mx=mx, ix=ix, m=m, c=c):
            if i == 0:
                if k >= nb:
                    # out of the unfused chain's contract (topk_indices faults at k == nb; the engine pools only when
                    # nbk > candidate_blocks, prompt.zig `pooled`), so these cases compare new against old only
                    return
                native(mod, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb)
                native(mod, "topk_indices", rows, 1024, mx, L(nb), nb, k, every, ix)
                native(mod, "pool_pick", rows, 1024, mx, L(nb), nb, ix, k, m, L(ms), c)
            else:
                native(mod, "idx_fused", rows, 1024, x, L(ss), width, 8, vis, every, k, m, L(ms), c,
                       shared=nb * 4)

        chain()
        chains.append(chain)
        states.append([mask] if mask_on else [blocks])
    ref_state = states[0] if k < nb else states[1]   # k >= nb: no unfused reference (see chain)
    diff_old = compare(f"idx old R={rows} nb={nb}", states[1], ref_state)
    diff_new = compare(f"idx new R={rows} nb={nb}", states[2], ref_state)
    diff_cross = compare(f"idx new-vs-old R={rows} nb={nb}", states[2], states[1])
    row = {"kernel": "0195-idx", "shape": label, "rows": rows, "nb": nb, "k": k,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old, "new_vs_old": diff_cross}}
    # bytes: score row read (width floats) + vis/every + the one output the launch writes
    row["bytes"] = rows * width * 4 + rows * 16 + (rows * nb if mask_on else rows * k * 4)
    if k >= nb:
        row["note"] = "k >= nb: no unfused reference; new compared with old"
    if timing and DO_TIMING and not special and k < nb:
        shared = nb * 4
        floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, shared)
        attach_timing(row, label, chains, ["ref", "old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0195 small-row form (block_max's 2D grid + the selection from global maxima) against the one-launch fused form

def case_idx_small(rows, nb, k, dist="random"):
    width, ss, ms = 8 * nb - 3, 8 * nb, nb + 7
    label = f"idx-small R={rows} nb={nb} k={k} {dist}"
    if not shape_ok(label):
        return None
    scores = rng.normal(size=(rows, ss)).astype("f4") if dist == "random" else rng.integers(-10, 10, (rows, ss)).astype("f4")
    x = Buffer(scores)
    vis = Buffer(np.linspace(0, width, rows, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    chains, states = [], []
    for i, mod in enumerate((opts.old_ops, opts.old_ops, opts.new_ops, opts.new_ops)):
        mx, ix = out((rows, nb), "u4"), out((rows, k), "i8")
        blocks = out((rows, k), "i4")
        bmax = out((rows, nb), "f4")

        def chain(i=i, mod=mod, mx=mx, ix=ix, blocks=blocks, bmax=bmax):
            if i == 0:
                native(mod, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb)
                native(mod, "topk_indices", rows, 1024, mx, L(nb), nb, k, every, ix)
                native(mod, "pool_pick", rows, 1024, mx, L(nb), nb, ix, k, P(), 0, blocks)
            elif i == 1:
                native(mod, "idx_fused", rows, 1024, x, L(ss), width, 8, vis, every, k, P(), 0, blocks, shared=nb * 4)
            elif i == 2:  # the one-launch fused form of the same tree (the current kernel)
                native(mod, "idx_fused", rows, 1024, x, L(ss), width, 8, vis, every, k, P(), 0, blocks, shared=nb * 4)
            else:
                native(mod, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, bmax, L(nb), nb)
                native(mod, "idx_pick", rows, 1024, bmax, L(nb), nb, k, every, blocks, shared=nb * 4)

        chain()
        chains.append(chain)
        states.append([blocks])
    diff_old = compare(f"idx-small old R={rows} nb={nb}", states[1], states[0])
    diff_fused = compare(f"idx-small fused R={rows} nb={nb}", states[2], states[0])
    diff_new = compare(f"idx-small new R={rows} nb={nb}", states[3], states[0])
    row = {"kernel": "0195-idx-small", "shape": label, "rows": rows, "nb": nb, "k": k,
           "parity": {"new_vs_ref": diff_new, "fused_vs_ref": diff_fused, "old_vs_ref": diff_old}}
    row["bytes"] = rows * width * 4 + rows * nb * 4 + rows * k * 4
    if DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, nb * 4)
        attach_timing(row, label, chains, ["ref", "old", "fused", "small"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0195 phase-1 cost alone (the unfused block_max kernel's 2D grid, for the small-row item)

def case_blockmax(rows, nb):
    width, ss = 8 * nb - 3, 8 * nb
    label = f"block_max R={rows} nb={nb} width={width}"
    if not shape_ok(label):
        return None
    x = Buffer(rng.normal(size=(rows, ss)).astype("f4"))
    vis = Buffer(np.full(rows, width, dtype="i8"))
    mx = out((rows, nb), "u4")

    def chain():
        native(opts.old_ops, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb)

    chain()
    cu("cuStreamSynchronize", stream)
    row = {"kernel": "0195-block-max", "shape": label, "rows": rows, "nb": nb, "parity": {"new_vs_ref": 0}}
    row["bytes"] = rows * width * 4 + rows * nb * 4
    if DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", ((nb + 255) // 256, rows), 256, 0)
        attach_timing(row, label, [chain], ["kit"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0195 histogram sweep (the warp-aggregated histogram at 256/512/1024-thread CTAs; the new kernel only)

def case_idx_hist(rows, nb, k, dist, block):
    width, ss, ms = 8 * nb - 3, 8 * nb, nb + 7
    label = f"idx-hist R={rows} nb={nb} block={block} {dist}"
    if not shape_ok(label):
        return None
    scores = rng.integers(-10, 10, (rows, ss)).astype("f4")
    if dist == "random":
        scores = rng.normal(size=(rows, ss)).astype("f4")
    elif dist == "flat":
        scores.fill(-np.inf)
    x = Buffer(scores)
    vis = Buffer(np.linspace(0, width, rows, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    mx, ix = out((rows, nb), "u4"), out((rows, k), "i8")
    ref = out((rows, k), "i4")
    native(opts.old_ops, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb)
    native(opts.old_ops, "topk_indices", rows, 1024, mx, L(nb), nb, k, every, ix)
    native(opts.old_ops, "pool_pick", rows, 1024, mx, L(nb), nb, ix, k, P(), 0, ref)
    new = out((rows, k), "i4")

    def chain():
        native(opts.new_ops, "idx_fused", rows, block, x, L(ss), width, 8, vis, every, k, P(), 0, new, shared=nb * 4)

    chain()
    diff = compare(f"idx-hist R={rows} nb={nb} block={block} {dist}", [new], [ref])
    row = {"kernel": "0195-idx-hist", "shape": label, "rows": rows, "nb": nb, "block": block,
           "parity": {"new_vs_ref": diff}}
    row["bytes"] = rows * width * 4 + rows * 16 + rows * k * 4
    if DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, block, nb * 4)
        attach_timing(row, label, [chain], ["new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0195 key fold (queries.py _score_keys' outputs written by the fused pool launch; the kit cubin is the byte reference)

def case_idx_fold(rows, width, dist="mixed", special=False, timing=True):
    """The fold's keys/tmax against the kit's _score_keys bytes, and the fused pool's cblk against the unfused chain.

    Widths are %4 == 0 (the kit faults otherwise) and inside the fused pool's regime: width > candidate_blocks * 8
    (16,384 columns) and nbk = ceil(width/8) <= 8192. Arms: ref = block_max+topk_indices+pool_pick + kit _score_keys,
    old = the fused pool without the fold + kit _score_keys, new = the fused pool with the fold. Timing is old vs new
    (the launch the fold removes is inside old), with the kit launch priced separately and the empty-kernel floor.
    """
    nbk = 1 + (width - 1) // 8
    nt = (width + 63) // 64
    k = min(2048, nbk)
    assert width % 4 == 0 and 16384 < width <= 65536, (width, nbk)
    label = f"idx-fold R={rows} width={width} nbk={nbk} k={k} {dist}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    scores = rng.integers(-10, 10, (rows, width)).astype("f4")
    if dist == "random":
        scores = rng.normal(size=(rows, width)).astype("f4")
    elif dist == "flat":
        scores.fill(-np.inf)
    elif dist == "mixed":
        scores = rng.normal(size=(rows, width)).astype("f4")  # both signs: where the old encoding's order flips
    if special:
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000], "u4")
        scores[:, ::7] = np.resize(bits.view("f4"), scores[:, ::7].shape)
        scores[0, 0], scores[-1, width - 1] = 0.0, np.inf
    x = Buffer(scores)
    vis = Buffer(np.full(rows, width, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    chains, states = [], []
    # WP1 closed the fold (0206) with no engine change: a tree whose idx_fused has no keys/tmax parameters cannot
    # write them, and the case then compares the mask/list only.
    fold_present = param_count(opts.new_ops, "tf_ds_idx_fused_kernel") >= 13
    if not fold_present:
        print(f"  {label}: the tree's idx_fused has no fold outputs; comparing the mask/list only", flush=True)
    for i, mod in enumerate((opts.old_ops, opts.old_ops, opts.new_ops, opts.new_ops)):
        mx, ix = out((rows, nbk), "u4"), out((rows, k), "i8")
        blocks = out((rows, k), "i4")
        keys, tmax = out((rows, width), "u8"), out((rows, nt), "u8")

        def chain(i=i, mod=mod, mx=mx, ix=ix, blocks=blocks, keys=keys, tmax=tmax):
            if i == 0:
                native(mod, "block_max", ((nbk + 255) // 256, rows), 256, x, L(width), width, 8, vis, mx, L(nbk), nbk)
                native(mod, "topk_indices", rows, 1024, mx, L(nbk), nbk, k, every, ix)
                native(mod, "pool_pick", rows, 1024, mx, L(nbk), nbk, ix, k, P(), 0, blocks)
            elif i == 1:
                native(mod, "idx_fused", rows, 1024, x, L(width), width, 8, vis, every, k, P(), 0, blocks, shared=nbk * 4)
            elif i == 2:
                if fold_present:
                    native(mod, "idx_fused", rows, 1024, x, L(width), width, 8, vis, every, k, P(), 0, blocks, keys,
                           tmax, shared=nbk * 4)
                else:
                    native(mod, "idx_fused", rows, 1024, x, L(width), width, 8, vis, every, k, P(), 0, blocks,
                           shared=nbk * 4)
            else:  # the same new kernel without the fold: the fold's marginal cost
                native(mod, "idx_fused", rows, 1024, x, L(width), width, 8, vis, every, k, P(), 0, blocks, P(), P(),
                       shared=nbk * 4)
            if i != 2:  # the kit's bytes are the reference for keys/tmax; the fold arm's are the fold's
                aot("scorekeys", (rows, (width + 1023) // 1024), x, keys, width, tmax, nt)

        chain()
        chains.append(chain)
        states.append([blocks, keys, tmax])
    ref_state = states[0]
    nlive = 3 if fold_present else 1
    diff_old = compare(f"idx-fold old R={rows} width={width}", states[1][:nlive], ref_state[:nlive])
    diff_new = compare(f"idx-fold new R={rows} width={width}", states[2][:nlive], ref_state[:nlive])
    diff_cross = compare(f"idx-fold new-vs-old R={rows} width={width}", states[2][:nlive], states[1][:nlive])
    # the kit wrote keys/tmax into the same buffers on every arm: the reference is the kit's own bytes
    row = {"kernel": "0195-idx-fold", "shape": label, "rows": rows, "width": width, "nbk": nbk, "k": k,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old, "new_vs_old": diff_cross},
           "fold_present": fold_present}
    row["bytes"] = rows * width * 4 + rows * width * 8 + rows * nt * 8 + rows * k * 4
    if timing and DO_TIMING and not special:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, nbk * 4)
        attach_timing(row, label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


def case_score_keys_only(rows, width):
    """The kit _score_keys launch alone at a fold width: the launch the fold removes."""
    label = f"kit _score_keys R={rows} width={width}"
    if not shape_ok(label):
        return None
    scores = rng.normal(size=(rows, width)).astype("f4")
    x = Buffer(scores)
    keys, tmax = out((rows, width), "u8"), out((rows, (width + 63) // 64), "u8")
    nt = (width + 63) // 64

    def chain():
        aot("scorekeys", (rows, (width + 1023) // 1024), x, keys, width, tmax, nt)

    chain()
    cu("cuStreamSynchronize", stream)
    row = {"kernel": "kit-scorekeys", "shape": label, "rows": rows, "width": width, "parity": {"new_vs_ref": 0}}
    row["bytes"] = rows * width * 4 + rows * width * 8 + rows * nt * 8
    if DO_TIMING:
        attach_timing(row, label, [chain], ["kit"], None, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0197 compressors

def bf(shape):
    return (rng.normal(size=shape).astype("f4").view("u4") >> 16).astype("u2")


def bf16_bits(v):
    return int(np.float32(v).view("u4") >> 16)


def bf16_round(v):
    u = int(np.float32(v).view("u4"))
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return np.uint32(u & 0xFFFFFFFF).view("f4")


def adv_query_hits():
    """(xe, xo, cos, sin) with bf16 xe/xo where the kit and torch odd-element contractions round to different bf16
    values and one of the two has an E2M1-boundary mantissa (1.000, 1.010, 1.100, 1.110): host search, f64 emulation
    of the f32 fma (the GPU is the arbiter)."""
    g = np.random.default_rng(7)
    out = []
    to_bf = lambda x: ((x.astype("f4").view("u4").astype("u8") + 0x7FFF + ((x.astype("f4").view("u4") >> 16) & 1))
                       >> 16 << 16).astype("u4").view("f4")
    while len(out) < 64:
        n = 2_000_000
        xe, xo = to_bf(g.standard_normal(n).astype("f4")), to_bf(g.standard_normal(n).astype("f4"))
        a = g.uniform(-np.pi, np.pi, n)
        cs, sn = np.cos(a).astype("f4"), np.sin(a).astype("f4")
        f = np.float64
        kit = to_bf((f(xe) * f(sn) + f((f(xo) * f(cs)).astype("f4"))).astype("f4"))
        tor = to_bf((f(xo) * f(cs) + f((f(xe) * f(sn)).astype("f4"))).astype("f4"))
        for i in np.nonzero(kit != tor)[0]:
            if any(((np.float32(abs(v)).view("u4") >> 16) & 0x7F) in (0x00, 0x20, 0x40, 0x60) for v in (kit[i], tor[i])):
                out.append((xe[i], xo[i], cs[i], sn[i]))
    return out[:64]


def case_compress_decode(rows, d, layout="mixed", timing=True, special=False):
    n = rows * 2
    stride = d + int(layout == "unaligned")
    label = f"compress-decode R={rows} D={d} {layout}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    slots = rng.permutation(n)[:rows].astype("i8")
    if layout == "cached":
        pair = np.tile(np.setdiff1d(np.arange(n), slots), (2, 1)).T.copy()
    elif layout in ("cross", "invalid", "stress"):
        pair = np.column_stack((np.roll(slots, 1), np.roll(slots, -1)))
        if layout == "invalid":
            pair[::5, 1] = -1
            pair[::7, 0] = n
    else:
        pair = np.column_stack((slots, np.setdiff1d(np.arange(n), slots)))
        pair[1::2] = pair[1::2, ::-1]
    inputs = [rng.normal(size=(rows, stride)).astype("f4") for _ in range(2)]
    rings = [rng.normal(size=(n, d)).astype("f4") for _ in range(2)]
    if special or layout == "stress":
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000], "u4")
        for a in (inputs[1], rings[1]):
            a[:, ::7] = np.resize(bits.view("f4"), a[:, ::7].shape)
        inputs[0][:, ::5] = np.resize(bits.view("f4"), inputs[0][:, ::5].shape)
    x, gate = map(Buffer, inputs)
    slot, gi = Buffer(slots), Buffer(pair)
    w = Buffer(bf(d))
    chains, states = [], []
    # newdup (4th arm): the new chain again into fresh buffers, so a nonzero new-vs-newdup isolates a nondeterministic
    # kernel (a cross-block ring read) from a parity difference.
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch, opts.new_torch)):
        rk, rg = map(Buffer, rings)
        flag = Buffer(np.zeros(1, "u4"))
        mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")
        prefix = (x, L(stride), gate, L(stride), slot, rk, rg, gi, L(n))

        def chain(i=i, mod=mod, prefix=prefix, dst=dst, mid=mid, flag=flag):
            if i == 0:
                native(mod, "compress2x", rows, 256, *prefix, mid, L(d), rows, d, flag)
                aot(f"norm{d}", rows, mid, d, w, dst, d + 16, eps)
            else:
                # the per-width kernels where a tree exports them, the older combined one otherwise
                native_any(mod, ["compress2x_norm128" if d == 128 else "compress2x_norm512", "compress2x_norm"],
                           rows, d, *prefix, w, dst, L(d + 16), rows, d, eps, flag)

        chain()
        chains.append(chain)
        states.append([dst, rk, rg, flag])
    diff_new = compare(f"decode compressor new D={d} R={rows}", states[2], states[0])
    diff_old = compare(f"decode compressor old D={d} R={rows}", states[1], states[0])
    diff_dup = compare(f"decode compressor new-vs-newdup D={d} R={rows}", states[3], states[2])
    row = {"kernel": "0197-compress-decode", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old, "new_vs_newdup": diff_dup}}
    row["bytes"] = rows * d * 4 * 4 + rows * 2 * d * 4 + rows * (d + 16) * 2 + 2 * n * d * 4
    if timing and DO_TIMING and not special and layout == "mixed":
        floor = lambda: launch(opts.floor, "empty_kernel", rows, d, 0)
        attach_timing(row, label, chains, ["ref", "old", "new", "newdup"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


def case_compress_prompt(rows, d, timing=True, special=False):
    count = 2 * rows
    label = f"compress-prompt R={rows} D={d}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    hx, hs = rng.normal(size=(count, d)).astype("f4"), rng.normal(size=(count, d)).astype("f4")
    if special:
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000], "u4")
        for a in (hx, hs):
            a[:, ::7] = np.resize(bits.view("f4"), a[:, ::7].shape)
    x, s = Buffer(hx), Buffer(hs)
    w = Buffer(bf(d))
    lanes = rms_lanes(d, rows)
    factor = F(np.float32(rows) / np.float32(rows * d))
    chains, states = [], []
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch)):
        mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")

        def chain(i=i, mod=mod, mid=mid, dst=dst):
            if i == 0:
                native(mod, "compress2", (rows * d + 255) // 256, 256, x, s, mid, rows, d)
                height = min(512 // lanes, 64)
                native(mod, "rms_norm", (rows + height - 1) // height, (lanes, height), mid, L(d), w, dst,
                       L(d + 16), rows, d, factor, eps, shared=lanes * height * 4 if lanes > 32 else 0)
            elif param_count(mod, "tf_ds_compress2_norm_kernel") >= 16:
                # the raw-ring window this tree's kernel folds in (zeros: the caller scatters them)
                groups = max(1, min(d, prompt_target(rows)) // lanes)
                native(mod, "compress2_norm", rows, (lanes, groups), x, s, w, dst, L(d + 16), rows, d,
                       factor, eps, 0, 0, 0, 0, L(0), 2 * rows, 2 * rows)
            else:  # a tree before the window fold
                groups = max(1, min(d, prompt_target(rows)) // lanes)
                native(mod, "compress2_norm", rows, (lanes, groups), x, s, w, dst, L(d + 16), rows, d,
                       factor, eps)

        chain()
        chains.append(chain)
        states.append([dst])
    diff_new = compare(f"prompt compressor new D={d} R={rows}", states[2], states[0])
    diff_old = compare(f"prompt compressor old D={d} R={rows}", states[1], states[0])
    row = {"kernel": "0197-compress-prompt", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old}}
    row["bytes"] = count * d * 4 * 2 + rows * (d + 16) * 2
    if timing and DO_TIMING and not special:
        groups = max(1, min(d, prompt_target(rows)) // lanes)
        floor = lambda: launch(opts.floor, "empty_kernel", rows, (lanes, groups), 0)
        attach_timing(row, label, chains, ["ref", "old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0198 rope-store

def case_rope(path, rows, special=False, timing=True):
    key = path.endswith("key")
    prompt = path.startswith("prompt")
    query = path == "query"
    heads = 32 if query else 1
    d = 128 if key or query else 512
    adv = special == "adv"
    label = f"rope {path} R={rows} D={d}{' adv' if adv else ' special' if special else ''}"
    if not shape_ok(label):
        return None
    src = bf((rows, heads, d))
    if adv:
        src[:] = 0
    elif special:
        src[..., :8] = [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0, 1, 0x7F7F, 0x8001]
        src[..., 32:64] = 0x7FC1
    weights = Buffer(bf(d))
    if prompt:  # the fused stores derive pos/group as base + r; the unfused chain reads the same contiguous run
        base = int(rng.integers(0, 256))
        run = np.arange(base, base + rows, dtype="i8")   # kept alive: the upload below is asynchronous
        pos = Buffer(run)
        dest = Buffer(run)
    else:
        pos = Buffer(rng.permutation(rows).astype("i8"))
        dest = Buffer(rng.permutation(rows).astype("i8"))
    # the rotation tables are indexed by absolute position, so a prompt case with a nonzero group base needs rows
    angle = rng.uniform(-np.pi, np.pi, ((base + rows) if prompt else rows, 32))
    cos_t, sin_t = np.cos(angle).astype("f4"), np.sin(angle).astype("f4")
    if adv:
        # B1 (review): inputs where the kit's odd-element contraction fma(xe, sn, xo*cs) and the torch form
        # fma(xo, cs, xe*sn) round to different bf16 values, at a value whose magnitude sits on an E2M1 boundary
        # pattern; head h sets the group's maximum A = 2^(e + h % 8 - 1) at another pair (cos 1, sin 0), so some
        # head's scale puts the boundary inside the code grid. Each row's position picks the hit.
        hits = adv_query_hits()
        cos_t[:] = 1.0
        sin_t[:] = 0.0
        pos_host = rng.permutation(rows).astype("i8")
        pos = Buffer(pos_host)
        for r in range(rows):
            xe, xo, c_, s_ = hits[int(pos_host[r]) % len(hits)]
            cos_t[pos_host[r], 0], sin_t[pos_host[r], 0] = c_, s_
            e = int(np.floor(np.log2(abs(float(c_) * float(xo) + float(s_) * float(xe)) + 1e-30)))
            for h in range(heads):
                src[r, h, d - 64] = bf16_bits(xe)
                src[r, h, d - 63] = bf16_bits(xo)
                src[r, h, d - 64 + 16] = bf16_bits(np.float32(2.0 ** (e + h % 8 - 1)))
    cos, sin = Buffer(cos_t), Buffer(sin_t)
    if os.environ.get("FUSED_BENCH_DEBUG"):
        print(f"  {label}: base={base if prompt else None} rows={rows} angle={angle.shape} src={src.shape}",
              flush=True)
    chains, states = [], []
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch)):
        x, rotated, dequant = Buffer(src), out(src.shape, "u2"), out(src.shape, "u2")
        # a prompt store writes row base + r (the group's slot in the ring), so the packed caches are indexed from 0
        packed_rows = base + rows if prompt else rows
        codes, scales = out((packed_rows, d // 2), "u1"), out((packed_rows, d // (32 if key else 16)), "u1")

        def chain(i=i, mod=mod, x=x, rot=rotated, dq=dequant, codes=codes, scales=scales):
            if prompt and i < 2:
                # prompt.zig stages the group ids and rotation positions as arrays for every path of the old tree;
                # this tree's fused stores derive them from the chunk, so only the old arms carry the two uploads.
                cu("cuMemcpyHtoDAsync_v2", pos.addr, P(run.ctypes.data), ct.c_size_t(rows * 8), stream,
                   ct.c_uint(0))
                cu("cuMemcpyHtoDAsync_v2", dest.addr, P(run.ctypes.data), ct.c_size_t(rows * 8), stream,
                   ct.c_uint(0))
            if path == "prompt latent" and i == 0:
                native(opts.old_ops, "copy_rows16", (1, rows), 256, x, L(d // 8), rot, L(d // 8), L(d // 8))
            if i:
                if key and prompt:
                    lanes = rms_lanes(d, rows)
                    y = pack_lanes(lanes, d // 2)
                    # both trees derive the group id and rotation position from the chunk (the pos/dest uploads
                    # above are the unfused chain's; the fused arms do not read them)
                    native(mod, "fused_prompt_idx_key", rows, (lanes, y), x, L(d), weights, eps, F(1 / d),
                           cos, sin, L(base), 1, codes, scales, d, 64, 0)
                elif key:
                    native(mod, "fused_idx_key", rows, 128, x, L(d), weights, cos, sin, pos, codes,
                           scales, dest, d, 64, eps, 0)
                elif query:
                    # the tree's own dispatch (0241: a warp a head above decode's largest window): grid.y is the head,
                    # so both arms launch the same geometry (a flat rows*heads grid is the pre-0241 shape and reads
                    # past the rows on every tree from 0241 on)
                    if rows >= 32:
                        native(mod, "fused_rope_qd_pair", (rows, heads // 8), 256, x, dq, cos, sin, pos, heads, d,
                               64, 0)
                    else:
                        native(mod, "fused_rope_qd", (rows, heads), 128, x, dq, cos, sin, pos, heads, d, 64, 0)
                elif prompt:
                    native(mod, "fused_prompt_latent", rows, min(d // 2, 256), x, cos, sin, L(base), 1, codes,
                           scales, d, 64, 0)
                else:
                    native(mod, "fused_rope_store", rows, min(d // 2, 256), x, cos, sin, pos, codes, scales,
                           dest, d, 64, 0)
                return
            if key:
                if prompt:
                    lanes = rms_lanes(d, rows)
                    height = min(512 // lanes, 64)
                    native(mod, "rms_norm", (rows + height - 1) // height, (lanes, height), x, L(d), weights,
                           rot, L(d), rows, d, F(1 / d), eps)
                else:
                    aot("norm128", rows, x, d, weights, rot, d, eps)
            target = rot if key or prompt else x
            if prompt:
                native(mod, "rope", (rows * 32 + 255) // 256, 256, target, L(d), d - 64, cos, sin, pos,
                       rows, 32, 0)
            else:
                aot("rope32" if query else f"rope{d}", (rows, 2 if query else 1), target, cos, sin, pos)
            if query:
                nblk = rows * heads * d // 32
                aot("qd", (nblk + 31) // 32, target, dq, nblk)
            else:
                aot(f"store{d}", rows, target, d, codes, scales, dest)

        chain()
        chains.append(chain)
        states.append([x, rotated, dequant, codes, scales])
    if query:
        live = [2]
    elif path == "prompt latent":
        live = [3, 4]
    else:
        live = [3, 4]
    if adv:
        cu("cuStreamSynchronize", stream)
        rot = states[0][0].readarr()[:, 0, d - 63].astype("u4") << 16   # the kit's rotated odd element, head 0
        tally = {"kit": 0, "torch": 0, "both": 0, "neither": 0}
        for r in range(rows):
            xe, xo, c_, s_ = hits[int(pos_host[r]) % len(hits)]
            f = np.float64
            kit = np.float32(f(xe) * f(s_) + f(np.float32(f(xo) * f(c_))))
            tor = np.float32(f(xo) * f(c_) + f(np.float32(f(xe) * f(s_))))
            kb, tb = bf16_bits(bf16_round(kit)), bf16_bits(bf16_round(tor))
            g = int(rot[r] >> 16)
            tally["both" if g == kb == tb else "kit" if g == kb else "torch" if g == tb else "neither"] += 1
        dq_ref, dq_new = states[0][2].readarr(), states[2][2].readarr()
        print(f"  adv R={rows}: kit rope matches {tally}; dq rows differing new vs ref: "
              f"{int((dq_ref != dq_new).any(axis=(1, 2)).sum())}", flush=True)
    diff_new = compare(f"{path} new R={rows}", [states[2][i] for i in live], [states[0][i] for i in live])
    diff_old = compare(f"{path} old R={rows}", [states[1][i] for i in live], [states[0][i] for i in live])
    row = {"kernel": f"0198-{path.replace(' ', '-')}", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old}}
    if query:
        row["bytes"] = rows * heads * d * 2 * 2
    else:
        row["bytes"] = rows * d * 2 + rows * (d // 2) + rows * (d // (32 if key else 16))
    if timing and DO_TIMING and not special:
        if key and prompt:
            lanes = rms_lanes(d, rows)
            block = (lanes, pack_lanes(lanes, d // 2))
        elif key or query:
            block = 128
        else:
            block = min(d // 2, 256)
        if query:
            grid = (rows, heads // 8) if rows >= 32 else (rows, heads)
            block = 256 if rows >= 32 else 128
        else:
            grid = rows
        floor = lambda: launch(opts.floor, "empty_kernel", grid, block, 0)
        attach_timing(row, label, chains, ["ref", "old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


def case_latent_values(timing=False):
    """Every one of the 65,536 bf16 encodings through the latent store, against the kit's rope + fp4 store."""
    label = "all 65,536 bf16 encodings"
    if not shape_ok(label):
        return None
    d = 512
    values = np.arange(65536, dtype="u2")
    blocks = np.repeat(values[:, None], 16, axis=1)
    blocks[:, 1::2] = rng.permutation(values)[:, None]
    source = blocks.reshape(-1, d)
    rows = len(source)
    x, rot = Buffer(source), Buffer(source)
    cos = Buffer(np.ones((1, 32), "f4"))
    sin = Buffer(np.zeros((1, 32), "f4"))
    pos = Buffer(np.zeros(rows, "i8"))
    dst = Buffer(np.arange(rows, dtype="i8"))
    codes, scales = out((rows, d // 2), "u1"), out((rows, d // 16), "u1")
    expected = [out((rows, d // 2), "u1"), out((rows, d // 16), "u1")]
    aot("rope512", rows, x, cos, sin, pos)
    aot("store512", rows, x, d, *expected, dst)
    native(opts.new_torch, "fused_rope_store", rows, min(d // 2, 256), rot, cos, sin, pos, codes, scales, dst, d, 64, 0)
    diff = compare("all bf16 encodings through packed latent", [codes, scales], expected)
    results.append({"kernel": "0198-latent", "shape": label, "parity": {"new_vs_ref": diff}})
    print(f"all 65,536 bf16 encodings through the latent store: {diff} differing bytes", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# 0199 six-slot prune

class TfDsPrune(ct.Structure):
    _fields_ = [("topp", ct.c_float), ("min_k", ct.c_int), ("renorm", ct.c_int), ("apply", ct.c_int),
                ("topk", ct.c_int), ("slots", ct.c_int), ("rows", ct.c_int), ("cap", ct.c_int),
                ("cap_min", ct.c_int)]


def case_prune(rows, dead_mode=False, special=False, timing=True):
    slots, topk = 7, 6
    label = f"prune R={rows}{' dead' if dead_mode else ''}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    pick = rng.integers(0, 512, (rows, slots), dtype="i4")
    wts = rng.random((rows, slots)).astype("f4") + 0.01
    if special:
        wts[::3] = np.resize(np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, 3.0], "f4"), wts[::3].shape)
        pick[::4, -1] = -1
        pick[::5, 1] = pick[::5, 2]  # tie on expert id
        wts[::2, 1] = wts[::2, 2]    # tie on weight
    p0, w0 = pick.copy(), wts.copy()
    dead = np.zeros(rows, "u1")
    if dead_mode:
        dead[::3] = 1
    args = TfDsPrune(0.85, 3, 1, 1, topk, slots, rows, 0, 0)
    chains, states = [], []
    for i, mod in enumerate((opts.old_ops, opts.old_ops, opts.new_ops, opts.new_ops)):
        pk, wt = Buffer(p0.copy()), Buffer(w0.copy())
        stats = Buffer(np.zeros(6, "u8"))
        dh = Buffer(dead)

        def chain(i=i, mod=mod, pk=pk, wt=wt, stats=stats, dh=dh):
            if i == 0:
                native(mod, "expert_prune", 1, 32 if rows <= 32 else 64, pk, wt, args, stats,
                       dh if dead_mode else P())
            else:
                native(mod, "prune_six", 1, 32 if rows <= 32 else 64, pk, wt, args, stats,
                       dh if dead_mode else P())

        chain()
        chains.append(chain)
        states.append([pk, wt, stats])
    diff_new = compare(f"prune new R={rows}", states[2], states[0])
    diff_old = compare(f"prune old R={rows}", states[1], states[0])
    row = {"kernel": "0199-prune", "shape": label, "rows": rows,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old}}
    row["bytes"] = rows * slots * 8
    if timing and DO_TIMING and not special:
        floor = lambda: launch(opts.floor, "empty_kernel", 1, 32 if rows <= 32 else 64, 0)
        attach_timing(row, label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# The unfused fallbacks WP1-WP3's switches fall back to: the new tree's own kernels, against the old tree's, so a change
# to a fused arm cannot quietly alter the arm that runs when a gate refuses (past max_nbk, a width with no kit norm).

def case_fallback_idx(rows, nb, k):
    """nbk past idx_fused.max_nbk (16385 blocks = 131,080 columns): the engine runs block_max + topk_indices + pool_pick.
    Both trees launch the same three kernels; this checks the new tree's bytes against the old tree's."""
    width, ss, ms = 8 * nb - 3, 8 * nb, nb + 7
    label = f"fallback idx R={rows} nb={nb} width={width} k={k}"
    if not shape_ok(label):
        return None
    x = Buffer(rng.normal(size=(rows, ss)).astype("f4"))
    vis = Buffer(np.linspace(0, width, rows, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    chains, states = [], []
    for mod in (opts.old_ops, opts.new_ops):
        mx, ix, blocks = out((rows, nb), "u4"), out((rows, k), "i8"), out((rows, k), "i4")

        topk = index_symbol(mod, "topk_indices", "topk_indices_cut")

        def chain(mod=mod, mx=mx, ix=ix, blocks=blocks, topk=topk):
            native(mod, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb)
            native(mod, topk, rows, 1024, mx, L(nb), nb, k, every, ix)
            native(mod, "pool_pick", rows, 1024, mx, L(nb), nb, ix, k, P(), 0, blocks)

        chain()
        chains.append(chain)
        states.append([blocks])
    diff = compare(f"{label} new_vs_old", states[1], states[0])
    row = {"kernel": "0195-idx-fallback", "shape": label, "rows": rows, "nb": nb, "parity": {"new_vs_old": diff}}
    row["bytes"] = rows * width * 4 + rows * nb * 4 + rows * k * 4
    if DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, 0)
        attach_timing(row, label, chains, ["old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


def case_fallback_compress(rows, d):
    """A width with no kit specialization (decodeOk takes 128 and 512): the engine runs compress2x + the kit norm. The
    norm is the kit's own launch in both arms, so the bytes this checks are compress2x's (the pair sum) and the raw
    ring rows the scatter writes; there is no AOT kit norm cubin at these widths to launch here."""
    n = rows * 2
    label = f"fallback compress-decode R={rows} D={d}"
    if not shape_ok(label):
        return None
    slots = rng.permutation(n)[:rows].astype("i8")
    pair = np.column_stack((slots, np.setdiff1d(np.arange(n), slots)))
    pair[1::2] = pair[1::2, ::-1]
    x, gate = map(Buffer, (rng.normal(size=(rows, d)).astype("f4"), rng.normal(size=(rows, d)).astype("f4")))
    slot, gi = Buffer(slots), Buffer(pair)
    ring0 = rng.normal(size=(n, d)).astype("f4")   # the same ring bytes for both arms
    ring1 = rng.normal(size=(n, d)).astype("f4")
    chains, states = [], []
    for mod in (opts.old_torch, opts.new_torch):
        rk, rg = Buffer(ring0), Buffer(ring1)
        flag = Buffer(np.zeros(1, "u4"))
        mid = out((rows, d), "u2")

        def chain(mod=mod, rk=rk, rg=rg, flag=flag, mid=mid):
            native(mod, "compress2x", rows, 256, x, L(d), gate, L(d), slot, rk, rg, gi, L(n), mid, L(d), rows, d, flag)

        chain()
        chains.append(chain)
        states.append([mid, rk, rg])
    diff = compare(f"{label} new_vs_old", states[1], states[0])
    row = {"kernel": "0197-compress-fallback", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_old": diff}}
    row["bytes"] = rows * d * 4 * 2 + 2 * n * d * 4 + rows * d * 2
    if DO_TIMING:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, d, 0)
        attach_timing(row, label, chains, ["old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# kit _score_keys (the launch 0195's fold replaces; not part of this tree's fused path, priced for the baseline)


def case_prompt_source(rows, d, special=False, timing=True, odd=False):
    """The ratio-2 KV-source step of a prompt chunk: the old chain (index upload + two raw-ring scatters +
    compress2 + rms_norm) against this tree's fused launch, which writes both raw tails and the normalized latent.
    Live outputs: both raw-ring buffers and the compressed latent row."""
    label = f"prompt-source R={rows} D={d}{' odd' if odd else ''}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    n = 2 * rows + int(odd)      # chunk rows at ratio 2 (odd: a trailing row with no pair)
    ring_rows = 64               # prompt.zig's raw_rows
    keep = min(n, ring_rows)
    first = n - keep
    start = 128                  # chunk start (a group boundary)
    hx = rng.normal(size=(n, d)).astype("f4")
    hs = rng.normal(size=(n, d)).astype("f4")
    if special:
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000], "u4")
        for a in (hx, hs):
            a[:, ::7] = np.resize(bits.view("f4"), a[:, ::7].shape)
    x, sc = Buffer(hx), Buffer(hs)
    tail_x, tail_s = Buffer(hx[first:]), Buffer(hs[first:])
    w = Buffer(bf(d))
    idx = np.array([(start + first + j) % ring_rows for j in range(keep)], "i8")
    idxb = Buffer(idx)
    lanes = rms_lanes(d, rows) if rows else 32   # rows == 0: a one-row chunk, no fused launch
    groups = max(1, min(d, prompt_target(rows)) // lanes)
    factor = F(np.float32(rows) / np.float32(rows * d)) if rows else F(1.0)
    raw_pos0 = (start + first) % ring_rows
    chains, states = [], []
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch)):
        rk = Buffer(np.zeros((ring_rows, d), "f4"))
        rg = Buffer(np.zeros((ring_rows, d), "f4"))
        lat = out((max(rows, 1), d + 16), "u2")
        mid = out((max(rows, 1), d), "u2")

        def chain(i=i, mod=mod, rk=rk, rg=rg, lat=lat, mid=mid):
            if i < 2:
                cu("cuMemcpyHtoDAsync_v2", idxb.addr, P(idx.ctypes.data), ct.c_size_t(idx.nbytes), stream,
                   ct.c_uint(0))
                native(opts.old_ops, "scatter_rows", keep, 256, tail_x, L(d * 4), idxb, rk, L(d * 4), L(d * 4), keep)
                native(opts.old_ops, "scatter_rows", keep, 256, tail_s, L(d * 4), idxb, rg, L(d * 4), L(d * 4), keep)
            if i == 0 and rows:  # the unfused chain: compress2 + the norm (no pair at rows == 0)
                native(mod, "compress2", (rows * d + 255) // 256, 256, x, sc, mid, rows, d)
                height = min(512 // lanes, 64)
                native(mod, "rms_norm", (rows + height - 1) // height, (lanes, height), mid, L(d), w, lat,
                       L(d + 16), rows, d, factor, eps, shared=lanes * height * 4 if lanes > 32 else 0)
            elif i == 1 and rows:  # the WP1 tree's fused compressor (the launch this item replaces)
                if param_count(mod, "tf_ds_compress2_norm_kernel") >= 16:
                    # the raw-ring window is already written by the scatters above: zeros leave it to them
                    native(mod, "compress2_norm", rows, (lanes, groups), x, sc, w, lat, L(d + 16), rows, d,
                           factor, eps, 0, 0, 0, 0, L(0), 2 * rows, 2 * rows)
                else:
                    native(mod, "compress2_norm", rows, (lanes, groups), x, sc, w, lat, L(d + 16), rows, d,
                           factor, eps)
            elif rows == 0:  # prompt.zig keeps the upload + scatters when the chunk has no complete pair
                cu("cuMemcpyHtoDAsync_v2", idxb.addr, P(idx.ctypes.data), ct.c_size_t(idx.nbytes), stream,
                   ct.c_uint(0))
                native(opts.old_ops, "scatter_rows", keep, 256, tail_x, L(d * 4), idxb, rk, L(d * 4), L(d * 4), keep)
                native(opts.old_ops, "scatter_rows", keep, 256, tail_s, L(d * 4), idxb, rg, L(d * 4), L(d * 4), keep)
            else:  # this tree: one launch, raw tails included
                native(mod, "compress2_norm", (n + 1) // 2, (lanes, groups), x, sc, w, lat, L(d + 16), rows, d,
                       factor, eps, 0, rk, rg, ring_rows, L(raw_pos0), first, n)

        chain()
        chains.append(chain)
        states.append([rk, rg, lat])
    diff_new = compare(f"prompt source new R={rows} D={d}", states[2], states[0])
    diff_old = compare(f"prompt source old R={rows} D={d}", states[1], states[0])
    row = {"kernel": "0197-compress-prompt", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old}}
    row["bytes"] = 2 * n * d * 4 + 2 * keep * d * 4 + rows * (d + 16) * 2
    if timing and DO_TIMING and not special:
        floor = lambda: launch(opts.floor, "empty_kernel", (n + 1) // 2, (lanes, groups), 0)
        attach_timing(row, label, chains, ["ref", "old", "new"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


def case_score_keys(rows, width):
    label = f"kit _score_keys R={rows} width={width}"
    if not shape_ok(label):
        return None
    scores = rng.normal(size=(rows, width)).astype("f4")
    x = Buffer(scores)
    keys = out((rows, width), "i8")
    nt = (width + 63) // 64
    tmax = out((rows, nt), "i8")

    def chain():
        # tri_index.scoreKeys: grid (rows, cdiv(n, 1024), 1), BN=1024, tmax [R, cdiv(n, 64)]
        aot("scorekeys", (rows, (width + 1023) // 1024), x, keys, width, tmax, nt)

    chain()
    cu("cuStreamSynchronize", stream)
    row = {"kernel": "kit-scorekeys", "shape": label, "rows": rows, "width": width,
           "parity": {"new_vs_ref": 0}}
    row["bytes"] = rows * width * 4 + rows * width * 8 + rows * nt * 8
    if DO_TIMING:
        attach_timing(row, label, [chain], ["kit"], None, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# case registry

IDX_SHAPES = [
    (1, 8, 8, False, "ties"), (7, 8, 8, False, "ties"), (32, 8, 8, False, "ties"), (48, 8, 8, False, "ties"),
    (7, 4096, 2048, False, "random"), (32, 8192, 2048, False, "ties"),
    (128, 4096, 2048, True, "ties"), (1024, 4096, 2048, True, "ties"), (2048, 4096, 2048, True, "ties"),
    (7, 2049, 2048, False, "random"), (7, 8192, 2048, True, "flat"),
]

DECODE_SHAPES = [(r, d) for r in (1, 7, 32, 48) for d in (128, 512)]
PROMPT_SHAPES = [(r, d) for r in (128, 1024, 2048, 4096) for d in (512, 2048)]
ROPE_SHAPES = [("key", r) for r in (1, 7, 32, 48)] + \
              [("latent", r) for r in (1, 7, 32, 48)] + \
              [("query", r) for r in (1, 7, 32, 48)] + \
              [("query", r) for r in (128, 1024)] + \
              [("prompt key", r) for r in (128, 1024, 2048, 4096)] + \
              [("prompt latent", r) for r in (128, 1024, 2048, 4096)]
PRUNE_SHAPES = [1, 7, 32, 48]
SCORE_KEYS_SHAPES = [(r, w) for r in (1, 7, 32, 128) for w in (16384, 32768, 65536)]

KERNELS = ("idx", "idx-hist", "block-max", "idx-small", "topk-indices", "cand-mask", "fold",
           "compress-decode", "compress-prompt", "prompt-source",
           "key", "latent", "query", "prompt-key", "prompt-latent", "prune", "score-keys", "fallback")

FOLD_SHAPES = [(r, w) for r in (1, 7, 32, 48) for w in (16388, 32768, 65536)]


def selected(kernel):
    return opts.kernel == "all" or kernel in opts.kernel.split(",")


def shape_ok(label):
    return opts.shape is None or opts.shape in label


def main():
    global opts
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_bin", type=Path, help="directory with old-*.cubin")
    parser.add_argument("new_bin", type=Path, help="directory with new-*.cubin")
    parser.add_argument("kit", type=Path, help="pinned kit aot/cubins directory")
    parser.add_argument("--kernel", default="all", help="comma list: " + ",".join(KERNELS) + " or all")
    parser.add_argument("--shape", default=None, help="substring filter on the shape label")
    parser.add_argument("--json", type=Path, default=None, help="write the result rows as JSON here")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--no-guard", action="store_true", help="skip the memory/health/lock guards")
    parser.add_argument("--wait", type=int, default=1800, help="seconds to wait for an idle production (0: wait forever)")
    parser.add_argument("--tries", type=int, default=20, help="clean timing windows to try per block")
    opts = parser.parse_args()
    opts.old_ops = opts.old_bin / "old-ops.cubin"
    opts.old_torch = opts.old_bin / "old-torch.cubin"
    opts.new_ops = opts.new_bin / "new-ops.cubin"
    opts.new_torch = opts.new_bin / "new-torch.cubin"
    opts.floor = opts.new_bin / "floor.cubin"
    for p in (opts.old_ops, opts.old_torch, opts.new_ops, opts.new_torch, opts.floor):
        if not p.exists():
            sys.exit(f"missing {p} (run build.sh)")

    lock = None
    h0 = None
    if not opts.no_guard:
        avail = mem_available_gib()
        if avail < 3.0:
            sys.exit(f"MemAvailable {avail:.2f} GiB < 3 GiB, refusing")
        print(f"MemAvailable {avail:.2f} GiB", flush=True)
        lock = open(os.path.expanduser("~/tf-bench/gpu.lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            h0 = wait_health(opts.wait)
        except TimeoutError as exc:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            sys.exit(str(exc))
        counters0 = {k: h0.get(k) for k in ("prompt_tokens_total", "generation_tokens_total")}
        print(f"health before: {counters0}", flush=True)

    cu("cuInit", 0)
    ctx = P()
    cu("cuDevicePrimaryCtxRetain", ref(ctx), 0)
    cu("cuCtxSetCurrent", ctx)
    cu("cuStreamCreate", ref(stream), 1)

    global DO_TIMING, DO_PARITY
    DO_TIMING = not opts.parity_only
    DO_PARITY = not opts.timing_only

    def run(fn, *a):
        fn(*a)
    try:
        if selected("topk-indices"):
            for rows, width, k, dist in (
                (7, 1, 0, "random"), (7, 1, 1, "special"), (7, 255, 1, "special"),
                (7, 256, 127, "ties"), (7, 257, 17, "flat"), (7, 512, 512, "random"),
                (32, 2048, 2048, "ties"), (32, 2049, 2048, "flat"),
                (7, 8192, 2048, "random"), (32, 16385, 2048, "ties"),
                (128, 4096, 512, "random"), (128, 32768, 512, "random"),
                (128, 32768, 512, "flat"), (7, 65535, 512, "special"),
                (7, 65536, 512, "ties"), (7, 65537, 513, "special"),
                (32, 131072, 512, "random"),
            ):
                case_topk_indices(rows, width, k, dist)
        if selected("cand-mask"):
            for rows in (1, 7, 32, 48):
                for blocks in (128, 512, 2048):
                    case_candidate_mask(rows, blocks)
            case_candidate_mask(7, 257, offset=1)
            case_candidate_mask(7, 255, pad=1)
            case_candidate_mask(7, 257, size=3)
        if selected("idx"):
            for rows, nb, k, mask, dist in IDX_SHAPES:
                run(case_idx, rows, nb, k, mask, dist)
            run(case_idx, 7, 8, 8, False, "random", False, True)
            run(case_idx, 7, 8, 8, False, "flat", False, True)
            run(case_idx, 128, 4096, 2048, True, "random", False, True)
            run(case_idx, 7, 4096, 2048, False, "random", True, False)
            # the 100K+/200K buckets: nbk 8193 (past the old cap), 16384 (the new cap) and 16385 (the unfused fallback)
            run(case_idx, 7, 8193, 2048, False, "random")
            run(case_idx, 32, 8193, 2048, True, "ties")
            run(case_idx, 7, 16384, 2048, False, "random")
            run(case_idx, 32, 16384, 2048, True, "ties")
            run(case_idx, 7, 16385, 2048, False, "random")
            run(case_idx, 7, 4096, 4096, False, "random")   # k == nb: no radix select, compaction only
            run(case_idx, 7, 4096, 4096, True, "random")    # k == nb, mask only: phase 1 alone
            run(case_idx, 7, 8192, 8192, True, "random")
            run(case_idx, 32, 8192, 2048, False, "flat", False, False)
            run(case_idx, 128, 4096, 2048, True, "flat", False, False)
            run(case_idx, 32, 8192, 2048, False, "same", False, False)
            run(case_idx, 128, 4096, 2048, True, "same", False, False)
            run(case_idx, 7, 4096, 2048, False, "random", False, False, 1)   # ss % 4 != 0: the scalar fallback
            run(case_idx, 32, 8192, 2048, True, "ties", False, False, 3)
            free_all()
        if selected("idx-small"):
            for rows in (1, 7):
                for nb in (2049, 4096, 8192):
                    run(case_idx_small, rows, nb, 2048)
            run(case_idx_small, 7, 4096, 2048, "ties")
            free_all()
        if selected("block-max"):
            for rows, nb in ((1, 4096), (7, 4096), (7, 8192), (32, 8192)):
                case_blockmax(rows, nb)
            free_all()
        if selected("idx-hist"):
            for dist in ("random", "ties", "flat"):
                for block in (256, 512, 1024):
                    run(case_idx_hist, 7, 4096, 2048, dist, block)
            for block in (256, 512, 1024):
                run(case_idx_hist, 32, 8192, 2048, "ties", block)
            free_all()
        # the key fold (old 0206) was closed without landing: its cases need a tree that has the fold variant, so they
        # run only when asked for by name (--kernel fold), never under `all`
        if "fold" in opts.kernel.split(","):
            for rows, w in FOLD_SHAPES:
                run(case_idx_fold, rows, w)
            for rows, w in ((7, 16388), (7, 32768), (32, 65536)):
                run(case_idx_fold, rows, w, "random", False, False)
                run(case_idx_fold, rows, w, "ties", False, False)
                run(case_idx_fold, rows, w, "flat", False, False)
            for rows, w in ((1, 16388), (7, 65536), (32, 16388), (48, 65532)):
                run(case_idx_fold, rows, w, "mixed", True, False)
            run(case_idx_fold, 7, 65532, "mixed", False, False)  # a partial last tile (60 keys)
            for rows, w in FOLD_SHAPES:
                case_score_keys_only(rows, w)
            free_all()
        if selected("compress-decode"):
            for rows, d in DECODE_SHAPES:
                run(case_compress_decode, rows, d)
            for layout in ("unaligned", "cached", "cross", "invalid"):
                for r in (1, 7, 33, 65):
                    run(case_compress_decode, r, 512, layout, False, False)
            run(case_compress_decode, 7, 512, "stress", False, True)
            free_all()
        if selected("compress-prompt"):
            for rows, d in PROMPT_SHAPES:
                run(case_compress_prompt, rows, d)
            run(case_compress_prompt, 128, 512, False, True)
            run(case_compress_prompt, 1024, 2048, False, True)
        if selected("prompt-source"):
            for rows, d in ((1, 512), (7, 512), (48, 512), (128, 512), (1024, 512), (128, 2048)):
                run(case_prompt_source, rows, d)
            run(case_prompt_source, 7, 512, True, False)      # NaN/+-inf/+-0 in both inputs
            run(case_prompt_source, 1, 128, False, False)
            run(case_prompt_source, 0, 128, False, False, True)  # a one-row chunk (n = 1): no pair, the scatter path
            for rows, d in ((1, 512), (7, 512), (32, 512), (64, 512), (128, 512)):
                run(case_prompt_source, rows, d, False, True, True)   # odd trailing row (+ ring wrap at 32/64)
            free_all()
            free_all()
        for name, tag in (("key", "key"), ("latent", "latent"), ("query", "query"),
                          ("prompt-key", "prompt key"), ("prompt-latent", "prompt latent")):
            if not selected(name):
                continue
            for path, rows in ROPE_SHAPES:
                if path == tag:
                    run(case_rope, path, rows)
            run(case_rope, tag, 128 if tag.startswith("prompt") else 7, True, False)
            if tag == "query":
                for rows in (32, 48, 64, 1024):
                    run(case_rope, "query", rows, "adv", False)
            free_all()
        if selected("latent"):
            case_latent_values()
            for rows, own in ((7, 4), (48, 24), (48, 1), (32, 0)):
                case_latent_owner(rows, own)
            free_all()
        if selected("prune"):
            for rows in PRUNE_SHAPES:
                run(case_prune, rows)
            for rows in (7, 48):
                run(case_prune, rows, True)
            run(case_prune, 7, False, True, False)
            run(case_prune, 48, True, True, False)
            free_all()
        if selected("score-keys"):
            for rows, w in SCORE_KEYS_SHAPES:
                case_score_keys(rows, w)
            free_all()
        if selected("fallback"):
            # the arms the switches fall back to: past idx_fused.max_nbk and at widths decodeOk does not take
            # k <= 2048 is the unfused chain's own bound (topk_indices refuses past it: ops.topkIndices)
            for rows, nb, k in ((7, 16385, 2048), (32, 16385, 2048)):
                case_fallback_idx(rows, nb, k)
            for rows, d in ((1, 256), (7, 1024), (48, 256)):
                case_fallback_compress(rows, d)
            free_all()
    finally:
        free_all()
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)

    print("\n== parity")
    for row in results:
        p = row.get("parity", {})
        ok = all(v == 0 for v in p.values())
        print(f"{'PASS' if ok else 'FAIL'}  {row['kernel']:<24} {row['shape']:<52} {p}")
    print("\n== timing (us, median[iqr])")
    for row in results:
        t = row.get("timing")
        if not t:
            if row.get("timing_valid") is False:
                print(f"{row['kernel']:<24} {row['shape']:<52} INVALID (retries={row.get('timing_retries', 0)})")
            continue
        s = " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in t.items())
        print(f"{row['kernel']:<24} {row['shape']:<52} {s}")
    guarded = not opts.no_guard
    h1 = health_now() if guarded else None
    health = summarize_health(results, guarded, h0, h1, timing_retries)
    if guarded:
        counters1 = health["after"]
        if health["timing_clean"] and timing_retries == 0:
            print(f"health after: {counters1}  every timing block verified idle")
        elif health["timing_clean"]:
            print(f"health after: {counters1}  every timing block verified idle ({timing_retries} retry/retries)")
        else:
            print(f"health after: {counters1}  timing NOT clean (retries={timing_retries})")
    if opts.json:
        opts.json.write_text(json.dumps({"results": results, "failures": failures, "health": health}, indent=1))
        print(f"\nwrote {opts.json}")
    if failures:
        print(f"\n{failures and len(failures)} parity failure(s)")
        for f in failures[:40]:
            print("  " + f)
        sys.exit(1)
    print("\nPASS")


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def health_counters(h):
    if not isinstance(h, dict):
        return (None, None)
    return (h.get("prompt_tokens_total"), h.get("generation_tokens_total"))


def counters_valid(c):
    return (isinstance(c, tuple) and len(c) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in c))


def check_idle_health(h):
    if not isinstance(h, dict):
        return False
    live = h.get("live")
    if not isinstance(live, dict):
        return False
    conns = live.get("connections")
    return (isinstance(conns, int) and not isinstance(conns, bool) and conns == 0
            and counters_valid(health_counters(h)))


def served_counters():
    return health_counters(health_now())


def summarize_health(rows, guarded, before_health, after_health, total_retries):
    timed_rows = [r for r in rows if isinstance(r, dict) and ("timing" in r or "timing_valid" in r)]
    if not guarded:
        return {"guarded": False, "timing_retries": total_retries, "timing_clean": False}
    before_c = health_counters(before_health)
    after_c = health_counters(after_health)
    clean = (counters_valid(before_c)
             and counters_valid(after_c)
             and all(r.get("timing_valid") is True for r in timed_rows))
    return {
        "guarded": True,
        "before": {"prompt_tokens_total": before_c[0], "generation_tokens_total": before_c[1]},
        "after": {"prompt_tokens_total": after_c[0], "generation_tokens_total": after_c[1]},
        "timing_retries": total_retries,
        "timing_clean": clean,
    }


def health_now():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return {}


def wait_health(timeout=1800):
    deadline = time.time() + timeout if timeout else None
    while True:
        h = health_now()
        if check_idle_health(h):
            print("health: connections=0", flush=True)
            return h
        live = h.get("live") if isinstance(h, dict) else None
        conns = live.get("connections") if isinstance(live, dict) else None
        print(f"production not idle/valid (connections={conns}, counters={health_counters(h)}), waiting", flush=True)
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("timed out waiting for idle production health with valid counters")
        time.sleep(20)


if __name__ == "__main__":
    main()
