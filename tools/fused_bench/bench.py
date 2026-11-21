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
from pathlib import Path
from statistics import median

import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
U64 = ct.c_ulonglong
ref = ct.byref
cuda = ct.CDLL("libcuda.so.1")
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

def cu(name, *args):
    err = getattr(cuda, name)(*args)
    if err:
        msg = ct.c_char_p()
        cuda.cuGetErrorString(err, ref(msg))
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
        cu("cuMemcpyHtoD_v2", self.addr, P(data.ctypes.data), ct.c_size_t(data.nbytes))
        allocations.append(self.addr)
        allocated += data.nbytes

    def read(self):
        host = np.empty(self.shape, self.dtype)
        cu("cuMemcpyDtoH_v2", P(host.ctypes.data), self.addr, ct.c_size_t(host.nbytes))
        return host.view("u1").ravel()

    def readarr(self):
        host = np.empty(self.shape, self.dtype)
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


def launch(path, name, grid, threads, shared, *args):
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
    cu("cuLaunchKernel", fn, gx, gy, 1, bx, by, 1, shared, stream, argv, None)
    if os.environ.get("FUSED_BENCH_DEBUG"):
        cu("cuStreamSynchronize", stream)


def aot(key, grid, *args):
    prefix, name, smem = kit[key]
    (path,) = opts.kit.glob(prefix + "*.cubin")
    launch(path, name, grid, 128, smem, *args, P(), P())


def native(mod, name, grid, block, *args, shared=0):
    launch(mod, "tf_ds_" + name + "_kernel", grid, block, shared, *args, 0)


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


# ---------------------------------------------------------------------------------------------------------------------
# timing

def time_graphs(label, chains, names, floor, batch, reps):
    """Interleaved CUDA-graph timings: one graph per arm of `batch` replays, plus the launch-floor graph."""
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
        names = list(names) + ["floor"]
    for e in exes:  # warm
        cu("cuGraphLaunch", e, stream)
        cu("cuStreamSynchronize", stream)
    start, stop = P(), P()
    for ev in (start, stop):
        cu("cuEventCreate", ref(ev), 0)
    # A timing block is only reported from a window with no production request: wait for connections == 0 and verify
    # the served-token counters did not move across the block, else rerun the block.
    samples = None
    for attempt in range(opts.tries):
        c0 = None
        if not opts.no_guard:
            wait_health(opts.wait)
            c0 = served_counters()
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
        if opts.no_guard or served_counters() == c0:
            break
        global timing_retries
        timing_retries += 1
        print(f"{label}: production served during the timing block, retry {attempt + 1}/{opts.tries}", flush=True)
    else:
        print(f"{label}: WARNING no clean timing window after {opts.tries} tries", flush=True)
    for e in exes:
        cu("cuGraphExecDestroy", e)
    for ev in (start, stop):
        cu("cuEventDestroy_v2", ev)
    stats = {}
    for name, s in zip(names, samples):
        a = np.asarray(s)
        stats[name] = {"median": round(float(median(s)), 3),
                       "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3),
                       "n": len(s)}
    print(f"{label}: " + " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in stats.items()), flush=True)
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
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
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
        row["timing"] = time_graphs(label, chains, ["ref", "old", "fused", "small"], floor, opts.batch, opts.reps)
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
        row["timing"] = time_graphs(label, [chain], ["kit"], floor, opts.batch, opts.reps)
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
        row["timing"] = time_graphs(label, [chain], ["new"], floor, opts.batch, opts.reps)
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
                native(mod, "idx_fused", rows, 1024, x, L(width), width, 8, vis, every, k, P(), 0, blocks, keys, tmax,
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
    diff_old = compare(f"idx-fold old R={rows} width={width}", states[1], ref_state)
    diff_new = compare(f"idx-fold new R={rows} width={width}", states[2], ref_state)
    diff_cross = compare(f"idx-fold new-vs-old R={rows} width={width}", states[2], states[1])
    # the kit wrote keys/tmax into the same buffers on every arm: the reference is the kit's own bytes
    row = {"kernel": "0195-idx-fold", "shape": label, "rows": rows, "width": width, "nbk": nbk, "k": k,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old, "new_vs_old": diff_cross}}
    row["bytes"] = rows * width * 4 + rows * width * 8 + rows * nt * 8 + rows * k * 4
    if timing and DO_TIMING and not special:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, 1024, nbk * 4)
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
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
        row["timing"] = time_graphs(label, [chain], ["kit"], None, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# 0197 compressors

def bf(shape):
    return (rng.normal(size=shape).astype("f4").view("u4") >> 16).astype("u2")


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
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch)):
        rk, rg = map(Buffer, rings)
        flag = Buffer(np.zeros(1, "u4"))
        mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")
        prefix = (x, L(stride), gate, L(stride), slot, rk, rg, gi, L(n))

        def chain(i=i, mod=mod, prefix=prefix, dst=dst, mid=mid, flag=flag):
            if i == 0:
                native(mod, "compress2x", rows, 256, *prefix, mid, L(d), rows, d, flag)
                aot(f"norm{d}", rows, mid, d, w, dst, d + 16, eps)
            else:
                native(mod, "compress2x_norm", rows, d, *prefix, w, dst, L(d + 16), rows, d, eps, flag)

        chain()
        chains.append(chain)
        states.append([dst, rk, rg, flag])
    diff_new = compare(f"decode compressor new D={d} R={rows}", states[2], states[0])
    diff_old = compare(f"decode compressor old D={d} R={rows}", states[1], states[0])
    row = {"kernel": "0197-compress-decode", "shape": label, "rows": rows, "d": d,
           "parity": {"new_vs_ref": diff_new, "old_vs_ref": diff_old}}
    row["bytes"] = rows * d * 4 * 4 + rows * 2 * d * 4 + rows * (d + 16) * 2 + 2 * n * d * 4
    if timing and DO_TIMING and not special and layout == "mixed":
        floor = lambda: launch(opts.floor, "empty_kernel", rows, d, 0)
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
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
            else:
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
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
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
    label = f"rope {path} R={rows} D={d}{' special' if special else ''}"
    if not shape_ok(label):
        return None
    src = bf((rows, heads, d))
    if special:
        src[..., :8] = [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0, 1, 0x7F7F, 0x8001]
        src[..., 32:64] = 0x7FC1
    weights = Buffer(bf(d))
    pos = Buffer(rng.permutation(rows).astype("i8"))
    dest = Buffer(rng.permutation(rows).astype("i8"))
    angle = rng.uniform(-np.pi, np.pi, (rows, 32))
    cos, sin = Buffer(np.cos(angle).astype("f4")), Buffer(np.sin(angle).astype("f4"))
    chains, states = [], []
    for i, mod in enumerate((opts.old_torch, opts.old_torch, opts.new_torch)):
        x, rotated, dequant = Buffer(src), out(src.shape, "u2"), out(src.shape, "u2")
        codes, scales = out((rows, d // 2), "u1"), out((rows, d // (32 if key else 16)), "u1")

        def chain(i=i, mod=mod, x=x, rot=rotated, dq=dequant, codes=codes, scales=scales):
            if path == "prompt latent" and i == 0:
                native(opts.old_ops, "copy_rows16", (1, rows), 256, x, L(d // 8), rot, L(d // 8), L(d // 8))
            if i:
                if key and prompt:
                    lanes = rms_lanes(d, rows)
                    y = pack_lanes(lanes, d // 2)
                    native(mod, "fused_prompt_idx_key", rows, (lanes, y), x, L(d), weights, eps, F(1 / d),
                           cos, sin, pos, codes, scales, dest, d, 64)
                elif key:
                    native(mod, "fused_idx_key", rows, 128, x, L(d), weights, cos, sin, pos, codes,
                           scales, dest, d, 64, eps)
                elif query:
                    native(mod, "fused_rope_qd", rows * heads, 128, x, dq, cos, sin, pos, heads, d, 64)
                elif prompt:
                    native(mod, "fused_prompt_latent", rows, min(d // 2, 256), x, cos, sin, pos, codes,
                           scales, dest, d, 64)
                else:
                    native(mod, "fused_rope_store", rows, min(d // 2, 256), x, cos, sin, pos, codes, scales,
                           dest, d, 64)
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
        grid = rows * heads if query else rows
        floor = lambda: launch(opts.floor, "empty_kernel", grid, block, 0)
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
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
    native(opts.new_torch, "fused_rope_store", rows, min(d // 2, 256), rot, cos, sin, pos, codes, scales, dst, d, 64)
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
        row["timing"] = time_graphs(label, chains, ["ref", "old", "new", "newnf"], floor, opts.batch, opts.reps)
    results.append(row)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# kit _score_keys (the launch 0195's fold replaces; not part of this tree's fused path, priced for the baseline)

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
        row["timing"] = time_graphs(label, [chain], ["kit"], None, opts.batch, opts.reps)
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
PROMPT_SHAPES = [(r, d) for r in (128, 1024) for d in (512, 2048)]
ROPE_SHAPES = [("key", r) for r in (1, 7, 32, 48)] + \
              [("latent", r) for r in (1, 7, 32, 48)] + \
              [("query", r) for r in (1, 7, 32, 48)] + \
              [("prompt key", r) for r in (128, 1024)] + \
              [("prompt latent", r) for r in (128, 1024)]
PRUNE_SHAPES = [1, 7, 32, 48]
SCORE_KEYS_SHAPES = [(r, w) for r in (1, 7, 32, 128) for w in (16384, 32768, 65536)]

KERNELS = ("idx", "idx-hist", "block-max", "idx-small", "fold", "compress-decode", "compress-prompt", "key", "latent", "query", "prompt-key",
           "prompt-latent", "prune", "score-keys")

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
    if not opts.no_guard:
        avail = mem_available_gib()
        if avail < 3.0:
            sys.exit(f"MemAvailable {avail:.2f} GiB < 3 GiB, refusing")
        print(f"MemAvailable {avail:.2f} GiB", flush=True)
        lock = open(os.path.expanduser("~/tf-bench/gpu.lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        h0 = wait_health(opts.wait)
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
            free_all()
        for name, tag in (("key", "key"), ("latent", "latent"), ("query", "query"),
                          ("prompt-key", "prompt key"), ("prompt-latent", "prompt latent")):
            if not selected(name):
                continue
            for path, rows in ROPE_SHAPES:
                if path == tag:
                    run(case_rope, path, rows)
            run(case_rope, tag, 128 if tag.startswith("prompt") else 7, True, False)
            free_all()
        if selected("latent"):
            case_latent_values()
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
            continue
        s = " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in t.items())
        print(f"{row['kernel']:<24} {row['shape']:<52} {s}")
    health = {}
    if not opts.no_guard:
        h1 = health_now()
        counters1 = {k: h1.get(k) for k in ("prompt_tokens_total", "generation_tokens_total")}
        health = {"before": counters0, "after": counters1, "timing_retries": timing_retries,
                  "timing_clean": timing_retries == 0}
        if timing_retries == 0:
            print(f"health after: {counters1}  every timing block verified idle")
        else:
            print(f"health after: {counters1}  {timing_retries} timing block(s) retried for a served request")
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


def served_counters():
    h = health_now()
    return (h.get("prompt_tokens_total"), h.get("generation_tokens_total"))


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
        conns = h.get("live", {}).get("connections")
        if conns == 0:
            print(f"health: connections={conns}", flush=True)
            return h
        print(f"production serving ({conns} connections), waiting", flush=True)
        if deadline is not None and time.time() > deadline:
            sys.exit("production stayed busy, refusing")
        time.sleep(20)


if __name__ == "__main__":
    main()
